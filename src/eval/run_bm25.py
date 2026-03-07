"""
run_bm25.py -- Evaluate a BM25 index across all 6 corruption buckets.

Usage:
    uv run python src/eval/run_bm25.py \\
        --index-dir results/indexes/bm25_pipe \\
        --eval-queries data/eval/eval_queries.parquet \\
        --output results/001_bm25_pipe.json \\
        --models-config configs/models.yaml \\
        --eval-config configs/eval.yaml \\
        --serialization pipe \\
        --top-k 10 \\
        --experiment-id 001

Output JSON matches the ADR-003 schema used by aggregate.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import aggregate_metrics, compute_metrics
from src.models.encoder import BM25Encoder, load_encoder

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_latency(
    encoder: BM25Encoder,
    queries: list[str],
    top_k: int,
    n_warmup: int = 100,
    n_measure: int = 1000,
) -> dict[str, float]:
    """
    Measure per-query latency in milliseconds (p50, p95, p99).

    Runs n_warmup queries without timing, then times n_measure queries
    using single-query calls (not batched) for realistic latency numbers.
    """
    n_all = len(queries)
    warmup_queries = queries[: min(n_warmup, n_all)]
    measure_queries = queries[: min(n_measure, n_all)]

    # Warmup
    for q in warmup_queries:
        tokens = encoder.tokenize(q)
        encoder.search(tokens, n=top_k)

    # Timed
    latencies_ms: list[float] = []
    for q in measure_queries:
        t0 = time.perf_counter()
        tokens = encoder.tokenize(q)
        encoder.search(tokens, n=top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_queries": len(latencies_ms),
    }


# ---------------------------------------------------------------------------
# Bucket evaluation
# ---------------------------------------------------------------------------


def evaluate_bucket(
    encoder: BM25Encoder,
    bucket_df: pl.DataFrame,
    query_col: str,
    top_k: int,
) -> dict[str, float]:
    """
    Compute retrieval metrics for all queries in a single bucket.

    Retrieves top_k results for each query, computes per-query metrics,
    then aggregates (mean) across all queries.
    """
    query_texts = bucket_df[query_col].to_list()
    ground_truth_ids = bucket_df["ground_truth_entity_id"].to_list()

    per_query: list[dict] = []
    for qt, gt_id in zip(query_texts, ground_truth_ids):
        tokens = encoder.tokenize(qt)
        results = encoder.search(tokens, n=top_k)
        retrieved_ids = [eid for eid, _score in results]
        metrics = compute_metrics(retrieved_ids, gt_id, ks=[1, 5, 10])
        per_query.append(metrics)

    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BM25 retrieval index across all corruption buckets."
    )
    parser.add_argument("--index-dir", required=True, help="Directory with bm25.pkl and entity_ids.json")
    parser.add_argument("--eval-queries", required=True, help="Parquet file with eval queries")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument(
        "--serialization", required=True, choices=["pipe", "kv"], help="Serialization format"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument(
        "--experiment-id", default="001", help="Experiment ID for the results JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load configs ----
    with open(args.models_config) as f:
        all_model_cfg = yaml.safe_load(f)
    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    model_key = "bm25_baseline"
    model_cfg = all_model_cfg[model_key]
    buckets: list[str] = eval_cfg["buckets"]
    top_k = args.top_k

    # ---- Load BM25 index ----
    index_dir = Path(args.index_dir)
    bm25_path = index_dir / "bm25.pkl"
    entity_ids_path = index_dir / "entity_ids.json"

    if not bm25_path.exists():
        console.print(f"[red]BM25 index not found: {bm25_path}")
        sys.exit(1)

    console.print(f"[bold cyan]Loading BM25 index from {bm25_path}...")
    encoder = BM25Encoder(model_key=model_key, model_cfg=model_cfg)
    encoder.load(str(bm25_path), str(entity_ids_path))
    n_records = len(encoder._entity_ids)
    console.print(f"[green]Loaded BM25 index with {n_records:,} documents.")

    # ---- Load eval queries ----
    console.print(f"[bold cyan]Loading eval queries from {args.eval_queries}...")
    df = pl.read_parquet(args.eval_queries)
    console.print(f"[cyan]Loaded {len(df):,} queries. Columns: {df.columns}")

    # ---- Select query column ----
    query_col = f"query_text_{args.serialization}"
    if query_col not in df.columns:
        # Try alternate column names
        if args.serialization == "kv" and "query_text_keyvalue" in df.columns:
            query_col = "query_text_keyvalue"
        else:
            console.print(f"[red]Column '{query_col}' not found. Available: {df.columns}")
            sys.exit(1)

    console.print(f"[cyan]Using query column: '{query_col}'")

    # ---- Index size ----
    import os
    index_size_bytes = sum(
        Path(root, f).stat().st_size
        for root, dirs, files in os.walk(index_dir)
        for f in files
    )
    index_size_mb = index_size_bytes / (1024 * 1024)

    # ---- Per-bucket evaluation ----
    per_bucket_metrics: dict[str, dict] = {}
    all_latency_ms: list[float] = []

    for bucket in buckets:
        bucket_df = df.filter(pl.col("bucket") == bucket)
        n_bucket = len(bucket_df)
        if n_bucket == 0:
            console.print(f"[yellow]Bucket '{bucket}' has no queries, skipping.")
            continue

        console.print(f"\n[bold cyan]Evaluating bucket: '{bucket}' ({n_bucket:,} queries)")

        # Timed eval for latency (on all queries in bucket; cap at n_measure)
        query_texts = bucket_df[query_col].to_list()
        n_warmup = min(100, n_bucket)
        n_measure = min(1000, n_bucket)

        latency_info = measure_latency(
            encoder=encoder,
            queries=query_texts,
            top_k=top_k,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        all_latency_ms.extend(
            # Approximation: use measure results as representative sample
            [latency_info["p50"]] * latency_info["n_queries"]
        )

        # Full eval on all queries
        bucket_metrics = evaluate_bucket(
            encoder=encoder,
            bucket_df=bucket_df,
            query_col=query_col,
            top_k=top_k,
        )
        per_bucket_metrics[bucket] = {
            **bucket_metrics,
            "n_queries": n_bucket,
            "latency_ms": latency_info,
        }

        console.print(
            f"  [green]R@1={bucket_metrics.get('recall_at_1', 0):.3f}  "
            f"R@10={bucket_metrics.get('recall_at_10', 0):.3f}  "
            f"MRR={bucket_metrics.get('mrr_at_10', 0):.3f}  "
            f"p50={latency_info['p50']:.2f}ms"
        )

    # ---- Overall metrics (mean across all buckets, weighted by n_queries) ----
    all_query_metrics: list[dict] = []
    for bucket in buckets:
        bucket_df = df.filter(pl.col("bucket") == bucket)
        if len(bucket_df) == 0:
            continue
        query_texts = bucket_df[query_col].to_list()
        ground_truth_ids = bucket_df["ground_truth_entity_id"].to_list()
        for qt, gt_id in zip(query_texts, ground_truth_ids):
            tokens = encoder.tokenize(qt)
            results = encoder.search(tokens, n=top_k)
            retrieved_ids = [eid for eid, _ in results]
            all_query_metrics.append(compute_metrics(retrieved_ids, gt_id, ks=[1, 5, 10]))

    overall_metrics = aggregate_metrics(all_query_metrics)

    # Combine latency across buckets
    if all_latency_ms:
        arr = np.array(all_latency_ms)
        overall_latency = {
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "n_queries": len(arr),
        }
    else:
        overall_latency = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n_queries": 0}

    # ---- Print summary table ----
    table = Table(title="BM25 Eval Results", show_header=True)
    table.add_column("Bucket")
    table.add_column("R@1", style="green")
    table.add_column("R@5", style="cyan")
    table.add_column("R@10", style="bold cyan")
    table.add_column("MRR@10", style="yellow")
    table.add_column("NDCG@10", style="magenta")
    table.add_column("p50ms")

    for bucket, bm in per_bucket_metrics.items():
        table.add_row(
            bucket,
            f"{bm.get('recall_at_1', 0):.3f}",
            f"{bm.get('recall_at_5', 0):.3f}",
            f"{bm.get('recall_at_10', 0):.3f}",
            f"{bm.get('mrr_at_10', 0):.3f}",
            f"{bm.get('ndcg_at_10', 0):.3f}",
            f"{bm.get('latency_ms', {}).get('p50', 0):.2f}",
        )

    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_metrics.get('recall_at_1', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_5', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('mrr_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('ndcg_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_latency['p50']:.2f}[/bold]",
    )
    console.print(table)

    # ---- Build output JSON (ADR-003 schema) ----
    # Strip latency_ms sub-dict and n_queries from per_bucket for clean schema
    per_bucket_clean: dict[str, dict] = {}
    per_bucket_latency: dict[str, dict] = {}
    for bucket, bm in per_bucket_metrics.items():
        per_bucket_clean[bucket] = {
            k: v for k, v in bm.items() if k not in ("latency_ms", "n_queries")
        }
        per_bucket_clean[bucket]["n_queries"] = bm.get("n_queries", 0)
        per_bucket_latency[bucket] = bm.get("latency_ms", {})

    result = {
        "experiment_id": args.experiment_id,
        "model": model_key,
        "serialization": args.serialization,
        "mode": "zero_shot",
        "quantization": "none",
        "dims": 0,
        "index_size": n_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": overall_metrics,
        "per_bucket": per_bucket_clean,
        "latency_ms": overall_latency,
        "per_bucket_latency_ms": per_bucket_latency,
        "index_size_mb": round(index_size_mb, 2),
    }

    # ---- Write output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    console.print(f"\n[bold green]Results written -> {output_path}")
    console.print(
        f"[bold green]Overall R@10={overall_metrics.get('recall_at_10', 0):.3f}  "
        f"MRR@10={overall_metrics.get('mrr_at_10', 0):.3f}"
    )


if __name__ == "__main__":
    main()
