"""
run_bm25.py -- Evaluate a BM25 (LanceDB FTS) index across all 6 corruption buckets.

The BM25 index is stored in LanceDB using native FTS (Tantivy). No pickle files.
Same storage format as the dense vector indexes.

Usage:
    uv run python src/eval/run_bm25.py \\
        --index-dir results/indexes/bm25_pipe \\
        --eval-queries data/eval/eval_queries.parquet \\
        --output results/001_bm25_pipe.json \\
        --serialization pipe \\
        --experiment-id 001

Optimizations:
    - LanceDB FTS queries are already fast (Tantivy inverted index)
    - Parallel metric computation via joblib
    - tqdm progress per bucket
    - Overall metrics from cached per-query results (no re-run)

Output JSON matches ADR-003 schema used by aggregate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from joblib import Parallel, delayed
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import aggregate_metrics, compute_metrics

console = Console()

BUCKETS = [
    "pristine",
    "missing_firstname",
    "missing_email_company",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query_fts(table, query_text: str, top_k: int) -> list[str]:
    """Run a single FTS query and return list of entity_ids."""
    results = (
        table.search(query_text, query_type="fts")
        .limit(top_k)
        .to_list()
    )
    return [r["entity_id"] for r in results]


def _eval_one(
    query_text: str,
    ground_truth_id: str,
    retrieved_ids: list[str],
    top_k: int,
) -> dict:
    return compute_metrics(retrieved_ids, ground_truth_id, ks=[5, 10])


# ---------------------------------------------------------------------------
# Bucket evaluation
# ---------------------------------------------------------------------------


def evaluate_bucket(
    table,
    query_texts: list[str],
    ground_truth_ids: list[str],
    top_k: int,
    n_jobs: int,
    bucket_name: str,
) -> list[dict]:
    """
    Run FTS queries for all queries in a bucket, then compute metrics in parallel.
    """
    # Run all queries sequentially (LanceDB FTS is already fast via Tantivy)
    retrieved_all: list[list[str]] = []
    for q in tqdm(query_texts, desc=f"  FTS {bucket_name}", leave=False):
        retrieved_all.append(_query_fts(table, q, top_k))

    # Parallel metric computation
    per_query = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_eval_one)(
            query_texts[i], ground_truth_ids[i], retrieved_all[i], top_k
        )
        for i in tqdm(range(len(query_texts)), desc=f"  Metrics {bucket_name}", leave=False)
    )
    return per_query


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_latency(
    table,
    query_texts: list[str],
    top_k: int,
    n_warmup: int = 100,
    n_measure: int = 1000,
) -> dict:
    n_all = len(query_texts)
    warmup = query_texts[: min(n_warmup, n_all)]
    measure = query_texts[: min(n_measure, n_all)]

    for q in warmup:
        _query_fts(table, q, top_k)

    latencies_ms: list[float] = []
    for q in measure:
        t0 = time.perf_counter()
        _query_fts(table, q, top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_queries": len(latencies_ms),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LanceDB FTS (BM25) index across all corruption buckets."
    )
    parser.add_argument("--index-dir", required=True, help="LanceDB index directory")
    parser.add_argument("--eval-queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--serialization", required=True, choices=["pipe", "kv"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--experiment-id", default="001")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for metric computation. -1 = all cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    top_k = args.top_k
    buckets = eval_cfg.get("buckets", BUCKETS)

    # Open LanceDB FTS index
    import lancedb

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        console.print(f"[red]Index directory not found: {index_dir}")
        sys.exit(1)

    console.print(f"[bold cyan]Opening LanceDB FTS index at {index_dir}...")
    db = lancedb.connect(str(index_dir))
    table = db.open_table("index")
    n_records = table.count_rows()
    console.print(f"[green]Loaded FTS index with {n_records:,} documents.")

    # Load eval queries
    console.print(f"[bold cyan]Loading eval queries from {args.eval_queries}...")
    df = pl.read_parquet(args.eval_queries)
    console.print(f"[cyan]Loaded {len(df):,} queries across {df['bucket'].n_unique()} buckets.")

    query_col = f"query_text_{args.serialization}"
    if query_col not in df.columns:
        console.print(f"[red]Column '{query_col}' not found. Available: {df.columns}")
        sys.exit(1)
    console.print(f"[cyan]Query column: '{query_col}' | n_jobs: {args.n_jobs}")

    index_size_mb = sum(
        Path(root, fname).stat().st_size
        for root, _, files in os.walk(index_dir)
        for fname in files
    ) / (1024 * 1024)

    # Per-bucket evaluation
    per_bucket_metrics: dict[str, dict] = {}
    all_per_query: list[dict] = []

    for bucket in buckets:
        bucket_df = df.filter(pl.col("bucket") == bucket)
        n_bucket = len(bucket_df)
        if n_bucket == 0:
            console.print(f"[yellow]Bucket '{bucket}' empty, skipping.")
            continue

        console.print(f"\n[bold cyan]Bucket: '{bucket}' ({n_bucket:,} queries)")
        query_texts = bucket_df[query_col].to_list()
        ground_truth_ids = bucket_df["ground_truth_entity_id"].to_list()

        latency_info = measure_latency(table, query_texts, top_k)
        per_query = evaluate_bucket(
            table=table,
            query_texts=query_texts,
            ground_truth_ids=ground_truth_ids,
            top_k=top_k,
            n_jobs=args.n_jobs,
            bucket_name=bucket,
        )
        all_per_query.extend(per_query)

        bucket_agg = aggregate_metrics(per_query)
        per_bucket_metrics[bucket] = {
            **bucket_agg,
            "n_queries": n_bucket,
            "latency_ms": latency_info,
        }

        console.print(
            f"  [green]R@1={bucket_agg.get('recall_at_1', 0):.3f}  "
            f"R@10={bucket_agg.get('recall_at_10', 0):.3f}  "
            f"MRR={bucket_agg.get('mrr_at_10', 0):.3f}  "
            f"p50={latency_info['p50']:.1f}ms"
        )

    overall_metrics = aggregate_metrics(all_per_query)
    latency_vals = [m.get("latency_ms", {}).get("p50", 0) for m in per_bucket_metrics.values()]
    arr = np.array(latency_vals) if latency_vals else np.array([0.0])
    overall_latency = {
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_queries": len(all_per_query),
    }

    # Summary table
    table_out = Table(title="BM25 (LanceDB FTS) Eval Results", show_header=True)
    for col in ["Bucket", "R@1", "R@5", "R@10", "MRR@10", "NDCG@10", "p50ms"]:
        table_out.add_column(col)
    for bucket, bm in per_bucket_metrics.items():
        table_out.add_row(
            bucket,
            f"{bm.get('recall_at_1', 0):.3f}",
            f"{bm.get('recall_at_5', 0):.3f}",
            f"{bm.get('recall_at_10', 0):.3f}",
            f"{bm.get('mrr_at_10', 0):.3f}",
            f"{bm.get('ndcg_at_10', 0):.3f}",
            f"{bm.get('latency_ms', {}).get('p50', 0):.1f}",
        )
    table_out.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_metrics.get('recall_at_1', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_5', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('mrr_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('ndcg_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_latency['p50']:.1f}[/bold]",
    )
    console.print(table_out)

    # ADR-003 schema output
    per_bucket_clean: dict[str, dict] = {}
    per_bucket_latency_ms: dict[str, dict] = {}
    for bucket, bm in per_bucket_metrics.items():
        per_bucket_clean[bucket] = {k: v for k, v in bm.items() if k != "latency_ms"}
        per_bucket_latency_ms[bucket] = bm.get("latency_ms", {})

    result = {
        "experiment_id": args.experiment_id,
        "model": "bm25_baseline",
        "index_type": "lancedb_fts",
        "serialization": args.serialization,
        "mode": "zero_shot",
        "quantization": None,
        "dims": None,
        "index_size": n_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": overall_metrics,
        "per_bucket": per_bucket_clean,
        "latency_ms": overall_latency,
        "per_bucket_latency_ms": per_bucket_latency_ms,
        "index_size_mb": round(index_size_mb, 2),
    }

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
