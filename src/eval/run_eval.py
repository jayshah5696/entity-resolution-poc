"""
run_eval.py -- Evaluate a dense embedding index across all 6 corruption buckets.

Usage:
    uv run python src/eval/run_eval.py \\
        --model gte_modernbert_base \\
        --index-dir results/indexes/gte_modernbert_base_pipe_fp32 \\
        --eval-queries data/eval/eval_queries.parquet \\
        --output results/004_gte_modernbert_pipe_fp32.json \\
        --models-config configs/models.yaml \\
        --eval-config configs/eval.yaml \\
        --serialization pipe \\
        --top-k 10 \\
        --experiment-id 004 \\
        --device mps

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
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


from src.eval.metrics import aggregate_metrics, compute_metrics
from src.models.encoder import load_encoder

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LanceDB search helpers
# ---------------------------------------------------------------------------


def search_batch(table, query_vecs: np.ndarray, top_k: int) -> list[list[str]]:
    """
    Run ANN search for a batch of query vectors.

    Returns a list of lists: for each query, an ordered list of retrieved entity_ids.
    """
    results_per_query: list[list[str]] = []
    for vec in query_vecs:
        result_df = (
            table.search(vec.tolist())
            .limit(top_k)
            .disable_scoring_autoprojection()
            .select(["entity_id"])
            .to_pandas()
        )
        results_per_query.append(result_df["entity_id"].tolist())
    return results_per_query


def search_single(table, query_vec: np.ndarray, top_k: int) -> list[str]:
    """Run ANN search for a single query vector. Returns ordered entity_ids."""
    result_df = (
        table.search(query_vec.tolist())
        .limit(top_k)
        .disable_scoring_autoprojection()
        .select(["entity_id"])
        .to_pandas()
    )
    return result_df["entity_id"].tolist()


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


def measure_latency_dense(
    encoder,
    table,
    queries: list[str],
    top_k: int,
    n_warmup: int = 100,
    n_measure: int = 1000,
    encode_batch_size: int = 32,
) -> dict[str, float]:
    """
    Measure single-query latency end-to-end (encode + search) in ms.

    Warmup phase discarded. Returns p50, p95, p99, n_queries.
    """
    all_queries = queries
    warmup_texts = all_queries[: min(n_warmup, len(all_queries))]
    measure_texts = all_queries[: min(n_measure, len(all_queries))]

    # Warmup
    for qt in warmup_texts:
        vec = encoder.encode_queries([qt], batch_size=1)
        search_single(table, vec[0], top_k)

    # Timed
    latencies_ms: list[float] = []
    for qt in measure_texts:
        t0 = time.perf_counter()
        vec = encoder.encode_queries([qt], batch_size=1)
        search_single(table, vec[0], top_k)
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


def evaluate_bucket_dense(
    encoder,
    table,
    bucket_df: pl.DataFrame,
    query_col: str,
    top_k: int,
    encode_batch_size: int = 32,
) -> dict[str, float]:
    """
    Encode all queries and retrieve top-k results, then compute metrics.
    Uses batched encoding for throughput; single-query search for correct ranks.
    """
    query_texts = bucket_df[query_col].to_list()
    ground_truth_ids = bucket_df["ground_truth_entity_id"].to_list()
    n = len(query_texts)

    # Encode all queries in batches
    all_vecs: list[np.ndarray] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Encoding {n} queries", total=n)
        for i in range(0, n, encode_batch_size):
            batch = query_texts[i : i + encode_batch_size]
            vecs = encoder.encode_queries(batch, batch_size=encode_batch_size)
            all_vecs.append(vecs)
            progress.advance(task, len(batch))

    query_vecs = np.concatenate(all_vecs, axis=0)

    # Search and collect metrics
    per_query: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Searching LanceDB", total=n)
        for i, (vec, gt_id) in enumerate(zip(query_vecs, ground_truth_ids)):
            retrieved_ids = search_single(table, vec, top_k)
            metrics = compute_metrics(retrieved_ids, gt_id)
            per_query.append(metrics)
            progress.advance(task, 1)

    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a dense embedding index across all 6 corruption buckets."
    )
    parser.add_argument("--model", required=True, help="Model key from models.yaml")
    parser.add_argument("--index-dir", required=True, help="Directory with LanceDB index")
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
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Compute device"
    )
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Query encoding batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Load configs ----
    with open(args.models_config) as f:
        all_model_cfg = yaml.safe_load(f)
    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    if args.model not in all_model_cfg:
        console.print(f"[red]Model '{args.model}' not found in {args.models_config}")
        sys.exit(1)

    model_cfg = all_model_cfg[args.model]
    buckets: list[str] = eval_cfg["buckets"]
    top_k = args.top_k
    index_dir = Path(args.index_dir)

    # ---- Load index metadata ----
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]metadata.json not found in {index_dir}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    dim = metadata.get("dim", 0)
    quantization = metadata.get("quantization", "fp32")
    n_records = metadata.get("n_records", 0)
    index_size_mb = metadata.get("index_size_mb", 0.0)

    console.print(
        f"[cyan]Index: model={args.model} dim={dim} quant={quantization} n={n_records:,}"
    )

    # ---- Open LanceDB table ----
    import lancedb

    console.print(f"[bold cyan]Opening LanceDB at {index_dir}...")
    db = lancedb.connect(str(index_dir))
    table = db.open_table("index")
    console.print(f"[green]LanceDB table opened. Rows: {table.count_rows():,}")

    # ---- Load encoder (queries only) ----
    console.print(f"[bold cyan]Loading encoder: {args.model}...")
    encoder = load_encoder(
        model_key=args.model,
        model_cfg=model_cfg,
        device=args.device,
        model_path=args.model_path,
    )

    # ---- Load eval queries ----
    console.print(f"[bold cyan]Loading eval queries from {args.eval_queries}...")
    df = pl.read_parquet(args.eval_queries)
    console.print(f"[cyan]Loaded {len(df):,} queries. Columns: {df.columns}")

    # ---- Select query column ----
    query_col = f"query_text_{args.serialization}"
    if query_col not in df.columns:
        if args.serialization == "kv" and "query_text_keyvalue" in df.columns:
            query_col = "query_text_keyvalue"
        else:
            console.print(f"[red]Column '{query_col}' not found. Available: {df.columns}")
            sys.exit(1)

    console.print(f"[cyan]Using query column: '{query_col}'")

    # ---- Per-bucket evaluation ----
    per_bucket_metrics: dict[str, dict] = {}
    all_per_query: list[dict] = []
    all_latency_info: list[dict] = []

    for bucket in buckets:
        bucket_df = df.filter(pl.col("bucket") == bucket)
        n_bucket = len(bucket_df)
        if n_bucket == 0:
            console.print(f"[yellow]Bucket '{bucket}' has no queries, skipping.")
            continue

        console.print(f"\n[bold cyan]Bucket: '{bucket}' ({n_bucket:,} queries)")

        # Latency measurement (single-query end-to-end)
        query_texts = bucket_df[query_col].to_list()
        latency_info = measure_latency_dense(
            encoder=encoder,
            table=table,
            queries=query_texts,
            top_k=top_k,
            n_warmup=min(100, n_bucket),
            n_measure=min(1000, n_bucket),
            encode_batch_size=args.batch_size,
        )

        # Full metrics on all queries
        bucket_metrics = evaluate_bucket_dense(
            encoder=encoder,
            table=table,
            bucket_df=bucket_df,
            query_col=query_col,
            top_k=top_k,
            encode_batch_size=args.batch_size,
        )

        # Collect per-query for overall metrics
        gt_ids = bucket_df["ground_truth_entity_id"].to_list()
        q_texts = bucket_df[query_col].to_list()
        for qt_text, gt_id in zip(q_texts, gt_ids):
            vec = encoder.encode_queries([qt_text], batch_size=1)
            retrieved = search_single(table, vec[0], top_k)
            all_per_query.append(compute_metrics(retrieved, gt_id))

        all_latency_info.append(latency_info)

        per_bucket_metrics[bucket] = {
            **bucket_metrics,
            "n_queries": n_bucket,
            "latency_ms": latency_info,
        }

        console.print(
            f"  [green]R@5={bucket_metrics.get('recall_at_5', 0):.3f}  "
            f"R@10={bucket_metrics.get('recall_at_10', 0):.3f}  "
            f"MRR={bucket_metrics.get('mrr_at_10', 0):.3f}  "
            f"p50={latency_info['p50']:.2f}ms"
        )

    # ---- Overall metrics ----
    overall_metrics = aggregate_metrics(all_per_query)

    # Combined latency across buckets
    if all_latency_info:
        all_p50 = [li["p50"] for li in all_latency_info]
        all_p95 = [li["p95"] for li in all_latency_info]
        all_p99 = [li["p99"] for li in all_latency_info]
        total_n = sum(li["n_queries"] for li in all_latency_info)
        overall_latency = {
            "p50": float(np.mean(all_p50)),
            "p95": float(np.mean(all_p95)),
            "p99": float(np.mean(all_p99)),
            "n_queries": total_n,
        }
    else:
        overall_latency = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n_queries": 0}

    # ---- Print summary table ----
    table_out = Table(title=f"Dense Eval: {args.model}", show_header=True)
    table_out.add_column("Bucket")
    table_out.add_column("R@5", style="cyan")
    table_out.add_column("R@10", style="bold cyan")
    table_out.add_column("MRR@10", style="yellow")
    table_out.add_column("NDCG@10", style="magenta")
    table_out.add_column("p50ms")

    for bucket, bm in per_bucket_metrics.items():
        table_out.add_row(
            bucket,
            f"{bm.get('recall_at_5', 0):.3f}",
            f"{bm.get('recall_at_10', 0):.3f}",
            f"{bm.get('mrr_at_10', 0):.3f}",
            f"{bm.get('ndcg_at_10', 0):.3f}",
            f"{bm.get('latency_ms', {}).get('p50', 0):.2f}",
        )

    table_out.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_metrics.get('recall_at_5', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('mrr_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('ndcg_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_latency['p50']:.2f}[/bold]",
    )
    console.print(table_out)

    # ---- Build output JSON (ADR-003 schema) ----
    per_bucket_clean: dict[str, dict] = {}
    per_bucket_latency_clean: dict[str, dict] = {}
    for bucket, bm in per_bucket_metrics.items():
        per_bucket_clean[bucket] = {
            k: v for k, v in bm.items() if k not in ("latency_ms",)
        }
        per_bucket_latency_clean[bucket] = bm.get("latency_ms", {})

    # Mode: zero_shot unless a fine-tuned model path was given
    mode = "fine_tuned" if args.model_path else "zero_shot"
    dims_cfg = model_cfg.get("dims")
    dims_val = int(dims_cfg[0]) if isinstance(dims_cfg, list) and dims_cfg else dim

    result = {
        "experiment_id": args.experiment_id,
        "model": args.model,
        "serialization": args.serialization,
        "mode": mode,
        "quantization": quantization,
        "dims": dims_val,
        "index_size": n_records,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": overall_metrics,
        "per_bucket": per_bucket_clean,
        "latency_ms": overall_latency,
        "per_bucket_latency_ms": per_bucket_latency_clean,
        "index_size_mb": index_size_mb,
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
