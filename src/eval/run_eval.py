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

Optimizations (vs. original):
    - Fix #1: No redundant third encode+search pass — evaluate_bucket_dense()
      returns per-query metrics directly, reused for overall aggregation.
    - Fix #2: LanceDB batch vector search — single call per bucket instead of
      N individual Python→Rust round-trips.
    - Fix #3: Parallel metric computation via joblib (pattern from run_bm25.py).
    - Fix #6: Device-aware default batch size (128 for MPS, 32 for CPU).
    - Fix #8: Default device is 'mps' (not 'cpu') since this runs on Apple Silicon.
    - 2026 Optimization: Native quantization support for queries and direct numpy search.
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
from joblib import Parallel, delayed
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
from src.eval.build_index import apply_quantization
from src.models.encoder import load_encoder

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LanceDB search helpers
# ---------------------------------------------------------------------------


def _search_batch_chunk(table, query_vecs: np.ndarray, top_k: int) -> list[list[str]]:
    """
    Run ANN search for a chunk of query vectors using LanceDB native batch search.

    Returns a list of lists: for each query, an ordered list of retrieved entity_ids.
    """
    n_queries = len(query_vecs)

    # 2026 Optimization: Pass numpy array directly, no .tolist()
    result_df = (
        table.search(query_vecs)
        .limit(top_k)
        .select(["entity_id", "_distance"])
        .to_pandas()
    )

    # Group results by query_index and extract entity_ids in order
    results_per_query: list[list[str]] = [[] for _ in range(n_queries)]
    if "query_index" in result_df.columns and len(result_df) > 0:
        for qi, group in result_df.groupby("query_index", sort=True):
            results_per_query[int(qi)] = group["entity_id"].tolist()
    elif n_queries == 1 and len(result_df) > 0:
        # Single-vector search doesn't include query_index
        results_per_query[0] = result_df["entity_id"].tolist()

    return results_per_query


def search_batch(
    table, query_vecs: np.ndarray, top_k: int, chunk_size: int = 512
) -> list[list[str]]:
    """
    Run ANN batch search, chunked to avoid LanceDB/Lance file-descriptor exhaustion.
    """
    n_queries = len(query_vecs)
    if n_queries <= chunk_size:
        return _search_batch_chunk(table, query_vecs, top_k)

    results_per_query: list[list[str]] = []
    for i in range(0, n_queries, chunk_size):
        chunk = query_vecs[i : i + chunk_size]
        chunk_results = _search_batch_chunk(table, chunk, top_k)
        results_per_query.extend(chunk_results)

    return results_per_query


def search_single(table, query_vec: np.ndarray, top_k: int) -> list[str]:
    """Run ANN search for a single query vector. Returns ordered entity_ids."""
    # 2026 Optimization: Pass numpy array directly
    result_df = (
        table.search(query_vec)
        .limit(top_k)
        .select(["entity_id", "_distance"])
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
    quantization: str = "fp32",
    n_warmup: int = 100,
    n_measure: int = 1000,
    encode_batch_size: int = 32,
) -> dict[str, float]:
    """
    Measure single-query latency end-to-end (encode + search) in ms.
    """
    all_queries = queries
    warmup_texts = all_queries[: min(n_warmup, len(all_queries))]
    measure_texts = all_queries[: min(n_measure, len(all_queries))]

    # Warmup
    for qt in warmup_texts:
        vec = encoder.encode_queries([qt], batch_size=1)
        vec_q = apply_quantization(vec, quantization)
        search_single(table, vec_q[0], top_k)

    # Timed
    latencies_ms: list[float] = []
    for qt in measure_texts:
        t0 = time.perf_counter()
        vec = encoder.encode_queries([qt], batch_size=1)
        vec_q = apply_quantization(vec, quantization)
        search_single(table, vec_q[0], top_k)
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
    quantization: str = "fp32",
    encode_batch_size: int = 128,
    n_jobs: int = -1,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Encode all queries and retrieve top-k results, then compute metrics.
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
    
    # Apply same quantization as index
    query_vecs = apply_quantization(query_vecs, quantization)

    # Batch search: single LanceDB call for all vectors
    console.print(f"  [cyan]Batch searching {n:,} vectors...")
    t0 = time.perf_counter()
    retrieved_all = search_batch(table, query_vecs, top_k)
    search_time = time.perf_counter() - t0
    console.print(f"  [green]Batch search done in {search_time:.1f}s")

    # Parallel metric computation (pattern from run_bm25.py)
    per_query: list[dict[str, float]] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_metrics)(retrieved_all[i], ground_truth_ids[i])
        for i in range(n)
    )

    return aggregate_metrics(per_query), per_query


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
    # Fix #8: Default to 'mps' since this project targets Apple Silicon
    parser.add_argument(
        "--device", default="mps", choices=["cpu", "cuda", "mps"], help="Compute device"
    )
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model checkpoint")
    # Fix #6: Device-aware batch size — None means auto (128 for MPS/CUDA, 32 for CPU)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Query encoding batch size (default: 128 for MPS/CUDA, 32 for CPU)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for metric computation. -1 = all cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Fix #6: Resolve batch size based on device
    if args.batch_size is None:
        args.batch_size = 128 if args.device in ("mps", "cuda") else 32
    console.print(f"[cyan]Batch size: {args.batch_size} (device={args.device})")

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
    # If index dim < encoder's native dim, truncate queries to match (MRL)
    encoder_native_dims = model_cfg.get("dims", [])
    if encoder_native_dims and dim < encoder_native_dims[0]:
        truncate_dim = dim
        console.print(f"[cyan]MRL truncation: encoder {encoder_native_dims[0]}d -> index {dim}d")
    else:
        truncate_dim = None

    console.print(f"[bold cyan]Loading encoder: {args.model}...")
    encoder = load_encoder(
        model_key=args.model,
        model_cfg=model_cfg,
        device=args.device,
        model_path=args.model_path,
        truncate_dim=truncate_dim,
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

    total_t0 = time.perf_counter()

    for bucket in buckets:
        bucket_df = df.filter(pl.col("bucket") == bucket)
        n_bucket = len(bucket_df)
        if n_bucket == 0:
            console.print(f"[yellow]Bucket '{bucket}' has no queries, skipping.")
            continue

        console.print(f"\n[bold cyan]Bucket: '{bucket}' ({n_bucket:,} queries)")

        # Latency measurement (single-query end-to-end — intentionally not batched)
        query_texts = bucket_df[query_col].to_list()
        latency_info = measure_latency_dense(
            encoder=encoder,
            table=table,
            queries=query_texts,
            top_k=top_k,
            quantization=quantization,
            n_warmup=min(100, n_bucket),
            n_measure=min(1000, n_bucket),
            encode_batch_size=args.batch_size,
        )

        # Fix #1 + #2 + #3: Full metrics on all queries — returns per-query list
        # so we can extend all_per_query without a redundant third pass.
        bucket_metrics, per_query = evaluate_bucket_dense(
            encoder=encoder,
            table=table,
            bucket_df=bucket_df,
            query_col=query_col,
            top_k=top_k,
            quantization=quantization,
            encode_batch_size=args.batch_size,
            n_jobs=args.n_jobs,
        )

        # Reuse per-query results for overall aggregation (no third pass!)
        all_per_query.extend(per_query)

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

    total_elapsed = time.perf_counter() - total_t0

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
    console.print(f"[bold cyan]Total eval time: {total_elapsed:.1f}s")

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
