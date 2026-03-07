"""
run_bm25.py -- Evaluate a BM25 index across all 6 corruption buckets.

Usage:
    uv run python src/eval/run_bm25.py \\
        --index-dir results/indexes/bm25_pipe \\
        --eval-queries data/eval/eval_queries.parquet \\
        --output results/001_bm25_pipe.json \\
        --serialization pipe \\
        --experiment-id 001

Optimizations:
    - Uses get_scores() + np.argpartition instead of get_top_n (no full sort)
    - Parallelizes across queries with joblib (uses all CPU cores by default)
    - tqdm progress bar per bucket
    - Overall metrics computed from cached per-query results, not re-run

Output JSON matches the ADR-003 schema used by aggregate.py.
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
from src.models.encoder import BM25Encoder

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
# Per-query worker (called in parallel)
# ---------------------------------------------------------------------------


def _score_one(
    query_tokens: list[str],
    entity_ids: list[str],
    scores: np.ndarray,
    ground_truth_id: str,
    top_k: int,
) -> dict:
    """
    Given precomputed BM25 scores for one query, extract top-k entity IDs
    and compute retrieval metrics.
    """
    n = min(top_k, len(scores))
    # argpartition is O(n), much faster than argsort for large n
    top_indices = np.argpartition(scores, -n)[-n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    retrieved_ids = [entity_ids[i] for i in top_indices]
    return compute_metrics(retrieved_ids, ground_truth_id, ks=[1, 5, 10])


def evaluate_bucket_parallel(
    bm25_obj,
    entity_ids: list[str],
    query_texts: list[str],
    ground_truth_ids: list[str],
    top_k: int,
    n_jobs: int,
    bucket_name: str,
) -> list[dict]:
    """
    Evaluate all queries in a bucket using joblib parallelism.

    Pre-computes get_scores() for all queries first (numpy, fast),
    then distributes metric computation across cores.
    """
    n = len(query_texts)

    # Tokenize all queries upfront (fast, single-threaded is fine)
    tokenized = [q.lower().split() for q in
                 tqdm(query_texts, desc=f"  Tokenizing {bucket_name}", leave=False)]

    # Compute BM25 scores for all queries with progress bar
    # get_scores() is the fast numpy path -- no Python sort
    all_scores = []
    for tokens in tqdm(tokenized, desc=f"  Scoring {bucket_name}", leave=False):
        all_scores.append(bm25_obj.get_scores(tokens))

    # Parallel metric computation (CPU-bound, benefits from multiprocessing)
    per_query = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_score_one)(
            tokenized[i], entity_ids, all_scores[i], ground_truth_ids[i], top_k
        )
        for i in tqdm(range(n), desc=f"  Metrics {bucket_name}", leave=False)
    )

    return per_query


# ---------------------------------------------------------------------------
# Latency measurement (single-threaded, realistic per-query numbers)
# ---------------------------------------------------------------------------


def measure_latency(
    bm25_obj,
    query_texts: list[str],
    top_k: int,
    n_warmup: int = 100,
    n_measure: int = 1000,
) -> dict:
    n_all = len(query_texts)
    warmup = query_texts[: min(n_warmup, n_all)]
    measure = query_texts[: min(n_measure, n_all)]

    for q in warmup:
        scores = bm25_obj.get_scores(q.lower().split())
        n = min(top_k, len(scores))
        np.argpartition(scores, -n)[-n:]

    latencies_ms: list[float] = []
    for q in measure:
        t0 = time.perf_counter()
        tokens = q.lower().split()
        scores = bm25_obj.get_scores(tokens)
        n = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx[np.argsort(scores[top_idx])[::-1]]
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
        description="Evaluate a BM25 retrieval index across all corruption buckets."
    )
    parser.add_argument("--index-dir", required=True)
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
        help="Number of parallel jobs for metric computation. -1 = all cores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.models_config) as f:
        all_model_cfg = yaml.safe_load(f)
    with open(args.eval_config) as f:
        eval_cfg = yaml.safe_load(f)

    top_k = args.top_k
    buckets = eval_cfg.get("buckets", BUCKETS)

    # Load BM25 index
    index_dir = Path(args.index_dir)
    bm25_path = index_dir / "bm25.pkl"
    entity_ids_path = index_dir / "entity_ids.json"

    if not bm25_path.exists():
        console.print(f"[red]BM25 index not found: {bm25_path}")
        sys.exit(1)

    console.print(f"[bold cyan]Loading BM25 index from {bm25_path}...")
    import pickle
    with open(bm25_path, "rb") as f:
        bm25_obj = pickle.load(f)
    with open(entity_ids_path) as f:
        entity_ids: list[str] = json.load(f)

    n_records = len(entity_ids)
    console.print(f"[green]Loaded BM25 index with {n_records:,} documents.")

    # Load eval queries
    console.print(f"[bold cyan]Loading eval queries from {args.eval_queries}...")
    df = pl.read_parquet(args.eval_queries)
    console.print(f"[cyan]Loaded {len(df):,} queries across {df['bucket'].n_unique()} buckets.")

    query_col = f"query_text_{args.serialization}"
    if query_col not in df.columns:
        console.print(f"[red]Column '{query_col}' not found. Available: {df.columns}")
        sys.exit(1)
    console.print(f"[cyan]Query column: '{query_col}' | n_jobs: {args.n_jobs}")

    # Index size
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

        # Latency (single-threaded, realistic)
        latency_info = measure_latency(bm25_obj, query_texts, top_k)

        # Parallel eval
        per_query = evaluate_bucket_parallel(
            bm25_obj=bm25_obj,
            entity_ids=entity_ids,
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

    # Overall metrics from cached results (no re-run)
    overall_metrics = aggregate_metrics(all_per_query)

    latency_samples = [
        m.get("latency_ms", {}).get("p50", 0) for m in per_bucket_metrics.values()
    ]
    arr = np.array(latency_samples) if latency_samples else np.array([0.0])
    overall_latency = {
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n_queries": len(all_per_query),
    }

    # Summary table
    table = Table(title="BM25 Eval Results", show_header=True)
    for col in ["Bucket", "R@1", "R@5", "R@10", "MRR@10", "NDCG@10", "p50ms"]:
        table.add_column(col)
    for bucket, bm in per_bucket_metrics.items():
        table.add_row(
            bucket,
            f"{bm.get('recall_at_1', 0):.3f}",
            f"{bm.get('recall_at_5', 0):.3f}",
            f"{bm.get('recall_at_10', 0):.3f}",
            f"{bm.get('mrr_at_10', 0):.3f}",
            f"{bm.get('ndcg_at_10', 0):.3f}",
            f"{bm.get('latency_ms', {}).get('p50', 0):.1f}",
        )
    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{overall_metrics.get('recall_at_1', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_5', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('recall_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('mrr_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_metrics.get('ndcg_at_10', 0):.3f}[/bold]",
        f"[bold]{overall_latency['p50']:.1f}[/bold]",
    )
    console.print(table)

    # Build output JSON (ADR-003 schema)
    per_bucket_clean: dict[str, dict] = {}
    per_bucket_latency_ms: dict[str, dict] = {}
    for bucket, bm in per_bucket_metrics.items():
        per_bucket_clean[bucket] = {
            k: v for k, v in bm.items() if k not in ("latency_ms",)
        }
        per_bucket_latency_ms[bucket] = bm.get("latency_ms", {})

    result = {
        "experiment_id": args.experiment_id,
        "model": "bm25_baseline",
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
