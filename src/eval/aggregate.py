"""
aggregate.py -- Aggregate all experiment result JSONs into a master CSV and Markdown report.

Usage:
    uv run python src/eval/aggregate.py \\
        --results-dir results/ \\
        --output-csv results/master_results.csv \\
        --output-report results/report.md

Globs all *.json files in results/ (top-level only; skips indexes/ subdirs).
Reads the ADR-003 result schema and flattens to CSV rows.
Writes a Markdown report with 5 sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

BUCKETS = [
    "pristine",
    "missing_firstname",
    "missing_email_company",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
]

METRIC_KEYS = [
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mrr_at_10",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
]


def flatten_result(data: dict, source_file: str) -> dict:
    """
    Flatten one result JSON into a single CSV row dict.

    Top-level fields are carried directly. Per-bucket metrics are
    flattened to columns like 'pristine_recall_at_1'.
    Latency comes from the top-level latency_ms block.
    """
    row: dict = {
        "experiment_id": data.get("experiment_id", ""),
        "model": data.get("model", ""),
        "serialization": data.get("serialization", ""),
        "mode": data.get("mode", ""),
        "quantization": data.get("quantization", ""),
        "dims": data.get("dims", 0),
        "index_size": data.get("index_size", 0),
        "index_size_mb": data.get("index_size_mb", 0.0),
        "timestamp": data.get("timestamp", ""),
        "source_file": source_file,
    }

    # Overall metrics
    overall = data.get("overall", {})
    for key in METRIC_KEYS:
        row[f"overall_{key}"] = overall.get(key, None)

    # Per-bucket metrics
    per_bucket = data.get("per_bucket", {})
    for bucket in BUCKETS:
        bm = per_bucket.get(bucket, {})
        for key in METRIC_KEYS:
            row[f"{bucket}_{key}"] = bm.get(key, None)

    # Latency
    latency = data.get("latency_ms", {})
    row["latency_p50"] = latency.get("p50", None)
    row["latency_p95"] = latency.get("p95", None)
    row["latency_p99"] = latency.get("p99", None)
    row["latency_n_queries"] = latency.get("n_queries", None)

    return row


def load_results(results_dir: Path) -> list[dict]:
    """
    Load all *.json files from the top level of results_dir.
    Skips subdirectories (like indexes/).
    """
    rows: list[dict] = []
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        console.print(f"[yellow]No JSON files found in {results_dir}")
        return rows

    for jf in json_files:
        # Skip metadata files inside index subdirs (they live in subdirs, not here)
        # But extra-safe: skip if they look like metadata
        try:
            with open(jf) as f:
                data = json.load(f)
            # A valid result JSON must have 'experiment_id' and 'overall'
            if "experiment_id" not in data or "overall" not in data:
                console.print(f"[yellow]Skipping {jf.name} (not a result file)")
                continue
            row = flatten_result(data, jf.name)
            rows.append(row)
            console.print(f"[cyan]Loaded: {jf.name} (exp={row['experiment_id']} model={row['model']})")
        except Exception as exc:
            console.print(f"[yellow]Failed to load {jf.name}: {exc}")

    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write master results CSV sorted by experiment_id."""
    if not rows:
        console.print("[yellow]No rows to write to CSV.")
        output_path.write_text("")
        return

    # Sort by experiment_id
    rows_sorted = sorted(rows, key=lambda r: str(r.get("experiment_id", "")))

    # Build fieldnames from union of all keys (deterministic order)
    base_fields = [
        "experiment_id", "model", "serialization", "mode", "quantization",
        "dims", "index_size", "index_size_mb", "timestamp", "source_file",
    ]
    overall_fields = [f"overall_{m}" for m in METRIC_KEYS]
    bucket_fields = [f"{b}_{m}" for b in BUCKETS for m in METRIC_KEYS]
    latency_fields = ["latency_p50", "latency_p95", "latency_p99", "latency_n_queries"]

    fieldnames = base_fields + overall_fields + bucket_fields + latency_fields

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    console.print(f"[green]CSV written -> {output_path} ({len(rows_sorted)} rows)")


# ---------------------------------------------------------------------------
# Markdown report writing
# ---------------------------------------------------------------------------


def fmt_metric(val, digits: int = 3) -> str:
    """Format a metric value for the report. Returns 'n/a' if None."""
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def fmt_delta(val: float | None) -> str:
    """Format a delta vs BM25 as +X.Xpp or -X.Xpp."""
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}pp"


def write_report(rows: list[dict], output_path: Path) -> None:
    """
    Write a 5-section Markdown report summarising all experiments.

    Section 1: Summary table (model, mode, R@10 overall, MRR@10, latency p50)
    Section 2: Per-bucket R@10 table
    Section 3: Delta vs BM25
    Section 4: Latency table
    Section 5: Key findings (auto-generated)
    """
    if not rows:
        output_path.write_text("# Experiment Results\n\nNo results found.\n")
        return

    rows_sorted = sorted(rows, key=lambda r: str(r.get("experiment_id", "")))
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Find BM25 baseline row(s)
    bm25_rows = [r for r in rows_sorted if r.get("model") == "bm25_baseline"]
    bm25_row = bm25_rows[0] if bm25_rows else None

    lines: list[str] = []

    # Header
    lines.append("# Experiment Results")
    lines.append(f"\nGenerated: {now_str}\n")
    lines.append(f"Experiments loaded: {len(rows_sorted)}\n")

    # ---- Section 1: Summary ----
    lines.append("## 1. Summary\n")
    lines.append("| Experiment | Model | Mode | Serialization | R@10 Overall | MRR@10 | Latency p50 ms |")
    lines.append("|------------|-------|------|---------------|:------------:|:------:|:--------------:|")
    for r in rows_sorted:
        lines.append(
            f"| {r.get('experiment_id', '')} "
            f"| {r.get('model', '')} "
            f"| {r.get('mode', '')} "
            f"| {r.get('serialization', '')} "
            f"| {fmt_metric(r.get('overall_recall_at_10'))} "
            f"| {fmt_metric(r.get('overall_mrr_at_10'))} "
            f"| {fmt_metric(r.get('latency_p50'), 2)} |"
        )
    lines.append("")

    # ---- Section 2: Per-bucket R@10 ----
    lines.append("## 2. Per-Bucket Recall@10\n")
    header_cols = " | ".join(BUCKETS)
    lines.append(f"| Model | Serialization | {header_cols} |")
    sep_cols = " | ".join([":---:"] * len(BUCKETS))
    lines.append(f"|-------|---------------|{sep_cols}|")
    for r in rows_sorted:
        bucket_vals = " | ".join(
            fmt_metric(r.get(f"{b}_recall_at_10")) for b in BUCKETS
        )
        lines.append(
            f"| {r.get('model', '')} "
            f"| {r.get('serialization', '')} "
            f"| {bucket_vals} |"
        )
    lines.append("")

    # ---- Section 3: Delta vs BM25 ----
    lines.append("## 3. Delta vs BM25 Baseline (R@10 per bucket)\n")
    if bm25_row is None:
        lines.append("No BM25 baseline found in results. Run experiment 001 first.\n")
    else:
        lines.append(f"| Model | Serialization | {header_cols} | Overall |")
        lines.append(f"|-------|---------------|{sep_cols}|:-------:|")
        for r in rows_sorted:
            if r.get("model") == "bm25_baseline":
                continue
            delta_cols: list[str] = []
            for b in BUCKETS:
                model_val = r.get(f"{b}_recall_at_10")
                bm25_val = bm25_row.get(f"{b}_recall_at_10")
                if model_val is not None and bm25_val is not None:
                    delta_cols.append(fmt_delta(model_val - bm25_val))
                else:
                    delta_cols.append("n/a")
            overall_model = r.get("overall_recall_at_10")
            overall_bm25 = bm25_row.get("overall_recall_at_10")
            if overall_model is not None and overall_bm25 is not None:
                overall_delta = fmt_delta(overall_model - overall_bm25)
            else:
                overall_delta = "n/a"
            lines.append(
                f"| {r.get('model', '')} "
                f"| {r.get('serialization', '')} "
                f"| {' | '.join(delta_cols)} "
                f"| {overall_delta} |"
            )
    lines.append("")

    # ---- Section 4: Latency ----
    lines.append("## 4. Latency\n")
    lines.append("| Model | Mode | p50 ms | p95 ms | p99 ms | Index MB |")
    lines.append("|-------|------|:------:|:------:|:------:|:--------:|")
    for r in rows_sorted:
        lines.append(
            f"| {r.get('model', '')} "
            f"| {r.get('mode', '')} "
            f"| {fmt_metric(r.get('latency_p50'), 2)} "
            f"| {fmt_metric(r.get('latency_p95'), 2)} "
            f"| {fmt_metric(r.get('latency_p99'), 2)} "
            f"| {fmt_metric(r.get('index_size_mb'), 1)} |"
        )
    lines.append("")

    # ---- Section 5: Key findings ----
    lines.append("## 5. Key Findings\n")

    # Which model wins each bucket (by R@10)
    lines.append("### Best Model per Bucket (by R@10)\n")
    for b in BUCKETS:
        col = f"{b}_recall_at_10"
        candidates = [(r.get(col), r.get("model"), r.get("experiment_id")) for r in rows_sorted]
        candidates = [(v, m, e) for v, m, e in candidates if v is not None]
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_val, best_model, best_exp = candidates[0]
            lines.append(
                f"- **{b}**: `{best_model}` (exp {best_exp}) with R@10={best_val:.3f}"
            )
        else:
            lines.append(f"- **{b}**: no data")
    lines.append("")

    # Largest gains over BM25
    lines.append("### Largest Gains over BM25 (Overall R@10)\n")
    if bm25_row:
        bm25_overall = bm25_row.get("overall_recall_at_10") or 0.0
        non_bm25 = [r for r in rows_sorted if r.get("model") != "bm25_baseline"]
        gains = []
        for r in non_bm25:
            model_overall = r.get("overall_recall_at_10")
            if model_overall is not None:
                gains.append((model_overall - bm25_overall, r.get("model"), r.get("experiment_id")))
        gains.sort(key=lambda x: x[0], reverse=True)
        for delta, model, exp_id in gains[:5]:
            direction = "gain" if delta >= 0 else "loss"
            lines.append(
                f"- `{model}` (exp {exp_id}): {fmt_delta(delta)} overall R@10 vs BM25"
            )
        if not gains:
            lines.append("- No non-BM25 results to compare.")
    else:
        lines.append("- BM25 baseline not found. Run experiment 001 first.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    console.print(f"[green]Report written -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all result JSONs into master CSV and Markdown report."
    )
    parser.add_argument(
        "--results-dir", default="results/", help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output-csv", default="results/master_results.csv", help="Path for master CSV"
    )
    parser.add_argument(
        "--output-report", default="results/report.md", help="Path for Markdown report"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        console.print(f"[red]Results directory not found: {results_dir}")
        sys.exit(1)

    console.print(f"[bold cyan]Scanning {results_dir} for result JSONs...")
    rows = load_results(results_dir)
    console.print(f"[green]Found {len(rows)} valid result files.")

    write_csv(rows, Path(args.output_csv))
    write_report(rows, Path(args.output_report))

    console.print("[bold green]Aggregation complete.")


if __name__ == "__main__":
    main()
