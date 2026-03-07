"""
Evaluation set builder.

For each eval profile, creates one query per bucket (6 buckets × N eval profiles).
Each query is the profile with a specific corruption applied, plus ground-truth entity_id.

Usage
-----
    uv run python src/data/eval_set.py \\
        --config configs/dataset.yaml \\
        --eval-profiles data/eval/eval_profiles.parquet \\
        --output-dir data/eval/
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.data.corrupt import corrupt_for_bucket
from src.data.serialize import serialize_pipe, serialize_kv

app = typer.Typer(add_completion=False)
console = Console()

BUCKETS = [
    "pristine",
    "missing_firstname",
    "missing_email_company",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
]


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _profile_to_dict(row: dict) -> dict:
    return {
        "entity_id": str(row.get("entity_id", "")),
        "first_name": str(row.get("first_name") or ""),
        "last_name": str(row.get("last_name") or ""),
        "company": str(row.get("company") or ""),
        "email": str(row.get("email") or ""),
        "country": str(row.get("country") or ""),
    }


def build_eval_queries(
    profiles: list[dict],
    rng: random.Random,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Build one query per bucket for each profile.

    Returns
    -------
    (all_queries, per_bucket_queries)
    """
    all_queries: list[dict] = []
    per_bucket: dict[str, list[dict]] = defaultdict(list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Building eval queries…"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("eval", total=len(profiles) * len(BUCKETS))

        for profile in profiles:
            for bucket in BUCKETS:
                corrupted, _ = corrupt_for_bucket(profile, bucket=bucket, rng=rng)

                query_id = f"{profile['entity_id']}_{bucket}"
                query = {
                    "query_id": query_id,
                    "entity_id": profile["entity_id"],
                    "bucket": bucket,
                    "query_text_pipe": serialize_pipe(corrupted),
                    "query_text_kv": serialize_kv(corrupted),
                    "ground_truth_entity_id": profile["entity_id"],
                }
                all_queries.append(query)
                per_bucket[bucket].append(query)
                progress.update(task, advance=1)

    return all_queries, dict(per_bucket)


@app.command()
def main(
    config: str = typer.Option(
        "configs/dataset.yaml", "--config", help="Path to dataset.yaml"
    ),
    eval_profiles: str = typer.Option(
        "data/eval/eval_profiles.parquet",
        "--eval-profiles",
        help="Path to eval_profiles.parquet",
    ),
    output_dir: str = typer.Option(
        "data/eval/", "--output-dir", help="Output directory"
    ),
) -> None:
    cfg = _load_config(config)
    seed: int = cfg.get("random_seed", 42)
    rng = random.Random(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Eval Set Builder")

    # ── Load eval profiles ────────────────────────────────────────────────────
    console.print(f"Loading eval profiles from [bold]{eval_profiles}[/]…")
    df = pl.read_parquet(eval_profiles)
    profile_list = [_profile_to_dict(row) for row in df.to_dicts()]
    console.print(
        f"Loaded [cyan]{len(profile_list):,}[/] eval profiles × "
        f"[cyan]{len(BUCKETS)}[/] buckets = "
        f"[bold]{len(profile_list) * len(BUCKETS):,}[/] queries."
    )

    # ── Build queries ─────────────────────────────────────────────────────────
    all_queries, per_bucket = build_eval_queries(profile_list, rng)
    console.print(f"Generated [bold green]{len(all_queries):,}[/] queries total.")

    # ── Save all queries ──────────────────────────────────────────────────────
    all_path = out / "eval_queries.parquet"
    pl.DataFrame(all_queries).write_parquet(all_path)
    console.print(f"Saved all queries → [bold]{all_path}[/]")

    # ── Save per-bucket files ─────────────────────────────────────────────────
    bucket_counts: dict[str, int] = {}
    for bucket, queries in per_bucket.items():
        bpath = out / f"eval_queries_{bucket}.parquet"
        pl.DataFrame(queries).write_parquet(bpath)
        bucket_counts[bucket] = len(queries)
        console.print(f"  Saved {bucket:30s} → {bpath}  ({len(queries):,} queries)")

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(all_queries),
        "n_eval_profiles": len(profile_list),
        "buckets": BUCKETS,
        "bucket_counts": bucket_counts,
    }
    manifest_path = out / "eval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"Saved manifest → [bold]{manifest_path}[/]")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(
        title="Eval Set Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Bucket", style="cyan")
    table.add_column("Queries", justify="right")
    for bucket in BUCKETS:
        table.add_row(bucket, f"{bucket_counts.get(bucket, 0):,}")
    table.add_row("[bold]TOTAL[/]", f"[bold]{len(all_queries):,}[/]")
    console.print(table)
    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    app()
