"""
Triplet generation pipeline.

For each profile in triplet_source:
  - Generate 3 positive variants (corrupted versions of the same identity)
  - Mine 2 hard negatives (same company prefix or same country)

Output: data/triplets/triplets.parquet

Usage
-----
    uv run python src/data/triplets.py \\
        --config configs/dataset.yaml \\
        --profiles data/processed/triplet_source.parquet \\
        --output-dir data/triplets/
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import typer
import yaml
from Levenshtein import distance as lev_distance
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.data.corrupt import corrupt_record, CORRUPTION_TYPES
from src.data.serialize import serialize_pipe, serialize_kv

app = typer.Typer(add_completion=False)
console = Console()

# ── Corruption combos used for the 3 positive variants ────────────────────────
# Each entry is a list of corruption types to apply together.
POSITIVE_COMBOS: list[list[str]] = [
    ["levenshtein_1"],
    ["abbreviation"],
    ["field_drop_single"],
    ["domain_swap"],
    ["case_mutation"],
    ["nickname"],
    ["truncation"],
    ["levenshtein_2"],
    ["company_abbrev"],
    ["field_drop_double"],
    ["levenshtein_1", "domain_swap"],
    ["abbreviation", "field_drop_single"],
    ["nickname", "levenshtein_1"],
    ["case_mutation", "truncation"],
]


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _profile_to_dict(row: dict) -> dict:
    """Ensure all string fields are str (not None)."""
    return {
        "entity_id": str(row.get("entity_id", "")),
        "first_name": str(row.get("first_name") or ""),
        "last_name": str(row.get("last_name") or ""),
        "company": str(row.get("company") or ""),
        "email": str(row.get("email") or ""),
        "country": str(row.get("country") or ""),
    }


def _build_company_prefix_index(
    profiles: list[dict], prefix_len: int = 4
) -> dict[str, list[int]]:
    """Map first `prefix_len` chars of company → list of profile indices."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(profiles):
        prefix = p["company"][:prefix_len].lower()
        if prefix:
            idx[prefix].append(i)
    return idx


def _build_country_index(profiles: list[dict]) -> dict[str, list[int]]:
    """Map country → list of profile indices."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(profiles):
        c = p.get("country", "")
        if c:
            idx[c].append(i)
    return idx


def _validate_positive(anchor: dict, positive: dict) -> bool:
    """
    Positive must differ from anchor in at least one field
    (edit distance > 0 or a field is dropped).
    """
    anchor_pipe = serialize_pipe(anchor)
    positive_pipe = serialize_pipe(positive)
    return anchor_pipe != positive_pipe


def _validate_negative(anchor: dict, negative: dict) -> bool:
    """Negative must not share the same email as anchor."""
    return anchor.get("email", "") != negative.get("email", "")


def generate_triplets(
    profiles: list[dict],
    cfg: dict,
    rng: random.Random,
) -> list[dict]:
    """
    Generate triplets for all profiles.

    For each profile:
      - 3 positive variants × 2 negatives = 6 triplets per profile
        (we pair each positive with a matched negative)
    """
    prefix_len: int = cfg.get("hard_negative_strategy", {}).get(
        "round1", {}
    ).get("company_prefix_len", 4)

    company_idx = _build_company_prefix_index(profiles, prefix_len)
    country_idx = _build_country_index(profiles)

    n = len(profiles)
    triplets: list[dict] = []

    corruption_counter: Counter = Counter()
    negative_source_counter: Counter = Counter()

    # Pre-shuffle combo order per profile using rng
    all_combos = POSITIVE_COMBOS[:]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Mining triplets…"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("triplet", total=n)

        for anchor_idx, profile in enumerate(profiles):
            # ── Generate 3 positive variants ──────────────────────────────────
            rng.shuffle(all_combos)
            chosen_combos = all_combos[:3]

            positives: list[tuple[dict, list[str]]] = []
            for combo in chosen_combos:
                pos, applied = corrupt_record(profile, corruption_types=combo, rng=rng)
                # Re-try once if validation fails
                if not _validate_positive(profile, pos):
                    pos, applied = corrupt_record(
                        profile, corruption_types=["levenshtein_1", "field_drop_single"], rng=rng
                    )
                positives.append((pos, applied))

            # ── Mine 2 negatives ──────────────────────────────────────────────
            # Negative 1: same company prefix
            prefix = profile["company"][:prefix_len].lower()
            candidates = [
                i for i in company_idx.get(prefix, [])
                if i != anchor_idx and profiles[i]["entity_id"] != profile["entity_id"]
            ]
            if candidates:
                neg1_idx = rng.choice(candidates)
                neg1 = profiles[neg1_idx]
                neg1_source = "same_company_prefix"
            else:
                # Fallback: random
                neg1_idx = rng.randint(0, n - 1)
                while neg1_idx == anchor_idx:
                    neg1_idx = rng.randint(0, n - 1)
                neg1 = profiles[neg1_idx]
                neg1_source = "random"

            # Negative 2: same country
            country = profile.get("country", "")
            country_candidates = [
                i for i in country_idx.get(country, [])
                if i != anchor_idx and profiles[i]["entity_id"] != profile["entity_id"]
            ]
            if country_candidates:
                neg2_idx = rng.choice(country_candidates)
                neg2 = profiles[neg2_idx]
                neg2_source = "same_country"
            else:
                neg2_idx = rng.randint(0, n - 1)
                while neg2_idx == anchor_idx:
                    neg2_idx = rng.randint(0, n - 1)
                neg2 = profiles[neg2_idx]
                neg2_source = "random"

            negatives = [(neg1, neg1_source), (neg2, neg2_source)]

            # ── Build triplet rows ────────────────────────────────────────────
            # We pair: pos[0]↔neg1, pos[1]↔neg2, pos[2]↔neg1 (cycle)
            neg_cycle = [negatives[0], negatives[1], negatives[0]]

            for (pos, applied), (neg, neg_src) in zip(positives, neg_cycle):
                # Validate
                if not _validate_negative(profile, neg):
                    # Find a replacement random negative
                    for _ in range(10):
                        rand_idx = rng.randint(0, n - 1)
                        if rand_idx != anchor_idx:
                            neg = profiles[rand_idx]
                            neg_src = "random"
                            if _validate_negative(profile, neg):
                                break

                corruption_counter.update(applied)
                negative_source_counter[neg_src] += 1

                triplets.append(
                    {
                        "anchor_id": profile["entity_id"],
                        "anchor_text_pipe": serialize_pipe(profile),
                        "anchor_text_kv": serialize_kv(profile),
                        "positive_text_pipe": serialize_pipe(pos),
                        "positive_text_kv": serialize_kv(pos),
                        "negative_text_pipe": serialize_pipe(neg),
                        "negative_text_kv": serialize_kv(neg),
                        "corruption_types": json.dumps(applied),
                        "negative_source": neg_src,
                    }
                )

            progress.update(task, advance=1)

    return triplets, corruption_counter, negative_source_counter


@app.command()
def main(
    config: str = typer.Option(
        "configs/dataset.yaml", "--config", help="Path to dataset.yaml"
    ),
    profiles: str = typer.Option(
        "data/processed/triplet_source.parquet",
        "--profiles",
        help="Path to triplet_source.parquet",
    ),
    output_dir: str = typer.Option(
        "data/triplets/", "--output-dir", help="Output directory"
    ),
) -> None:
    cfg = _load_config(config)
    seed: int = cfg.get("random_seed", 42)
    rng = random.Random(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Triplet Generation Pipeline")

    # ── Load profiles ─────────────────────────────────────────────────────────
    console.print(f"Loading profiles from [bold]{profiles}[/]…")
    df = pl.read_parquet(profiles)
    profile_list = [_profile_to_dict(row) for row in df.to_dicts()]
    console.print(f"Loaded [cyan]{len(profile_list):,}[/] profiles.")

    # ── Generate triplets ─────────────────────────────────────────────────────
    triplets, corruption_counts, neg_source_counts = generate_triplets(
        profile_list, cfg, rng
    )
    console.print(f"Generated [bold green]{len(triplets):,}[/] triplets.")

    # ── Save parquet ──────────────────────────────────────────────────────────
    out_path = out / "triplets.parquet"
    df_triplets = pl.DataFrame(triplets)
    df_triplets.write_parquet(out_path)
    console.print(f"Saved → [bold]{out_path}[/]")

    # ── Save stats ────────────────────────────────────────────────────────────
    stats: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_triplets": len(triplets),
        "source_profiles": len(profile_list),
        "triplets_per_profile": len(triplets) / max(len(profile_list), 1),
        "corruption_type_distribution": dict(corruption_counts.most_common()),
        "negative_source_distribution": dict(neg_source_counts),
    }
    stats_path = out / "triplets_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"Saved stats → [bold]{stats_path}[/]")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(
        title="Triplet Generation Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total triplets", f"{len(triplets):,}")
    table.add_row("Source profiles", f"{len(profile_list):,}")
    table.add_row("Triplets / profile", f"{len(triplets)/max(len(profile_list),1):.1f}")
    for ctype, cnt in corruption_counts.most_common(5):
        table.add_row(f"  corruption:{ctype}", f"{cnt:,}")
    for src, cnt in neg_source_counts.most_common():
        table.add_row(f"  negative:{src}", f"{cnt:,}")
    console.print(table)
    console.print("[bold green]Done![/]")


if __name__ == "__main__":
    app()
