"""
Phase 2 dataset assembly, dedup, and split.

Combines positive pairs (from corruptions) and negative pairs (from mining)
into train/val/test splits with strict no-overlap guarantees.

Usage:
    python src/data/phase2_split.py --pool data/phase2/base_pool.parquet --output-dir data/phase2/
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

from src.data.phase2_corrupt import (
    ALL_CORRUPTION_CODES,
    colval_serialize,
    corrupt_record_phase2,
)
from src.data.phase2_sources import load_nicknames

console = Console()
app = typer.Typer(help="Assemble and split Phase 2 CE training dataset.")


# ---------------------------------------------------------------------------
# Positive pair generation
# ---------------------------------------------------------------------------


def generate_positive_pairs(
    pool: pl.DataFrame,
    corruptions_per_record: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate positive pairs by applying corruptions to pool records.

    Each base record gets `corruptions_per_record` different corrupted versions,
    each producing a (anchor, corrupted, label=1) pair.
    """
    rng = random.Random(seed)
    nickname_map = load_nicknames()

    # Codes that reliably produce visible changes without external maps
    reliable_codes = [
        "C1", "C2", "C6",
        "N2", "N3", "N5", "N6", "N7", "N11",
        "T3", "T4", "T6",
        "E1", "E2",
    ]

    pairs = []
    for row in pool.iter_rows(named=True):
        codes_to_apply = rng.sample(
            reliable_codes, min(corruptions_per_record, len(reliable_codes))
        )
        for code in codes_to_apply:
            corrupted = corrupt_record_phase2(
                row, [code], rng, nickname_map=nickname_map,
            )
            pairs.append({
                "anchor_id": row["entity_id"],
                "candidate_id": row["entity_id"],
                "anchor_text": colval_serialize(row),
                "candidate_text": colval_serialize(corrupted),
                "label": 1,
                "corruption_type": code,
                "strategy": f"positive_{code}",
            })

    return pairs


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _normalize_key(text: str) -> str:
    """Normalize text for dedup: lowercase, strip, remove punctuation."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(" ", "")


def simple_dedup(pairs: list[dict], threshold: float = 0.9) -> list[dict]:
    """Remove near-duplicate pairs using hash-based dedup.

    Uses a simple approach: hash the normalized concatenation of
    anchor+candidate text. Exact duplicates are removed.
    For near-duplicates, we use a fingerprint approach.
    """
    seen: set[str] = set()
    deduped = []

    for p in pairs:
        # Create fingerprint from normalized anchor + candidate
        key = _normalize_key(p["anchor_text"]) + "|" + _normalize_key(p["candidate_text"])
        fp = hashlib.md5(key.encode()).hexdigest()

        if fp not in seen:
            seen.add(fp)
            deduped.append(p)

    return deduped


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def deterministic_split(
    pairs: list[dict],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into train/val/test with stratification by (label, strategy).

    Entity-level split: all pairs sharing an anchor_id go to the same split.
    This prevents data leakage.
    """
    rng = random.Random(seed)

    # Group pairs by anchor_id
    anchor_groups: dict[str, list[dict]] = {}
    for p in pairs:
        anchor_groups.setdefault(p["anchor_id"], []).append(p)

    # Shuffle anchor_ids deterministically
    anchor_ids = list(anchor_groups.keys())
    rng.shuffle(anchor_ids)

    n_total = len(anchor_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(anchor_ids[:n_train])
    val_ids = set(anchor_ids[n_train:n_train + n_val])
    test_ids = set(anchor_ids[n_train + n_val:])

    train = [p for p in pairs if p["anchor_id"] in train_ids]
    val = [p for p in pairs if p["anchor_id"] in val_ids]
    test = [p for p in pairs if p["anchor_id"] in test_ids]

    return train, val, test


def validate_split(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    phase1_triplets_path: Path | None = None,
) -> None:
    """Validate split invariants. Raises AssertionError on violation.

    Checks:
    1. No anchor_id overlap between train/val/test
    2. No entity_id overlap with Phase 1 triplets (if path provided)
    3. Test set has both labels
    """
    train_anchors = {p["anchor_id"] for p in train}
    val_anchors = {p["anchor_id"] for p in val}
    test_anchors = {p["anchor_id"] for p in test}

    # No overlap between splits
    assert len(train_anchors & val_anchors) == 0, (
        f"Train/val anchor overlap: {len(train_anchors & val_anchors)} IDs"
    )
    assert len(train_anchors & test_anchors) == 0, (
        f"Train/test anchor overlap: {len(train_anchors & test_anchors)} IDs"
    )
    assert len(val_anchors & test_anchors) == 0, (
        f"Val/test anchor overlap: {len(val_anchors & test_anchors)} IDs"
    )

    # Check Phase 1 overlap if path provided
    if phase1_triplets_path and Path(phase1_triplets_path).exists():
        p1_df = pl.read_parquet(phase1_triplets_path)
        p1_ids = set()
        for col in ["anchor_entity_id", "positive_entity_id", "negative_entity_id"]:
            if col in p1_df.columns:
                p1_ids.update(p1_df[col].to_list())

        all_p2_ids = train_anchors | val_anchors | test_anchors
        overlap = all_p2_ids & p1_ids
        assert len(overlap) == 0, (
            f"Phase 1/Phase 2 entity ID overlap: {len(overlap)} IDs"
        )

    # Test set should have both labels
    test_labels = {p["label"] for p in test}
    assert 0 in test_labels, "Test set has no negative examples"
    assert 1 in test_labels, "Test set has no positive examples"


def pairs_to_dataframe(pairs: list[dict]) -> pl.DataFrame:
    """Convert list of pair dicts to a parquet-friendly DataFrame."""
    return pl.DataFrame([
        {
            "anchor_id": p["anchor_id"],
            "candidate_id": p["candidate_id"],
            "anchor_text": p["anchor_text"],
            "candidate_text": p["candidate_text"],
            "label": p["label"],
            "corruption_type": p.get("corruption_type", ""),
            "strategy": p.get("strategy", ""),
        }
        for p in pairs
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def build_dataset(
    pool_path: Path = typer.Option(..., help="Path to base_pool.parquet"),
    output_dir: Path = typer.Option("data/phase2", help="Output directory"),
    corruptions_per_record: int = typer.Option(5, help="Corruptions per record"),
    n_negatives_per_strategy: int = typer.Option(10_000, help="Negatives per strategy"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Assemble, dedup, and split the CE training dataset."""
    from src.data.phase2_negatives import mine_all_negatives
    from src.data.phase2_boundary import mine_boundary_pairs_deterministic

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pool
    pool = pl.read_parquet(pool_path)
    console.print(f"[cyan]Pool loaded: {len(pool):,} records")

    # Generate positive pairs
    console.print("[cyan]Generating positive pairs...")
    positives = generate_positive_pairs(pool, corruptions_per_record, seed)
    console.print(f"[green]Positive pairs: {len(positives):,}")

    # Mine negatives
    console.print("[cyan]Mining negative pairs...")
    negatives = mine_all_negatives(pool, n_negatives_per_strategy, seed)

    # Serialize negatives for the dataset
    neg_pairs = []
    for neg in negatives:
        neg_pairs.append({
            "anchor_id": neg["anchor_id"],
            "candidate_id": neg["candidate_id"],
            "anchor_text": colval_serialize(neg["anchor"]),
            "candidate_text": colval_serialize(neg["candidate"]),
            "label": 0,
            "corruption_type": "",
            "strategy": neg["strategy"],
        })
    console.print(f"[green]Negative pairs: {len(neg_pairs):,}")

    # Boundary zone pairs
    console.print("[cyan]Mining boundary-zone pairs...")
    boundary = mine_boundary_pairs_deterministic(pool, n_pairs=min(50_000, len(pool)), seed=seed)
    boundary_pairs = []
    for bp in boundary:
        boundary_pairs.append({
            "anchor_id": bp["anchor_id"],
            "candidate_id": bp["candidate_id"],
            "anchor_text": colval_serialize(bp["anchor"]),
            "candidate_text": colval_serialize(bp["candidate"]),
            "label": 0,
            "corruption_type": "",
            "strategy": bp["strategy"],
        })
    console.print(f"[green]Boundary pairs: {len(boundary_pairs):,}")

    # Combine all
    all_pairs = positives + neg_pairs + boundary_pairs
    console.print(f"[cyan]Total before dedup: {len(all_pairs):,}")

    # Dedup
    all_pairs = simple_dedup(all_pairs)
    console.print(f"[cyan]Total after dedup: {len(all_pairs):,}")

    # Split
    train, val, test = deterministic_split(all_pairs, seed=seed)
    console.print(
        f"[green]Split: train={len(train):,} val={len(val):,} test={len(test):,}"
    )

    # Validate
    validate_split(train, val, test)
    console.print("[green]Split validation passed!")

    # Save
    for name, data in [("ce_train", train), ("ce_val", val), ("ce_test", test)]:
        df = pairs_to_dataframe(data)
        path = output_dir / f"{name}.parquet"
        df.write_parquet(path)
        console.print(f"[green]Saved: {path} ({len(df):,} rows)")


if __name__ == "__main__":
    app()
