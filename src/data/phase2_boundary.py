"""
Phase 2 boundary-zone mining.

Deterministic approach: uses heuristic rules to find "hard" pairs that
are near the decision boundary. LLM judge is NOT implemented yet.

TODO: Add LLM judge pipeline (Claude/GPT-4) for boundary-zone labeling.
      Estimated cost: ~$10-15 for ~50K pairs.
      See plan.md Section 6 for the prompt template and approach.
"""

from __future__ import annotations

import random
from copy import deepcopy

import polars as pl
from rich.console import Console

console = Console()


def deterministic_filter(
    anchor: dict, candidate: dict,
) -> bool:
    """Pre-filter: should this pair be EXCLUDED as a negative?

    Returns True if the pair is likely a TRUE MATCH (exclude from negatives).
    """
    # Email domain exact match → likely same person
    a_email = anchor.get("email", "")
    c_email = candidate.get("email", "")
    if a_email and c_email and "@" in a_email and "@" in c_email:
        a_domain = a_email.split("@")[1]
        c_domain = c_email.split("@")[1]
        a_local = a_email.split("@")[0].lower()
        c_local = c_email.split("@")[0].lower()
        # Same email exactly → exclude
        if a_email.lower() == c_email.lower():
            return True

    # Exact company name match + exact name match → exclude
    a_name = f"{anchor.get('first_name', '')} {anchor.get('last_name', '')}".lower().strip()
    c_name = f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".lower().strip()
    a_co = anchor.get("company", "").lower().strip()
    c_co = candidate.get("company", "").lower().strip()
    if a_name == c_name and a_co == c_co and a_name:
        return True

    return False


def mine_boundary_pairs_deterministic(
    pool: pl.DataFrame,
    n_pairs: int = 50_000,
    seed: int = 42,
) -> list[dict]:
    """Mine boundary-zone pairs using deterministic heuristics.

    This is a placeholder for the full boundary mining pipeline.
    It creates pairs that are "confusable" — similar names at similar companies,
    but provably different entities.

    For the full pipeline with bi-encoder cosine similarity scoring
    and LLM judge verification, see plan.md Section 6.
    """
    rng = random.Random(seed)
    rows = list(pool.iter_rows(named=True))
    n_pool = len(rows)

    console.print("[cyan]Mining boundary-zone pairs (deterministic mode)...")
    console.print("[yellow]NOTE: LLM judge pipeline not yet implemented.")
    console.print("[yellow]Using heuristic-based boundary mining only.")

    pairs = []

    # Strategy: find records with similar (but not identical) names
    # Group by first 2 chars of last_name for near-match candidates
    prefix_groups: dict[str, list[dict]] = {}
    for r in rows:
        prefix = r.get("last_name", "")[:2].lower()
        if prefix:
            prefix_groups.setdefault(prefix, []).append(r)

    for prefix, group in prefix_groups.items():
        if len(group) < 2:
            continue
        for _ in range(min(10, len(group))):
            a, b = rng.sample(group, 2)
            if a["entity_id"] != b["entity_id"]:
                if not deterministic_filter(a, b):
                    pairs.append({
                        "anchor_id": a["entity_id"],
                        "candidate_id": b["entity_id"],
                        "anchor": a,
                        "candidate": b,
                        "label": 0,
                        "strategy": "NEG7_boundary_zone_deterministic",
                    })
                    if len(pairs) >= n_pairs:
                        break
        if len(pairs) >= n_pairs:
            break

    console.print(f"[green]Mined {len(pairs):,} boundary-zone pairs (deterministic).")
    return pairs[:n_pairs]
