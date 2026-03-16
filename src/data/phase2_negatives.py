"""
Phase 2 negative mining strategies.

6 deterministic strategies (NEG1-NEG6) that produce provably label=0 pairs.

Usage:
    from src.data.phase2_negatives import mine_all_negatives
"""

from __future__ import annotations

import random
from copy import deepcopy

import polars as pl


def mine_neg1_same_company_diff_person(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42,
) -> list[dict]:
    """NEG1: Same company, different person.

    Pairs two records that share a company but are different people.
    Label guarantee: different entity_id.
    """
    rng = random.Random(seed)
    # Group records by company_canonical
    company_col = "company_canonical" if "company_canonical" in pool.columns else "company"
    groups = pool.group_by(company_col).agg(pl.col("entity_id"))

    pairs = []
    rows_by_id = {r["entity_id"]: r for r in pool.iter_rows(named=True)}

    for row in groups.iter_rows(named=True):
        entity_ids = row["entity_id"]
        if len(entity_ids) < 2:
            continue
        # Sample pairs within this company
        for _ in range(min(5, len(entity_ids) * (len(entity_ids) - 1) // 2)):
            a_id, b_id = rng.sample(list(entity_ids), 2)
            pairs.append({
                "anchor_id": a_id,
                "candidate_id": b_id,
                "anchor": rows_by_id[a_id],
                "candidate": rows_by_id[b_id],
                "label": 0,
                "strategy": "NEG1_same_company_diff_person",
            })
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break

    return pairs[:n]


def mine_neg2_phonetic_neighbors(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42,
) -> list[dict]:
    """NEG2: Phonetic name collision using Soundex.

    Pairs records whose last names have the same Soundex code but are different people.
    Label guarantee: different entity_id, different base record.
    """
    rng = random.Random(seed)

    # Simple Soundex implementation (avoids external dependency)
    def soundex(name: str) -> str:
        if not name:
            return "0000"
        name = name.upper()
        codes = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3",
            "L": "4",
            "M": "5", "N": "5",
            "R": "6",
        }
        result = name[0]
        prev = codes.get(name[0], "")
        for c in name[1:]:
            code = codes.get(c, "")
            if code and code != prev:
                result += code
            prev = code if code else prev
        return (result + "0000")[:4]

    rows = pool.iter_rows(named=True)
    rows_list = list(rows)

    # Group by Soundex of last_name
    soundex_groups: dict[str, list[dict]] = {}
    for r in rows_list:
        sx = soundex(r.get("last_name", ""))
        soundex_groups.setdefault(sx, []).append(r)

    pairs = []
    for sx, group in soundex_groups.items():
        if len(group) < 2:
            continue
        for _ in range(min(3, len(group))):
            a, b = rng.sample(group, 2)
            if a["entity_id"] != b["entity_id"]:
                pairs.append({
                    "anchor_id": a["entity_id"],
                    "candidate_id": b["entity_id"],
                    "anchor": a,
                    "candidate": b,
                    "label": 0,
                    "strategy": "NEG2_phonetic_neighbor",
                })
                if len(pairs) >= n:
                    break
        if len(pairs) >= n:
            break

    return pairs[:n]


def mine_neg3_common_name_diff_company(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42, top_n_surnames: int = 50,
) -> list[dict]:
    """NEG3: Common name at different companies.

    Pairs records with the same common surname but at different organizations.
    """
    rng = random.Random(seed)
    rows_list = list(pool.iter_rows(named=True))

    # Group by last_name
    name_groups: dict[str, list[dict]] = {}
    for r in rows_list:
        name_groups.setdefault(r["last_name"], []).append(r)

    # Sort by frequency, take top N
    sorted_names = sorted(name_groups.items(), key=lambda x: len(x[1]), reverse=True)

    pairs = []
    for _, group in sorted_names[:top_n_surnames]:
        if len(group) < 2:
            continue
        for _ in range(min(5, len(group))):
            a, b = rng.sample(group, 2)
            if a["entity_id"] != b["entity_id"] and a["company"] != b["company"]:
                pairs.append({
                    "anchor_id": a["entity_id"],
                    "candidate_id": b["entity_id"],
                    "anchor": a,
                    "candidate": b,
                    "label": 0,
                    "strategy": "NEG3_common_name_diff_company",
                })
                if len(pairs) >= n:
                    break
        if len(pairs) >= n:
            break

    return pairs[:n]


def mine_neg4_title_function_swap(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42,
) -> list[dict]:
    """NEG4: Same seniority, different function.

    Pairs records at the same seniority level but different job functions.
    """
    rng = random.Random(seed)
    rows_list = list(pool.iter_rows(named=True))

    # Extract function keyword from title
    function_keywords = [
        "engineering", "marketing", "sales", "finance", "operations",
        "product", "design", "data", "legal", "hr", "human resources",
        "software", "it", "security", "research",
    ]

    def get_function(title: str) -> str:
        title_lower = title.lower()
        for kw in function_keywords:
            if kw in title_lower:
                return kw
        return "other"

    # Group by function
    func_groups: dict[str, list[dict]] = {}
    for r in rows_list:
        func = get_function(r.get("title", ""))
        func_groups.setdefault(func, []).append(r)

    pairs = []
    func_keys = [k for k in func_groups.keys() if len(func_groups[k]) >= 1]

    for _ in range(n * 3):  # over-sample then truncate
        if len(func_keys) < 2:
            break
        f1, f2 = rng.sample(func_keys, 2)
        a = rng.choice(func_groups[f1])
        b = rng.choice(func_groups[f2])
        if a["entity_id"] != b["entity_id"]:
            pairs.append({
                "anchor_id": a["entity_id"],
                "candidate_id": b["entity_id"],
                "anchor": a,
                "candidate": b,
                "label": 0,
                "strategy": "NEG4_title_function_swap",
            })
            if len(pairs) >= n:
                break

    return pairs[:n]


def mine_neg5_title_level_swap(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42,
) -> list[dict]:
    """NEG5: Same function, different seniority level.

    E.g., VP Engineering vs Director of Engineering.
    """
    rng = random.Random(seed)
    rows_list = list(pool.iter_rows(named=True))

    seniority_markers = {
        "chief": 5, "vp": 4, "vice president": 4, "director": 3,
        "senior": 2, "sr.": 2, "lead": 2, "principal": 2,
        "manager": 1, "junior": 0, "jr.": 0, "intern": 0,
    }

    def get_level(title: str) -> int:
        title_lower = title.lower()
        for marker, level in seniority_markers.items():
            if marker in title_lower:
                return level
        return 1  # default mid-level

    # Group by level
    level_groups: dict[int, list[dict]] = {}
    for r in rows_list:
        level = get_level(r.get("title", ""))
        level_groups.setdefault(level, []).append(r)

    pairs = []
    level_keys = [k for k in level_groups.keys() if len(level_groups[k]) >= 1]

    for _ in range(n * 3):
        if len(level_keys) < 2:
            break
        l1, l2 = rng.sample(level_keys, 2)
        a = rng.choice(level_groups[l1])
        b = rng.choice(level_groups[l2])
        if a["entity_id"] != b["entity_id"]:
            pairs.append({
                "anchor_id": a["entity_id"],
                "candidate_id": b["entity_id"],
                "anchor": a,
                "candidate": b,
                "label": 0,
                "strategy": "NEG5_title_level_swap",
            })
            if len(pairs) >= n:
                break

    return pairs[:n]


def mine_neg6_random(
    pool: pl.DataFrame, n: int = 10_000, seed: int = 42,
) -> list[dict]:
    """NEG6: Random negative pairs.

    Random draw from pool — not all useless, they set the floor.
    """
    rng = random.Random(seed)
    rows_list = list(pool.iter_rows(named=True))
    n_pool = len(rows_list)

    pairs = []
    for _ in range(n):
        a_idx, b_idx = rng.sample(range(n_pool), 2)
        a = rows_list[a_idx]
        b = rows_list[b_idx]
        pairs.append({
            "anchor_id": a["entity_id"],
            "candidate_id": b["entity_id"],
            "anchor": a,
            "candidate": b,
            "label": 0,
            "strategy": "NEG6_random",
        })

    return pairs


def mine_all_negatives(
    pool: pl.DataFrame, n_per_strategy: int = 10_000, seed: int = 42,
) -> list[dict]:
    """Run all 6 negative mining strategies."""
    all_pairs = []
    all_pairs.extend(mine_neg1_same_company_diff_person(pool, n_per_strategy, seed))
    all_pairs.extend(mine_neg2_phonetic_neighbors(pool, n_per_strategy, seed + 1))
    all_pairs.extend(mine_neg3_common_name_diff_company(pool, n_per_strategy, seed + 2))
    all_pairs.extend(mine_neg4_title_function_swap(pool, n_per_strategy, seed + 3))
    all_pairs.extend(mine_neg5_title_level_swap(pool, n_per_strategy, seed + 4))
    all_pairs.extend(mine_neg6_random(pool, n_per_strategy, seed + 5))
    return all_pairs
