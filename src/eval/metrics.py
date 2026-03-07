"""
Retrieval metrics. Single-relevant-document assumption (entity resolution:
each query has exactly one correct match in the index).

All functions operate on a single query. Use list comprehension + np.mean
to aggregate over a query set.

No external dependencies beyond numpy.
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Per-query metric functions
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """1.0 if relevant_id appears in top-k retrieved_ids, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """
    Fraction of top-k that are relevant.
    With a single relevant document this equals recall_at_k / k.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    n_relevant = sum(1 for rid in top_k if rid == relevant_id)
    return n_relevant / k


def reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    """1/rank if relevant_id is found, else 0.0. Used for MRR computation."""
    for i, rid in enumerate(retrieved_ids):
        if rid == relevant_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    """
    NDCG@k for a single relevant document.

    DCG = 1 / log2(rank + 1) at the position of the relevant doc.
    IDCG = 1 / log2(2) = 1.0 (ideal: relevant doc at position 1).
    NDCG = DCG / IDCG.

    Returns 0.0 if the relevant doc is not in top-k.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    for i, rid in enumerate(top_k):
        if rid == relevant_id:
            # rank is 1-indexed; log base 2 of (rank + 1)
            dcg = 1.0 / math.log2(i + 2)
            idcg = 1.0  # ideal: doc at rank 1 -> 1/log2(2) = 1.0
            return dcg / idcg
    return 0.0


# ---------------------------------------------------------------------------
# Composite metric computation for a single query
# ---------------------------------------------------------------------------


def compute_metrics(
    retrieved_ids: list[str],
    relevant_id: str,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ordered list of retrieved entity IDs, most relevant first.
    relevant_id : str
        The single ground-truth entity ID.
    ks : list[int]
        Cutoffs to evaluate at. Defaults to [1, 5, 10].

    Returns
    -------
    dict with keys:
        recall_at_1, recall_at_5, recall_at_10
        precision_at_5
        mrr_at_10
        ndcg_at_1, ndcg_at_5, ndcg_at_10
    """
    if ks is None:
        ks = [1, 5, 10]

    result: dict[str, float] = {}

    for k in ks:
        result[f"recall_at_{k}"] = recall_at_k(retrieved_ids, relevant_id, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, relevant_id, k)

    # Precision only at k=5 by default (matches ADR-003 schema)
    result["precision_at_5"] = precision_at_k(retrieved_ids, relevant_id, 5)

    # MRR capped at 10 by convention
    mrr_list = retrieved_ids[:10]
    result["mrr_at_10"] = reciprocal_rank(mrr_list, relevant_id)

    return result


# ---------------------------------------------------------------------------
# Aggregation across queries
# ---------------------------------------------------------------------------


def aggregate_metrics(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Compute mean of each metric across all queries.

    Parameters
    ----------
    per_query_metrics : list[dict]
        One dict per query, as returned by compute_metrics().

    Returns
    -------
    dict with same keys as input dicts, values are means.
    Empty input returns empty dict.
    """
    if not per_query_metrics:
        return {}

    # Collect all keys from first dict (assume all dicts have same keys)
    keys = list(per_query_metrics[0].keys())
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in per_query_metrics]
        aggregated[key] = float(np.mean(values))
    return aggregated
