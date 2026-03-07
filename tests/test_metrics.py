"""
Tests for src/eval/metrics.py

Covers:
- recall_at_k: relevant in position 1, in position k, not in top-k, k=1/5/10
- precision_at_k: same cases plus mathematical relationship to recall
- reciprocal_rank: ranks 1, 2, 5, not found
- ndcg_at_k: rank 1 = 1.0, rank 2 < 1.0, not found = 0.0
- compute_metrics: correct dict keys, all values in [0, 1]
- aggregate_metrics: correct mean, handles empty list
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.metrics import (
    aggregate_metrics,
    compute_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RELEVANT_ID = "entity_42"

# Retrieved lists for controlled testing
RETRIEVED_R1 = [RELEVANT_ID, "a", "b", "c", "d", "e", "f", "g", "h", "i"]
RETRIEVED_R2 = ["a", RELEVANT_ID, "b", "c", "d", "e", "f", "g", "h", "i"]
RETRIEVED_R5 = ["a", "b", "c", "d", RELEVANT_ID, "e", "f", "g", "h", "i"]
RETRIEVED_R6 = ["a", "b", "c", "d", "e", RELEVANT_ID, "f", "g", "h", "i"]
RETRIEVED_R10 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", RELEVANT_ID]
RETRIEVED_R11 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", RELEVANT_ID]  # rank 11
RETRIEVED_MISS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]  # not present


# ============================================================================
# recall_at_k
# ============================================================================


class TestRecallAtK:
    def test_relevant_at_rank1_k1(self):
        assert recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=1) == 1.0

    def test_relevant_at_rank1_k5(self):
        assert recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=5) == 1.0

    def test_relevant_at_rank1_k10(self):
        assert recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=10) == 1.0

    def test_relevant_at_rank5_k5(self):
        # Rank 5 is the last position for k=5
        assert recall_at_k(RETRIEVED_R5, RELEVANT_ID, k=5) == 1.0

    def test_relevant_at_rank5_k4(self):
        # Rank 5 is outside k=4
        assert recall_at_k(RETRIEVED_R5, RELEVANT_ID, k=4) == 0.0

    def test_relevant_at_rank10_k10(self):
        assert recall_at_k(RETRIEVED_R10, RELEVANT_ID, k=10) == 1.0

    def test_relevant_at_rank10_k9(self):
        assert recall_at_k(RETRIEVED_R10, RELEVANT_ID, k=9) == 0.0

    def test_relevant_at_rank2_k1(self):
        assert recall_at_k(RETRIEVED_R2, RELEVANT_ID, k=1) == 0.0

    def test_not_in_retrieved_k10(self):
        assert recall_at_k(RETRIEVED_MISS, RELEVANT_ID, k=10) == 0.0

    def test_not_in_retrieved_k1(self):
        assert recall_at_k(RETRIEVED_MISS, RELEVANT_ID, k=1) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], RELEVANT_ID, k=5) == 0.0

    def test_k_zero_returns_zero(self):
        assert recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=0) == 0.0

    def test_relevant_beyond_k(self):
        # Rank 11 is outside k=10
        assert recall_at_k(RETRIEVED_R11, RELEVANT_ID, k=10) == 0.0

    def test_return_type_is_float(self):
        result = recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=10)
        assert isinstance(result, float)

    def test_single_result_hit(self):
        assert recall_at_k([RELEVANT_ID], RELEVANT_ID, k=1) == 1.0

    def test_single_result_miss(self):
        assert recall_at_k(["other"], RELEVANT_ID, k=1) == 0.0


# ============================================================================
# precision_at_k
# ============================================================================


class TestPrecisionAtK:
    def test_relevant_at_rank1_k1(self):
        # 1 relevant in top-1 -> precision = 1/1 = 1.0
        assert precision_at_k(RETRIEVED_R1, RELEVANT_ID, k=1) == 1.0

    def test_relevant_at_rank1_k5(self):
        # 1 relevant in top-5 -> precision = 1/5 = 0.2
        assert abs(precision_at_k(RETRIEVED_R1, RELEVANT_ID, k=5) - 0.2) < 1e-9

    def test_relevant_at_rank5_k5(self):
        # 1 relevant in top-5 -> precision = 1/5 = 0.2
        assert abs(precision_at_k(RETRIEVED_R5, RELEVANT_ID, k=5) - 0.2) < 1e-9

    def test_relevant_at_rank2_k1(self):
        # Not in top-1 -> precision = 0
        assert precision_at_k(RETRIEVED_R2, RELEVANT_ID, k=1) == 0.0

    def test_not_in_top_k(self):
        assert precision_at_k(RETRIEVED_MISS, RELEVANT_ID, k=5) == 0.0

    def test_relevant_at_rank10_k10(self):
        # 1 relevant in top-10 -> precision = 1/10 = 0.1
        assert abs(precision_at_k(RETRIEVED_R10, RELEVANT_ID, k=10) - 0.1) < 1e-9

    def test_empty_retrieved(self):
        assert precision_at_k([], RELEVANT_ID, k=5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(RETRIEVED_R1, RELEVANT_ID, k=0) == 0.0

    def test_precision_equals_recall_over_k(self):
        # precision@k = recall@k / k for single relevant doc
        for k in [1, 5, 10]:
            p = precision_at_k(RETRIEVED_R1, RELEVANT_ID, k=k)
            r = recall_at_k(RETRIEVED_R1, RELEVANT_ID, k=k)
            assert abs(p - r / k) < 1e-9

    def test_return_type_is_float(self):
        result = precision_at_k(RETRIEVED_R1, RELEVANT_ID, k=5)
        assert isinstance(result, float)


# ============================================================================
# reciprocal_rank
# ============================================================================


class TestReciprocalRank:
    def test_rank1(self):
        assert reciprocal_rank(RETRIEVED_R1, RELEVANT_ID) == 1.0

    def test_rank2(self):
        assert abs(reciprocal_rank(RETRIEVED_R2, RELEVANT_ID) - 0.5) < 1e-9

    def test_rank5(self):
        assert abs(reciprocal_rank(RETRIEVED_R5, RELEVANT_ID) - 0.2) < 1e-9

    def test_rank10(self):
        assert abs(reciprocal_rank(RETRIEVED_R10, RELEVANT_ID) - 0.1) < 1e-9

    def test_not_found(self):
        assert reciprocal_rank(RETRIEVED_MISS, RELEVANT_ID) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], RELEVANT_ID) == 0.0

    def test_single_match(self):
        assert reciprocal_rank([RELEVANT_ID], RELEVANT_ID) == 1.0

    def test_value_range(self):
        # RR should be in (0, 1] or 0 when not found
        for retrieved in [RETRIEVED_R1, RETRIEVED_R2, RETRIEVED_R5, RETRIEVED_MISS]:
            rr = reciprocal_rank(retrieved, RELEVANT_ID)
            assert 0.0 <= rr <= 1.0

    def test_return_type_is_float(self):
        result = reciprocal_rank(RETRIEVED_R1, RELEVANT_ID)
        assert isinstance(result, float)

    def test_rank_from_list(self):
        """RR at rank i+1 equals 1/(i+1) for 0-indexed position i."""
        for i, rank in enumerate([1, 2, 5, 10]):
            lst = ["irrelevant"] * (rank - 1) + [RELEVANT_ID]
            expected = 1.0 / rank
            assert abs(reciprocal_rank(lst, RELEVANT_ID) - expected) < 1e-9


# ============================================================================
# ndcg_at_k
# ============================================================================


class TestNdcgAtK:
    def test_rank1_is_1(self):
        # Relevant at rank 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0 => NDCG = 1.0
        assert abs(ndcg_at_k(RETRIEVED_R1, RELEVANT_ID, k=10) - 1.0) < 1e-9

    def test_rank1_k1_is_1(self):
        assert abs(ndcg_at_k(RETRIEVED_R1, RELEVANT_ID, k=1) - 1.0) < 1e-9

    def test_rank2_less_than_1(self):
        # DCG = 1/log2(3) < 1.0
        result = ndcg_at_k(RETRIEVED_R2, RELEVANT_ID, k=10)
        expected = 1.0 / math.log2(3)  # approx 0.631
        assert abs(result - expected) < 1e-9
        assert result < 1.0

    def test_rank5_value(self):
        # DCG = 1/log2(6)
        result = ndcg_at_k(RETRIEVED_R5, RELEVANT_ID, k=10)
        expected = 1.0 / math.log2(6)  # approx 0.387
        assert abs(result - expected) < 1e-9

    def test_rank10_value(self):
        # DCG = 1/log2(11)
        result = ndcg_at_k(RETRIEVED_R10, RELEVANT_ID, k=10)
        expected = 1.0 / math.log2(11)  # approx 0.289
        assert abs(result - expected) < 1e-9

    def test_not_found_is_0(self):
        assert ndcg_at_k(RETRIEVED_MISS, RELEVANT_ID, k=10) == 0.0

    def test_relevant_beyond_cutoff(self):
        # Rank 11 is outside k=10
        assert ndcg_at_k(RETRIEVED_R11, RELEVANT_ID, k=10) == 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k([], RELEVANT_ID, k=10) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(RETRIEVED_R1, RELEVANT_ID, k=0) == 0.0

    def test_monotone_in_rank(self):
        # Higher rank -> lower NDCG
        ndcg1 = ndcg_at_k(RETRIEVED_R1, RELEVANT_ID, k=10)
        ndcg2 = ndcg_at_k(RETRIEVED_R2, RELEVANT_ID, k=10)
        ndcg5 = ndcg_at_k(RETRIEVED_R5, RELEVANT_ID, k=10)
        assert ndcg1 > ndcg2 > ndcg5 > 0

    def test_value_range(self):
        for retrieved in [RETRIEVED_R1, RETRIEVED_R2, RETRIEVED_R5, RETRIEVED_MISS]:
            score = ndcg_at_k(retrieved, RELEVANT_ID, k=10)
            assert 0.0 <= score <= 1.0

    def test_return_type_is_float(self):
        result = ndcg_at_k(RETRIEVED_R1, RELEVANT_ID, k=10)
        assert isinstance(result, float)


# ============================================================================
# compute_metrics
# ============================================================================


class TestComputeMetrics:
    EXPECTED_KEYS = {
        "recall_at_5",
        "recall_at_10",
        "precision_at_5",
        "mrr_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
    }

    def test_returns_dict(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        assert self.EXPECTED_KEYS.issubset(result.keys())

    def test_all_values_in_unit_interval(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_miss_gives_zeros(self):
        result = compute_metrics(RETRIEVED_MISS, RELEVANT_ID)
        for key, val in result.items():
            assert val == 0.0, f"{key}={val} expected 0.0 for miss"

    def test_rank1_gives_all_ones(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        assert result["recall_at_5"] == 1.0
        assert result["recall_at_10"] == 1.0
        assert result["mrr_at_10"] == 1.0
        assert abs(result["ndcg_at_5"] - 1.0) < 1e-9
        assert abs(result["ndcg_at_10"] - 1.0) < 1e-9

    def test_rank2_recall_at_5_is_1(self):
        result = compute_metrics(RETRIEVED_R2, RELEVANT_ID)
        assert result["recall_at_5"] == 1.0
        assert result["recall_at_10"] == 1.0

    def test_custom_ks(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID, ks=[1, 3])
        assert "recall_at_1" in result
        assert "recall_at_3" in result
        assert "recall_at_5" not in result

    def test_mrr_capped_at_10(self):
        # Relevant at rank 11 -> MRR = 0 (capped at 10)
        result = compute_metrics(RETRIEVED_R11, RELEVANT_ID)
        assert result.get("mrr_at_10") == 0.0

    def test_precision_at_5_for_rank1(self):
        result = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        # 1 relevant in top-5 -> 1/5 = 0.2
        assert abs(result["precision_at_5"] - 0.2) < 1e-9

    def test_precision_at_5_for_miss(self):
        result = compute_metrics(RETRIEVED_MISS, RELEVANT_ID)
        assert result["precision_at_5"] == 0.0


# ============================================================================
# aggregate_metrics
# ============================================================================


class TestAggregateMetrics:
    def test_empty_list_returns_empty_dict(self):
        result = aggregate_metrics([])
        assert result == {}

    def test_single_query_returns_same_values(self):
        single = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        result = aggregate_metrics([single])
        for key in single:
            assert abs(result[key] - single[key]) < 1e-9

    def test_mean_is_correct(self):
        """Mean of two complementary results should be 0.5 for binary metrics."""
        m1 = compute_metrics(RETRIEVED_R1, RELEVANT_ID)  # all 1s
        m2 = compute_metrics(RETRIEVED_MISS, RELEVANT_ID)  # all 0s
        result = aggregate_metrics([m1, m2])
        assert abs(result["recall_at_1"] - 0.5) < 1e-9
        assert abs(result["recall_at_10"] - 0.5) < 1e-9
        assert abs(result["mrr_at_10"] - 0.5) < 1e-9

    def test_all_values_in_unit_interval(self):
        metrics_list = [
            compute_metrics(RETRIEVED_R1, RELEVANT_ID),
            compute_metrics(RETRIEVED_R2, RELEVANT_ID),
            compute_metrics(RETRIEVED_R5, RELEVANT_ID),
            compute_metrics(RETRIEVED_MISS, RELEVANT_ID),
        ]
        result = aggregate_metrics(metrics_list)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"aggregate {key}={val} out of [0,1]"

    def test_correct_mean_for_ndcg(self):
        """Mean NDCG is the average of per-query NDCG values."""
        m1 = compute_metrics(RETRIEVED_R1, RELEVANT_ID)
        m2 = compute_metrics(RETRIEVED_R2, RELEVANT_ID)
        result = aggregate_metrics([m1, m2])
        expected_ndcg10 = (m1["ndcg_at_10"] + m2["ndcg_at_10"]) / 2
        assert abs(result["ndcg_at_10"] - expected_ndcg10) < 1e-9

    def test_consistent_with_numpy_mean(self):
        """aggregate_metrics must match np.mean() for each key."""
        metrics_list = [
            compute_metrics(r, RELEVANT_ID)
            for r in [RETRIEVED_R1, RETRIEVED_R2, RETRIEVED_R5, RETRIEVED_R10, RETRIEVED_MISS]
        ]
        result = aggregate_metrics(metrics_list)
        for key in result:
            expected = float(np.mean([m[key] for m in metrics_list]))
            assert abs(result[key] - expected) < 1e-9, f"Mismatch for {key}"

    def test_return_type(self):
        metrics_list = [compute_metrics(RETRIEVED_R1, RELEVANT_ID)]
        result = aggregate_metrics(metrics_list)
        assert isinstance(result, dict)
        for val in result.values():
            assert isinstance(val, float)

    def test_large_batch(self):
        """100 queries with rank-1 hit -> all aggregated metrics = 1.0."""
        metrics_list = [compute_metrics(RETRIEVED_R1, RELEVANT_ID) for _ in range(100)]
        result = aggregate_metrics(metrics_list)
        for key in ["recall_at_1", "recall_at_10", "mrr_at_10"]:
            assert abs(result[key] - 1.0) < 1e-9

    def test_all_misses(self):
        """100 queries with no hits -> all aggregated metrics = 0.0."""
        metrics_list = [compute_metrics(RETRIEVED_MISS, RELEVANT_ID) for _ in range(100)]
        result = aggregate_metrics(metrics_list)
        for val in result.values():
            assert val == 0.0
