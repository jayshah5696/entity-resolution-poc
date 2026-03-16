"""
Tests for Phase 2 dataset assembly, dedup, and split.

CRITICAL invariant tests — never skip these.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.data.phase2_corrupt import colval_serialize
from src.data.phase2_negatives import mine_all_negatives
from src.data.phase2_pool import build_base_pool, build_company_pool
from src.data.phase2_sources import (
    load_census_surnames,
    load_ssa_names,
    parse_gleif,
    parse_onet_reported,
)
from src.data.phase2_split import (
    deterministic_split,
    generate_positive_pairs,
    pairs_to_dataframe,
    simple_dedup,
    validate_split,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def small_pool():
    gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
    companies = build_company_pool(gleif)
    surnames = load_census_surnames(FIXTURES / "census_surnames.csv")
    first_names = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
    titles = parse_onet_reported(FIXTURES / "onet_reported_titles.txt")
    return build_base_pool(companies, surnames, first_names, titles, n=50, seed=42)


@pytest.fixture
def small_pairs(small_pool):
    """Generate a small set of positive + negative pairs."""
    positives = generate_positive_pairs(small_pool, corruptions_per_record=3, seed=42)
    negatives = mine_all_negatives(small_pool, n_per_strategy=5, seed=42)
    neg_pairs = [
        {
            "anchor_id": n["anchor_id"],
            "candidate_id": n["candidate_id"],
            "anchor_text": colval_serialize(n["anchor"]),
            "candidate_text": colval_serialize(n["candidate"]),
            "label": 0,
            "corruption_type": "",
            "strategy": n["strategy"],
        }
        for n in negatives
    ]
    return positives + neg_pairs


# ── Positive pair generation ─────────────────────────────────────────────────


class TestGeneratePositivePairs:
    def test_returns_list(self, small_pool):
        pairs = generate_positive_pairs(small_pool, corruptions_per_record=2, seed=42)
        assert isinstance(pairs, list)
        assert len(pairs) > 0

    def test_all_label_one(self, small_pool):
        pairs = generate_positive_pairs(small_pool, corruptions_per_record=2, seed=42)
        for p in pairs:
            assert p["label"] == 1

    def test_anchor_id_matches_candidate_id(self, small_pool):
        """Positive pairs: anchor and candidate are the same entity."""
        pairs = generate_positive_pairs(small_pool, corruptions_per_record=2, seed=42)
        for p in pairs:
            assert p["anchor_id"] == p["candidate_id"]

    def test_text_differs(self, small_pool):
        """Anchor text and candidate text should differ (corruption applied)."""
        pairs = generate_positive_pairs(small_pool, corruptions_per_record=2, seed=42)
        differ_count = sum(1 for p in pairs if p["anchor_text"] != p["candidate_text"])
        # Some codes (T4, T6=empty) may be identity on certain titles
        assert differ_count > len(pairs) * 0.80, "Most pairs should have different text"

    def test_has_corruption_type(self, small_pool):
        pairs = generate_positive_pairs(small_pool, corruptions_per_record=2, seed=42)
        for p in pairs:
            assert p["corruption_type"], "Missing corruption_type"


# ── Dedup ─────────────────────────────────────────────────────────────────────


class TestSimpleDedup:
    def test_removes_exact_duplicates(self):
        pairs = [
            {"anchor_text": "hello", "candidate_text": "world", "label": 1},
            {"anchor_text": "hello", "candidate_text": "world", "label": 1},
            {"anchor_text": "foo", "candidate_text": "bar", "label": 0},
        ]
        result = simple_dedup(pairs)
        assert len(result) == 2

    def test_keeps_unique_pairs(self):
        pairs = [
            {"anchor_text": "a", "candidate_text": "b", "label": 1},
            {"anchor_text": "c", "candidate_text": "d", "label": 0},
            {"anchor_text": "e", "candidate_text": "f", "label": 1},
        ]
        result = simple_dedup(pairs)
        assert len(result) == 3


# ── Split ─────────────────────────────────────────────────────────────────────


class TestDeterministicSplit:
    def test_split_ratios(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(small_pairs)

    def test_no_anchor_overlap_train_val(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        train_ids = {p["anchor_id"] for p in train}
        val_ids = {p["anchor_id"] for p in val}
        assert len(train_ids & val_ids) == 0

    def test_no_anchor_overlap_train_test(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        train_ids = {p["anchor_id"] for p in train}
        test_ids = {p["anchor_id"] for p in test}
        assert len(train_ids & test_ids) == 0

    def test_no_anchor_overlap_val_test(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        val_ids = {p["anchor_id"] for p in val}
        test_ids = {p["anchor_id"] for p in test}
        assert len(val_ids & test_ids) == 0

    def test_deterministic_with_seed(self, small_pairs):
        t1, v1, te1 = deterministic_split(small_pairs, seed=42)
        t2, v2, te2 = deterministic_split(small_pairs, seed=42)
        assert len(t1) == len(t2)
        assert len(v1) == len(v2)
        assert len(te1) == len(te2)

    def test_test_set_has_both_labels(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        labels = {p["label"] for p in test}
        assert 0 in labels, "Test set missing negatives"
        assert 1 in labels, "Test set missing positives"


# ── Validate split ───────────────────────────────────────────────────────────


class TestValidateSplit:
    def test_validate_passes_clean_split(self, small_pairs):
        train, val, test = deterministic_split(small_pairs, seed=42)
        validate_split(train, val, test)  # should not raise

    def test_validate_fails_on_overlap(self):
        shared = {"anchor_id": "same-id", "candidate_id": "x", "label": 1}
        train = [shared]
        test = [shared]
        val = []
        with pytest.raises(AssertionError, match="overlap"):
            validate_split(train, val, test)


# ── DataFrame conversion ────────────────────────────────────────────────────


class TestPairsToDataFrame:
    def test_returns_dataframe(self, small_pairs):
        df = pairs_to_dataframe(small_pairs[:10])
        assert isinstance(df, pl.DataFrame)

    def test_has_required_columns(self, small_pairs):
        df = pairs_to_dataframe(small_pairs[:10])
        required = {"anchor_id", "candidate_id", "anchor_text", "candidate_text", "label"}
        assert required.issubset(set(df.columns))

    def test_label_values(self, small_pairs):
        df = pairs_to_dataframe(small_pairs[:10])
        labels = set(df["label"].to_list())
        assert labels.issubset({0, 1})
