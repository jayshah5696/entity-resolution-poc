"""Tests for Phase 2 negative mining strategies."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.data.phase2_negatives import (
    mine_all_negatives,
    mine_neg1_same_company_diff_person,
    mine_neg2_phonetic_neighbors,
    mine_neg3_common_name_diff_company,
    mine_neg4_title_function_swap,
    mine_neg5_title_level_swap,
    mine_neg6_random,
)
from src.data.phase2_boundary import (
    deterministic_filter,
    mine_boundary_pairs_deterministic,
)
from src.data.phase2_pool import build_base_pool, build_company_pool
from src.data.phase2_sources import (
    load_census_surnames,
    load_ssa_names,
    parse_gleif,
    parse_onet_reported,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def small_pool():
    """Build a small pool from fixtures for testing."""
    gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
    companies = build_company_pool(gleif)
    surnames = load_census_surnames(FIXTURES / "census_surnames.csv")
    first_names = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
    titles = parse_onet_reported(FIXTURES / "onet_reported_titles.txt")
    return build_base_pool(companies, surnames, first_names, titles, n=200, seed=42)


class TestNeg1SameCompanyDiffPerson:
    def test_returns_list_of_dicts(self, small_pool):
        pairs = mine_neg1_same_company_diff_person(small_pool, n=10)
        assert isinstance(pairs, list)
        if pairs:
            assert isinstance(pairs[0], dict)

    def test_all_label_zero(self, small_pool):
        pairs = mine_neg1_same_company_diff_person(small_pool, n=10)
        for p in pairs:
            assert p["label"] == 0

    def test_different_entity_ids(self, small_pool):
        pairs = mine_neg1_same_company_diff_person(small_pool, n=10)
        for p in pairs:
            assert p["anchor_id"] != p["candidate_id"]

    def test_same_company(self, small_pool):
        pairs = mine_neg1_same_company_diff_person(small_pool, n=10)
        for p in pairs:
            assert p["anchor"]["company"] == p["candidate"]["company"]


class TestNeg2PhoneticNeighbors:
    def test_all_label_zero(self, small_pool):
        pairs = mine_neg2_phonetic_neighbors(small_pool, n=10)
        for p in pairs:
            assert p["label"] == 0

    def test_different_entity_ids(self, small_pool):
        pairs = mine_neg2_phonetic_neighbors(small_pool, n=10)
        for p in pairs:
            assert p["anchor_id"] != p["candidate_id"]

    def test_has_strategy_tag(self, small_pool):
        pairs = mine_neg2_phonetic_neighbors(small_pool, n=5)
        for p in pairs:
            assert p["strategy"] == "NEG2_phonetic_neighbor"


class TestNeg3CommonNameDiffCompany:
    def test_all_label_zero(self, small_pool):
        pairs = mine_neg3_common_name_diff_company(small_pool, n=10)
        for p in pairs:
            assert p["label"] == 0

    def test_different_companies(self, small_pool):
        pairs = mine_neg3_common_name_diff_company(small_pool, n=10)
        for p in pairs:
            assert p["anchor"]["company"] != p["candidate"]["company"]


class TestNeg4TitleFunctionSwap:
    def test_all_label_zero(self, small_pool):
        pairs = mine_neg4_title_function_swap(small_pool, n=10)
        for p in pairs:
            assert p["label"] == 0

    def test_different_entity_ids(self, small_pool):
        pairs = mine_neg4_title_function_swap(small_pool, n=10)
        for p in pairs:
            assert p["anchor_id"] != p["candidate_id"]


class TestNeg5TitleLevelSwap:
    def test_all_label_zero(self, small_pool):
        pairs = mine_neg5_title_level_swap(small_pool, n=10)
        for p in pairs:
            assert p["label"] == 0


class TestNeg6Random:
    def test_returns_correct_count(self, small_pool):
        pairs = mine_neg6_random(small_pool, n=50)
        assert len(pairs) == 50

    def test_all_label_zero(self, small_pool):
        pairs = mine_neg6_random(small_pool, n=50)
        for p in pairs:
            assert p["label"] == 0

    def test_has_anchor_and_candidate(self, small_pool):
        pairs = mine_neg6_random(small_pool, n=10)
        for p in pairs:
            assert "anchor" in p
            assert "candidate" in p
            assert "anchor_id" in p
            assert "candidate_id" in p


class TestMineAllNegatives:
    def test_returns_from_multiple_strategies(self, small_pool):
        pairs = mine_all_negatives(small_pool, n_per_strategy=5, seed=42)
        strategies = {p["strategy"] for p in pairs}
        # Should have at least random negatives
        assert "NEG6_random" in strategies
        assert len(strategies) >= 2

    def test_all_label_zero(self, small_pool):
        pairs = mine_all_negatives(small_pool, n_per_strategy=5, seed=42)
        for p in pairs:
            assert p["label"] == 0, f"Strategy {p['strategy']} produced label != 0"


class TestDeterministicFilter:
    def test_exact_email_excluded(self):
        a = {"email": "jay@acme.com", "first_name": "Jay", "last_name": "Smith", "company": "X"}
        b = {"email": "jay@acme.com", "first_name": "John", "last_name": "Doe", "company": "Y"}
        assert deterministic_filter(a, b) is True

    def test_same_name_same_company_excluded(self):
        a = {"email": "a@x.com", "first_name": "Jay", "last_name": "Smith", "company": "Acme"}
        b = {"email": "b@y.com", "first_name": "Jay", "last_name": "Smith", "company": "Acme"}
        assert deterministic_filter(a, b) is True

    def test_different_people_not_excluded(self):
        a = {"email": "jay@acme.com", "first_name": "Jay", "last_name": "Smith", "company": "Acme"}
        b = {"email": "bob@xyz.com", "first_name": "Bob", "last_name": "Jones", "company": "XYZ"}
        assert deterministic_filter(a, b) is False


class TestBoundaryMiningDeterministic:
    def test_returns_pairs(self, small_pool):
        pairs = mine_boundary_pairs_deterministic(small_pool, n_pairs=10)
        assert isinstance(pairs, list)
        assert len(pairs) <= 10

    def test_all_label_zero(self, small_pool):
        pairs = mine_boundary_pairs_deterministic(small_pool, n_pairs=10)
        for p in pairs:
            assert p["label"] == 0

    def test_different_entity_ids(self, small_pool):
        pairs = mine_boundary_pairs_deterministic(small_pool, n_pairs=10)
        for p in pairs:
            assert p["anchor_id"] != p["candidate_id"]
