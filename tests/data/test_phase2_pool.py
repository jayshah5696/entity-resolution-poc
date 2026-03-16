"""Tests for Phase 2 base entity pool builder."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.data.phase2_pool import build_base_pool, build_company_pool
from src.data.phase2_sources import (
    load_census_surnames,
    load_ssa_names,
    parse_gleif,
    parse_onet_reported,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def source_data():
    """Load all source data from fixtures."""
    gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
    companies = build_company_pool(gleif)
    surnames = load_census_surnames(FIXTURES / "census_surnames.csv")
    first_names = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
    titles = parse_onet_reported(FIXTURES / "onet_reported_titles.txt")
    return companies, surnames, first_names, titles


class TestBuildCompanyPool:
    def test_returns_dataframe(self):
        gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
        pool = build_company_pool(gleif)
        assert isinstance(pool, pl.DataFrame)

    def test_has_required_columns(self):
        gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
        pool = build_company_pool(gleif)
        required = {"company", "country", "company_canonical"}
        assert required.issubset(set(pool.columns))

    def test_deduplicates_on_canonical(self):
        gleif = parse_gleif(FIXTURES / "gleif_sample.csv")
        pool = build_company_pool(gleif)
        # Should have unique canonical names
        assert pool["company_canonical"].n_unique() == len(pool)


class TestBuildBasePool:
    def test_returns_correct_size(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert len(pool) == 100

    def test_has_required_columns(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=50, seed=42)
        required = {
            "entity_id", "first_name", "last_name", "company",
            "title", "email", "country",
        }
        assert required.issubset(set(pool.columns))

    def test_entity_id_unique(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert pool["entity_id"].n_unique() == 100

    def test_no_email_duplicates(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert pool["email"].n_unique() == 100

    def test_names_not_empty(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert pool.filter(pl.col("first_name") == "").height == 0
        assert pool.filter(pl.col("last_name") == "").height == 0

    def test_company_not_empty(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert pool.filter(pl.col("company") == "").height == 0

    def test_email_has_at_sign(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        for email in pool["email"].to_list():
            assert "@" in email, f"Email missing @: {email}"

    def test_country_present(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=100, seed=42)
        assert pool.filter(pl.col("country") == "").height == 0

    def test_deterministic_with_seed(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool1 = build_base_pool(companies, surnames, first_names, titles, n=50, seed=42)
        pool2 = build_base_pool(companies, surnames, first_names, titles, n=50, seed=42)
        assert pool1["first_name"].to_list() == pool2["first_name"].to_list()
        assert pool1["last_name"].to_list() == pool2["last_name"].to_list()

    def test_middle_name_about_20_percent(self, source_data):
        companies, surnames, first_names, titles = source_data
        pool = build_base_pool(companies, surnames, first_names, titles, n=500, seed=42)
        has_middle = pool.filter(pl.col("middle_name") != "").height
        pct = has_middle / 500
        assert 0.10 < pct < 0.35, f"Middle name %: {pct:.2%}"
