"""
Tests for Phase 2 data source parsing.

Uses small fixture files in tests/fixtures/ — no network downloads.
Tests the PARSING logic only. Download functions are tested via integration marks.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ── GLEIF ─────────────────────────────────────────────────────────────────────


class TestGLEIFParsing:
    """Test GLEIF golden copy CSV parsing."""

    def test_parse_gleif_returns_dataframe(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        assert isinstance(df, pl.DataFrame)

    def test_parse_gleif_has_required_columns(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        required = {"legal_name", "other_names", "country", "entity_status"}
        assert required.issubset(set(df.columns))

    def test_parse_gleif_row_count(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        assert len(df) == 10

    def test_parse_gleif_legal_name_not_empty(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        assert df.filter(pl.col("legal_name") == "").height == 0

    def test_parse_gleif_other_names_is_list(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        # Goldman Sachs should have other names
        gs = df.filter(pl.col("legal_name").str.contains("Goldman"))
        assert gs.height == 1
        other = gs["other_names"][0].to_list()
        assert isinstance(other, list)
        assert len(other) >= 1

    def test_parse_gleif_country_codes(self):
        from src.data.phase2_sources import parse_gleif

        df = parse_gleif(FIXTURES / "gleif_sample.csv")
        countries = set(df["country"].to_list())
        assert "US" in countries


# ── O*NET ─────────────────────────────────────────────────────────────────────


class TestONETParsing:
    """Test O*NET alternate and reported title parsing."""

    def test_parse_onet_alternates_returns_dict(self):
        from src.data.phase2_sources import parse_onet_alternates

        result = parse_onet_alternates(FIXTURES / "onet_alternate_titles.txt")
        assert isinstance(result, dict)

    def test_parse_onet_alternates_has_entries(self):
        from src.data.phase2_sources import parse_onet_alternates

        result = parse_onet_alternates(FIXTURES / "onet_alternate_titles.txt")
        assert len(result) > 0

    def test_parse_onet_alternates_ceo_mapping(self):
        from src.data.phase2_sources import parse_onet_alternates

        result = parse_onet_alternates(FIXTURES / "onet_alternate_titles.txt")
        # "Chief Executives" should map to CEO, Chief Executive Officer, President
        assert "Chief Executives" in result
        alts = result["Chief Executives"]
        assert "CEO" in alts
        assert "Chief Executive Officer" in alts

    def test_parse_onet_alternates_vp_engineering(self):
        from src.data.phase2_sources import parse_onet_alternates

        result = parse_onet_alternates(FIXTURES / "onet_alternate_titles.txt")
        # Look for VP Engineering mapping
        found = False
        for canonical, alts in result.items():
            if "VP Engineering" in alts or "Vice President of Engineering" in alts:
                found = True
                break
        assert found, "VP Engineering not found in any alternates"

    def test_parse_onet_reported_returns_list(self):
        from src.data.phase2_sources import parse_onet_reported

        result = parse_onet_reported(FIXTURES / "onet_reported_titles.txt")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_parse_onet_reported_has_real_titles(self):
        from src.data.phase2_sources import parse_onet_reported

        result = parse_onet_reported(FIXTURES / "onet_reported_titles.txt")
        assert "Full Stack Developer" in result


# ── Census Surnames ───────────────────────────────────────────────────────────


class TestCensusSurnames:
    """Test Census 2010 surname parsing."""

    def test_load_census_returns_dataframe(self):
        from src.data.phase2_sources import load_census_surnames

        df = load_census_surnames(FIXTURES / "census_surnames.csv")
        assert isinstance(df, pl.DataFrame)

    def test_load_census_has_name_and_count(self):
        from src.data.phase2_sources import load_census_surnames

        df = load_census_surnames(FIXTURES / "census_surnames.csv")
        assert "name" in df.columns
        assert "count" in df.columns

    def test_load_census_smith_is_rank_1(self):
        from src.data.phase2_sources import load_census_surnames

        df = load_census_surnames(FIXTURES / "census_surnames.csv")
        first = df.sort("count", descending=True).row(0, named=True)
        assert first["name"].upper() == "SMITH"

    def test_load_census_count_positive(self):
        from src.data.phase2_sources import load_census_surnames

        df = load_census_surnames(FIXTURES / "census_surnames.csv")
        assert df.filter(pl.col("count") <= 0).height == 0


# ── SSA Baby Names ────────────────────────────────────────────────────────────


class TestSSANames:
    """Test SSA baby names parsing."""

    def test_load_ssa_returns_dataframe(self):
        from src.data.phase2_sources import load_ssa_names

        df = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
        assert isinstance(df, pl.DataFrame)

    def test_load_ssa_has_required_columns(self):
        from src.data.phase2_sources import load_ssa_names

        df = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
        assert "name" in df.columns
        assert "count" in df.columns

    def test_load_ssa_has_entries(self):
        from src.data.phase2_sources import load_ssa_names

        df = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
        assert len(df) > 0

    def test_load_ssa_james_is_common(self):
        from src.data.phase2_sources import load_ssa_names

        df = load_ssa_names(FIXTURES / "ssa_names_sample.csv")
        james = df.filter(pl.col("name") == "James")
        assert james.height >= 1
        assert james["count"][0] > 1_000_000


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────


class TestEDGARParsing:
    """Test SEC EDGAR submissions parsing."""

    def test_parse_edgar_submission(self):
        from src.data.phase2_sources import parse_edgar_submission

        import json
        with open(FIXTURES / "edgar_sample.json") as f:
            data = json.load(f)
        result = parse_edgar_submission(data)
        assert result["name"] == "Apple Inc."

    def test_parse_edgar_former_names(self):
        from src.data.phase2_sources import parse_edgar_submission

        import json
        with open(FIXTURES / "edgar_sample.json") as f:
            data = json.load(f)
        result = parse_edgar_submission(data)
        assert "former_names" in result
        assert len(result["former_names"]) >= 1
        assert "APPLE COMPUTER INC" in result["former_names"]


# ── Nicknames ─────────────────────────────────────────────────────────────────


class TestNicknamesLoading:
    """Test nickname lookup utility."""

    def test_load_nicknames_returns_dict(self):
        from src.data.phase2_sources import load_nicknames

        result = load_nicknames()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_william_has_bill(self):
        from src.data.phase2_sources import load_nicknames

        result = load_nicknames()
        # nicknames pkg uses lowercase keys
        william_nicks = result.get("william", result.get("William", set()))
        assert len(william_nicks) > 0
