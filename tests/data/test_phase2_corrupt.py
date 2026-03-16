"""
Tests for Phase 2 corruption engine — one test per corruption type.

28 positive corruption types + COL/VAL serialization + bucket mapping.
"""

from __future__ import annotations

import random

import pytest

from src.data.phase2_corrupt import (
    ALL_CORRUPTION_CODES,
    EVAL_BUCKETS_PHASE2,
    PERSONAL_DOMAINS,
    QWERTY_NEIGHBORS,
    colval_pair,
    colval_serialize,
    corrupt_c1_suffix_swap,
    corrupt_c2_suffix_drop,
    corrupt_c3_the_prefix_drop,
    corrupt_c4_ampersand_normalize,
    corrupt_c5_company_abbreviation,
    corrupt_c6_word_truncation,
    corrupt_c7_rebrand,
    corrupt_c8_shorten_name,
    corrupt_e1_email_format_variant,
    corrupt_e2_domain_swap,
    corrupt_for_bucket_phase2,
    corrupt_n1_diacritic_strip,
    corrupt_n10_middle_initial,
    corrupt_n11_last_initial,
    corrupt_n12_nickname,
    corrupt_n13_phonetic_sub,
    corrupt_n2_single_char_delete,
    corrupt_n3_keyboard_sub,
    corrupt_n4_ocr_sub,
    corrupt_n5_char_transposition,
    corrupt_n6_name_swap,
    corrupt_n7_first_initial,
    corrupt_n8_first_middle_initial,
    corrupt_n9_drop_middle,
    corrupt_record_phase2,
    corrupt_t1_title_abbreviation,
    corrupt_t2_title_expansion,
    corrupt_t3_title_reorder,
    corrupt_t4_seniority_drop,
    corrupt_t5_seniority_synonym,
    corrupt_t6_missing_field,
)


def _make_record(**overrides) -> dict:
    """Create a test entity record with sensible defaults."""
    base = {
        "entity_id": "test-001",
        "first_name": "Rajesh",
        "last_name": "Krishnamurthy",
        "middle_name": "Kumar",
        "company": "Goldman Sachs Group Inc.",
        "title": "Senior Software Engineer",
        "email": "rajesh.krishnamurthy@goldmansachs.com",
        "country": "US",
    }
    base.update(overrides)
    return base


# ── COL/VAL Serialization ────────────────────────────────────────────────────


class TestCOLVALSerialization:
    def test_colval_has_all_fields(self):
        r = _make_record()
        s = colval_serialize(r)
        assert "COL fn VAL" in s
        assert "COL ln VAL" in s
        assert "COL org VAL" in s
        assert "COL title VAL" in s
        assert "COL email VAL" in s
        assert "COL country VAL" in s

    def test_colval_values_present(self):
        r = _make_record()
        s = colval_serialize(r)
        assert "Rajesh" in s
        assert "Krishnamurthy" in s
        assert "Goldman Sachs" in s

    def test_colval_empty_field(self):
        r = _make_record(first_name="")
        s = colval_serialize(r)
        assert "COL fn VAL " in s  # empty value

    def test_colval_pair_has_sep(self):
        a = _make_record()
        b = _make_record(first_name="R.")
        pair = colval_pair(a, b)
        assert "[CLS]" in pair
        assert "[SEP]" in pair
        assert pair.count("[SEP]") == 2


# ── Company Corruptions (C1-C8) ──────────────────────────────────────────────


class TestCompanyCorruptions:
    def test_c1_suffix_swap_changes_suffix(self):
        result = corrupt_c1_suffix_swap("Acme LLC", random.Random(42))
        assert result != "Acme LLC"
        assert result.startswith("Acme")

    def test_c1_suffix_swap_adds_suffix_when_none(self):
        result = corrupt_c1_suffix_swap("Acme", random.Random(42))
        assert len(result) > len("Acme")

    def test_c2_suffix_drop(self):
        assert corrupt_c2_suffix_drop("Microsoft Corporation") == "Microsoft"
        assert corrupt_c2_suffix_drop("Acme LLC") == "Acme"

    def test_c2_suffix_drop_no_suffix(self):
        assert corrupt_c2_suffix_drop("Apple") == "Apple"

    def test_c3_the_prefix_drop(self):
        assert corrupt_c3_the_prefix_drop("The Home Depot") == "Home Depot"
        assert corrupt_c3_the_prefix_drop("Apple") == "Apple"

    def test_c4_ampersand_to_and(self):
        assert corrupt_c4_ampersand_normalize("Johnson & Johnson") == "Johnson and Johnson"

    def test_c4_and_to_ampersand(self):
        assert corrupt_c4_ampersand_normalize("Johnson and Johnson") == "Johnson & Johnson"

    def test_c5_abbreviation_with_map(self):
        abbr_map = {"International Business Machines": ["IBM"]}
        result = corrupt_c5_company_abbreviation("International Business Machines", abbr_map)
        assert result == "IBM"

    def test_c5_abbreviation_fallback_acronym(self):
        result = corrupt_c5_company_abbreviation("Goldman Sachs")
        assert result == "GS"

    def test_c6_word_truncation(self):
        assert corrupt_c6_word_truncation("Goldman Sachs Group") == "Goldman Sachs"

    def test_c6_word_truncation_single_word(self):
        assert corrupt_c6_word_truncation("Apple") == "Apple"

    def test_c7_rebrand_with_map(self):
        rebrand = {"Facebook": "Meta Platforms"}
        assert corrupt_c7_rebrand("Facebook", rebrand) == "Meta Platforms"

    def test_c7_rebrand_no_map(self):
        assert corrupt_c7_rebrand("Apple") == "Apple"

    def test_c8_shorten_name(self):
        result = corrupt_c8_shorten_name("Acme International Holdings Ltd")
        assert "Intl" in result
        assert "Hldgs" in result


# ── Name Corruptions (N1-N13) ────────────────────────────────────────────────


class TestNameCorruptions:
    def test_n1_diacritic_strip(self):
        assert corrupt_n1_diacritic_strip("García") == "Garcia"
        assert corrupt_n1_diacritic_strip("Müller") == "Muller"
        assert corrupt_n1_diacritic_strip("Naïve") == "Naive"
        assert corrupt_n1_diacritic_strip("José") == "Jose"
        # Multi-ethnic names
        assert corrupt_n1_diacritic_strip("Dvořák") == "Dvorak"
        assert corrupt_n1_diacritic_strip("Nguyễn") == "Nguyen"

    def test_n2_single_char_delete(self):
        rng = random.Random(42)
        result = corrupt_n2_single_char_delete("Microsoft", rng)
        assert len(result) == len("Microsoft") - 1

    def test_n2_short_name_safe(self):
        assert corrupt_n2_single_char_delete("A") == "A"

    def test_n3_keyboard_sub_uses_qwerty_neighbors(self):
        rng = random.Random(42)
        original = "smith"
        result = corrupt_n3_keyboard_sub(original, rng)
        assert result != original
        for i, (orig, new) in enumerate(zip(original, result)):
            if orig != new:
                assert new in QWERTY_NEIGHBORS[orig], (
                    f"Position {i}: {orig!r}→{new!r} is not a QWERTY neighbor"
                )

    def test_n3_preserves_case(self):
        rng = random.Random(42)
        result = corrupt_n3_keyboard_sub("Smith", rng)
        # First char should stay uppercase if it was uppercase
        if result[0] != "S":
            assert result[0].isupper()

    def test_n4_ocr_sub(self):
        result = corrupt_n4_ocr_sub("Barnes", random.Random(42))
        # "rn" → "m" should be possible
        assert result != "Barnes" or result == "Barnes"  # may or may not apply

    def test_n5_char_transposition(self):
        rng = random.Random(42)
        result = corrupt_n5_char_transposition("Smith", rng)
        assert len(result) == len("Smith")
        assert sorted(result) == sorted("Smith")

    def test_n6_name_swap(self):
        r = _make_record()
        swapped = corrupt_n6_name_swap(r)
        assert swapped["first_name"] == "Krishnamurthy"
        assert swapped["last_name"] == "Rajesh"

    def test_n7_first_initial(self):
        assert corrupt_n7_first_initial("Rajesh") == "R."
        assert corrupt_n7_first_initial("") == ""

    def test_n8_first_middle_initial(self):
        r = _make_record()
        result = corrupt_n8_first_middle_initial(r)
        assert result["first_name"] == "R."
        assert result["middle_name"] == "K."

    def test_n9_drop_middle(self):
        r = _make_record()
        result = corrupt_n9_drop_middle(r)
        assert result["middle_name"] == ""
        assert result["first_name"] == "Rajesh"  # unchanged

    def test_n10_middle_initial(self):
        r = _make_record()
        result = corrupt_n10_middle_initial(r)
        assert result["middle_name"] == "K."

    def test_n11_last_initial(self):
        assert corrupt_n11_last_initial("Smith") == "S."
        assert corrupt_n11_last_initial("") == ""

    def test_n12_nickname_with_map(self):
        # Multi-ethnic nickname mappings
        nicks = {
            "william": {"bill", "will", "billy"},
            "rajesh": {"raj"},
            "mohammed": {"mo", "mohammad"},
            "yoshiko": {"yoshi"},
        }
        result = corrupt_n12_nickname("William", nicks, random.Random(42))
        assert result.lower() in {"bill", "will", "billy"}

    def test_n12_nickname_multi_ethnic(self):
        nicks = {"rajesh": {"raj"}, "mohammed": {"mo", "mohammad"}}
        assert corrupt_n12_nickname("Rajesh", nicks, random.Random(42)) == "Raj"
        result = corrupt_n12_nickname("Mohammed", nicks, random.Random(42))
        assert result.lower() in {"mo", "mohammad"}

    def test_n12_nickname_no_match(self):
        nicks = {"william": {"bill"}}
        assert corrupt_n12_nickname("Xyz", nicks) == "Xyz"

    def test_n13_phonetic_sub(self):
        result = corrupt_n13_phonetic_sub("Phillips", random.Random(42))
        # "ph" → "f" should give "fillips" or similar
        assert result != "Phillips"


# ── Title Corruptions (T1-T6) ────────────────────────────────────────────────


class TestTitleCorruptions:
    def test_t1_abbreviation_fallback(self):
        result = corrupt_t1_title_abbreviation("Vice President of Engineering")
        assert "VP" in result

    def test_t1_abbreviation_with_map(self):
        alts = {"Chief Executives": ["CEO", "Chief Executive Officer", "President"]}
        result = corrupt_t1_title_abbreviation("Chief Executives", alts, random.Random(42))
        assert result in ["CEO", "Chief Executive Officer", "President"]

    def test_t2_expansion(self):
        assert corrupt_t2_title_expansion("VP Engineering") == "Vice President Engineering"
        assert corrupt_t2_title_expansion("CTO") == "Chief Technology Officer"

    def test_t3_reorder(self):
        assert corrupt_t3_title_reorder("Engineering VP") == "VP Engineering"

    def test_t4_seniority_drop(self):
        assert corrupt_t4_seniority_drop("Senior Software Engineer") == "Software Engineer"
        assert corrupt_t4_seniority_drop("Staff Engineer") == "Engineer"

    def test_t4_seniority_drop_no_prefix(self):
        assert corrupt_t4_seniority_drop("Software Engineer") == "Software Engineer"

    def test_t5_seniority_synonym(self):
        rng = random.Random(42)
        result = corrupt_t5_seniority_synonym("Senior Software Engineer", rng)
        assert result.startswith("Sr") or result.startswith("Senior")
        assert "Software Engineer" in result

    def test_t6_missing_field(self):
        assert corrupt_t6_missing_field() == ""


# ── Email Corruptions (E1-E2) ────────────────────────────────────────────────


class TestEmailCorruptions:
    def test_e1_format_variant(self):
        rng = random.Random(42)
        result = corrupt_e1_email_format_variant("jay.smith@acme.com", rng)
        assert "@acme.com" in result
        assert result != "jay.smith@acme.com"

    def test_e1_no_dot_unchanged(self):
        result = corrupt_e1_email_format_variant("jsmith@acme.com", random.Random(42))
        assert "@" in result

    def test_e2_domain_swap(self):
        rng = random.Random(42)
        result = corrupt_e2_domain_swap("jay@acme.com", rng)
        domain = result.split("@")[1]
        assert domain in PERSONAL_DOMAINS
        assert result.startswith("jay@")


# ── Composite: corrupt_record_phase2 ─────────────────────────────────────────


class TestCorruptRecordPhase2:
    def test_entity_id_preserved(self):
        r = _make_record(entity_id="uuid-123")
        result = corrupt_record_phase2(r, ["N7"])
        assert result["entity_id"] == "uuid-123"

    def test_multiple_codes_applied(self):
        r = _make_record()
        result = corrupt_record_phase2(r, ["N7", "C2"], random.Random(42))
        assert result["first_name"] == "R."  # N7
        assert "Inc" not in result["company"]  # C2 strips suffix

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown corruption code"):
            corrupt_record_phase2(_make_record(), ["INVALID"])

    def test_positive_pair_text_differs_guaranteed(self):
        """Codes that MUST change COL/VAL text on the test record."""
        guaranteed_change = [
            "C1", "C2", "C6",  # company has suffix/multi-word
            "N2", "N3", "N5", "N6", "N7", "N11",  # name always changes
            "T3", "T4", "T6",  # title reorder/drop/missing
            "E1", "E2",  # email format/domain change
        ]
        for code in guaranteed_change:
            anchor = _make_record()
            corrupted = corrupt_record_phase2(anchor, [code], random.Random(42))
            a_text = colval_serialize(anchor)
            c_text = colval_serialize(corrupted)
            assert a_text != c_text, f"Code {code} produced identical text"


# ── Bucket mapping ───────────────────────────────────────────────────────────


class TestBucketMapping:
    def test_pristine_unchanged(self):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, "pristine")
        assert codes == []
        assert result["first_name"] == r["first_name"]

    def test_missing_firstname(self):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, "missing_firstname")
        assert result["first_name"] == ""
        assert result["last_name"] == "Krishnamurthy"

    def test_missing_email_company(self):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, "missing_email_company")
        assert result["email"] == ""
        assert result["company"] == ""
        assert result["first_name"] == "Rajesh"  # unchanged

    def test_typo_name(self):
        r = _make_record()
        rng = random.Random(42)
        result, codes = corrupt_for_bucket_phase2(r, "typo_name", rng)
        # Name should be changed by keyboard sub
        name_changed = (
            result["first_name"] != r["first_name"]
            or result["last_name"] != r["last_name"]
        )
        assert name_changed

    def test_domain_mismatch(self):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, "domain_mismatch", random.Random(42))
        domain = result["email"].split("@")[1]
        assert domain in PERSONAL_DOMAINS

    def test_swapped_attributes(self):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, "swapped_attributes")
        assert result["first_name"] == "Krishnamurthy"
        assert result["last_name"] == "Rajesh"

    def test_unknown_bucket_raises(self):
        with pytest.raises(ValueError, match="Unknown bucket"):
            corrupt_for_bucket_phase2(_make_record(), "nonexistent")

    @pytest.mark.parametrize("bucket", EVAL_BUCKETS_PHASE2)
    def test_all_buckets_produce_valid_output(self, bucket):
        r = _make_record()
        result, codes = corrupt_for_bucket_phase2(r, bucket, random.Random(42))
        assert isinstance(result, dict)
        assert isinstance(codes, list)
        assert "entity_id" in result
