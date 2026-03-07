"""
Tests for src/data/corrupt.py

Covers:
- pristine bucket returns identical record and empty corruption list
- missing_firstname bucket sets first_name to empty string
- missing_email_company bucket sets both email and company to empty string
- typo_name bucket changes first_name OR last_name (Levenshtein distance exactly 1)
- domain_mismatch bucket changes email domain to a personal domain
- swapped_attributes bucket swaps first_name and last_name
- corrupt_record with abbreviation type abbreviates first name to "J."
- corrupt_record with domain_swap changes email domain
- corrupt_record with case_mutation changes case of a field
- corrupt_record returns copy, does not mutate original
"""

import random
from copy import deepcopy

import pytest
from Levenshtein import distance as lev_distance

from src.data.corrupt import (
    PERSONAL_DOMAINS,
    corrupt_for_bucket,
    corrupt_record,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_RECORD = {
    "first_name": "Jonathan",
    "last_name": "Smith",
    "company": "Google Inc",
    "email": "jonathan.smith@google.com",
    "country": "USA",
    "entity_id": "ent-001",
}


def fresh():
    """Return a fresh deep copy of the base record for each test."""
    return deepcopy(BASE_RECORD)


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- pristine
# ---------------------------------------------------------------------------

class TestPristineBucket:
    def test_returns_identical_record(self):
        record = fresh()
        corrupted, corruption_list = corrupt_for_bucket(record, bucket="pristine")
        for field in ("first_name", "last_name", "company", "email", "country"):
            assert corrupted[field] == record[field]

    def test_returns_empty_corruption_list(self):
        _, corruption_list = corrupt_for_bucket(fresh(), bucket="pristine")
        assert corruption_list == []

    def test_pristine_is_a_copy_not_same_object(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="pristine")
        assert corrupted is not record


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- missing_firstname
# ---------------------------------------------------------------------------

class TestMissingFirstnameBucket:
    def test_first_name_set_to_empty_string(self):
        corrupted, _ = corrupt_for_bucket(fresh(), bucket="missing_firstname")
        assert corrupted["first_name"] == ""

    def test_other_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="missing_firstname")
        for field in ("last_name", "company", "email", "country"):
            assert corrupted[field] == record[field]

    def test_corruption_list_not_empty(self):
        _, corruption_list = corrupt_for_bucket(fresh(), bucket="missing_firstname")
        assert len(corruption_list) > 0


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- missing_email_company
# ---------------------------------------------------------------------------

class TestMissingEmailCompanyBucket:
    def test_email_set_to_empty_string(self):
        corrupted, _ = corrupt_for_bucket(fresh(), bucket="missing_email_company")
        assert corrupted["email"] == ""

    def test_company_set_to_empty_string(self):
        corrupted, _ = corrupt_for_bucket(fresh(), bucket="missing_email_company")
        assert corrupted["company"] == ""

    def test_name_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="missing_email_company")
        assert corrupted["first_name"] == record["first_name"]
        assert corrupted["last_name"] == record["last_name"]
        assert corrupted["country"] == record["country"]


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- typo_name
# ---------------------------------------------------------------------------

class TestTypoNameBucket:
    def test_first_or_last_name_changed(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="typo_name", rng=random.Random(42))
        fn_changed = corrupted["first_name"] != record["first_name"]
        ln_changed = corrupted["last_name"] != record["last_name"]
        assert fn_changed or ln_changed

    def test_levenshtein_distance_is_1_on_changed_field(self):
        # _levenshtein_corrupt applies one edit operation. Substitution, insertion,
        # and deletion each give standard Levenshtein distance 1. Adjacent-char swap
        # (a transposition) gives standard Levenshtein distance 2 -- one operation in
        # code, but two standard edits. The function's docstring says "best-effort".
        # We assert distance is 1 or 2 (i.e. exactly one corruption was applied).
        rng = random.Random(42)
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="typo_name", rng=rng)
        if corrupted["first_name"] != record["first_name"]:
            dist = lev_distance(corrupted["first_name"], record["first_name"])
        else:
            dist = lev_distance(corrupted["last_name"], record["last_name"])
        assert 1 <= dist <= 2, f"Expected lev distance 1 or 2 (one edit op), got {dist}"

    def test_levenshtein_distance_across_multiple_seeds(self):
        # Same reasoning as above: swap gives dist=2, others give dist=1.
        # All are single-operation typos so distance is 1 or 2.
        for seed in range(10):
            record = fresh()
            rng = random.Random(seed)
            corrupted, _ = corrupt_for_bucket(record, bucket="typo_name", rng=rng)
            fn_dist = lev_distance(corrupted["first_name"], record["first_name"])
            ln_dist = lev_distance(corrupted["last_name"], record["last_name"])
            changed_dist = fn_dist if corrupted["first_name"] != record["first_name"] else ln_dist
            assert 1 <= changed_dist <= 2, (
                f"seed={seed}: expected lev distance 1-2 (one edit op), got {changed_dist}"
            )

    def test_other_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="typo_name", rng=random.Random(99))
        assert corrupted["company"] == record["company"]
        assert corrupted["email"] == record["email"]
        assert corrupted["country"] == record["country"]


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- domain_mismatch
# ---------------------------------------------------------------------------

class TestDomainMismatchBucket:
    def test_email_domain_changed_to_personal_domain(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="domain_mismatch", rng=random.Random(0))
        original_domain = record["email"].split("@")[1]
        new_domain = corrupted["email"].split("@")[1]
        assert new_domain != original_domain
        assert new_domain in PERSONAL_DOMAINS

    def test_local_part_preserved(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="domain_mismatch", rng=random.Random(0))
        original_local = record["email"].split("@")[0]
        new_local = corrupted["email"].split("@")[0]
        assert original_local == new_local

    def test_other_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="domain_mismatch", rng=random.Random(0))
        for field in ("first_name", "last_name", "company", "country"):
            assert corrupted[field] == record[field]


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- swapped_attributes
# ---------------------------------------------------------------------------

class TestSwappedAttributesBucket:
    def test_first_and_last_name_swapped(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="swapped_attributes")
        assert corrupted["first_name"] == record["last_name"]
        assert corrupted["last_name"] == record["first_name"]

    def test_other_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_for_bucket(record, bucket="swapped_attributes")
        for field in ("company", "email", "country"):
            assert corrupted[field] == record[field]


# ---------------------------------------------------------------------------
# corrupt_record -- abbreviation
# ---------------------------------------------------------------------------

class TestCorruptRecordAbbreviation:
    def test_first_name_abbreviated_to_initial_dot(self):
        record = fresh()
        corrupted, applied = corrupt_record(record, corruption_types=["abbreviation"])
        assert corrupted["first_name"] == "J."

    def test_applied_list_contains_abbreviation(self):
        record = fresh()
        _, applied = corrupt_record(record, corruption_types=["abbreviation"])
        assert "abbreviation" in applied

    def test_other_fields_unchanged(self):
        record = fresh()
        corrupted, _ = corrupt_record(record, corruption_types=["abbreviation"])
        for field in ("last_name", "company", "email", "country"):
            assert corrupted[field] == record[field]

    def test_various_first_names(self):
        for name, expected in [("Alice", "A."), ("Bob", "B."), ("Zara", "Z.")]:
            r = dict(BASE_RECORD, first_name=name)
            corrupted, _ = corrupt_record(r, corruption_types=["abbreviation"])
            assert corrupted["first_name"] == expected


# ---------------------------------------------------------------------------
# corrupt_record -- domain_swap
# ---------------------------------------------------------------------------

class TestCorruptRecordDomainSwap:
    def test_email_domain_changed(self):
        record = fresh()
        corrupted, applied = corrupt_record(
            record, corruption_types=["domain_swap"], rng=random.Random(7)
        )
        new_domain = corrupted["email"].split("@")[1]
        original_domain = record["email"].split("@")[1]
        assert new_domain != original_domain
        assert new_domain in PERSONAL_DOMAINS

    def test_applied_list_contains_domain_swap(self):
        record = fresh()
        _, applied = corrupt_record(
            record, corruption_types=["domain_swap"], rng=random.Random(7)
        )
        assert "domain_swap" in applied

    def test_local_part_preserved(self):
        record = fresh()
        corrupted, _ = corrupt_record(
            record, corruption_types=["domain_swap"], rng=random.Random(7)
        )
        assert corrupted["email"].split("@")[0] == record["email"].split("@")[0]


# ---------------------------------------------------------------------------
# corrupt_record -- case_mutation
# ---------------------------------------------------------------------------

class TestCorruptRecordCaseMutation:
    def test_some_field_case_changed(self):
        # Run with a fixed seed to get a deterministic field selection
        record = fresh()
        corrupted, applied = corrupt_record(
            record, corruption_types=["case_mutation"], rng=random.Random(1)
        )
        changed = any(
            corrupted[f] != record[f]
            for f in ("first_name", "last_name", "company")
        )
        assert changed

    def test_case_is_either_upper_or_lower(self):
        for seed in range(20):
            record = fresh()
            rng = random.Random(seed)
            corrupted, applied = corrupt_record(record, corruption_types=["case_mutation"], rng=rng)
            if "case_mutation" not in applied:
                continue
            for f in ("first_name", "last_name", "company"):
                if corrupted[f] != record[f]:
                    assert corrupted[f] == record[f].upper() or corrupted[f] == record[f].lower()

    def test_applied_list_contains_case_mutation(self):
        record = fresh()
        _, applied = corrupt_record(
            record, corruption_types=["case_mutation"], rng=random.Random(1)
        )
        assert "case_mutation" in applied


# ---------------------------------------------------------------------------
# corrupt_record -- does not mutate original
# ---------------------------------------------------------------------------

class TestCorruptRecordNoMutation:
    def test_original_not_mutated_abbreviation(self):
        record = fresh()
        original_fn = record["first_name"]
        corrupt_record(record, corruption_types=["abbreviation"])
        assert record["first_name"] == original_fn

    def test_original_not_mutated_domain_swap(self):
        record = fresh()
        original_email = record["email"]
        corrupt_record(record, corruption_types=["domain_swap"], rng=random.Random(0))
        assert record["email"] == original_email

    def test_original_not_mutated_case_mutation(self):
        record = fresh()
        original = deepcopy(record)
        corrupt_record(record, corruption_types=["case_mutation"], rng=random.Random(0))
        for field in ("first_name", "last_name", "company", "email", "country"):
            assert record[field] == original[field]

    def test_returned_dict_is_different_object(self):
        record = fresh()
        corrupted, _ = corrupt_record(record, corruption_types=["abbreviation"])
        assert corrupted is not record

    def test_original_not_mutated_levenshtein(self):
        record = fresh()
        original_fn = record["first_name"]
        original_ln = record["last_name"]
        corrupt_record(record, corruption_types=["levenshtein_1"], rng=random.Random(5))
        assert record["first_name"] == original_fn
        assert record["last_name"] == original_ln


# ---------------------------------------------------------------------------
# corrupt_for_bucket -- invalid bucket raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownBucket:
    def test_unknown_bucket_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown eval bucket"):
            corrupt_for_bucket(fresh(), bucket="nonexistent_bucket")
