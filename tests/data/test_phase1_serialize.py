"""
Tests for src/data/serialize.py

Covers:
- serialize_pipe with all 5 fields present
- serialize_kv with all 5 fields present
- serialize_pipe with missing first_name (empty string slot preserved)
- serialize_kv with missing first_name (fn: explicit)
- serialize_pipe with multiple missing fields
- round-trip pipe: serialize then deserialize equals original
- round-trip kv: serialize then deserialize equals original
- serialize() dispatcher with both fmt values
- serialize() with invalid fmt raises ValueError
- field order is correct in pipe output
"""

import pytest

from src.data.serialize import (
    FIELD_ORDER,
    deserialize_kv,
    deserialize_pipe,
    serialize,
    serialize_kv,
    serialize_pipe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FULL_RECORD = {
    "first_name": "Jonathan",
    "last_name": "Smith",
    "company": "Google Inc",
    "email": "jonathan.smith@google.com",
    "country": "USA",
}

MISSING_FN_RECORD = {
    "first_name": "",
    "last_name": "Smith",
    "company": "Google Inc",
    "email": "jonathan.smith@google.com",
    "country": "USA",
}

MULTI_MISSING_RECORD = {
    "first_name": "",
    "last_name": "Smith",
    "company": "",
    "email": "",
    "country": "USA",
}


# ---------------------------------------------------------------------------
# serialize_pipe
# ---------------------------------------------------------------------------

class TestSerializePipe:
    def test_all_fields_present(self):
        result = serialize_pipe(FULL_RECORD)
        assert result == "Jonathan | Smith | Google Inc | jonathan.smith@google.com | USA"

    def test_missing_first_name_slot_preserved(self):
        result = serialize_pipe(MISSING_FN_RECORD)
        # Empty first_name slot: leading " | "
        assert result.startswith(" | ")
        parts = result.split(" | ")
        assert len(parts) == 5
        assert parts[0] == ""
        assert parts[1] == "Smith"

    def test_multiple_missing_fields(self):
        result = serialize_pipe(MULTI_MISSING_RECORD)
        parts = result.split(" | ")
        assert len(parts) == 5
        assert parts[0] == ""       # first_name
        assert parts[1] == "Smith"  # last_name
        assert parts[2] == ""       # company
        assert parts[3] == ""       # email
        assert parts[4] == "USA"    # country

    def test_field_order_correct(self):
        result = serialize_pipe(FULL_RECORD)
        parts = result.split(" | ")
        assert parts[0] == FULL_RECORD["first_name"]
        assert parts[1] == FULL_RECORD["last_name"]
        assert parts[2] == FULL_RECORD["company"]
        assert parts[3] == FULL_RECORD["email"]
        assert parts[4] == FULL_RECORD["country"]

    def test_field_order_matches_field_order_constant(self):
        # FIELD_ORDER drives the split — verify index alignment
        result = serialize_pipe(FULL_RECORD)
        parts = result.split(" | ")
        assert len(parts) == len(FIELD_ORDER)
        for i, field in enumerate(FIELD_ORDER):
            assert parts[i] == FULL_RECORD[field]

    def test_extra_keys_ignored(self):
        record = dict(FULL_RECORD)
        record["entity_id"] = "abc-123"
        record["score"] = 0.99
        result = serialize_pipe(record)
        # Should not contain the extra keys
        assert "entity_id" not in result
        assert "abc-123" not in result

    def test_none_value_treated_as_empty(self):
        record = dict(FULL_RECORD)
        record["first_name"] = None
        result = serialize_pipe(record)
        parts = result.split(" | ")
        assert parts[0] == ""


# ---------------------------------------------------------------------------
# serialize_kv
# ---------------------------------------------------------------------------

class TestSerializeKv:
    def test_all_fields_present(self):
        result = serialize_kv(FULL_RECORD)
        assert result == (
            "fn:Jonathan ln:Smith org:Google Inc em:jonathan.smith@google.com co:USA"
        )

    def test_missing_first_name_fn_explicit(self):
        result = serialize_kv(MISSING_FN_RECORD)
        # fn key must be present even when empty
        assert "fn:" in result
        tokens = result.split(" ")
        fn_token = next(t for t in tokens if t.startswith("fn:"))
        assert fn_token == "fn:"

    def test_all_keys_always_present(self):
        result = serialize_kv(MULTI_MISSING_RECORD)
        for key in ("fn:", "ln:", "org:", "em:", "co:"):
            assert key in result

    def test_key_order(self):
        result = serialize_kv(FULL_RECORD)
        fn_pos = result.index("fn:")
        ln_pos = result.index("ln:")
        org_pos = result.index("org:")
        em_pos = result.index("em:")
        co_pos = result.index("co:")
        assert fn_pos < ln_pos < org_pos < em_pos < co_pos

    def test_none_value_treated_as_empty(self):
        record = dict(FULL_RECORD)
        record["email"] = None
        result = serialize_kv(record)
        assert "em:" in result
        # Value after em: should be empty before the next key
        assert "em: co:" in result or result.endswith("em:")


# ---------------------------------------------------------------------------
# Round-trip: pipe
# ---------------------------------------------------------------------------

class TestRoundTripPipe:
    def test_full_record_round_trip(self):
        serialized = serialize_pipe(FULL_RECORD)
        recovered = deserialize_pipe(serialized)
        for field in FIELD_ORDER:
            assert recovered[field] == FULL_RECORD[field], (
                f"Field {field!r} mismatch: {recovered[field]!r} != {FULL_RECORD[field]!r}"
            )

    def test_missing_first_name_round_trip(self):
        serialized = serialize_pipe(MISSING_FN_RECORD)
        recovered = deserialize_pipe(serialized)
        assert recovered["first_name"] == ""
        assert recovered["last_name"] == MISSING_FN_RECORD["last_name"]
        assert recovered["email"] == MISSING_FN_RECORD["email"]

    def test_multi_missing_round_trip(self):
        serialized = serialize_pipe(MULTI_MISSING_RECORD)
        recovered = deserialize_pipe(serialized)
        for field in FIELD_ORDER:
            assert recovered[field] == MULTI_MISSING_RECORD[field]


# ---------------------------------------------------------------------------
# Round-trip: kv
# ---------------------------------------------------------------------------

class TestRoundTripKv:
    def test_full_record_round_trip(self):
        serialized = serialize_kv(FULL_RECORD)
        recovered = deserialize_kv(serialized)
        for field in FIELD_ORDER:
            assert recovered[field] == FULL_RECORD[field], (
                f"Field {field!r} mismatch: {recovered[field]!r} != {FULL_RECORD[field]!r}"
            )

    def test_missing_first_name_round_trip(self):
        serialized = serialize_kv(MISSING_FN_RECORD)
        recovered = deserialize_kv(serialized)
        assert recovered["first_name"] == ""
        assert recovered["last_name"] == MISSING_FN_RECORD["last_name"]

    def test_multi_missing_round_trip(self):
        serialized = serialize_kv(MULTI_MISSING_RECORD)
        recovered = deserialize_kv(serialized)
        for field in FIELD_ORDER:
            assert recovered[field] == MULTI_MISSING_RECORD[field]

    def test_company_with_spaces_round_trip(self):
        record = dict(FULL_RECORD)
        record["company"] = "Acme Global Solutions"
        serialized = serialize_kv(record)
        recovered = deserialize_kv(serialized)
        assert recovered["company"] == "Acme Global Solutions"


# ---------------------------------------------------------------------------
# serialize() dispatcher
# ---------------------------------------------------------------------------

class TestSerializeDispatcher:
    def test_pipe_fmt_dispatches_to_pipe(self):
        result = serialize(FULL_RECORD, fmt="pipe")
        expected = serialize_pipe(FULL_RECORD)
        assert result == expected

    def test_kv_fmt_dispatches_to_kv(self):
        result = serialize(FULL_RECORD, fmt="kv")
        expected = serialize_kv(FULL_RECORD)
        assert result == expected

    def test_invalid_fmt_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown serialization format"):
            serialize(FULL_RECORD, fmt="json")

    def test_invalid_fmt_empty_string_raises(self):
        with pytest.raises(ValueError):
            serialize(FULL_RECORD, fmt="")

    def test_invalid_fmt_case_sensitive(self):
        with pytest.raises(ValueError):
            serialize(FULL_RECORD, fmt="PIPE")
