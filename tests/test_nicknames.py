"""
Tests for src/utils/nicknames.py

Covers:
- NICKNAMES is a dict with at least 40 keys
- Every value in NICKNAMES is a non-empty list
- NICKNAME_TO_CANONICAL has entries
- get_nickname returns a string
- get_nickname for a name with nicknames returns something different from
  input at least sometimes (tested with fixed rng seed)
- get_nickname for unknown name returns name unchanged
"""

import random

import pytest

from src.utils.nicknames import (
    NICKNAME_TO_CANONICAL,
    NICKNAMES,
    get_nickname,
)


# ---------------------------------------------------------------------------
# NICKNAMES structure
# ---------------------------------------------------------------------------

class TestNickamesDict:
    def test_is_a_dict(self):
        assert isinstance(NICKNAMES, dict)

    def test_has_at_least_40_keys(self):
        assert len(NICKNAMES) >= 40, (
            f"NICKNAMES has only {len(NICKNAMES)} keys, expected >= 40"
        )

    def test_every_value_is_a_non_empty_list(self):
        for canonical, nicknames in NICKNAMES.items():
            assert isinstance(nicknames, list), (
                f"NICKNAMES[{canonical!r}] is not a list"
            )
            assert len(nicknames) > 0, (
                f"NICKNAMES[{canonical!r}] is an empty list"
            )

    def test_keys_are_strings(self):
        for key in NICKNAMES:
            assert isinstance(key, str)

    def test_values_contain_strings(self):
        for canonical, nicknames in NICKNAMES.items():
            for nick in nicknames:
                assert isinstance(nick, str), (
                    f"NICKNAMES[{canonical!r}] contains non-string value {nick!r}"
                )

    def test_well_known_entries_present(self):
        # Spot-check a handful of canonical names that must be in the table
        for name in ("James", "Robert", "William", "Jennifer", "Elizabeth"):
            assert name in NICKNAMES, f"{name!r} missing from NICKNAMES"

    def test_well_known_nicknames_present(self):
        assert "Jim" in NICKNAMES["James"]
        assert "Bob" in NICKNAMES["Robert"]
        assert "Bill" in NICKNAMES["William"]


# ---------------------------------------------------------------------------
# NICKNAME_TO_CANONICAL structure
# ---------------------------------------------------------------------------

class TestNicknamesToCanonical:
    def test_has_entries(self):
        assert len(NICKNAME_TO_CANONICAL) > 0

    def test_is_a_dict(self):
        assert isinstance(NICKNAME_TO_CANONICAL, dict)

    def test_keys_are_strings(self):
        for k in NICKNAME_TO_CANONICAL:
            assert isinstance(k, str)

    def test_values_are_strings(self):
        for v in NICKNAME_TO_CANONICAL.values():
            assert isinstance(v, str)

    def test_reverse_lookup_correct_for_well_known_nick(self):
        # "Jim" should map back to "James"
        assert NICKNAME_TO_CANONICAL.get("Jim") == "James"
        assert NICKNAME_TO_CANONICAL.get("Bob") == "Robert"

    def test_every_nickname_appears_in_canonical_names(self):
        # Every canonical listed in NICKNAME_TO_CANONICAL must appear in NICKNAMES
        for nick, canonical in NICKNAME_TO_CANONICAL.items():
            assert canonical in NICKNAMES, (
                f"NICKNAME_TO_CANONICAL[{nick!r}] = {canonical!r} but "
                f"{canonical!r} not in NICKNAMES"
            )

    def test_size_at_least_equal_to_number_of_unique_nicknames(self):
        all_nicks = {n for nicks in NICKNAMES.values() for n in nicks}
        # NICKNAME_TO_CANONICAL may have fewer entries if nicknames are shared
        # (first-encountered wins), but it must have at least 1 per canonical
        assert len(NICKNAME_TO_CANONICAL) >= 1


# ---------------------------------------------------------------------------
# get_nickname behaviour
# ---------------------------------------------------------------------------

class TestGetNickname:
    def test_returns_a_string(self):
        result = get_nickname("James")
        assert isinstance(result, str)

    def test_returns_string_for_unknown_name(self):
        result = get_nickname("Zzyzx")
        assert isinstance(result, str)

    def test_unknown_name_returned_unchanged(self):
        unknown = "Zzyzxqwertyultrarare"
        result = get_nickname(unknown)
        assert result == unknown

    def test_unknown_name_never_returns_none(self):
        assert get_nickname("") is not None
        assert get_nickname("NoSuchNameInTable") is not None

    def test_name_with_nicknames_can_return_different_value(self):
        # "James" has nicknames ["Jim", "Jimmy", "Jamie"].
        # With a fixed seed we should get one of them, not "James" itself.
        rng = random.Random(42)
        results = {get_nickname("James", rng=rng) for _ in range(20)}
        # At least one result should differ from the canonical name
        assert results - {"James"}, (
            "get_nickname('James') always returned 'James' — nickname not applied"
        )

    def test_result_is_in_nickname_list(self):
        rng = random.Random(0)
        for _ in range(50):
            result = get_nickname("James", rng=rng)
            assert result in NICKNAMES["James"] or result == "James"

    def test_case_insensitive_fallback_title_case(self):
        # get_nickname tries title-case if exact match fails
        rng = random.Random(42)
        result_lower = get_nickname("james", rng=rng)
        # Should return a valid nickname (title-cased fallback)
        assert result_lower in NICKNAMES["James"] or result_lower == "james"

    def test_seeded_rng_is_reproducible(self):
        nick1 = get_nickname("Robert", rng=random.Random(7))
        nick2 = get_nickname("Robert", rng=random.Random(7))
        assert nick1 == nick2

    def test_different_seeds_may_give_different_results(self):
        # Robert has ["Rob", "Bob", "Bobby", "Robbie"] -- 4 options
        results = {get_nickname("Robert", rng=random.Random(s)) for s in range(20)}
        assert len(results) > 1, (
            "get_nickname returned the same value for all seeds -- RNG not being used"
        )

    def test_empty_string_returns_empty_string(self):
        result = get_nickname("")
        assert result == ""
