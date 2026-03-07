"""
Corruption engine for entity profiles.

Implements all corruption types defined in configs/dataset.yaml and provides:
  corrupt_record()      — apply one or more corruptions to a record
  corrupt_for_bucket()  — apply bucket-specific corruption for eval set

All output field values are strings; empty string signals a dropped field.
"""

from __future__ import annotations

import random
import re
import string
from copy import deepcopy

from Levenshtein import distance as lev_distance  # noqa: F401  (also used for validation)
from src.utils.nicknames import get_nickname

# ── Constants ─────────────────────────────────────────────────────────────────

PERSONAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]

_COMPANY_SUFFIXES = [
    "Inc", "LLC", "Ltd", "Corp", "Corporation", "Co", "Company",
    "Group", "Holdings", "International", "Global", "Solutions",
    "Technologies", "Services", "Consulting", "Partners",
]

# All valid corruption type names (mirrors dataset.yaml keys + bucket names)
CORRUPTION_TYPES = [
    "abbreviation",
    "truncation",
    "levenshtein_1",
    "levenshtein_2",
    "field_drop_single",
    "field_drop_double",
    "domain_swap",
    "company_abbrev",
    "case_mutation",
    "nickname",
]

EVAL_BUCKETS = [
    "pristine",
    "missing_firstname",
    "missing_email_company",
    "typo_name",
    "domain_mismatch",
    "swapped_attributes",
]

PROFILE_FIELDS = ["first_name", "last_name", "company", "email", "country"]


# ── Private corruption primitives ─────────────────────────────────────────────

def _abbreviate_first_name(name: str) -> str:
    """'Jay' → 'J.'"""
    if not name:
        return name
    return name[0].upper() + "."


def _truncate(s: str, min_len: int = 2, rng: random.Random | None = None) -> str:
    """Random truncation keeping at least min_len chars."""
    _rng = rng or random
    if not s or len(s) <= min_len:
        return s
    # Keep at least min_len, at most len(s)-1 chars
    max_keep = max(min_len, len(s) - 1)
    keep = _rng.randint(min_len, max_keep)
    return s[:keep]


def _levenshtein_corrupt(s: str, n_edits: int = 1, rng: random.Random | None = None) -> str:
    """
    Apply n_edits random edits to s, producing a string with
    lev_distance(result, s) == n_edits (best-effort; may be less if s is very short).

    Edit operations (chosen uniformly):
      - swap adjacent chars
      - insert random lowercase letter at random position
      - delete a random character
      - substitute a random character with a random letter
    """
    _rng = rng or random
    if not s:
        return s

    result = list(s)

    for _ in range(n_edits):
        if not result:
            break
        op = _rng.randint(0, 3)

        if op == 0 and len(result) >= 2:
            # Swap two adjacent chars
            i = _rng.randint(0, len(result) - 2)
            result[i], result[i + 1] = result[i + 1], result[i]

        elif op == 1:
            # Insert random char
            i = _rng.randint(0, len(result))
            ch = _rng.choice(string.ascii_lowercase)
            result.insert(i, ch)

        elif op == 2 and len(result) > 1:
            # Delete a char
            i = _rng.randint(0, len(result) - 1)
            result.pop(i)

        else:
            # Substitute a char
            i = _rng.randint(0, len(result) - 1)
            ch = _rng.choice(string.ascii_lowercase)
            result[i] = ch

    return "".join(result)


def _drop_field(record: dict, field: str) -> dict:
    """Return a copy of record with `field` set to empty string."""
    r = dict(record)
    r[field] = ""
    return r


def _swap_email_domain(email: str, rng: random.Random | None = None) -> str:
    """Replace the domain part of an email with a random personal domain."""
    _rng = rng or random
    if not email or "@" not in email:
        return email
    local = email.split("@")[0]
    domain = _rng.choice(PERSONAL_DOMAINS)
    return f"{local}@{domain}"


def _abbreviate_company(company: str, rng: random.Random | None = None) -> str:
    """
    'Acme Corporation' → 'Acme Corp' or 'Acme' (random).

    Strategy:
      1. Strip known suffixes, yielding a "base" (e.g. "Acme").
      2. 50% chance: return base only; 50%: base + shortened suffix.
    """
    _rng = rng or random
    if not company:
        return company

    # Strip trailing punctuation
    company = company.rstrip(".,")

    # Try to strip a known suffix
    words = company.split()
    base_words = words[:]
    stripped_suffix: str | None = None

    for suffix in sorted(_COMPANY_SUFFIXES, key=len, reverse=True):
        if len(words) > 1 and words[-1].rstrip(".") == suffix:
            base_words = words[:-1]
            stripped_suffix = suffix
            break
        # Multi-word suffix check (e.g. "and Company")
        if len(words) > 2 and " ".join(words[-2:]).rstrip(".") in ("and Company", "& Company"):
            base_words = words[:-2]
            stripped_suffix = " ".join(words[-2:])
            break

    base = " ".join(base_words)

    roll = _rng.random()
    if roll < 0.35 or not stripped_suffix:
        # Return base only
        return base
    else:
        # Return base + abbreviated suffix
        abbrev_suffixes = ["Inc", "Corp", "LLC", "Ltd", "Co"]
        return base + " " + _rng.choice(abbrev_suffixes)


def _mutate_case(s: str, rng: random.Random | None = None) -> str:
    """Randomly UPPERCASE or lowercase the entire string."""
    _rng = rng or random
    if not s:
        return s
    return s.upper() if _rng.random() < 0.5 else s.lower()


def _apply_nickname(name: str, rng: random.Random | None = None) -> str:
    """Replace `name` with a nickname if available."""
    return get_nickname(name, rng=rng)


# ── Public API ────────────────────────────────────────────────────────────────

def corrupt_record(
    record: dict,
    corruption_types: list[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[dict, list[str]]:
    """
    Apply one or more corruptions to a profile record.

    Parameters
    ----------
    record : dict
        Must contain keys: first_name, last_name, company, email, country, entity_id
    corruption_types : list[str] | None
        Which corruption types to apply. If None, a single random type is chosen.
        Valid values: abbreviation, truncation, levenshtein_1, levenshtein_2,
                      field_drop_single, field_drop_double, domain_swap,
                      company_abbrev, case_mutation, nickname
    rng : random.Random | None
        Optional seeded RNG for reproducibility.

    Returns
    -------
    (corrupted_record, list_of_applied_types)
        corrupted_record is a new dict; entity_id is preserved unchanged.
        All field values are strings (empty string = field dropped).
    """
    _rng = rng or random
    r = deepcopy(record)
    # Ensure all profile fields exist and are strings
    for f in PROFILE_FIELDS:
        if r.get(f) is None:
            r[f] = ""

    if corruption_types is None:
        corruption_types = [_rng.choice(CORRUPTION_TYPES)]

    applied: list[str] = []

    for ctype in corruption_types:
        if ctype == "abbreviation":
            if r["first_name"]:
                r["first_name"] = _abbreviate_first_name(r["first_name"])
                applied.append("abbreviation")

        elif ctype == "truncation":
            field = _rng.choice(["last_name", "company"])
            if r[field]:
                r[field] = _truncate(r[field], min_len=3, rng=_rng)
                applied.append("truncation")

        elif ctype == "levenshtein_1":
            field = _rng.choice(["first_name", "last_name"])
            if r[field]:
                r[field] = _levenshtein_corrupt(r[field], n_edits=1, rng=_rng)
                applied.append("levenshtein_1")

        elif ctype == "levenshtein_2":
            field = _rng.choice(["first_name", "last_name", "company"])
            if r[field]:
                r[field] = _levenshtein_corrupt(r[field], n_edits=2, rng=_rng)
                applied.append("levenshtein_2")

        elif ctype == "field_drop_single":
            droppable = [f for f in PROFILE_FIELDS if r.get(f)]
            if droppable:
                field = _rng.choice(droppable)
                r[field] = ""
                applied.append("field_drop_single")

        elif ctype == "field_drop_double":
            droppable = [f for f in PROFILE_FIELDS if r.get(f)]
            n_drop = min(2, len(droppable))
            fields = _rng.sample(droppable, n_drop)
            for field in fields:
                r[field] = ""
            if fields:
                applied.append("field_drop_double")

        elif ctype == "domain_swap":
            if r.get("email") and "@" in r["email"]:
                r["email"] = _swap_email_domain(r["email"], rng=_rng)
                applied.append("domain_swap")

        elif ctype == "company_abbrev":
            if r.get("company"):
                r["company"] = _abbreviate_company(r["company"], rng=_rng)
                applied.append("company_abbrev")

        elif ctype == "case_mutation":
            field = _rng.choice(["first_name", "last_name", "company"])
            if r[field]:
                r[field] = _mutate_case(r[field], rng=_rng)
                applied.append("case_mutation")

        elif ctype == "nickname":
            if r.get("first_name"):
                r["first_name"] = _apply_nickname(r["first_name"], rng=_rng)
                applied.append("nickname")

        else:
            raise ValueError(f"Unknown corruption type: {ctype!r}")

    return r, applied


def corrupt_for_bucket(
    record: dict,
    bucket: str,
    rng: random.Random | None = None,
) -> tuple[dict, list[str]]:
    """
    Apply corruption specifically targeting an eval bucket.

    Parameters
    ----------
    record : dict
        Pristine profile record.
    bucket : str
        One of: pristine, missing_firstname, missing_email_company,
                typo_name, domain_mismatch, swapped_attributes
    rng : random.Random | None
        Optional seeded RNG.

    Returns
    -------
    (corrupted_record, list_of_applied_types)
    """
    _rng = rng or random
    r = deepcopy(record)
    for f in PROFILE_FIELDS:
        if r.get(f) is None:
            r[f] = ""

    if bucket == "pristine":
        return r, []

    elif bucket == "missing_firstname":
        r["first_name"] = ""
        return r, ["field_drop_single:first_name"]

    elif bucket == "missing_email_company":
        r["email"] = ""
        r["company"] = ""
        return r, ["field_drop_double:email+company"]

    elif bucket == "typo_name":
        field = _rng.choice(["first_name", "last_name"])
        if r[field]:
            r[field] = _levenshtein_corrupt(r[field], n_edits=1, rng=_rng)
        return r, [f"levenshtein_1:{field}"]

    elif bucket == "domain_mismatch":
        if r.get("email") and "@" in r["email"]:
            r["email"] = _swap_email_domain(r["email"], rng=_rng)
        return r, ["domain_swap"]

    elif bucket == "swapped_attributes":
        # Swap first_name ↔ last_name
        r["first_name"], r["last_name"] = r["last_name"], r["first_name"]
        return r, ["swapped_attributes:first_last"]

    else:
        raise ValueError(
            f"Unknown eval bucket: {bucket!r}. "
            f"Must be one of: {EVAL_BUCKETS}"
        )
