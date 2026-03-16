"""
Phase 2 corruption engine for cross-encoder training data.

Implements 28 positive corruption types (all produce label=1 pairs) and
the COL/VAL serialization format for cross-encoder input.

Usage:
    from src.data.phase2_corrupt import corrupt_record_phase2, colval_serialize
"""

from __future__ import annotations

import random
import re
import unicodedata
from copy import deepcopy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yijh", "i": "uojk", "o": "iplk", "p": "ol",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
    "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "ioljm",
    "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}

OCR_PAIRS: list[tuple[str, str]] = [
    ("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("1", "l"), ("l", "1"),
    ("rn", "m"), ("m", "rn"), ("cl", "d"), ("d", "cl"), ("vv", "w"), ("w", "vv"),
    ("5", "S"), ("S", "5"), ("8", "B"), ("B", "8"),
]

PHONETIC_PAIRS: list[tuple[str, str]] = [
    ("ph", "f"), ("f", "ph"), ("ck", "k"), ("k", "ck"), ("ce", "se"), ("se", "ce"),
    ("ks", "x"), ("x", "ks"), ("tion", "shun"), ("sion", "shun"),
    ("ght", "te"), ("ough", "ow"), ("ei", "ie"), ("ie", "ei"),
    ("ae", "e"), ("oe", "e"), ("ll", "l"), ("ss", "s"), ("tt", "t"),
]

SENIORITY_MAP: dict[str, list[str]] = {
    "Senior": ["Sr.", "Sr"],
    "Sr.": ["Senior", "Sr"],
    "Sr": ["Senior", "Sr."],
    "Junior": ["Jr.", "Jr"],
    "Jr.": ["Junior", "Jr"],
    "Jr": ["Junior", "Jr."],
    "Staff": ["Lead", "Principal"],
    "Lead": ["Staff", "Principal"],
    "Principal": ["Staff", "Lead"],
}

SENIORITY_PREFIXES = ["Senior", "Sr.", "Sr", "Junior", "Jr.", "Jr", "Staff", "Lead", "Principal"]

LEGAL_SUFFIXES = [
    "Inc", "Inc.", "LLC", "Ltd", "Ltd.", "Corp", "Corp.", "Corporation",
    "Co", "Co.", "Company", "Group", "Holdings", "International",
    "GmbH", "AG", "SA", "S.A.", "PLC", "Plc", "Limited", "Pty",
    "N.V.", "B.V.", "S.p.A.", "S.r.l.",
]

COLVAL_FIELDS = [
    ("fn", "first_name"),
    ("ln", "last_name"),
    ("org", "company"),
    ("title", "title"),
    ("email", "email"),
    ("country", "country"),
]

ALL_CORRUPTION_CODES = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13",
    "T1", "T2", "T3", "T4", "T5", "T6",
    "E1", "E2",
]

EVAL_BUCKETS_PHASE2 = [
    "pristine", "missing_firstname", "missing_email_company",
    "typo_name", "domain_mismatch", "swapped_attributes",
]

PERSONAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]


# ---------------------------------------------------------------------------
# COL/VAL Serialization
# ---------------------------------------------------------------------------


def colval_serialize(record: dict) -> str:
    """Serialize a record to COL/VAL format (Ditto-style)."""
    parts = []
    for col_key, field_name in COLVAL_FIELDS:
        val = record.get(field_name, "") or ""
        parts.append(f"COL {col_key} VAL {val}")
    return " ".join(parts)


def colval_pair(record_a: dict, record_b: dict) -> str:
    """Serialize a pair for cross-encoder input with [CLS]/[SEP]."""
    return f"[CLS] {colval_serialize(record_a)} [SEP] {colval_serialize(record_b)} [SEP]"


# ---------------------------------------------------------------------------
# Company corruptions (C1-C8)
# ---------------------------------------------------------------------------


def _strip_suffix(company: str) -> str:
    """Strip known legal suffix from company name."""
    words = company.rstrip(".,").split()
    for i in range(len(words) - 1, -1, -1):
        cleaned = words[i].rstrip(".,")
        if cleaned in LEGAL_SUFFIXES:
            return " ".join(words[:i]).strip()
    return company


def corrupt_c1_suffix_swap(company: str, rng: random.Random | None = None) -> str:
    """C1: Legal suffix swap. 'Acme LLC' → 'Acme Ltd'."""
    _rng = rng or random
    base = _strip_suffix(company)
    if base == company:
        return company + " " + _rng.choice(["Inc", "LLC", "Ltd", "Corp"])
    new_suffix = _rng.choice(["Inc", "LLC", "Ltd", "Corp", "Co"])
    return f"{base} {new_suffix}"


def corrupt_c2_suffix_drop(company: str) -> str:
    """C2: Drop legal suffix. 'Microsoft Corporation' → 'Microsoft'."""
    return _strip_suffix(company)


def corrupt_c3_the_prefix_drop(company: str) -> str:
    """C3: Drop leading 'The'. 'The Home Depot' → 'Home Depot'."""
    return company[4:] if company.startswith("The ") else company


def corrupt_c4_ampersand_normalize(company: str, rng: random.Random | None = None) -> str:
    """C4: Normalize ampersand. 'Johnson & Johnson' ↔ 'Johnson and Johnson'."""
    if " & " in company:
        return company.replace(" & ", " and ")
    if " and " in company:
        return company.replace(" and ", " & ")
    return company


def corrupt_c5_company_abbreviation(
    company: str, abbreviation_map: dict[str, list[str]] | None = None,
) -> str:
    """C5: Use known abbreviation. 'International Business Machines' → 'IBM'."""
    if abbreviation_map and company in abbreviation_map:
        return abbreviation_map[company][0]
    words = company.split()
    if len(words) >= 2:
        acronym = "".join(w[0].upper() for w in words if w[0].isalpha())
        if len(acronym) >= 2:
            return acronym
    return company


def corrupt_c6_word_truncation(company: str) -> str:
    """C6: Drop last word. 'Goldman Sachs Group' → 'Goldman Sachs'."""
    words = company.split()
    return " ".join(words[:-1]) if len(words) > 1 else company


def corrupt_c7_rebrand(company: str, rebrand_map: dict[str, str] | None = None) -> str:
    """C7: Apply rebrand. 'Facebook' → 'Meta Platforms'."""
    if rebrand_map and company in rebrand_map:
        return rebrand_map[company]
    return company


def corrupt_c8_shorten_name(company: str) -> str:
    """C8: Shorten + abbreviate. 'Acme International Holdings' → 'Acme Intl Hldgs'."""
    base = _strip_suffix(company)
    abbrevs = {
        "International": "Intl", "Corporation": "Corp", "Technologies": "Tech",
        "Solutions": "Sol", "Holdings": "Hldgs", "Manufacturing": "Mfg",
        "Engineering": "Eng", "Services": "Svc", "Management": "Mgmt",
    }
    words = base.split()
    return " ".join(abbrevs.get(w, w) for w in words)


# ---------------------------------------------------------------------------
# Name corruptions (N1-N13)
# ---------------------------------------------------------------------------


def corrupt_n1_diacritic_strip(name: str) -> str:
    """N1: Strip diacritics. 'García' → 'Garcia'."""
    nfkd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def corrupt_n2_single_char_delete(name: str, rng: random.Random | None = None) -> str:
    """N2: Delete a single random character."""
    _rng = rng or random
    if len(name) <= 1:
        return name
    pos = _rng.randint(0, len(name) - 1)
    return name[:pos] + name[pos + 1:]


def corrupt_n3_keyboard_sub(name: str, rng: random.Random | None = None) -> str:
    """N3: QWERTY adjacent key substitution. 'Smith' → 'Smkth'."""
    _rng = rng or random
    if not name:
        return name
    chars = list(name)
    candidates = [i for i, c in enumerate(chars) if c.lower() in QWERTY_NEIGHBORS]
    if not candidates:
        return name
    pos = _rng.choice(candidates)
    original = chars[pos]
    neighbors = QWERTY_NEIGHBORS.get(original.lower(), "")
    if not neighbors:
        return name
    replacement = _rng.choice(neighbors)
    chars[pos] = replacement.upper() if original.isupper() else replacement
    return "".join(chars)


def corrupt_n4_ocr_sub(name: str, rng: random.Random | None = None) -> str:
    """N4: OCR-like substitution. 'rn' → 'm', '0' → 'O'."""
    _rng = rng or random
    applicable = [(old, new) for old, new in OCR_PAIRS if old in name]
    if not applicable:
        return name
    old, new = _rng.choice(applicable)
    return name.replace(old, new, 1)


def corrupt_n5_char_transposition(name: str, rng: random.Random | None = None) -> str:
    """N5: Swap adjacent characters. 'Smith' → 'Smiht'."""
    _rng = rng or random
    if len(name) < 2:
        return name
    pos = _rng.randint(0, len(name) - 2)
    chars = list(name)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def corrupt_n6_name_swap(record: dict) -> dict:
    """N6: Swap first_name ↔ last_name."""
    r = dict(record)
    r["first_name"], r["last_name"] = r.get("last_name", ""), r.get("first_name", "")
    return r


def corrupt_n7_first_initial(name: str) -> str:
    """N7: First initial only. 'Jay' → 'J.'."""
    return name[0].upper() + "." if name else name


def corrupt_n8_first_middle_initial(record: dict) -> dict:
    """N8: Both first and middle to initials."""
    r = dict(record)
    fn = r.get("first_name", "")
    mn = r.get("middle_name", "")
    if fn:
        r["first_name"] = fn[0].upper() + "."
    if mn:
        r["middle_name"] = mn[0].upper() + "."
    return r


def corrupt_n9_drop_middle(record: dict) -> dict:
    """N9: Drop middle name."""
    r = dict(record)
    r["middle_name"] = ""
    return r


def corrupt_n10_middle_initial(record: dict) -> dict:
    """N10: Middle name to initial."""
    r = dict(record)
    mn = r.get("middle_name", "")
    if mn:
        r["middle_name"] = mn[0].upper() + "."
    return r


def corrupt_n11_last_initial(name: str) -> str:
    """N11: Last name to initial. 'Smith' → 'S.'."""
    return name[0].upper() + "." if name else name


def corrupt_n12_nickname(
    name: str, nickname_map: dict[str, set[str]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """N12: Replace with nickname. 'William' → 'Bill'."""
    _rng = rng or random
    if not nickname_map:
        return name
    nicks = nickname_map.get(name.lower(), set())
    if not nicks:
        return name
    chosen = _rng.choice(sorted(nicks))
    return chosen.capitalize() if name[0].isupper() else chosen


def corrupt_n13_phonetic_sub(name: str, rng: random.Random | None = None) -> str:
    """N13: Phonetic substitution. 'ph' → 'f', 'ck' → 'k'."""
    _rng = rng or random
    name_lower = name.lower()
    applicable = [(old, new) for old, new in PHONETIC_PAIRS if old in name_lower]
    if not applicable:
        return name
    old, new = _rng.choice(applicable)
    idx = name_lower.find(old)
    if idx == -1:
        return name
    return name[:idx] + new + name[idx + len(old):]


# ---------------------------------------------------------------------------
# Title corruptions (T1-T6)
# ---------------------------------------------------------------------------


def corrupt_t1_title_abbreviation(
    title: str, alternates_map: dict[str, list[str]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """T1: Use alternate/abbreviated title from O*NET."""
    _rng = rng or random
    if alternates_map:
        for canonical, alts in alternates_map.items():
            if title == canonical and alts:
                return _rng.choice(alts)
            if title in alts:
                options = [canonical] + [a for a in alts if a != title]
                if options:
                    return _rng.choice(options)
    abbrevs = {
        "Vice President": "VP", "Senior Vice President": "SVP",
        "Chief Executive Officer": "CEO", "Chief Technology Officer": "CTO",
        "Chief Financial Officer": "CFO", "Chief Operating Officer": "COO",
    }
    for full, short in abbrevs.items():
        if full in title:
            return title.replace(full, short)
    return title


def corrupt_t2_title_expansion(title: str) -> str:
    """T2: Expand abbreviation. 'VP' → 'Vice President'."""
    expansions = {
        "VP": "Vice President", "SVP": "Senior Vice President",
        "CEO": "Chief Executive Officer", "CTO": "Chief Technology Officer",
        "CFO": "Chief Financial Officer", "COO": "Chief Operating Officer",
        "EVP": "Executive Vice President", "Sr.": "Senior", "Jr.": "Junior",
        "Mgr": "Manager", "Dir": "Director", "Eng": "Engineer",
    }
    for short, full in expansions.items():
        pattern = r"\b" + re.escape(short) + r"\b"
        if re.search(pattern, title):
            return re.sub(pattern, full, title, count=1)
    return title


def corrupt_t3_title_reorder(title: str) -> str:
    """T3: Reorder title tokens. 'Engineering VP' → 'VP Engineering'."""
    words = title.split()
    if len(words) >= 2:
        words.reverse()
    return " ".join(words)


def corrupt_t4_seniority_drop(title: str) -> str:
    """T4: Drop seniority prefix. 'Senior Software Engineer' → 'Software Engineer'."""
    for prefix in SENIORITY_PREFIXES:
        if title.startswith(prefix + " "):
            return title[len(prefix) + 1:]
    return title


def corrupt_t5_seniority_synonym(title: str, rng: random.Random | None = None) -> str:
    """T5: Swap seniority prefix synonym. 'Sr.' → 'Senior'."""
    _rng = rng or random
    for prefix in SENIORITY_PREFIXES:
        if title.startswith(prefix + " ") or title == prefix:
            alternatives = [s for s in SENIORITY_MAP.get(prefix, []) if s != prefix]
            if alternatives:
                replacement = _rng.choice(alternatives)
                return replacement + title[len(prefix):] if title != prefix else replacement
    return title


def corrupt_t6_missing_field() -> str:
    """T6: Drop the title entirely."""
    return ""


# ---------------------------------------------------------------------------
# Email corruptions (E1-E2)
# ---------------------------------------------------------------------------


def corrupt_e1_email_format_variant(email: str, rng: random.Random | None = None) -> str:
    """E1: Change email format. 'jay.smith@acme.com' → 'j.smith@acme.com'."""
    _rng = rng or random
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    parts = local.split(".")
    if len(parts) >= 2:
        patterns = [
            f"{parts[0][0]}.{parts[1]}",
            f"{parts[0]}{parts[1]}",
            f"{parts[0][0]}{parts[1]}",
            f"{parts[0]}_{parts[1]}",
        ]
        return f"{_rng.choice(patterns)}@{domain}"
    return email


def corrupt_e2_domain_swap(email: str, rng: random.Random | None = None) -> str:
    """E2: Swap to personal domain. 'jay@acme.com' → 'jay@gmail.com'."""
    _rng = rng or random
    if not email or "@" not in email:
        return email
    local = email.split("@")[0]
    return f"{local}@{_rng.choice(PERSONAL_DOMAINS)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def corrupt_record_phase2(
    record: dict,
    corruption_codes: list[str],
    rng: random.Random | None = None,
    nickname_map: dict[str, set[str]] | None = None,
    abbreviation_map: dict[str, list[str]] | None = None,
    rebrand_map: dict[str, str] | None = None,
    alternates_map: dict[str, list[str]] | None = None,
) -> dict:
    """Apply corruption codes to a record. Returns a new corrupted record."""
    _rng = rng or random
    r = deepcopy(record)

    for code in corruption_codes:
        code_upper = code.upper()

        if code_upper == "C1":
            r["company"] = corrupt_c1_suffix_swap(r.get("company", ""), _rng)
        elif code_upper == "C2":
            r["company"] = corrupt_c2_suffix_drop(r.get("company", ""))
        elif code_upper == "C3":
            r["company"] = corrupt_c3_the_prefix_drop(r.get("company", ""))
        elif code_upper == "C4":
            r["company"] = corrupt_c4_ampersand_normalize(r.get("company", ""), _rng)
        elif code_upper == "C5":
            r["company"] = corrupt_c5_company_abbreviation(r.get("company", ""), abbreviation_map)
        elif code_upper == "C6":
            r["company"] = corrupt_c6_word_truncation(r.get("company", ""))
        elif code_upper == "C7":
            r["company"] = corrupt_c7_rebrand(r.get("company", ""), rebrand_map)
        elif code_upper == "C8":
            r["company"] = corrupt_c8_shorten_name(r.get("company", ""))
        elif code_upper == "N1":
            r["first_name"] = corrupt_n1_diacritic_strip(r.get("first_name", ""))
            r["last_name"] = corrupt_n1_diacritic_strip(r.get("last_name", ""))
        elif code_upper == "N2":
            field = _rng.choice(["first_name", "last_name"])
            r[field] = corrupt_n2_single_char_delete(r.get(field, ""), _rng)
        elif code_upper == "N3":
            field = _rng.choice(["first_name", "last_name"])
            r[field] = corrupt_n3_keyboard_sub(r.get(field, ""), _rng)
        elif code_upper == "N4":
            field = _rng.choice(["first_name", "last_name", "company"])
            r[field] = corrupt_n4_ocr_sub(r.get(field, ""), _rng)
        elif code_upper == "N5":
            field = _rng.choice(["first_name", "last_name"])
            r[field] = corrupt_n5_char_transposition(r.get(field, ""), _rng)
        elif code_upper == "N6":
            r = corrupt_n6_name_swap(r)
        elif code_upper == "N7":
            r["first_name"] = corrupt_n7_first_initial(r.get("first_name", ""))
        elif code_upper == "N8":
            r = corrupt_n8_first_middle_initial(r)
        elif code_upper == "N9":
            r = corrupt_n9_drop_middle(r)
        elif code_upper == "N10":
            r = corrupt_n10_middle_initial(r)
        elif code_upper == "N11":
            r["last_name"] = corrupt_n11_last_initial(r.get("last_name", ""))
        elif code_upper == "N12":
            r["first_name"] = corrupt_n12_nickname(r.get("first_name", ""), nickname_map, _rng)
        elif code_upper == "N13":
            field = _rng.choice(["first_name", "last_name"])
            r[field] = corrupt_n13_phonetic_sub(r.get(field, ""), _rng)
        elif code_upper == "T1":
            r["title"] = corrupt_t1_title_abbreviation(r.get("title", ""), alternates_map, _rng)
        elif code_upper == "T2":
            r["title"] = corrupt_t2_title_expansion(r.get("title", ""))
        elif code_upper == "T3":
            r["title"] = corrupt_t3_title_reorder(r.get("title", ""))
        elif code_upper == "T4":
            r["title"] = corrupt_t4_seniority_drop(r.get("title", ""))
        elif code_upper == "T5":
            r["title"] = corrupt_t5_seniority_synonym(r.get("title", ""), _rng)
        elif code_upper == "T6":
            r["title"] = corrupt_t6_missing_field()
        elif code_upper == "E1":
            r["email"] = corrupt_e1_email_format_variant(r.get("email", ""), _rng)
        elif code_upper == "E2":
            r["email"] = corrupt_e2_domain_swap(r.get("email", ""), _rng)
        elif code_upper == "N_MISSING_FN":
            r["first_name"] = ""
        elif code_upper == "E_DROP":
            r["email"] = ""
        elif code_upper == "C_DROP":
            r["company"] = ""
        else:
            raise ValueError(f"Unknown corruption code: {code!r}")

    return r


def corrupt_for_bucket_phase2(
    record: dict, bucket: str, rng: random.Random | None = None,
) -> tuple[dict, list[str]]:
    """Apply bucket-specific corruption matching Phase 1's 6 eval buckets."""
    _rng = rng or random
    if bucket == "pristine":
        return deepcopy(record), []
    elif bucket == "missing_firstname":
        return corrupt_record_phase2(record, ["N_MISSING_FN"], _rng), ["N_MISSING_FN"]
    elif bucket == "missing_email_company":
        return corrupt_record_phase2(record, ["E_DROP", "C_DROP"], _rng), ["E_DROP", "C_DROP"]
    elif bucket == "typo_name":
        return corrupt_record_phase2(record, ["N3"], _rng), ["N3"]
    elif bucket == "domain_mismatch":
        return corrupt_record_phase2(record, ["E2"], _rng), ["E2"]
    elif bucket == "swapped_attributes":
        return corrupt_record_phase2(record, ["N6"], _rng), ["N6"]
    else:
        raise ValueError(f"Unknown bucket: {bucket!r}. Valid: {EVAL_BUCKETS_PHASE2}")
