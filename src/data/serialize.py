"""
Serialization / deserialization for entity profiles.

Two formats:
  pipe  — "Jay Shah | Acme Corp | jay@acme.com | USA"
  kv    — "fn:Jay ln:Shah org:Acme Corp em:jay@acme.com co:USA"

Missing / empty fields are always preserved so position is unambiguous.
"""

from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────

FIELD_ORDER: list[str] = ["first_name", "last_name", "company", "email", "country"]

FIELD_KEYS: dict[str, str] = {
    "first_name": "fn",
    "last_name": "ln",
    "company": "org",
    "email": "em",
    "country": "co",
}

# Reverse mapping: short key → field name
_KEY_TO_FIELD: dict[str, str] = {v: k for k, v in FIELD_KEYS.items()}

_PIPE_SEP = " | "


# ── Helpers ───────────────────────────────────────────────────────────────────

def _val(record: dict, field: str) -> str:
    """Return the field value as a string; None → empty string."""
    v = record.get(field)
    if v is None:
        return ""
    return str(v)


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_pipe(record: dict) -> str:
    """
    Serialize a record to pipe-delimited format.

    Example (all fields):
        "Jay Shah | Acme Corp | jay@acme.com | USA"

    Missing first_name:
        " | Shah | Acme Corp | jay@acme.com | USA"
    """
    parts = [_val(record, f) for f in FIELD_ORDER]
    return _PIPE_SEP.join(parts)


def serialize_kv(record: dict) -> str:
    """
    Serialize a record to key-value format.

    Example (all fields):
        "fn:Jay ln:Shah org:Acme Corp em:jay@acme.com co:USA"

    Missing first_name:
        "fn: ln:Shah org:Acme Corp em:jay@acme.com co:USA"
    """
    parts = []
    for field in FIELD_ORDER:
        key = FIELD_KEYS[field]
        val = _val(record, field)
        parts.append(f"{key}:{val}")
    return " ".join(parts)


def serialize(record: dict, fmt: str) -> str:
    """
    Dispatch serialization.

    Parameters
    ----------
    record : dict
        Profile dict; may contain keys outside FIELD_ORDER (e.g. entity_id) —
        those are ignored.
    fmt : str
        "pipe" or "kv".

    Returns
    -------
    str
        Serialized string.

    Raises
    ------
    ValueError
        If fmt is not "pipe" or "kv".
    """
    if fmt == "pipe":
        return serialize_pipe(record)
    if fmt == "kv":
        return serialize_kv(record)
    raise ValueError(f"Unknown serialization format: {fmt!r}. Must be 'pipe' or 'kv'.")


# ── Deserialization ───────────────────────────────────────────────────────────

def deserialize_pipe(text: str) -> dict:
    """
    Reverse of serialize_pipe.

    Returns a dict with keys from FIELD_ORDER. Empty slots become empty strings.
    """
    parts = text.split(_PIPE_SEP)
    # If the text was produced by serialize_pipe, len(parts) == len(FIELD_ORDER).
    # Be lenient: pad or truncate.
    result: dict[str, str] = {}
    for i, field in enumerate(FIELD_ORDER):
        result[field] = parts[i].strip() if i < len(parts) else ""
    return result


def deserialize_kv(text: str) -> dict:
    """
    Reverse of serialize_kv.

    Parses tokens of the form "key:value", where value may contain spaces
    (everything until the next known key token).
    """
    result: dict[str, str] = {f: "" for f in FIELD_ORDER}

    if not text:
        return result

    # Split on known key prefixes.  We walk left-to-right and accumulate
    # value tokens between consecutive key markers.
    known_keys = list(FIELD_KEYS.values())  # ["fn", "ln", "org", "em", "co"]

    # Tokenise on whitespace boundaries where a token starts with "key:"
    tokens = text.split(" ")

    current_key: str | None = None
    value_parts: list[str] = []

    def _flush() -> None:
        if current_key and current_key in _KEY_TO_FIELD:
            field = _KEY_TO_FIELD[current_key]
            result[field] = " ".join(value_parts).strip()

    for token in tokens:
        # Check if this token starts a new key
        matched_key: str | None = None
        for k in known_keys:
            prefix = k + ":"
            if token.startswith(prefix):
                matched_key = k
                remainder = token[len(prefix):]
                break

        if matched_key is not None:
            _flush()
            current_key = matched_key
            value_parts = [remainder] if remainder else []
        else:
            if current_key is not None:
                value_parts.append(token)
            # else: stray token before any key — ignore

    _flush()
    return result
