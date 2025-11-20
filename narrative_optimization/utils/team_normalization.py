"""
Shared helpers for canonicalizing team names across odds feeds and narrative data.

The goal is to make cross-source joins deterministic without maintaining an
exhaustive alias table for every sport.  We normalize punctuation, handle the
common `St.` → `State` variations, and collapse extraneous whitespace so that
`Penn St Nittany Lions` and `Penn State Nittany Lions` map to the same key.
"""

from __future__ import annotations

import re
from typing import Optional


_REPLACEMENTS = [
    ("&", "and"),
    ("St.", "State"),
    ("St ", "State "),
    ("Univ.", "University"),
    (" Univ", " University"),
    ("-St", " State"),
]


def normalize_team_name(name: Optional[str]) -> str:
    """Return a canonicalized version of the provided team name."""
    if not name:
        return ""
    normalized = name.strip()
    for old, new in _REPLACEMENTS:
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\bSt\b", "State", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def team_key(name: Optional[str]) -> str:
    """Lowercase key trimmed of punctuation for dictionary lookups."""
    normalized = normalize_team_name(name)
    key = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    return " ".join(key.split())

