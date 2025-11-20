"""
WikiPlots narrative extractor.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

MIN_LENGTH = 200


def extract_wikiplots(records: Iterable[Dict]) -> Tuple[List[str], np.ndarray, int]:
    narratives: List[str] = []
    outcomes: List[float] = []
    total = 0
    for record in records:
        total += 1
        narrative = record.get("narrative") or ""
        if len(narrative) < MIN_LENGTH:
            continue
        impact = record.get("impact_score")
        try:
            impact_value = float(impact)
        except (TypeError, ValueError):
            impact_value = len(narrative.split()) / 400
        narratives.append(narrative)
        outcomes.append(impact_value)
    return narratives, np.asarray(outcomes, dtype=float), total

