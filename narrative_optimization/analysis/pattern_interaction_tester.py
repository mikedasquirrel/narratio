"""
Pattern Interaction Tester
==========================

Evaluates whether combining two discovered contexts amplifies or cancels the
edge. Outputs interaction matrices for UI + betting engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class InteractionResult:
    pattern_a: str
    pattern_b: str
    win_rate: float
    baseline: float
    lift: float
    sample_size: int
    classification: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "pattern_a": self.pattern_a,
            "pattern_b": self.pattern_b,
            "win_rate": self.win_rate,
            "baseline": self.baseline,
            "lift": self.lift,
            "sample_size": self.sample_size,
            "classification": self.classification,
        }


class PatternInteractionTester:
    """
    Test pairwise interactions between binary pattern columns (1 = matches).
    """

    def __init__(self, outcome_col: str, min_samples: int = 20):
        self.outcome_col = outcome_col
        self.min_samples = min_samples

    def evaluate(self, df: pd.DataFrame, pattern_cols: Sequence[str]) -> List[Dict]:
        if df.empty or len(pattern_cols) < 2:
            return []

        baseline = df[self.outcome_col].mean()
        results: List[InteractionResult] = []

        for a, b in combinations(pattern_cols, 2):
            mask = (df[a] > 0) & (df[b] > 0)
            n = mask.sum()
            if n < self.min_samples:
                continue

            win_rate = df.loc[mask, self.outcome_col].mean()
            lift = win_rate - baseline
            classification = self._classify(lift)

            results.append(
                InteractionResult(
                    pattern_a=a,
                    pattern_b=b,
                    win_rate=float(win_rate),
                    baseline=float(baseline),
                    lift=float(lift),
                    sample_size=int(n),
                    classification=classification,
                )
            )

        results.sort(key=lambda r: abs(r.lift), reverse=True)
        return [r.to_dict() for r in results]

    def _classify(self, lift: float) -> str:
        if lift >= 0.15:
            return "amplifying"
        if lift >= 0.05:
            return "supporting"
        if lift <= -0.15:
            return "cancelling"
        if lift <= -0.05:
            return "dampening"
        return "neutral"


__all__ = ["PatternInteractionTester", "InteractionResult"]


