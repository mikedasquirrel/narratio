"""
Market Edge Calculator
----------------------

Wraps the lower-level betting edge helpers with context awareness (dynamic π,
context strat filters, risk guardrails).
"""

from __future__ import annotations

from typing import Dict, Optional

from utils import betting_edge_calculator as base


class MarketEdgeCalculator:
    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge

    def evaluate_matchup(
        self,
        model_probability_home: float,
        odds_home: Optional[int],
        odds_away: Optional[int],
        context_label: str = "",
        pi_effective: Optional[float] = None,
    ) -> Dict[str, Dict]:
        home = self._evaluate_side(
            "home", model_probability_home, odds_home, context_label, pi_effective
        )
        away = self._evaluate_side(
            "away",
            1 - model_probability_home,
            odds_away,
            context_label,
            pi_effective,
        )
        return {"home": home, "away": away}

    def _evaluate_side(
        self,
        side: str,
        model_prob: float,
        odds: Optional[int],
        context_label: str,
        pi_effective: Optional[float],
    ) -> Dict:
        if odds is None:
            return {"action": "SKIP", "reason": "No odds"}

        rec = base.betting_recommendation(
            model_probability=model_prob, odds=odds, min_edge=self.min_edge
        )
        rec["side"] = side
        rec["context"] = context_label
        rec["pi_effective"] = pi_effective
        return rec


__all__ = ["MarketEdgeCalculator"]


