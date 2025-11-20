"""
Dynamic π Calculator
====================

Turns the theoretical finding (π varies per instance) into production code.

Responsibilities
----------------
1. Score instance complexity using domain-specific heuristics.
2. Convert complexity → π_effective via calibrated sensitivity curves.
3. Annotate records with π metadata for downstream transformers/UI.
4. Persist aggregate diagnostics (range, std, buckets) for monitoring.

This module is intentionally lightweight so it can be imported from Flask
routes, batch scripts, or transformer pipelines without creating dependencies
on the heavier `DynamicNarrativityAnalyzer` plotting stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import numpy as np

from narrative_optimization.domain_registry import get_domain
from narrative_optimization.src.analysis.dynamic_narrativity import (
    DynamicNarrativityAnalyzer,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


DEFAULT_SENSITIVITY = 0.22

DOMAIN_COMPLEXITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "nhl": {
        "is_playoff": 0.25,
        "is_rivalry": 0.20,
        "rest_advantage": 0.10,
        "record_pressure": 0.15,
        "travel_penalty": 0.05,
        "betting_delta": 0.25,
    },
    "nfl": {
        "is_playoff": 0.30,
        "divisional": 0.15,
        "qb_edge": 0.20,
        "spread_abs": 0.20,
        "rest_advantage": 0.10,
        "prime_time": 0.05,
    },
    "supreme_court": {
        "vote_margin": 0.25,
        "issue_salience": 0.20,
        "precedent": 0.15,
        "public_attention": 0.15,
        "amicus_density": 0.10,
        "historical_pressure": 0.15,
    },
}

DOMAIN_SENSITIVITY_OVERRIDES: Dict[str, float] = {
    "supreme_court": 0.32,
    "nfl": 0.26,
    "nhl": 0.20,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return 0


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #


@dataclass
class DynamicPiResult:
    domain: str
    pi_base: float
    pi_sensitivity: float
    pi_effective: np.ndarray
    complexities: np.ndarray
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "pi_base": self.pi_base,
            "pi_sensitivity": self.pi_sensitivity,
            "pi_min": float(self.pi_effective.min()),
            "pi_max": float(self.pi_effective.max()),
            "pi_mean": float(self.pi_effective.mean()),
            "pi_std": float(self.pi_effective.std()),
            "complexity_mean": float(self.complexities.mean()),
            "complexity_std": float(self.complexities.std()),
            "pi_effective": self.pi_effective.tolist(),
            "complexities": self.complexities.tolist(),
            "stats": self.stats,
        }


class DynamicPiCalculator:
    """
    Production-ready π_effective annotator.

    Usage:
        calculator = DynamicPiCalculator("nhl")
        result = calculator.annotate(records)
        records_with_pi = result["records"]
    """

    def __init__(self, domain_name: str, cache_dir: Optional[Path] = None):
        self.domain_name = domain_name
        self.domain_config = get_domain(domain_name)
        if not self.domain_config:
            raise ValueError(f"Domain '{domain_name}' is not registered.")

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path("narrative_optimization/results/dynamic_pi")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sensitivity = DOMAIN_SENSITIVITY_OVERRIDES.get(
            domain_name, DEFAULT_SENSITIVITY
        )
        self.weights = DOMAIN_COMPLEXITY_WEIGHTS.get(domain_name, {})
        self.analyzer = DynamicNarrativityAnalyzer(self.domain_config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def annotate(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate π_effective for each record and return enriched payload.
        """
        records_list = list(records)
        if not records_list:
            return {"records": [], "pi_effective": [], "stats": {}}

        complexities = self._score_complexities(records_list)
        pi_base = getattr(self.domain_config, "estimated_pi", 0.5)
        pi_effective = np.clip(pi_base + self.sensitivity * complexities, 0.0, 1.0)

        # Attach to records
        for idx, record in enumerate(records_list):
            complexity = float(complexities[idx])
            pi_eff = float(pi_effective[idx])
            record.setdefault("pi_metadata", {})
            record["pi_metadata"].update(
                {
                    "pi_base": pi_base,
                    "pi_effective": float(pi_eff),
                    "complexity": float(complexity),
                    "pi_bucket": self._bucketize_pi(pi_eff),
                }
            )

        stats = self._summarize(pi_effective, complexities)
        result = DynamicPiResult(
            domain=self.domain_name,
            pi_base=pi_base,
            pi_sensitivity=self.sensitivity,
            pi_effective=pi_effective,
            complexities=complexities,
            stats=stats,
        )

        self._persist(result)
        return {
            "records": records_list,
            "pi_effective": pi_effective.tolist(),
            "stats": stats,
        }

    # ------------------------------------------------------------------ #
    # Complexity scoring
    # ------------------------------------------------------------------ #

    def _score_complexities(self, records: List[Dict[str, Any]]) -> np.ndarray:
        scores = []
        for record in records:
            if self.domain_name == "nhl":
                scores.append(self._complexity_nhl(record))
            elif self.domain_name == "nfl":
                scores.append(self._complexity_nfl(record))
            elif self.domain_name == "supreme_court":
                scores.append(self._complexity_supreme_court(record))
            else:
                scores.append(self._complexity_generic(record))
        return np.clip(np.array(scores, dtype=float), 0.0, 1.0)

    def _complexity_nhl(self, record: Dict[str, Any]) -> float:
        context = record.get("temporal_context") or {}
        odds = record.get("betting_odds") or {}
        weights = self.weights

        playoff = _bool_int(record.get("is_playoff"))
        rivalry = _bool_int(record.get("is_rivalry"))
        rest_adv = abs(_safe_float(context.get("rest_advantage")))
        record_diff = abs(_safe_float(context.get("record_differential")))
        travel = abs(_safe_float(context.get("travel_diff")))
        implied = _safe_float(odds.get("implied_prob_home"), 0.5)
        betting_delta = abs(implied - 0.5)

        score = (
            weights.get("is_playoff", 0.2) * playoff
            + weights.get("is_rivalry", 0.2) * rivalry
            + weights.get("rest_advantage", 0.1) * min(rest_adv / 3, 1)
            + weights.get("record_pressure", 0.15) * min(record_diff / 0.5, 1)
            + weights.get("travel_penalty", 0.05) * min(travel / 3, 1)
            + weights.get("betting_delta", 0.2) * min(betting_delta / 0.25, 1)
        )
        return min(score, 1.0)

    def _complexity_nfl(self, record: Dict[str, Any]) -> float:
        weights = self.weights
        playoff = _bool_int(record.get("playoff"))
        div_game = _bool_int(record.get("div_game"))
        rest_adv = abs(_safe_float(record.get("rest_advantage")))
        spread = abs(_safe_float(record.get("spread_line")))
        prime_time = _bool_int(record.get("prime_time"))
        qb_block = record.get("home_qb") or {}
        qb_edge = _safe_float(qb_block.get("epa_per_play")) - _safe_float(
            (record.get("away_qb") or {}).get("epa_per_play")
        )

        score = (
            weights.get("is_playoff", 0.3) * playoff
            + weights.get("divisional", 0.15) * div_game
            + weights.get("rest_advantage", 0.1) * min(rest_adv / 4, 1)
            + weights.get("spread_abs", 0.2) * min(spread / 7, 1)
            + weights.get("qb_edge", 0.2) * min(abs(qb_edge) / 0.25, 1)
            + weights.get("prime_time", 0.05) * prime_time
        )
        return min(score, 1.0)

    def _complexity_supreme_court(self, record: Dict[str, Any]) -> float:
        weights = self.weights
        outcome = record.get("outcome") or {}
        metadata = record.get("metadata") or {}
        vote_margin = _safe_float(outcome.get("vote_margin"), 0)
        unanimous = _bool_int(outcome.get("unanimous"))
        precedent = _bool_int(outcome.get("precedent_setting"))
        overturned = _bool_int(outcome.get("overturned"))
        area = (metadata.get("area_of_law") or "").lower()
        high_salience = int(
            any(keyword in area for keyword in ["constitutional", "rights", "election"])
        )
        citation = _safe_float(outcome.get("citation_count") or metadata.get("citation_count"), 0)

        score = (
            weights.get("vote_margin", 0.2) * min(vote_margin / 5, 1)
            + weights.get("issue_salience", 0.2) * high_salience
            + weights.get("precedent", 0.1) * precedent
            + weights.get("public_attention", 0.15) * unanimous
            + weights.get("historic_pressure", 0.15) * overturned
            + weights.get("amicus_density", 0.1) * min(citation / 50000, 1)
        )
        return min(score, 1.0)

    def _complexity_generic(self, record: Dict[str, Any]) -> float:
        text = str(record.get("narrative") or record.get("description") or "")
        length = len(text.split())
        outcome = record.get("outcome")
        return min(0.15 + 0.0005 * length + 0.05 * _bool_int(outcome), 1.0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bucketize_pi(pi_value: float) -> str:
        if pi_value < 0.35:
            return "low"
        if pi_value < 0.55:
            return "medium"
        if pi_value < 0.75:
            return "high"
        return "extreme"

    def _summarize(
        self, pi_effective: np.ndarray, complexities: np.ndarray
    ) -> Dict[str, Any]:
        tertiles = np.percentile(complexities, [33, 67])
        buckets = {
            "low": float(pi_effective[complexities <= tertiles[0]].mean())
            if (complexities <= tertiles[0]).any()
            else None,
            "mid": float(
                pi_effective[
                    (complexities > tertiles[0]) & (complexities <= tertiles[1])
                ].mean()
            )
            if (
                (complexities > tertiles[0]) & (complexities <= tertiles[1])
            ).any()
            else None,
            "high": float(pi_effective[complexities > tertiles[1]].mean())
            if (complexities > tertiles[1]).any()
            else None,
        }

        return {
            "pi_range": (float(pi_effective.min()), float(pi_effective.max())),
            "pi_std": float(pi_effective.std()),
            "complexity_range": (
                float(complexities.min()),
                float(complexities.max()),
            ),
            "pi_by_complexity_bucket": buckets,
        }

    def _persist(self, result: DynamicPiResult) -> None:
        output_path = self.cache_dir / f"{self.domain_name}_dynamic_pi.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)


__all__ = ["DynamicPiCalculator", "DynamicPiResult"]


