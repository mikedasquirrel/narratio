"""
Literary Alignment Calibrator
=============================

Leverages the newly ingested literary corpora (WikiPlots, Stereotropes,
CMU Movie Summaries, ML Research) to derive normative narrative anchors.
These anchors act as lightweight "embeddings" that quantify how closely a
record's storytelling mirrors high-performing literary patterns. The
scores can be injected into any domain pipeline (sports, legal, etc.)
without reprocessing the raw corpora on each run.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class LiteraryAlignmentCalibrator:
    """
    Computes scalar alignment scores between any record's narrative
    surface features and the reference distribution learned from
    literary corpora.
    """

    DEFAULT_DOMAINS = ("wikiplots", "stereotropes", "cmu_movies")

    def __init__(
        self,
        domains: Optional[Iterable[str]] = None,
        context_dir: Optional[Path] = None,
        dynamic_pi_dir: Optional[Path] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        results_root = project_root / "narrative_optimization" / "results"

        self.domains = tuple(domains or self.DEFAULT_DOMAINS)
        self.context_dir = context_dir or results_root / "context_stratification"
        self.dynamic_pi_dir = dynamic_pi_dir or results_root / "dynamic_pi"
        self.fallbacks = {
            "narrative_length": 2800.0,
            "sentence_count": 18.0,
            "word_count": 520.0,
        }

        self.references = self._build_condition_reference()
        self.pi_reference = self._build_pi_reference()
        self.density_anchor = self._build_density_anchor()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(self, record: Dict) -> Dict[str, float]:
        """
        Returns a deterministic feature dict that can be merged directly
        into existing model rows.
        """

        narrative = self._extract_narrative(record)
        word_count = float(len(narrative.split())) if narrative else 0.0
        sentence_count = float(
            narrative.count(".") + narrative.count("!") + narrative.count("?")
        )
        char_count = float(len(narrative))
        pi_effective = self._extract_pi(record)

        length_alignment = self._ratio(char_count, "narrative_length")
        sentence_alignment = self._ratio(sentence_count, "sentence_count")
        word_alignment = self._ratio(word_count, "word_count")
        density_delta = self._density_delta(word_count, sentence_count)
        pi_alignment = self._pi_alignment(pi_effective)
        pi_zscore = self._pi_zscore(pi_effective)

        story_coverage = (
            length_alignment + sentence_alignment + word_alignment
        ) / 3.0 if any(
            value > 0 for value in (length_alignment, sentence_alignment, word_alignment)
        ) else 0.0

        return {
            "literary_length_alignment": length_alignment,
            "literary_sentence_alignment": sentence_alignment,
            "literary_word_alignment": word_alignment,
            "literary_density_delta": density_delta,
            "literary_pi_alignment": pi_alignment,
            "literary_pi_zscore": pi_zscore,
            "literary_story_coverage": story_coverage,
        }

    # ------------------------------------------------------------------ #
    # Reference builders
    # ------------------------------------------------------------------ #

    def _build_condition_reference(self) -> Dict[str, Dict[str, Optional[float]]]:
        aggregates: Dict[str, Dict[str, List[float]]] = {}
        for domain in self.domains:
            path = self.context_dir / f"{domain}_contexts.json"
            if not path.exists():
                continue
            data = self._safe_load(path)
            for pattern in data.get("patterns", []):
                for feature, condition in (pattern.get("conditions") or {}).items():
                    stats = aggregates.setdefault(
                        feature, {"min": [], "max": [], "eq": []}
                    )
                    if isinstance(condition, dict):
                        for bound in ("min", "max", "eq"):
                            value = condition.get(bound)
                            self._maybe_store(stats[bound], value)
                    else:
                        self._maybe_store(stats["eq"], condition)

        summary: Dict[str, Dict[str, Optional[float]]] = {}
        for feature, stats in aggregates.items():
            summary[feature] = {
                "min_anchor": self._median(stats["min"]),
                "max_anchor": self._median(stats["max"]),
                "eq_anchor": self._median(stats["eq"]),
            }
        return summary

    def _build_pi_reference(self) -> Dict[str, float]:
        means: List[float] = []
        stds: List[float] = []
        for domain in self.domains:
            path = self.dynamic_pi_dir / f"{domain}_dynamic_pi.json"
            if not path.exists():
                continue
            payload = self._safe_load(path)
            self._maybe_store(means, payload.get("pi_mean"))
            self._maybe_store(stds, payload.get("pi_std"))

        pi_mean = float(statistics.mean(means)) if means else 0.78
        pi_std = float(statistics.mean(stds)) if stds else 0.04
        if pi_std == 0:
            pi_std = 0.01
        return {"mean": pi_mean, "std": pi_std}

    def _build_density_anchor(self) -> float:
        word_anchor = (
            self.references.get("word_count", {}).get("min_anchor")
            or self.fallbacks["word_count"]
        )
        sentence_anchor = (
            self.references.get("sentence_count", {}).get("min_anchor")
            or self.fallbacks["sentence_count"]
        )
        if sentence_anchor == 0:
            sentence_anchor = 1.0
        return float(word_anchor / sentence_anchor)

    # ------------------------------------------------------------------ #
    # Feature helpers
    # ------------------------------------------------------------------ #

    def _ratio(self, value: float, feature_key: str) -> float:
        if value <= 0:
            return 0.0
        anchor = (
            self.references.get(feature_key, {}).get("min_anchor")
            or self.fallbacks.get(feature_key)
        )
        if not anchor:
            return 0.0
        ratio = value / anchor
        return round(min(max(ratio, 0.0), 4.0), 6)

    def _density_delta(self, word_count: float, sentence_count: float) -> float:
        if word_count <= 0 or sentence_count <= 0 or not self.density_anchor:
            return 0.0
        density = word_count / sentence_count
        delta = (density - self.density_anchor) / self.density_anchor
        return round(max(min(delta, 4.0), -4.0), 6)

    def _pi_alignment(self, pi_effective: float) -> float:
        if pi_effective <= 0 or not self.pi_reference["mean"]:
            return 0.0
        ratio = pi_effective / self.pi_reference["mean"]
        return round(min(max(ratio, 0.0), 3.0), 6)

    def _pi_zscore(self, pi_effective: float) -> float:
        if pi_effective <= 0:
            return 0.0
        mean = self.pi_reference["mean"]
        std = self.pi_reference["std"]
        return round((pi_effective - mean) / std, 6)

    @staticmethod
    def _extract_narrative(record: Dict) -> str:
        return (
            record.get("synthetic_narrative")
            or record.get("narrative")
            or record.get("majority_opinion")
            or record.get("summary")
            or ""
        )

    @staticmethod
    def _extract_pi(record: Dict) -> float:
        pi_meta = record.get("pi_metadata") or {}
        try:
            return float(pi_meta.get("pi_effective") or pi_meta.get("pi_base") or 0.5)
        except (TypeError, ValueError):
            return 0.5

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def _maybe_store(bucket: List[float], value) -> None:
        if value is None:
            return
        try:
            bucket.append(float(value))
        except (TypeError, ValueError):
            return

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        try:
            return float(statistics.median(values))
        except statistics.StatisticsError:
            return None

    @staticmethod
    def _safe_load(path: Path) -> Dict:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


__all__ = ["LiteraryAlignmentCalibrator"]


