"""
Multi-Scale Enhanced Transformer
================================

Extends the legacy multi-scale extractor with:
- explicit macro/meso/micro/nano separation
- scale interactions + volatility
- hooks for temporal checkpoints + dynamic π
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MultiScaleEnhancedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = [
            "macro_season_norm",
            "macro_is_playoff_window",
            "macro_league_hash",
            "macro_story_density",
            "macro_pi_signal",
            "macro_variance_flag",
            "meso_division_hash",
            "meso_rivalry_flag",
            "meso_streak_norm",
            "meso_context_prestige",
            "micro_rest_delta",
            "micro_record_delta",
            "micro_spread_norm",
            "micro_moneyline_delta",
            "micro_story_length",
            "micro_sentiment_proxy",
            "nano_checkpoint_count",
            "nano_momentum_shift",
            "nano_edge_delta",
            "nano_complexity_flag",
            "scale_alignment",
            "scale_tension",
            "scale_entropy",
            "scale_depth",
            "pi_alignment",
            "contextual_gravity",
            "prestige_alignment",
            "temporal_position",
            "temporal_compression",
            "narrative_drift",
            "interaction_macro_micro",
            "interaction_macro_nano",
        ]

    def fit(self, X: Iterable[Dict], y=None):
        return self

    def transform(self, X: Iterable[Dict]):
        features = []
        for record in X:
            features.append(self._featurize(record))
        return np.array(features, dtype=float)

    # ------------------------------------------------------------------ #
    # Feature helpers
    # ------------------------------------------------------------------ #

    def _featurize(self, record: Dict) -> List[float]:
        season = self._parse_season(record.get("season"))
        macro_playoff = float(record.get("is_playoff") or record.get("playoff") or 0)
        league = record.get("league") or record.get("court") or record.get("game_type")
        narrative = (
            record.get("synthetic_narrative")
            or record.get("narrative")
            or record.get("majority_opinion")
            or ""
        )
        pi_metadata = record.get("pi_metadata") or {}
        pi_effective = float(pi_metadata.get("pi_effective") or pi_metadata.get("pi_base") or 0.5)

        temporal_context = record.get("temporal_context") or {}
        rest_adv = float(temporal_context.get("rest_advantage") or record.get("rest_advantage") or 0.0)
        record_diff = float(temporal_context.get("record_differential") or 0.0)
        spread = float(record.get("spread_line") or 0.0)
        moneyline = self._moneyline_delta(record.get("betting_odds") or {})
        checkpoints = record.get("checkpoints") or []

        macro_story_density = len(narrative.split()) / 400.0
        macro_variance_flag = float(abs(rest_adv) > 2 or macro_playoff)
        meso_division = (
            record.get("division")
            or record.get("conference")
            or record.get("metadata", {}).get("area_of_law")
        )
        rivalry = float(
            record.get("is_rivalry")
            or record.get("div_game")
            or (record.get("metadata") or {}).get("precedent_setting", False)
        )
        streak = float(
            temporal_context.get("home_streak", 0) or record.get("streak", 0)
        )
        prestige = float(
            (record.get("home_brand_weight") or 0)
            if "home_brand_weight" in record
            else (record.get("metadata") or {}).get("citation_count", 0) / 50000.0
        )
        micro_story_length = len(narrative)
        micro_sentiment_proxy = narrative.lower().count("must-win") + narrative.lower().count("historic")

        nano_checkpoint_count = len(checkpoints)
        nano_momentum_shift = self._momentum_shift(checkpoints)
        nano_edge_delta = self._edge_delta(checkpoints)
        nano_complexity_flag = float(nano_checkpoint_count >= 4 or nano_momentum_shift > 0.1)

        scale_alignment = float(np.sign(rest_adv) == np.sign(record_diff))
        scale_tension = float(np.sign(rest_adv) != np.sign(spread))
        scale_entropy = self._entropy([macro_playoff, rivalry, nano_complexity_flag])
        scale_depth = float(bool(season)) + float(bool(meso_division)) + float(bool(checkpoints))

        pi_alignment = pi_effective * (1 + 0.1 * macro_playoff)
        contextual_gravity = abs(rest_adv) + abs(record_diff)
        prestige_alignment = prestige * (1 + rivalry * 0.3)

        temporal_position = temporal_context.get("season_progress") or self._temporal_position(record.get("date"))
        temporal_compression = float(abs(spread) < 3 and macro_playoff)
        narrative_drift = float(micro_sentiment_proxy > 0 and nano_momentum_shift < 0)

        interaction_macro_micro = rest_adv * macro_story_density
        interaction_macro_nano = macro_playoff * nano_momentum_shift

        return [
            season,
            macro_playoff,
            self._hash_float(league),
            macro_story_density,
            pi_effective,
            macro_variance_flag,
            self._hash_float(meso_division),
            rivalry,
            streak / 10.0,
            prestige,
            rest_adv / 5.0,
            record_diff,
            spread / 14.0,
            moneyline / 400.0,
            micro_story_length / 600.0,
            micro_sentiment_proxy,
            nano_checkpoint_count,
            nano_momentum_shift,
            nano_edge_delta,
            nano_complexity_flag,
            scale_alignment,
            scale_tension,
            scale_entropy,
            scale_depth,
            pi_alignment,
            contextual_gravity,
            prestige_alignment,
            temporal_position,
            temporal_compression,
            narrative_drift,
            interaction_macro_micro,
            interaction_macro_nano,
        ]

    @staticmethod
    def _parse_season(season) -> float:
        try:
            if isinstance(season, str) and len(season) == 8:
                season = int(season[:4])
            season = int(season)
            return (season - 2000) / 50.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _hash_float(value: Optional[str]) -> float:
        if not value:
            return 0.0
        return (hash(value) % 1000) / 1000.0

    @staticmethod
    def _moneyline_delta(odds: Dict) -> float:
        home = odds.get("moneyline_home") or 0
        away = odds.get("moneyline_away") or 0
        try:
            return float(home) - float(away)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _momentum_shift(checkpoints: List[Dict]) -> float:
        if len(checkpoints) < 2:
            return 0.0
        probs = [
            cp.get("metrics", {}).get("win_probability_home")
            for cp in checkpoints
            if cp.get("metrics")
        ]
        probs = [p for p in probs if p is not None]
        if len(probs) < 2:
            return 0.0
        return probs[-1] - probs[0]

    @staticmethod
    def _edge_delta(checkpoints: List[Dict]) -> float:
        edges = [
            cp.get("metrics", {}).get("pre_game_edge")
            for cp in checkpoints
            if cp.get("metrics")
        ]
        edges = [e for e in edges if e is not None]
        if not edges:
            return 0.0
        return edges[-1] - edges[0]

    @staticmethod
    def _temporal_position(date_value) -> float:
        if not date_value:
            return 0.0
        try:
            dt = datetime.fromisoformat(str(date_value))
        except ValueError:
            return 0.0
        return dt.timetuple().tm_yday / 366.0

    @staticmethod
    def _entropy(values: Iterable[float]) -> float:
        array = np.array(list(values))
        probs = array / (array.sum() + 1e-9)
        return float(-np.nansum(probs * np.log(probs + 1e-9)))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"multi_scale_enhanced_{name}" for name in self.feature_names])


__all__ = ["MultiScaleEnhancedTransformer"]


