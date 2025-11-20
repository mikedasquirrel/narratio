"""
NHL Checkpoint Narrative Builder
================================

Produces per-period checkpoint summaries from existing NHL structured data so
the narrative optimization pipeline can surface actionable updates after each
period (P1, P2, Final) without needing true live play-by-play yet.

The builder deterministically:
- Derives lightweight scoring splits per period from final totals + context
- Computes updated win probabilities vs pre-game odds
- Emits concise narrative blurbs plus machine-friendly metrics

All logic is deterministic and only depends on fields already present in
`data/domains/nhl_games_with_odds.json`.

Author: Narrative Optimization Framework
Date: November 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from textwrap import shorten


BASE_PERIOD_WEIGHTS = np.array([0.36, 0.34, 0.30])


@dataclass(frozen=True)
class TemporalPhase:
    code: str
    label: str
    minutes: float
    progress: float


PERIODS = (
    TemporalPhase("PREGAME", "Pregame Thesis", 0, 0.0),
    TemporalPhase("P1", "End of 1st period", 20, 0.33),
    TemporalPhase("P2", "End of 2nd period", 40, 0.67),
    TemporalPhase("FINAL", "End of regulation", 60, 1.00),
)


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Container for period-level snapshot data."""

    game_id: str
    checkpoint_id: str
    sequence: int
    narrative: str
    score: Dict[str, int]
    metrics: Dict[str, float]
    metadata: Dict[str, float]
    target: int


class NHLCheckpointBuilder:
    """
    Synthesizes NHL checkpoint narratives from static game records.

    The approximation strategy:
    - Use contextual edges (rest, record differential, odds) to bias which team
      is expected to surge early vs late.
    - Deterministically allocate final goals across periods using biased
      weights so replay/backtest runs remain reproducible.
    - Translate each checkpoint into a structured snapshot with text +
      analytics suitable for models and UI.
    """

    def __init__(self):
        self._base_weights = BASE_PERIOD_WEIGHTS

    def build(
        self,
        games: Iterable[Dict],
        checkpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Build snapshots for all games (optionally filtered)."""
        snapshots: List[Dict] = []
        for game in games:
            per_game = self._build_snapshots_for_game(game)
            if checkpoint:
                per_game = [
                    snap for snap in per_game if snap["checkpoint_id"] == checkpoint
                ]
            snapshots.extend(per_game)
            if limit and len(snapshots) >= limit:
                snapshots = snapshots[:limit]
                break
        return snapshots

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _build_snapshots_for_game(self, game: Dict) -> List[Dict]:
        game_id = str(game.get("game_id"))
        home = game.get("home_team", "HOME")
        away = game.get("away_team", "AWAY")
        target = 1 if game.get("home_won") else 0

        context = game.get("temporal_context", {}) or {}
        odds = game.get("betting_odds", {}) or {}

        home_bias = self._compute_bias(context, odds_edge="home")
        away_bias = -home_bias  # From away perspective

        home_scores = self._split_goals(game.get("home_score", 0), home_bias)
        away_scores = self._split_goals(game.get("away_score", 0), away_bias)

        p_snapshots: List[Dict] = []
        cumulative_home = 0
        cumulative_away = 0
        base_prob = self._sanitize_probability(odds.get("implied_prob_home", 0.5))

        score_idx = -1
        sequence_counter = 0
        for phase in PERIODS:
            checkpoint_id = phase.code
            label = phase.label
            minutes_elapsed = phase.minutes
            progress = phase.progress

            if checkpoint_id == "PREGAME":
                snapshot = self._pregame_snapshot(
                    game_id,
                    home,
                    away,
                    base_prob,
                    context,
                    odds,
                    target,
                    sequence_counter,
                )
                p_snapshots.append(snapshot)
                sequence_counter += 1
                continue

            score_idx += 1
            cumulative_home += home_scores[score_idx]
            cumulative_away += away_scores[score_idx]

            score_diff = cumulative_home - cumulative_away
            win_probability = self._estimate_win_probability(
                base_prob,
                score_diff,
                progress,
            )

            metrics = {
                "home_score": float(cumulative_home),
                "away_score": float(cumulative_away),
                "score_differential": float(score_diff),
                "win_probability_home": win_probability,
                "pre_game_edge": base_prob - 0.5,
                "rest_advantage": float(context.get("rest_advantage", 0) or 0.0),
                "record_differential": float(
                    context.get("record_differential", 0.0) or 0.0
                ),
                "leverage_index": self._leverage_index(score_diff, progress),
            }

            metadata = {
                "minutes_elapsed": float(minutes_elapsed),
                "game_progress": float(progress),
                "checkpoint_label": label,
                "implied_edge": float(odds.get("implied_prob_home", 0.5) - 0.5),
            }

            narrative = self._compose_narrative(
                label,
                home,
                away,
                cumulative_home,
                cumulative_away,
                metrics,
                win_probability,
                target,
                context,
            )

            snapshot = CheckpointSnapshot(
                game_id=game_id,
                checkpoint_id=checkpoint_id,
                sequence=sequence_counter,
                narrative=narrative,
                score={"home": cumulative_home, "away": cumulative_away},
                metrics=metrics,
                metadata=metadata,
                target=target,
            )

            p_snapshots.append(self._to_dict(snapshot))
            sequence_counter += 1

        return p_snapshots

    def _pregame_snapshot(
        self,
        game_id: str,
        home: str,
        away: str,
        base_prob: float,
        context: Dict,
        odds: Dict,
        target: int,
        sequence: int,
    ) -> Dict:
        metrics = {
            "home_score": 0.0,
            "away_score": 0.0,
            "score_differential": 0.0,
            "win_probability_home": base_prob,
            "pre_game_edge": base_prob - 0.5,
            "rest_advantage": float(context.get("rest_advantage", 0.0) or 0.0),
            "record_differential": float(context.get("record_differential", 0.0) or 0.0),
            "leverage_index": 0.0,
        }
        metadata = {
            "minutes_elapsed": 0.0,
            "game_progress": 0.0,
            "checkpoint_label": "Pregame Thesis",
            "implied_edge": float(odds.get("implied_prob_home", 0.5) - 0.5),
        }
        return {
            "game_id": game_id,
            "checkpoint_id": "PREGAME",
            "sequence": sequence,
            "narrative": (
                f"{home} hosts {away}. Rest edge {metrics['rest_advantage']:+.1f} days, "
                f"record delta {metrics['record_differential']:+.2f}. "
                f"Pregame win probability {base_prob:.1%}."
            ),
            "score": {"home": 0, "away": 0},
            "metrics": metrics,
            "metadata": metadata,
            "target": target,
        }

    def _compute_bias(self, context: Dict, odds_edge: str = "home") -> float:
        rest_adv = float(context.get("rest_advantage", 0.0) or 0.0)
        record_diff = float(context.get("record_differential", 0.0) or 0.0)
        form_diff = float(context.get("form_differential", 0.0) or 0.0)

        raw_bias = 0.15 * rest_adv + 0.10 * record_diff + 0.08 * form_diff

        return float(np.clip(raw_bias, -1.5, 1.5))

    def _split_goals(self, total_goals: int, bias: float) -> List[int]:
        if total_goals <= 0:
            return [0, 0, 0]

        bias_vector = np.array([0.05, 0.0, -0.05]) * bias
        weights = self._base_weights + bias_vector
        weights = np.clip(weights, 0.15, 0.55)
        weights = weights / weights.sum()

        raw_values = weights * total_goals
        allocations = [int(math.floor(val)) for val in raw_values]
        remainder = total_goals - sum(allocations)

        # Distribute remainder by largest fractional parts to keep determinism
        fractional = [
            (raw_values[i] - allocations[i], -i) for i in range(len(raw_values))
        ]
        fractional.sort(reverse=True)
        idx = 0
        while remainder > 0 and idx < len(fractional):
            allocations[-fractional[idx][1]] += 1
            remainder -= 1
            idx += 1

        return allocations

    def _estimate_win_probability(
        self,
        base_prob: float,
        score_diff: int,
        progress: float,
    ) -> float:
        # Convert base prob to logit space for additive adjustments
        base_prob = self._sanitize_probability(base_prob)
        logit = math.log(base_prob / (1 - base_prob))

        # Score differential impact (roughly 0.6 logit swing per goal)
        logit += score_diff * (0.6 + 0.15 * progress)

        # Progress penalty (less time remaining amplifies current edge)
        logit += progress * 0.25 * np.sign(score_diff)

        return self._sigmoid(logit)

    def _leverage_index(self, score_diff: int, progress: float) -> float:
        leverage = abs(score_diff) + 0.5
        time_remaining = max(0.01, 1.0 - progress)
        return float(leverage / (1 + 4 * time_remaining))

    def _compose_narrative(
        self,
        label: str,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        metrics: Dict[str, float],
        win_probability: float,
        target: int,
        context: Dict,
    ) -> str:
        score_phrase = (
            f"{home} {home_score}-{away_score} {away}"
            if home_score >= away_score
            else f"{away} {away_score}-{home_score} {home}"
        )

        rest_adv = metrics["rest_advantage"]
        rest_callout = ""
        if abs(rest_adv) >= 1:
            advantaged = home if rest_adv > 0 else away
            rest_callout = f"{advantaged} showing the expected rest edge. "

        rivalry = "Rivalry stakes intensify. " if context.get("is_rivalry") else ""

        direction = (
            "Home crowd surging."
            if win_probability >= 0.55
            else "Visitors flip momentum."
            if win_probability <= 0.45
            else "Game remains balanced."
        )

        narrative = (
            f"{label}: {score_phrase}. "
            f"{rest_callout}{rivalry}"
            f"{direction} "
            f"Win probability for {home} now {win_probability:.1%} "
            f"(pre-game {(metrics['pre_game_edge'] + 0.5):.1%})."
        )

        return shorten(narrative, width=420, placeholder="…")

    @staticmethod
    def _sigmoid(value: float) -> float:
        try:
            result = 1 / (1 + math.exp(-value))
        except OverflowError:
            result = 0.0 if value < 0 else 1.0
        return float(min(max(result, 0.001), 0.999))

    @staticmethod
    def _sanitize_probability(prob: float) -> float:
        if prob is None:
            return 0.5
        return float(min(max(prob, 0.001), 0.999))

    @staticmethod
    def _to_dict(snapshot: CheckpointSnapshot) -> Dict:
        return {
            "game_id": snapshot.game_id,
            "checkpoint_id": snapshot.checkpoint_id,
            "sequence": snapshot.sequence,
            "narrative": snapshot.narrative,
            "score": snapshot.score,
            "metrics": snapshot.metrics,
            "metadata": snapshot.metadata,
            "target": snapshot.target,
        }


def build_nhl_checkpoint_snapshots(
    games: Iterable[Dict],
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Convenience wrapper used by the domain registry.
    """
    builder = NHLCheckpointBuilder()
    return builder.build(games, checkpoint=checkpoint, limit=limit)


