"""
NFL Checkpoint Narrative Builder
--------------------------------

Creates deterministic checkpoint snapshots (pregame + quarters) from the
structured NFL dataset so temporal decomposition can operate without live feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import math
import numpy as np


PHASES = (
    ("PREGAME", "Pregame Thesis", 0, 0.0),
    ("Q1", "End of 1st quarter", 15, 0.25),
    ("HALF", "Halftime", 30, 0.50),
    ("Q3", "End of 3rd quarter", 45, 0.75),
    ("FINAL", "Final", 60, 1.0),
)


@dataclass(frozen=True)
class NFLSnapshot:
    game_id: str
    checkpoint_id: str
    sequence: int
    narrative: str
    score: Dict[str, int]
    metrics: Dict[str, float]
    metadata: Dict[str, float]
    target: int


class NFLCheckpointBuilder:
    def build(
        self,
        games: Iterable[Dict],
        checkpoint: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        snapshots: List[Dict] = []
        for game in games:
            per_game = self._build_game_snapshots(game)
            if checkpoint:
                per_game = [snap for snap in per_game if snap["checkpoint_id"] == checkpoint]
            snapshots.extend(per_game)
            if limit and len(snapshots) >= limit:
                return snapshots[:limit]
        return snapshots

    def _build_game_snapshots(self, game: Dict) -> List[Dict]:
        home, away = game.get("home_team"), game.get("away_team")
        if not home or not away:
            return []

        home_total = int(game.get("home_score", 0))
        away_total = int(game.get("away_score", 0))
        target = 1 if home_total > away_total else 0

        spread = float(game.get("spread_line") or (game.get("betting_odds") or {}).get("spread_home") or 0.0)
        rest_adv = float(game.get("rest_advantage") or (game.get("temporal_context") or {}).get("rest_advantage") or 0.0)
        odds = game.get("betting_odds") or {}
        implied_prob = float(odds.get("implied_prob_home") or 0.5)

        home_split = self._split_points(home_total)
        away_split = self._split_points(away_total)

        snapshots: List[Dict] = []
        sequence = 0
        for idx, (code, label, minutes, progress) in enumerate(PHASES):
            if code == "PREGAME":
                snapshots.append(
                    self._pregame_snapshot(
                        game, home, away, implied_prob, rest_adv, spread, sequence, target
                    )
                )
                sequence += 1
                continue

            quarter_idx = idx - 1
            score_home = sum(home_split[: quarter_idx + 1])
            score_away = sum(away_split[: quarter_idx + 1])

            win_prob = self._estimate_win_probability(implied_prob, spread, score_home - score_away, progress)
            metrics = {
                "home_score": float(score_home),
                "away_score": float(score_away),
                "score_differential": float(score_home - score_away),
                "win_probability_home": win_prob,
                "spread": spread,
                "rest_advantage": rest_adv,
            }
            metadata = {
                "minutes_elapsed": float(minutes),
                "game_progress": float(progress),
                "checkpoint_label": label,
            }
            narrative = self._narrative(label, home, away, score_home, score_away, win_prob, spread)
            snapshots.append(
                self._to_dict(
                    NFLSnapshot(
                        game_id=str(game.get("game_id")),
                        checkpoint_id=code,
                        sequence=sequence,
                        narrative=narrative,
                        score={"home": score_home, "away": score_away},
                        metrics=metrics,
                        metadata=metadata,
                        target=target,
                    )
                )
            )
            sequence += 1

        return snapshots

    def _split_points(self, total: int) -> List[int]:
        if total <= 0:
            return [0, 0, 0, 0]
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        raw = weights * total
        ints = [int(math.floor(x)) for x in raw]
        remainder = total - sum(ints)
        idx = 0
        while remainder > 0:
            ints[idx % 4] += 1
            remainder -= 1
            idx += 1
        return ints

    def _estimate_win_probability(self, base_prob, spread, score_diff, progress):
        logit = math.log(max(min(base_prob, 0.999), 0.001) / (1 - max(min(base_prob, 0.999), 0.001)))
        logit += 0.12 * score_diff
        logit += 0.04 * spread
        logit += progress * 0.5 * np.sign(score_diff)
        odds = math.exp(logit)
        return float(odds / (1 + odds))

    def _pregame_snapshot(self, game, home, away, implied_prob, rest_adv, spread, sequence, target):
        metrics = {
            "home_score": 0.0,
            "away_score": 0.0,
            "score_differential": 0.0,
            "win_probability_home": implied_prob,
            "spread": spread,
            "rest_advantage": rest_adv,
        }
        narrative = (
            f"Pregame: {home} favored by {spread:+.1f}, rest edge {rest_adv:+.1f} days. "
            f"Win probability {implied_prob:.1%} vs {away}."
        )
        metadata = {
            "minutes_elapsed": 0.0,
            "game_progress": 0.0,
            "checkpoint_label": "Pregame Thesis",
        }
        return self._to_dict(
            NFLSnapshot(
                game_id=str(game.get("game_id")),
                checkpoint_id="PREGAME",
                sequence=sequence,
                narrative=narrative,
                score={"home": 0, "away": 0},
                metrics=metrics,
                metadata=metadata,
                target=target,
            )
        )

    @staticmethod
    def _narrative(label, home, away, home_score, away_score, win_prob, spread):
        leader = home if home_score >= away_score else away
        return (
            f"{label}: {home} {home_score}-{away_score} {away}. "
            f"{leader} asserting script; win probability {win_prob:.1%} (spread {spread:+.1f})."
        )

    @staticmethod
    def _to_dict(snapshot: NFLSnapshot) -> Dict:
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


def build_nfl_checkpoint_snapshots(
    games: Iterable[Dict],
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    return NFLCheckpointBuilder().build(games, checkpoint=checkpoint, limit=limit)


