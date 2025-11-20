"""
NFL Live Betting Engine
-----------------------

Adapts the NHL live engine heuristics for football (four quarters,
touchdown-heavy scoring).
"""

from __future__ import annotations

from narrative_optimization.domains.nhl.live_betting_engine import LiveBettingEngine


class NFLLiveBettingEngine(LiveBettingEngine):
    def _calculate_comeback_probability(self, game_state):
        home_score = game_state["home_score"]
        away_score = game_state["away_score"]
        period = int(float(game_state.get("current_period", game_state.get("period", 1))))
        time_remaining = float(game_state.get("time_remaining", 0))
        score_diff = home_score - away_score

        if score_diff >= 0:
            base = 0.75 if period < 4 else 0.82
            return min(base + (time_remaining / 60.0) * 0.05, 0.97)

        deficit = abs(score_diff)
        if deficit <= 3:
            base_prob = 0.42
        elif deficit <= 7:
            base_prob = 0.30
        elif deficit <= 10:
            base_prob = 0.18
        else:
            base_prob = 0.10

        if period <= 2:
            time_factor = 1.0
        elif period == 3:
            time_factor = 0.6
        else:
            time_factor = 0.35 if time_remaining > 5 else 0.15

        comeback_prob = base_prob * time_factor

        momentum = self._calculate_momentum(game_state)
        if momentum > 1:
            comeback_prob *= 1.2
        elif momentum < -1:
            comeback_prob *= 0.8

        return min(comeback_prob, 0.9)

