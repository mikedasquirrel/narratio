"""
NBA Live Betting Engine
-----------------------

Reuses the NHL engine infrastructure but adapts comeback math for
point-based scoring and four quarters.
"""

from __future__ import annotations

from narrative_optimization.domains.nhl.live_betting_engine import LiveBettingEngine


class NBALiveBettingEngine(LiveBettingEngine):
    def _calculate_comeback_probability(self, game_state):
        home_score = game_state["home_score"]
        away_score = game_state["away_score"]
        period = int(float(game_state.get("current_period", game_state.get("period", 1))))
        time_remaining = float(game_state.get("time_remaining", 0))
        score_diff = home_score - away_score

        if score_diff >= 0:
            base = 0.7 if period < 4 else 0.8
            return min(base + (time_remaining / 48.0) * 0.1, 0.95)

        deficit = abs(score_diff)
        if deficit <= 4:
            base_prob = 0.45
        elif deficit <= 8:
            base_prob = 0.30
        elif deficit <= 14:
            base_prob = 0.18
        else:
            base_prob = 0.08

        if period <= 2:
            time_factor = 1.0
        elif period == 3:
            time_factor = 0.55
        else:
            time_factor = 0.35 if time_remaining > 2 else 0.15

        comeback_prob = base_prob * time_factor

        momentum = self._calculate_momentum(game_state)
        if momentum > 2:
            comeback_prob *= 1.25
        elif momentum < -2:
            comeback_prob *= 0.75

        return min(comeback_prob, 0.85)

