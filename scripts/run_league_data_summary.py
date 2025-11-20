#!/usr/bin/env python3
"""
League Data Summary
===================

Parses the raw NBA/NFL domain datasets and computes headline metrics:
  - Win rates
  - Simple ROI from moneyline prices (NBA)
  - Favorite vs underdog splits
  - Spread cover rates (NFL)

Outputs a JSON summary @ analysis/nba_nfl_data_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import ijson
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NBA_PATH = PROJECT_ROOT / "data" / "domains" / "nba_enhanced_betting_data.json"
NFL_PATH = PROJECT_ROOT / "data" / "domains" / "nfl_games_with_odds.json"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "nba_nfl_data_summary.json"


def calc_ml_profit(moneyline: float, won: bool, stake: float = 100.0) -> float:
    if moneyline is None:
        return 0.0
    if won:
        if moneyline > 0:
            return stake * (moneyline / 100.0)
        else:
            return stake * (100.0 / abs(moneyline))
    return -stake


def analyze_nba() -> Dict[str, Any]:
    total = wins = 0
    ml_bets = ml_profit = 0.0
    fav_bets = fav_profit = 0.0
    dog_bets = dog_profit = 0.0
    forms = {}

    with NBA_PATH.open("rb") as f:
        for entry in ijson.items(f, "item"):
            total += 1
            won = bool(entry.get("won"))
            if won:
                wins += 1

            odds = entry.get("betting_odds", {}) or {}
            moneyline = odds.get("moneyline")
            if moneyline is not None:
                profit = calc_ml_profit(moneyline, won)
                ml_profit += profit
                ml_bets += 1
                if moneyline < 0:
                    fav_profit += profit
                    fav_bets += 1
                else:
                    dog_profit += profit
                    dog_bets += 1

            form = entry.get("temporal_context", {}).get("form")
            if form:
                forms.setdefault(form, {"games": 0, "wins": 0})
                forms[form]["games"] += 1
                if won:
                    forms[form]["wins"] += 1

    def pct(x: float, y: float) -> float:
        return (x / y) if y else 0.0

    return {
        "file": str(NBA_PATH),
        "total_entries": total,
        "win_rate": pct(wins, total),
        "moneyline_roi": pct(ml_profit, ml_bets * 100.0),
        "favorite_roi": pct(fav_profit, fav_bets * 100.0),
        "underdog_roi": pct(dog_profit, dog_bets * 100.0),
        "form_breakdown": {
            form: {
                "games": info["games"],
                "win_rate": pct(info["wins"], info["games"]),
            }
            for form, info in forms.items()
        },
    }


def analyze_nfl() -> Dict[str, Any]:
    total = home_wins = 0
    fav_games = fav_covers = fav_wins = 0
    dog_games = dog_covers = dog_wins = 0
    close_games = 0

    with NFL_PATH.open("rb") as f:
        for entry in ijson.items(f, "item"):
            total += 1
            home_score = entry.get("home_score", 0) or 0
            away_score = entry.get("away_score", 0) or 0
            odds = entry.get("betting_odds", {}) or {}
            spread = odds.get("spread")
            home_win = bool(entry.get("home_won"))
            if home_win:
                home_wins += 1

            if spread is None:
                continue

            if abs(spread) <= 3:
                close_games += 1

            if spread < 0:
                fav_games += 1
                if home_win:
                    fav_wins += 1
                if home_score - away_score + spread > 0:
                    fav_covers += 1
            else:
                dog_games += 1
                if home_win:
                    dog_wins += 1
                if home_score - away_score + spread > 0:
                    dog_covers += 1

    def pct(x: float, y: float) -> float:
        return (x / y) if y else 0.0

    return {
        "file": str(NFL_PATH),
        "total_games": total,
        "home_win_rate": pct(home_wins, total),
        "favorite_games": fav_games,
        "favorite_win_rate": pct(fav_wins, fav_games),
        "favorite_cover_rate": pct(fav_covers, fav_games),
        "underdog_games": dog_games,
        "underdog_win_rate": pct(dog_wins, dog_games),
        "underdog_cover_rate": pct(dog_covers, dog_games),
        "close_spread_games": close_games,
    }


def main() -> None:
    summary = {
        "nba": analyze_nba(),
        "nfl": analyze_nfl(),
        "generated_at": datetime.utcnow().isoformat(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

