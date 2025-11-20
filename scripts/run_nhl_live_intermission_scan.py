#!/usr/bin/env python3
"""
NHL Live Intermission Scanner
=============================

Pulls:
  1. Current NHL game states from the public StatsAPI
  2. Live (or latest) FanDuel moneylines from The Odds API
Then runs the NHL LiveBettingEngine at intermissions / period ends to flag edges.

Usage:
    python scripts/run_nhl_live_intermission_scan.py --date 2025-11-19

Notes:
  - Only checks games whose status is "In Progress" and currently in intermission
    (or just ended a period), which matches the user's half-time / period-end
    workflow.
  - Requires the Odds API key configured in config/odds_api_config.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CONFIG_DIR = PROJECT_ROOT / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from odds_api_config import BASE_URL, ODDS_API_KEY, SPORTS
from narrative_optimization.domains.nhl.live_betting_engine import LiveBettingEngine


STATS_API_SCHEDULE = "https://statsapi.web.nhl.com/api/v1/schedule"
STATS_API_LINESCORE = "https://statsapi.web.nhl.com/api/v1/game/{game_pk}/linescore"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan NHL intermissions for live betting edges.")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Target date (YYYY-MM-DD). Defaults to today (UTC).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/nhl_live_recommendations.json"),
        help="File to append latest recommendations.",
    )
    return parser.parse_args()


def fetch_schedule(date_str: str) -> List[Dict]:
    resp = requests.get(STATS_API_SCHEDULE, params={"date": date_str})
    resp.raise_for_status()
    data = resp.json()
    dates = data.get("dates", [])
    if not dates:
        return []
    return dates[0].get("games", [])


def fetch_linescore(game_pk: int) -> Dict:
    resp = requests.get(STATS_API_LINESCORE.format(game_pk=game_pk))
    resp.raise_for_status()
    return resp.json()


def fetch_fanduel_odds() -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel",
        "dateFormat": "iso",
    }
    resp = requests.get(f"{BASE_URL}/sports/{SPORTS['nhl']}/odds", params=params, timeout=15)
    resp.raise_for_status()
    odds_data = resp.json()
    odds_map: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for event in odds_data:
        home = event.get("home_team")
        away = event.get("away_team")
        if not home or not away:
            continue
        key = (home, away)
        price_home = price_away = None
        for bookmaker in event.get("bookmakers", []):
            if bookmaker.get("key") != "fanduel":
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home:
                        price_home = outcome.get("price")
                    elif outcome.get("name") == away:
                        price_away = outcome.get("price")
        odds_map[key] = {
            "home_moneyline": price_home,
            "away_moneyline": price_away,
        }
    return odds_map


def in_intermission(linescore: Dict) -> bool:
    intermission = linescore.get("intermissionInfo", {})
    if intermission.get("inIntermission"):
        return True
    # Also treat exact end-of-period (clock reads 00:00) as a valid pause
    return linescore.get("currentPeriodTimeRemaining") == "END"


def build_game_state(game: Dict, linescore: Dict, odds_lookup: Dict) -> Optional[Dict]:
    status = game.get("status", {}).get("detailedState", "")
    if not status.startswith("In Progress"):
        return None
    if not in_intermission(linescore):
        return None

    home_info = linescore["teams"]["home"]
    away_info = linescore["teams"]["away"]
    home_name = home_info["team"]["name"]
    away_name = away_info["team"]["name"]

    odds = odds_lookup.get((home_name, away_name), {})

    period_scores = []
    for period in linescore.get("periods", []):
        period_scores.append({
            "home": period.get("home", {}).get("goals", 0),
            "away": period.get("away", {}).get("goals", 0),
        })

    intermission = linescore.get("intermissionInfo", {})
    time_remaining = intermission.get("intermissionTimeRemaining", 0) / 60.0

    return {
        "home_team": home_name,
        "away_team": away_name,
        "current_period": linescore.get("currentPeriod", 1),
        "time_remaining": time_remaining,
        "home_score": home_info.get("goals", 0),
        "away_score": away_info.get("goals", 0),
        "period_scores": period_scores,
        "live_odds": {
            "home_moneyline": odds.get("home_moneyline"),
            "away_moneyline": odds.get("away_moneyline"),
        },
    }


def main() -> None:
    args = parse_args()
    games = fetch_schedule(args.date)
    if not games:
        print(f"No NHL games found for {args.date}")
        return

    odds_lookup = fetch_fanduel_odds()
    engine = LiveBettingEngine()
    recommendations = []

    for game in games:
        game_pk = game.get("gamePk")
        if not game_pk:
            continue
        linescore = fetch_linescore(game_pk)
        game_state = build_game_state(game, linescore, odds_lookup)
        if not game_state:
            continue
        rec = engine.analyze_live_game(game_state)
        if rec:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            recommendations.append(rec)

    if recommendations:
        print(f"\nDetected {len(recommendations)} live opportunities.")
    else:
        print("\nNo live edges detected at current intermissions.")

    if recommendations:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if args.output.exists():
            with args.output.open() as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        existing.extend(recommendations)
        with args.output.open("w") as f:
            json.dump(existing, f, indent=2)
        print(f"Recommendations appended to {args.output}")


if __name__ == "__main__":
    main()

