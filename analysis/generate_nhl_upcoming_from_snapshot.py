#!/usr/bin/env python3
"""
Generate Upcoming NHL Games from Absolute Maximum Snapshot
===========================================================

This utility consumes the latest Absolute Maximum odds snapshot and converts
future games into the canonical `nhl_upcoming_*.json` format expected by the
scoring pipeline. It enriches each matchup with aggregated bookmaker odds and
current-season records derived from the historical results file.

Usage:
    python analysis/generate_nhl_upcoming_from_snapshot.py \
        --snapshot analysis/nhl_absolute_max_snapshot.json \
        --history data/domains/nhl_games_with_odds.json \
        --output data/domains/nhl_upcoming_latest.json

Key features:
  - Filters to games with commence_time in the future (UTC).
  - Aggregates bookmaker markets into consensus moneylines, totals, and puck lines.
  - Maps full team names to NHL abbreviations (handling legacy and relocated clubs).
  - Injects overall/home/away records based on the latest completed games in the
    historical dataset.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TEAM_ABBR_TO_NAME = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "STL": "St Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}

# Additional aliases seen in the snapshot feed
NAME_ALIASES = {
    "Montréal Canadiens": "Montreal Canadiens",
    "Utah Mammoth": "Utah Hockey Club",
}

NAME_TO_ABBR = {}
for abbr, name in TEAM_ABBR_TO_NAME.items():
    NAME_TO_ABBR[name.lower()] = abbr
for alias, canonical in NAME_ALIASES.items():
    NAME_TO_ABBR[alias.lower()] = NAME_TO_ABBR.get(canonical.lower(), "UTA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive upcoming NHL games from Absolute Max snapshot.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("analysis/nhl_absolute_max_snapshot.json"),
        help="Path to the Absolute Maximum snapshot JSON.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/domains/nhl_games_with_odds.json"),
        help="Path to the historical NHL games file for record calculations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/domains/nhl_upcoming_latest.json"),
        help="Destination file for the generated upcoming games.",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=3,
        help="Maximum number of days ahead to include (to avoid bloating the file).",
    )
    return parser.parse_args()


def infer_season(commence_dt: datetime) -> str:
    if commence_dt.month >= 7:
        start = commence_dt.year
        end = commence_dt.year + 1
    else:
        start = commence_dt.year - 1
        end = commence_dt.year
    return f"{start}{end}"


def summarize(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def aggregate_bookmakers(bookmakers: List[Dict], home_name: str, away_name: str) -> Dict[str, Optional[float]]:
    home_ml, away_ml = [], []
    pl_home_pts, pl_home_prices, pl_away_pts, pl_away_prices = [], [], [], []
    totals_pts, totals_over_prices, totals_under_prices = [], [], []

    for book in bookmakers or []:
        for market in book.get("markets", []) or []:
            key = market.get("key")
            outcomes = market.get("outcomes", []) or []
            if key == "h2h":
                for outcome in outcomes:
                    if outcome.get("name") == home_name:
                        home_ml.append(outcome.get("price"))
                    elif outcome.get("name") == away_name:
                        away_ml.append(outcome.get("price"))
            elif key == "spreads":
                for outcome in outcomes:
                    if outcome.get("name") == home_name:
                        pl_home_pts.append(outcome.get("point"))
                        pl_home_prices.append(outcome.get("price"))
                    elif outcome.get("name") == away_name:
                        pl_away_pts.append(outcome.get("point"))
                        pl_away_prices.append(outcome.get("price"))
            elif key == "totals":
                for outcome in outcomes:
                    name = (outcome.get("name") or "").lower()
                    if "over" in name:
                        totals_pts.append(outcome.get("point"))
                        totals_over_prices.append(outcome.get("price"))
                    elif "under" in name:
                        totals_under_prices.append(outcome.get("price"))

    return {
        "moneyline_home": summarize(home_ml),
        "moneyline_away": summarize(away_ml),
        "puck_line_home": summarize(pl_home_pts),
        "puck_line_home_odds": summarize(pl_home_prices),
        "puck_line_away": summarize(pl_away_pts),
        "puck_line_away_odds": summarize(pl_away_prices),
        "total": summarize(totals_pts),
        "over_odds": summarize(totals_over_prices),
        "under_odds": summarize(totals_under_prices),
    }


def build_records(history_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    with history_path.open() as f:
        games = json.load(f)

    records: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "home_wins": 0,
        "home_losses": 0,
        "away_wins": 0,
        "away_losses": 0,
    }))

    for game in games:
        season = game.get("season")
        home = game.get("home_team")
        away = game.get("away_team")
        home_won = bool(game.get("home_won"))
        if not season or not home or not away:
            continue

        home_stats = records[season][home]
        away_stats = records[season][away]

        if home_won:
            home_stats["wins"] += 1
            home_stats["home_wins"] += 1
            away_stats["losses"] += 1
            away_stats["away_losses"] += 1
        else:
            away_stats["wins"] += 1
            away_stats["away_wins"] += 1
            home_stats["losses"] += 1
            home_stats["home_losses"] += 1

    return records


def to_abbr(team_name: str) -> Optional[str]:
    if not team_name:
        return None
    key = team_name.lower()
    return NAME_TO_ABBR.get(key)


def format_odds(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return int(round(value))


def build_upcoming_entries(
    snapshot_games: List[Dict],
    records: Dict[str, Dict[str, Dict[str, int]]],
    lookahead_days: int,
) -> List[Dict]:
    now = datetime.now(timezone.utc)
    today = now.date()
    cutoff = now + timedelta(days=lookahead_days)
    upcoming = []

    for game in snapshot_games:
        commence_raw = game.get("commence_time")
        if not commence_raw:
            continue
        commence_dt = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
        if commence_dt.date() < today or commence_dt > cutoff:
            continue

        home_abbr = to_abbr(game.get("home_team"))
        away_abbr = to_abbr(game.get("away_team"))
        if not home_abbr or not away_abbr:
            continue

        season = infer_season(commence_dt)
        record_table = records.get(season, {})
        home_stats = record_table.get(home_abbr, {})
        away_stats = record_table.get(away_abbr, {})

        odds_summary = aggregate_bookmakers(game.get("bookmakers", []), game.get("home_team"), game.get("away_team"))

        entry = {
            "game_id": f"{commence_dt.strftime('%Y%m%d')}-{away_abbr}-{home_abbr}",
            "season": season,
            "date": commence_dt.date().isoformat(),
            "commence_time": commence_raw,
            "game_type": "regular",
            "venue": "",
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_record": {
                "wins": int(home_stats.get("wins", 0)),
                "losses": int(home_stats.get("losses", 0)),
            },
            "away_record": {
                "wins": int(away_stats.get("wins", 0)),
                "losses": int(away_stats.get("losses", 0)),
            },
            "home_home_record": {
                "wins": int(home_stats.get("home_wins", 0)),
                "losses": int(home_stats.get("home_losses", 0)),
            },
            "away_away_record": {
                "wins": int(away_stats.get("away_wins", 0)),
                "losses": int(away_stats.get("away_losses", 0)),
            },
            "betting_odds": {
                "moneyline_home": format_odds(odds_summary["moneyline_home"]),
                "moneyline_away": format_odds(odds_summary["moneyline_away"]),
                "total": odds_summary["total"],
                "over_odds": format_odds(odds_summary["over_odds"]),
                "under_odds": format_odds(odds_summary["under_odds"]),
                "puck_line_home": odds_summary["puck_line_home"],
                "puck_line_home_odds": format_odds(odds_summary["puck_line_home_odds"]),
                "puck_line_away": odds_summary["puck_line_away"],
                "puck_line_away_odds": format_odds(odds_summary["puck_line_away_odds"]),
            },
        }
        upcoming.append(entry)

    upcoming.sort(key=lambda g: g["commence_time"])
    return upcoming


def main() -> None:
    args = parse_args()

    if not args.snapshot.exists():
        raise SystemExit(f"Snapshot file not found: {args.snapshot}")
    if not args.history.exists():
        raise SystemExit(f"History file not found: {args.history}")

    with args.snapshot.open() as f:
        snapshot = json.load(f)
    snapshot_games = snapshot.get("games", [])

    records = build_records(args.history)
    upcoming = build_upcoming_entries(snapshot_games, records, args.lookahead_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(upcoming, f, indent=2)

    print(f"✓ Generated {len(upcoming)} upcoming games -> {args.output}")


if __name__ == "__main__":
    main()

