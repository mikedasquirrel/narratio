#!/usr/bin/env python3
"""
Join canonical NBA narrative features with closing odds for modeling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.utils.team_normalization import normalize_team_name


DATA_ROOT = PROJECT_ROOT / "data"
ODDS_FILE = DATA_ROOT / "modeling_datasets" / "nba_games_with_closing_odds.jsonl"
META_FILE = DATA_ROOT / "domains" / "nba_all_seasons_real.json"
FEATURES_FILE = PROJECT_ROOT / "narrative_optimization" / "data" / "features" / "nba_all_features.npz"
OUTPUT_PARQUET = DATA_ROOT / "modeling_datasets" / "nba_feature_matrix.parquet"
SUMMARY_FILE = DATA_ROOT / "modeling_datasets" / "nba_feature_matrix_summary.json"

VALID_NBA_TEAMS = {
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
}

TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}


def normalize_team(name: str) -> str:
    normalized = normalize_team_name(name)
    return TEAM_ALIASES.get(normalized, normalized)


def american_to_prob(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return -price / (-price + 100.0)
    return None


def load_odds_rows() -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    with ODDS_FILE.open() as f:
        for line in f:
            if not line.strip():
                continue
            game = json.loads(line)
            home_team = normalize_team(game["home_team"])
            away_team = normalize_team(game["away_team"])
            if home_team not in VALID_NBA_TEAMS or away_team not in VALID_NBA_TEAMS:
                continue
            date = game["event_time"][:10]
            home_row = {
                "date": date,
                "team_name": home_team,
                "opponent": away_team,
                "is_home": True,
                "team_score": game["home_score"],
                "opponent_score": game["away_score"],
                "margin": game["home_margin"],
                "closing_moneyline": game["closing_moneyline_home"],
                "closing_moneyline_implied": game.get("closing_moneyline_home_implied"),
                "closing_spread": game["closing_spread_home"],
                "closing_spread_price": game["closing_spread_home_price"],
                "closing_total": game["closing_total"],
        "closing_total_price": game["closing_total_price"],
        "total_under_price": game["total_under_price"],
                "moneyline_result": game["moneyline_result"],
                "spread_result": game["spread_result"],
                "total_result": game["total_result"],
                "espn_game_id": game["espn_game_id"],
                "season_year": game["season_year"],
            }
            away_row = {
                "date": date,
                "team_name": away_team,
                "opponent": home_team,
                "is_home": False,
                "team_score": game["away_score"],
                "opponent_score": game["home_score"],
                "margin": -game["home_margin"] if game["home_margin"] is not None else None,
                "closing_moneyline": game["closing_moneyline_away"],
                "closing_moneyline_implied": american_to_prob(game["closing_moneyline_away"]),
                "closing_spread": -game["closing_spread_home"] if game["closing_spread_home"] is not None else None,
                "closing_spread_price": game["closing_spread_away_price"],
        "closing_total": game["closing_total"],
        "closing_total_price": game["closing_total_price"],
        "total_under_price": game["total_under_price"],
                "moneyline_result": 1.0 - game["moneyline_result"] if game["moneyline_result"] is not None else None,
                "spread_result": _invert_result(game["spread_result"]),
                "total_result": game["total_result"],
                "espn_game_id": game["espn_game_id"],
                "season_year": game["season_year"],
            }
            records.append(home_row)
            records.append(away_row)
    return pd.DataFrame(records)


def _invert_result(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if abs(value - 0.5) < 1e-6:
        return 0.5
    if value == 1.0:
        return 0.0
    if value == 0.0:
        return 1.0
    return None


def main() -> None:
    if not META_FILE.exists():
        raise FileNotFoundError("Missing nba_all_seasons_real.json metadata.")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError("Missing nba_all_features.npz.")
    if not ODDS_FILE.exists():
        raise FileNotFoundError("Missing nba_games_with_closing_odds.jsonl. Run build_nba_modeling_dataset.py.")

    meta_entries = json.loads(META_FILE.read_text())
    meta_df = pd.DataFrame(meta_entries)
    meta_df = meta_df.rename(columns={"won": "actual_win", "points": "team_points"})
    meta_df["team_name"] = meta_df["team_name"].apply(normalize_team)
    meta_df = meta_df[meta_df["team_name"].isin(VALID_NBA_TEAMS)]

    feature_data = np.load(FEATURES_FILE, allow_pickle=True)
    feature_df = pd.DataFrame(feature_data["features"], columns=feature_data["feature_names"])
    if len(feature_df) != len(meta_df):
        raise ValueError("Feature matrix length does not match metadata length.")
    meta_df = pd.concat([meta_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    odds_team_df = load_odds_rows()

    merged = pd.merge(
        meta_df,
        odds_team_df,
        left_on=["date", "team_name", "home_game"],
        right_on=["date", "team_name", "is_home"],
        how="inner",
        suffixes=("", "_odds"),
    )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    coverage = len(merged) / len(meta_df) if len(meta_df) else 0.0
    SUMMARY_FILE.write_text(
        json.dumps(
            {
                "meta_rows": int(len(meta_df)),
                "odds_rows": int(len(odds_team_df)),
                "merged_rows": int(len(merged)),
                "coverage": coverage,
            },
            indent=2,
        )
    )

    print(
        f"Built NBA feature matrix with {len(merged)} rows "
        f"({coverage:.1%} of canonical dataset) → {OUTPUT_PARQUET}"
    )


if __name__ == "__main__":
    main()

