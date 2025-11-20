#!/usr/bin/env python3
"""
Merge the canonical NHL narrative feature set with newly scraped closing odds.

Inputs:
  - narrative_optimization/domains/nhl/nhl_narrative_betting_dataset.parquet
  - data/modeling_datasets/nhl_games_with_closing_odds.jsonl

Output:
  - data/modeling_datasets/nhl_feature_matrix.parquet
  - data/modeling_datasets/nhl_feature_matrix_summary.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.utils.team_normalization import team_key


DATA_ROOT = PROJECT_ROOT / "data"
NARRATIVE_DATASET = (
    PROJECT_ROOT
    / "narrative_optimization"
    / "domains"
    / "nhl"
    / "nhl_narrative_betting_dataset.parquet"
)
ODDS_DATASET = DATA_ROOT / "modeling_datasets" / "nhl_games_with_closing_odds.jsonl"
OUTPUT_PARQUET = DATA_ROOT / "modeling_datasets" / "nhl_feature_matrix.parquet"
SUMMARY_FILE = DATA_ROOT / "modeling_datasets" / "nhl_feature_matrix_summary.json"

TEAM_NAME_MAP = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "State Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}
TEAM_KEY_TO_ABBR = {team_key(name): abbr for name, abbr in TEAM_NAME_MAP.items()}


def load_odds_df() -> pd.DataFrame:
    records = []
    with ODDS_DATASET.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record["event_date"] = record["event_time"][:10]
            record["home_abbr"] = TEAM_KEY_TO_ABBR.get(team_key(record["home_team"]))
            record["away_abbr"] = TEAM_KEY_TO_ABBR.get(team_key(record["away_team"]))
            if not record["home_abbr"] or not record["away_abbr"]:
                continue
            records.append(record)
    df = pd.DataFrame(records)
    return df


def main() -> None:
    if not NARRATIVE_DATASET.exists():
        raise FileNotFoundError("Missing NHL narrative dataset.")
    if not ODDS_DATASET.exists():
        raise FileNotFoundError("Missing NHL odds dataset. Run build_nhl_modeling_dataset.py.")

    odds_df = load_odds_df()
    narrative_df = pd.read_parquet(NARRATIVE_DATASET)

    narrative_df["key_date"] = narrative_df["date"]
    narrative_df["home_abbr"] = narrative_df["home_team"]
    narrative_df["away_abbr"] = narrative_df["away_team"]

    merged = pd.merge(
        narrative_df,
        odds_df[
            [
                "event_date",
                "home_abbr",
                "away_abbr",
                "closing_moneyline_home",
                "closing_moneyline_home_implied",
                "closing_moneyline_away",
                "closing_puckline_home",
                "closing_puckline_home_price",
                "closing_puckline_away_price",
                "closing_total",
                "closing_total_price",
                "total_under_price",
                "moneyline_result",
                "matching_time_delta_seconds",
            ]
        ],
        left_on=["key_date", "home_abbr", "away_abbr"],
        right_on=["event_date", "home_abbr", "away_abbr"],
        how="inner",
    )

    coverage = len(merged) / len(narrative_df) if len(narrative_df) else 0

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    SUMMARY_FILE.write_text(
        json.dumps(
            {
                "narrative_rows": int(len(narrative_df)),
                "odds_rows": int(len(odds_df)),
                "merged_rows": int(len(merged)),
                "coverage": coverage,
            },
            indent=2,
        )
    )
    print(
        f"Built NHL feature matrix with {len(merged)} rows "
        f"({coverage:.1%} of narrative dataset) → {OUTPUT_PARQUET}"
    )


if __name__ == "__main__":
    main()

