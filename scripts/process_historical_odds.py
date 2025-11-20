#!/usr/bin/env python3
"""
Normalize and summarize The Odds API historical exports for downstream modeling.

This script ingests the large JSON dumps produced by the absolute scraping runs,
derives bookmaker-level consensus lines, and writes clean JSONL outputs with the
exact same schema for every supported sport.  The resulting datasets can be
joined against narrative features, temporal frameworks, or any other modeling
inputs without having to re-interpret market structures each time.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.utils.team_normalization import normalize_team_name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
HISTORICAL_ROOT = DATA_ROOT / "historical_odds_complete"
PROCESSED_ROOT = DATA_ROOT / "processed_odds"


SUPPORTED_SPORTS: Dict[str, Dict[str, str]] = {
    # (sport_key, friendly_name)
    "basketball_ncaab": {
        "friendly_name": "NCAAB",
        "input_file": HISTORICAL_ROOT / "basketball_ncaab_complete.json",
        "output_file": PROCESSED_ROOT / "basketball_ncaab_normalized.jsonl",
        "summary_file": PROCESSED_ROOT / "basketball_ncaab_summary.json",
    },
    "basketball_nba": {
        "friendly_name": "NBA",
        "input_file": HISTORICAL_ROOT / "basketball_nba_complete.json",
        "output_file": PROCESSED_ROOT / "basketball_nba_normalized.jsonl",
        "summary_file": PROCESSED_ROOT / "basketball_nba_summary.json",
    },
    "icehockey_nhl": {
        "friendly_name": "NHL",
        "input_file": HISTORICAL_ROOT / "icehockey_nhl_complete.json",
        "output_file": PROCESSED_ROOT / "icehockey_nhl_normalized.jsonl",
        "summary_file": PROCESSED_ROOT / "icehockey_nhl_summary.json",
    },
    "baseball_mlb": {
        "friendly_name": "MLB",
        "input_file": HISTORICAL_ROOT / "baseball_mlb_complete.json",
        "output_file": PROCESSED_ROOT / "baseball_mlb_normalized.jsonl",
        "summary_file": PROCESSED_ROOT / "baseball_mlb_summary.json",
    },
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    """Parse API timestamps while gracefully handling trailing 'Z'."""
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def american_to_implied(price: Optional[float]) -> Optional[float]:
    """Convert American odds to implied probability."""
    if price is None:
        return None
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def aggregate_price_stats(prices: Sequence[float]) -> Dict[str, Optional[float]]:
    if not prices:
        return {"best": None, "worst": None, "median": None}
    return {
        "best": max(prices),
        "worst": min(prices),
        "median": statistics.median(prices),
    }


def aggregate_line_stats(
    entries: Sequence[Tuple[Optional[float], Optional[float]]]
) -> Dict[str, Optional[float]]:
    if not entries:
        return {
            "min_point": None,
            "max_point": None,
            "median_point": None,
            "min_price": None,
            "max_price": None,
            "median_price": None,
            "point_range": None,
            "price_range": None,
            "sample_size": 0,
        }

    points = [pt for pt, _ in entries if pt is not None]
    prices = [pr for _, pr in entries if pr is not None]

    def _range(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        if len(values) == 1:
            return 0.0
        return max(values) - min(values)

    return {
        "min_point": min(points) if points else None,
        "max_point": max(points) if points else None,
        "median_point": statistics.median(points) if points else None,
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
        "median_price": statistics.median(prices) if prices else None,
        "point_range": _range(points),
        "price_range": _range(prices),
        "sample_size": len(entries),
    }


# ---------------------------------------------------------------------------
# Core normalizer
# ---------------------------------------------------------------------------

@dataclass
class HistoricalOddsNormalizer:
    sport_key: str
    friendly_name: str
    input_file: Path
    output_file: Path
    summary_file: Path

    def process(self) -> Dict[str, object]:
        games_payload = self._load_games()
        processed_records: Dict[str, Dict[str, object]] = {}
        missing_market_counts = {"h2h": 0, "spreads": 0, "totals": 0}
        commence_times: List[datetime] = []

        for game in games_payload:
            record = self._process_single_game(game)
            if not record:
                continue
            game_id = record["game_id"]
            existing = processed_records.get(game_id)
            if existing:
                record = self._prefer_record(existing, record)
            processed_records[game_id] = record
            commence_times.append(record["commence_time_dt"])
            for market in ("h2h", "spreads", "totals"):
                if record[f"{market}_market_count"] == 0:
                    missing_market_counts[market] += 1

        records_list = list(processed_records.values())
        self._write_records(records_list)
        summary = self._build_summary(records_list, missing_market_counts, commence_times)
        self.summary_file.write_text(json.dumps(summary, indent=2))
        return summary

    def _load_games(self) -> List[Dict[str, object]]:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Missing odds dump: {self.input_file}")
        payload = json.loads(self.input_file.read_text())
        games = payload.get("games", [])
        if not games:
            raise ValueError(f"No games found in {self.input_file}")
        return games

    def _process_single_game(self, game: Dict[str, object]) -> Optional[Dict[str, object]]:
        commence = parse_iso8601(game.get("commence_time"))
        bookmakers = game.get("bookmakers", [])
        if not commence or not bookmakers:
            return None

        home_raw = game.get("home_team")
        away_raw = game.get("away_team")
        if not home_raw or not away_raw:
            return None

        market_counts = self._count_markets(bookmakers)
        market_summary = self._summarize_markets(bookmakers, home_raw, away_raw)

        record: Dict[str, object] = {
            "game_id": game.get("id"),
            "sport_key": game.get("sport_key"),
            "friendly_sport": self.friendly_name,
            "commence_time": commence.astimezone(timezone.utc).isoformat(),
            "commence_time_dt": commence,
            "season_year": commence.astimezone(timezone.utc).year,
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_team": normalize_team_name(home_raw),
            "away_team": normalize_team_name(away_raw),
            "bookmaker_count": len(bookmakers),
            **market_counts,
            **market_summary,
        }

        last_updates = [
            parse_iso8601(bk.get("last_update")) for bk in bookmakers if bk.get("last_update")
        ]
        if last_updates:
            latest = max(dt for dt in last_updates if dt is not None)
            record["final_update_time"] = latest.astimezone(timezone.utc).isoformat()
            record["final_update_dt"] = latest
            record["update_latency_minutes"] = (
                (commence - latest).total_seconds() / 60.0
                if commence and latest
                else None
            )
        else:
            record["final_update_time"] = None
            record["final_update_dt"] = None
            record["update_latency_minutes"] = None

        return record

    @staticmethod
    def _count_markets(bookmakers: Sequence[Dict[str, object]]) -> Dict[str, int]:
        market_counts = {"h2h_market_count": 0, "spreads_market_count": 0, "totals_market_count": 0}
        for bookmaker in bookmakers:
            unique_markets = {market.get("key") for market in bookmaker.get("markets", [])}
            for key in unique_markets:
                if key == "h2h":
                    market_counts["h2h_market_count"] += 1
                elif key == "spreads":
                    market_counts["spreads_market_count"] += 1
                elif key == "totals":
                    market_counts["totals_market_count"] += 1
        return market_counts

    def _summarize_markets(
        self,
        bookmakers: Sequence[Dict[str, object]],
        home_raw: str,
        away_raw: str,
    ) -> Dict[str, Optional[float]]:
        h2h_prices = {"home": [], "away": []}
        spreads = {"home": [], "away": []}
        totals = {"over": [], "under": []}

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")

                    if key == "h2h":
                        if name == home_raw:
                            h2h_prices["home"].append(price)
                        elif name == away_raw:
                            h2h_prices["away"].append(price)
                    elif key == "spreads":
                        if name == home_raw:
                            spreads["home"].append((point, price))
                        elif name == away_raw:
                            spreads["away"].append((point, price))
                    elif key == "totals":
                        label = (name or "").lower()
                        if label.startswith("over"):
                            totals["over"].append((point, price))
                        elif label.startswith("under"):
                            totals["under"].append((point, price))

        summary: Dict[str, Optional[float]] = {}
        for side in ("home", "away"):
            stats = aggregate_price_stats(h2h_prices[side])
            summary[f"h2h_{side}_best_price"] = stats["best"]
            summary[f"h2h_{side}_worst_price"] = stats["worst"]
            summary[f"h2h_{side}_median_price"] = stats["median"]
            summary[f"h2h_{side}_median_implied_prob"] = american_to_implied(stats["median"])

            spread_stats = aggregate_line_stats(spreads[side])
            for metric, value in spread_stats.items():
                summary[f"spread_{side}_{metric}"] = value

        for direction in ("over", "under"):
            total_stats = aggregate_line_stats(totals[direction])
            for metric, value in total_stats.items():
                summary[f"total_{direction}_{metric}"] = value

        return summary

    @staticmethod
    def _prefer_record(
        existing: Dict[str, object],
        candidate: Dict[str, object],
    ) -> Dict[str, object]:
        """Choose the richer record when duplicate game_ids appear."""
        existing_score = (
            existing.get("bookmaker_count", 0)
            + existing.get("h2h_market_count", 0)
            + existing.get("spreads_market_count", 0)
            + existing.get("totals_market_count", 0)
        )
        candidate_score = (
            candidate.get("bookmaker_count", 0)
            + candidate.get("h2h_market_count", 0)
            + candidate.get("spreads_market_count", 0)
            + candidate.get("totals_market_count", 0)
        )

        if candidate_score > existing_score:
            return candidate
        if candidate_score < existing_score:
            return existing

        existing_update = existing.get("final_update_dt")
        candidate_update = candidate.get("final_update_dt")
        if existing_update and candidate_update:
            return candidate if candidate_update > existing_update else existing
        if candidate_update and not existing_update:
            return candidate
        return existing

    def _write_records(self, records: Sequence[Dict[str, object]]) -> None:
        PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w") as f:
            for record in records:
                payload = dict(record)
                payload.pop("commence_time_dt", None)
                payload.pop("final_update_dt", None)
                f.write(json.dumps(payload) + "\n")

    def _build_summary(
        self,
        records: Sequence[Dict[str, object]],
        missing_market_counts: Dict[str, int],
        commence_times: Sequence[datetime],
    ) -> Dict[str, object]:
        if not records:
            raise ValueError(f"No normalized records produced for {self.sport_key}")
        commence_sorted = sorted(commence_times)
        return {
            "sport_key": self.sport_key,
            "friendly_name": self.friendly_name,
            "records": len(records),
            "start_date": commence_sorted[0].astimezone(timezone.utc).isoformat(),
            "end_date": commence_sorted[-1].astimezone(timezone.utc).isoformat(),
            "missing_market_counts": missing_market_counts,
            "output_file": str(self.output_file),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize historical odds outputs into modeling-ready JSONL files."
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        default=None,
        help="Subset of sport keys to process (default = all configured sports).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_sports = args.sports or list(SUPPORTED_SPORTS.keys())
    summaries: Dict[str, Dict[str, object]] = {}

    for sport_key in selected_sports:
        if sport_key not in SUPPORTED_SPORTS:
            raise ValueError(f"Unsupported sport_key: {sport_key}")
        config = SUPPORTED_SPORTS[sport_key]
        normalizer = HistoricalOddsNormalizer(
            sport_key=sport_key,
            friendly_name=config["friendly_name"],
            input_file=Path(config["input_file"]),
            output_file=Path(config["output_file"]),
            summary_file=Path(config["summary_file"]),
        )
        summary = normalizer.process()
        summaries[sport_key] = summary
        print(
            f"[{sport_key}] Normalized {summary['records']} games "
            f"from {summary['start_date']} to {summary['end_date']} "
            f"→ {summary['output_file']}"
        )

    (PROCESSED_ROOT / "processing_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

