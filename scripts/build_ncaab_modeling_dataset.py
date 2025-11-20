#!/usr/bin/env python3
"""
Merge normalized The Odds API history with verified NCAA game results.

Outputs a modeling-ready JSONL file with:
  * ESPN scoreboard metadata (home/away, scores, venue context)
  * Closing market consensus (moneyline, spreads, totals)
  * Derived labels (cover result, total outcome, deltas)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.utils.team_normalization import (
    normalize_team_name,
    team_key,
)


DATA_ROOT = PROJECT_ROOT / "data"
SCOREBOARD_SOURCES = [
    DATA_ROOT / "college_basketball_season" / "espn_scoreboard_complete.json",
    DATA_ROOT / "college_basketball_season" / "complete_seasons_MULTIPLE.json",
]
ENRICHED_STATS_FILE = DATA_ROOT / "college_basketball_season" / "season_ENRICHED_complete.json"
ODDS_FILE = DATA_ROOT / "processed_odds" / "basketball_ncaab_normalized.jsonl"
OUTPUT_DIR = DATA_ROOT / "modeling_datasets"
OUTPUT_FILE = OUTPUT_DIR / "ncaab_games_with_closing_odds.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "ncaab_games_with_closing_odds_summary.json"

MAX_TIME_DELTA_SECONDS = 60 * 60 * 30  # 30 hours cushion for timezone mismatches
ENRICHED_STAT_FIELDS = [
    "points",
    "rebounds",
    "assists",
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "fieldGoalPct",
    "threePointFieldGoalsAttempted",
    "threePointFieldGoalsMade",
    "threePointFieldGoalPct",
    "freeThrowsAttempted",
    "freeThrowsMade",
    "freeThrowPct",
    "avgPoints",
    "avgRebounds",
    "avgAssists",
]


def parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def load_enriched_stats() -> Dict[str, Dict[str, Dict[str, float]]]:
    if not ENRICHED_STATS_FILE.exists():
        return {}
    payload = json.loads(ENRICHED_STATS_FILE.read_text())
    enriched: Dict[str, Dict[str, Dict[str, float]]] = {}
    for season_payload in payload.get("seasons", []):
        for game in season_payload.get("games", []):
            event_id = game.get("id")
            competitions = game.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            stat_entry: Dict[str, Dict[str, float]] = {}
            for competitor in comp.get("competitors", []):
                side = competitor.get("homeAway")
                if side not in ("home", "away"):
                    continue
                stat_entry[side] = _extract_stat_block(competitor.get("statistics", []))
            if stat_entry:
                enriched[event_id] = stat_entry
    return enriched


def _extract_stat_block(stat_entries: Sequence[Dict[str, object]]) -> Dict[str, float]:
    block: Dict[str, float] = {}
    for entry in stat_entries or []:
        name = entry.get("name")
        if name not in ENRICHED_STAT_FIELDS:
            continue
        raw_value = entry.get("displayValue") or entry.get("value")
        value = _coerce_number(raw_value)
        if value is not None:
            block[name] = value
    return block


def _coerce_number(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if cleaned in {"", "--", "null"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def load_scoreboard_games(
    enriched_stats: Dict[str, Dict[str, Dict[str, float]]]
) -> List[Dict[str, object]]:
    payload = None
    for path in SCOREBOARD_SOURCES:
        if path.exists():
            payload = json.loads(path.read_text())
            break
    if payload is None:
        raise FileNotFoundError(
            "No scoreboard dataset found. Run fetch_ncaab_scoreboard.py or provide complete_seasons_MULTIPLE.json."
        )
    games: List[Dict[str, object]] = []

    for season_payload in payload.get("seasons", []):
        season_label = season_payload.get("season")
        for game in season_payload.get("games", []):
            competitions = game.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            event_time = parse_iso8601(comp.get("date") or game.get("date"))
            if not event_time:
                continue

            home_name = home.get("team", {}).get("displayName") or home.get("team", {}).get(
                "shortDisplayName"
            )
            away_name = away.get("team", {}).get("displayName") or away.get("team", {}).get(
                "shortDisplayName"
            )
            if not home_name or not away_name:
                continue

            home_score = _safe_score(home.get("score"))
            away_score = _safe_score(away.get("score"))

            record = {
                "espn_game_id": game.get("id"),
                "season_label": season_label,
                "season_year": (game.get("season") or {}).get("year"),
                "event_time": event_time.astimezone(timezone.utc),
                "home_team": normalize_team_name(home_name),
                "away_team": normalize_team_name(away_name),
                "home_key": team_key(home_name),
                "away_key": team_key(away_name),
                "home_score": home_score,
                "away_score": away_score,
                "venue": (comp.get("venue") or {}).get("fullName"),
                "neutral_site": comp.get("neutralSite", False),
                "conference_game": comp.get("conferenceCompetition", False),
                "tournament_type": (comp.get("type") or {}).get("abbreviation"),
                "home_rank": (home.get("curatedRank") or {}).get("current"),
                "away_rank": (away.get("curatedRank") or {}).get("current"),
            }
            stat_entry = enriched_stats.get(record["espn_game_id"])
            if stat_entry:
                for stat_name in ENRICHED_STAT_FIELDS:
                    home_value = stat_entry.get("home", {}).get(stat_name)
                    away_value = stat_entry.get("away", {}).get(stat_name)
                    if home_value is not None:
                        record[f"home_stat_{stat_name}"] = home_value
                    if away_value is not None:
                        record[f"away_stat_{stat_name}"] = away_value
            games.append(record)

    return games


def _safe_score(score_payload) -> Optional[int]:
    if score_payload is None:
        return None
    if isinstance(score_payload, dict):
        value = score_payload.get("value")
    else:
        value = score_payload
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_odds_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with ODDS_FILE.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            commence_dt = parse_iso8601(payload.get("commence_time"))
            payload["commence_dt"] = commence_dt
            payload["home_key"] = team_key(payload.get("home_team"))
            payload["away_key"] = team_key(payload.get("away_team"))
            records.append(payload)
    return records


def build_odds_index(records: Sequence[Dict[str, object]]) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    index: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        index[(record["home_key"], record["away_key"])].append(record)
    for entries in index.values():
        entries.sort(key=lambda r: r.get("commence_dt") or datetime.min.replace(tzinfo=timezone.utc))
    return index


def find_match(
    odds_index: Dict[Tuple[str, str], List[Dict[str, object]]],
    key: Tuple[str, str],
    target_time: datetime,
    used_ids: set,
) -> Tuple[Optional[Dict[str, object]], Optional[float]]:
    candidates = odds_index.get(key, [])
    best = None
    best_delta = None
    for record in candidates:
        game_id = record.get("game_id")
        commence_dt = record.get("commence_dt")
        if not commence_dt or game_id in used_ids:
            continue
        delta = abs((commence_dt - target_time).total_seconds())
        if delta > MAX_TIME_DELTA_SECONDS:
            continue
        if best is None or delta < best_delta:
            best = record
            best_delta = delta
    return best, best_delta


def determine_spread_outcome(home_margin: Optional[int], spread_line: Optional[float]) -> Optional[float]:
    if home_margin is None or spread_line is None:
        return None
    delta = home_margin + spread_line
    if abs(delta) < 1e-6:
        return 0.5  # push
    return 1.0 if delta > 0 else 0.0


def determine_total_outcome(total_points: Optional[int], total_line: Optional[float]) -> Optional[float]:
    if total_points is None or total_line is None:
        return None
    delta = total_points - total_line
    if abs(delta) < 1e-6:
        return 0.5
    return 1.0 if delta > 0 else 0.0


def build_modeling_record(
    scoreboard: Dict[str, object],
    odds: Dict[str, object],
    time_delta: Optional[float],
) -> Dict[str, object]:
    home_score = scoreboard["home_score"]
    away_score = scoreboard["away_score"]
    home_margin = (home_score - away_score) if (home_score is not None and away_score is not None) else None
    total_points = (home_score + away_score) if (home_score is not None and away_score is not None) else None
    spread_line = odds.get("spread_home_median_point")
    total_line = odds.get("total_over_median_point")
    spread_result = determine_spread_outcome(home_margin, spread_line)
    total_result = determine_total_outcome(total_points, total_line)

    record = {
        "espn_game_id": scoreboard["espn_game_id"],
        "season": scoreboard["season_label"],
        "season_year": scoreboard["season_year"],
        "event_time": scoreboard["event_time"].isoformat(),
        "home_team": scoreboard["home_team"],
        "away_team": scoreboard["away_team"],
        "home_score": home_score,
        "away_score": away_score,
        "home_margin": home_margin,
        "total_points": total_points,
        "neutral_site": scoreboard["neutral_site"],
        "conference_game": scoreboard["conference_game"],
        "tournament_type": scoreboard["tournament_type"],
        "home_rank": scoreboard["home_rank"],
        "away_rank": scoreboard["away_rank"],
        "venue": scoreboard["venue"],
        "odds_game_id": odds.get("game_id"),
        "closing_commence_time": odds.get("commence_time"),
        "bookmaker_count": odds.get("bookmaker_count"),
        "h2h_market_count": odds.get("h2h_market_count"),
        "spreads_market_count": odds.get("spreads_market_count"),
        "totals_market_count": odds.get("totals_market_count"),
        "closing_moneyline_home": odds.get("h2h_home_median_price"),
        "closing_moneyline_home_implied": odds.get("h2h_home_median_implied_prob"),
        "closing_moneyline_away": odds.get("h2h_away_median_price"),
        "closing_spread_home": spread_line,
        "closing_spread_home_price": odds.get("spread_home_median_price"),
        "closing_spread_away": odds.get("spread_away_median_point"),
        "closing_spread_away_price": odds.get("spread_away_median_price"),
        "closing_total": total_line,
        "closing_total_price": odds.get("total_over_median_price"),
        "total_under_price": odds.get("total_under_median_price"),
        "update_latency_minutes": odds.get("update_latency_minutes"),
        "spread_result": spread_result,
        "total_result": total_result,
        "spread_delta": (home_margin + spread_line) if (home_margin is not None and spread_line is not None) else None,
        "total_delta": (total_points - total_line) if (total_points is not None and total_line is not None) else None,
        "matching_time_delta_seconds": time_delta,
    }

    for key, value in scoreboard.items():
        if key.startswith("home_stat_") or key.startswith("away_stat_"):
            record[key] = value

    return record


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched_stats = load_enriched_stats()
    scoreboard_games = load_scoreboard_games(enriched_stats)
    odds_records = load_odds_records()
    odds_index = build_odds_index(odds_records)

    used_ids = set()
    merged_records: List[Dict[str, object]] = []
    time_deltas: List[float] = []
    unmatched_games: List[Dict[str, object]] = []

    for game in scoreboard_games:
        match, delta = find_match(odds_index, (game["home_key"], game["away_key"]), game["event_time"], used_ids)
        if not match:
            unmatched_games.append(game)
            continue
        used_ids.add(match["game_id"])
        merged_records.append(build_modeling_record(game, match, delta))
        if delta is not None:
            time_deltas.append(delta)

    with OUTPUT_FILE.open("w") as f:
        for record in merged_records:
            f.write(json.dumps(record) + "\n")

    summary = {
        "scoreboard_games": len(scoreboard_games),
        "odds_records": len(odds_records),
        "merged_games": len(merged_records),
        "match_rate": len(merged_records) / len(scoreboard_games) if scoreboard_games else 0,
        "unused_odds_records": len(odds_records) - len(used_ids),
        "median_time_delta_seconds": median(time_deltas) if time_deltas else None,
        "max_time_delta_seconds": max(time_deltas) if time_deltas else None,
        "unmatched_games": len(unmatched_games),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(
        f"Matched {summary['merged_games']} of {summary['scoreboard_games']} NCAA games "
        f"({summary['match_rate']:.1%}) → {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

