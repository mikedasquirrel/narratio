#!/usr/bin/env python3
"""
Merge ESPN NHL scoreboard data with normalized Odds API history to produce a
modeling-ready dataset (results + closing lines + metadata).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
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
SCOREBOARD_FILE = DATA_ROOT / "nhl_season" / "espn_scoreboard_complete.json"
ODDS_FILE = DATA_ROOT / "processed_odds" / "icehockey_nhl_normalized.jsonl"
OUTPUT_DIR = DATA_ROOT / "modeling_datasets"
OUTPUT_FILE = OUTPUT_DIR / "nhl_games_with_closing_odds.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "nhl_games_with_closing_odds_summary.json"
UNMATCHED_FILE = OUTPUT_DIR / "nhl_games_without_closing_odds.json"

MAX_TIME_DELTA_SECONDS = 60 * 60 * 30


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(cleaned).astimezone(timezone.utc)
    except ValueError:
        return None


def load_scoreboard_games() -> List[Dict[str, object]]:
    if not SCOREBOARD_FILE.exists():
        raise FileNotFoundError(
            "Missing NHL scoreboard data. Run fetch_nhl_scoreboard.py first."
        )
    payload = json.loads(SCOREBOARD_FILE.read_text())
    games: List[Dict[str, object]] = []

    for season_payload in payload.get("seasons", []):
        season_label = season_payload.get("season")
        for game in season_payload.get("games", []):
            comps = game.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            event_time = parse_iso(comp.get("date") or game.get("date"))
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

            season_info = game.get("season") or {}
            season_type = int(season_info.get("type", 2))

            games.append(
                {
                    "espn_game_id": game.get("id"),
                    "season_label": season_label,
                    "season_year": (game.get("season") or {}).get("year"),
                    "season_type": season_type,
                    "event_time": event_time,
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
                }
            )

    return games


def _safe_score(score_payload) -> Optional[int]:
    if score_payload is None:
        return None
    try:
        return int(score_payload)
    except (TypeError, ValueError):
        if isinstance(score_payload, dict):
            value = score_payload.get("value")
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def load_odds_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with ODDS_FILE.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            commence = parse_iso(payload.get("commence_time"))
            payload["commence_dt"] = commence
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
        commence = record.get("commence_dt")
        if not commence or record.get("game_id") in used_ids:
            continue
        delta = abs((commence - target_time).total_seconds())
        if delta > MAX_TIME_DELTA_SECONDS:
            continue
        if best is None or delta < best_delta:
            best = record
            best_delta = delta
    return best, best_delta


def determine_moneyline_result(home_score: Optional[int], away_score: Optional[int]) -> Optional[float]:
    if home_score is None or away_score is None:
        return None
    return 1.0 if home_score > away_score else 0.0


def build_modeling_record(
    scoreboard: Dict[str, object],
    odds: Dict[str, object],
    time_delta: Optional[float],
) -> Dict[str, object]:
    home_score = scoreboard["home_score"]
    away_score = scoreboard["away_score"]
    home_margin = (
        home_score - away_score if (home_score is not None and away_score is not None) else None
    )
    moneyline_result = determine_moneyline_result(home_score, away_score)
    total_points = (
        home_score + away_score if (home_score is not None and away_score is not None) else None
    )

    return {
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
        "closing_puckline_home": odds.get("spread_home_median_point"),
        "closing_puckline_home_price": odds.get("spread_home_median_price"),
        "closing_puckline_away_price": odds.get("spread_away_median_price"),
        "closing_total": odds.get("total_over_median_point"),
        "closing_total_price": odds.get("total_over_median_price"),
        "total_under_price": odds.get("total_under_median_price"),
        "moneyline_result": moneyline_result,
        "spread_result": None,  # placeholder for future puckline labeling
        "total_result": None,
        "matching_time_delta_seconds": time_delta,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scoreboard_games = load_scoreboard_games()
    initial_count = len(scoreboard_games)
    scoreboard_games = [
        g for g in scoreboard_games if g.get("season_type") in (2, 3)
    ]
    print(
        f"Filtered preseason games: {initial_count - len(scoreboard_games)} removed "
        f"({len(scoreboard_games)} remaining)"
    )
    odds_records = load_odds_records()
    odds_index = build_odds_index(odds_records)
    commence_times = [r.get("commence_dt") for r in odds_records if r.get("commence_dt")]
    earliest_commence = min(commence_times) if commence_times else None
    if earliest_commence:
        before_filter = len(scoreboard_games)
        cutoff = earliest_commence - timedelta(days=2)
        scoreboard_games = [g for g in scoreboard_games if g["event_time"] >= cutoff]
        print(
            f"Trimmed games earlier than odds coverage: {before_filter - len(scoreboard_games)} removed "
            f"(cutoff={cutoff.date()})"
        )

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

    def _serialize_game(game: Dict[str, object]) -> Dict[str, object]:
        payload = dict(game)
        event_time = payload.get("event_time")
        if isinstance(event_time, datetime):
            payload["event_time"] = event_time.isoformat()
        return {
            "espn_game_id": payload.get("espn_game_id"),
            "season": payload.get("season"),
            "season_label": payload.get("season_label"),
            "season_year": payload.get("season_year"),
            "season_type": payload.get("season_type"),
            "event_time": payload.get("event_time"),
            "home_team": payload.get("home_team"),
            "away_team": payload.get("away_team"),
            "neutral_site": payload.get("neutral_site"),
            "tournament_type": payload.get("tournament_type"),
        }

    UNMATCHED_FILE.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "count": len(unmatched_games),
                "games": [_serialize_game(game) for game in unmatched_games[:1000]],
            },
            indent=2,
        )
    )

    summary = {
        "scoreboard_games": len(scoreboard_games),
        "odds_records": len(odds_records),
        "merged_games": len(merged_records),
        "match_rate": len(merged_records) / len(scoreboard_games) if scoreboard_games else 0,
        "unused_odds_records": len(odds_records) - len(used_ids),
        "median_time_delta_seconds": median(time_deltas) if time_deltas else None,
        "max_time_delta_seconds": max(time_deltas) if time_deltas else None,
        "unmatched_games": len(unmatched_games),
        "unmatched_sample": [_serialize_game(game) for game in unmatched_games[:25]],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(
        f"Matched {summary['merged_games']} of {summary['scoreboard_games']} NHL games "
        f"({summary['match_rate']:.1%}) → {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

