#!/usr/bin/env python3
"""
Augment the NBA narrative corpus with ESPN scoreboard data (2020-2025).

Generates templated narratives for each team per game, appends them to
`data/domains/nba_pregame_narratives.json`, and updates the simplified
metadata file `data/domains/nba_all_seasons_real.json` used for feature merging.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from narrative_optimization.utils.team_normalization import normalize_team_name


SCOREBOARD_FILE = PROJECT_ROOT / "data" / "nba_season" / "espn_scoreboard_complete.json"
NARRATIVE_FILE = PROJECT_ROOT / "data" / "domains" / "nba_pregame_narratives.json"
META_FILE = PROJECT_ROOT / "data" / "domains" / "nba_all_seasons_real.json"
BATCH_FILE = PROJECT_ROOT / "narrative_optimization" / "BATCH_EXECUTION_STATUS.json"

VALID_TEAM_NAMES = {
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

NAME_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Los Angeles Clipers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}
SCOREBOARD_START_DATE = "2020-07-30"


def normalize_team(name: str) -> str:
    """Normalize ESPN team names to canonical NBA names."""
    normalized = normalize_team_name(name)
    normalized = NAME_ALIASES.get(normalized, normalized)
    return normalized


def season_label(year: int) -> str:
    """Convert ESPN season year (ending year) to NBA season label."""
    return f"{year-1}-{str(year)[-2:]}"


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def build_narrative(entry: dict) -> str:
    location = "host" if entry["home_game"] else "travel to"
    focus = "continue shaping their identity" if entry["home_game"] else "test themselves on the road"
    return (
        f"On {entry['date']}, the {entry['team_name']} {location} the {entry['opponent_name']} at {entry['venue']}."
        f" With the {entry['season']} campaign evolving, {entry['team_name']} look to {focus}"
        f" against a familiar conference foe."
    )


def build_rich_narrative(entry: dict) -> str:
    context = "home floor" if entry["home_game"] else "road stage"
    neutral_note = " (neutral site)" if entry["neutral_site"] else ""
    return (
        f"{entry['date']} | {entry['venue']}{neutral_note}\n"
        f"Season: {entry['season']} | Event ID: {entry['game_id']}\n\n"
        f"{entry['team_name']} and {entry['opponent_name']} square off as the league navigates a unique schedule cadence."
        f" {entry['team_name']} enter the matchup focused on rhythm and execution on the {context},"
        f" while the {entry['opponent_name']} present a contrasting style that keeps scouts intrigued."
        f" Storylines around rotations, pace, and adaptability headline this meeting."
    )


def build_entry(event_id: str, season_label_str: str, date_str: str, venue: str, neutral: bool,
                team_block: dict, opponent_block: dict, home_game: bool) -> dict:
    team_name = normalize_team(team_block["team"]["displayName"])
    opponent_name = normalize_team(opponent_block["team"]["displayName"])
    points_for = int(team_block.get("score") or 0)
    points_against = int(opponent_block.get("score") or 0)
    won = team_block.get("winner", False)
    plus_minus = points_for - points_against

    return {
        "game_id": event_id,
        "season": season_label_str,
        "date": date_str,
        "team_abbreviation": team_block["team"].get("abbreviation"),
        "team_name": team_name,
        "matchup": f"{team_block['team']['abbreviation']} {'vs.' if home_game else '@'} {opponent_block['team']['abbreviation']}",
        "won": bool(won),
        "points": points_for,
        "plus_minus": plus_minus,
        "home_game": home_game,
        "opponent_points": points_against,
        "opponent_name": opponent_name,
        "venue": venue,
        "neutral_site": neutral,
    }


def append_if_new(collection: List[dict], keyset: Set[Tuple[str, str, bool]], entry: dict, text_fields: bool = False):
    key = (entry["date"], entry["team_name"], entry["home_game"])
    if key in keyset:
        return False
    keyset.add(key)
    if text_fields:
        entry["narrative"] = build_narrative(entry)
        entry["rich_narrative"] = build_rich_narrative(entry)
    else:
        entry["narrative"] = build_narrative(entry)
    # remove helper fields not needed in final output
    entry.pop("opponent_name", None)
    entry.pop("opponent_points", None)
    entry.pop("neutral_site", None)
    entry.pop("venue", None)
    collection.append(entry)
    return True


def main() -> None:
    if not SCOREBOARD_FILE.exists():
        raise FileNotFoundError(f"Missing scoreboard data: {SCOREBOARD_FILE}")

    scoreboard_payload = json.loads(SCOREBOARD_FILE.read_text())
    existing_narratives = load_json(NARRATIVE_FILE, [])
    existing_meta = load_json(META_FILE, [])

    existing_narratives = [
        entry
        for entry in existing_narratives
        if entry.get("date", "") < SCOREBOARD_START_DATE
    ]
    existing_meta = [
        entry
        for entry in existing_meta
        if entry.get("date", "") < SCOREBOARD_START_DATE
    ]

    narrative_keys = {
        (item["date"], normalize_team(item["team_name"]), bool(item["home_game"]))
        for item in existing_narratives
    }
    meta_keys = {
        (item["date"], normalize_team(item["team_name"]), bool(item["home_game"]))
        for item in existing_meta
    }

    new_narratives = 0

    for season in scoreboard_payload.get("seasons", []):
        for event in season.get("games", []):
            competitions = event.get("competitions") or []
            if not competitions:
                continue
            comp = competitions[0]
            date_obj = datetime.fromisoformat((comp.get("date") or event.get("date")).replace("Z", "+00:00"))
            date_str = date_obj.strftime("%Y-%m-%d")
            event_id = event.get("id")
            venue = (comp.get("venue") or {}).get("fullName", "Unknown venue")
            neutral = bool(comp.get("neutralSite"))
            season_year = int(event.get("season", {}).get("year", 0) or 0)
            if not season_year:
                continue
            season_label_str = season_label(season_year)

            competitors = comp.get("competitors") or []
            if len(competitors) != 2:
                continue
            home_block = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_block = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home_block or not away_block:
                continue

            for block, opp, home_flag in [(home_block, away_block, True), (away_block, home_block, False)]:
                team_name = normalize_team(block["team"]["displayName"])
                opponent_name = normalize_team(opp["team"]["displayName"])
                if team_name not in VALID_TEAM_NAMES or opponent_name not in VALID_TEAM_NAMES:
                    continue
                entry = build_entry(
                    event_id=event_id,
                    season_label_str=season_label_str,
                    date_str=date_str,
                    venue=venue,
                    neutral=neutral,
                    team_block=block,
                    opponent_block=opp,
                    home_game=home_flag,
                )

                if append_if_new(existing_narratives, narrative_keys, entry.copy(), text_fields=True):
                    append_if_new(existing_meta, meta_keys, entry.copy(), text_fields=False)
                    new_narratives += 1

    existing_narratives.sort(key=lambda x: (x["date"], x["team_name"]))
    existing_meta.sort(key=lambda x: (x["date"], x["team_name"]))

    NARRATIVE_FILE.write_text(json.dumps(existing_narratives, indent=2))
    META_FILE.write_text(json.dumps(existing_meta, indent=2))

    # Update batch config sample size + timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    batch_data = load_json(BATCH_FILE, {})
    if batch_data:
        batch_data["last_updated"] = timestamp
        nba_cfg = batch_data.get("domains", {}).get("nba")
        if nba_cfg:
            nba_cfg["sample_size"] = len(existing_narratives)
            nba_cfg["data_path"] = "data/domains/nba_pregame_narratives.json"
            nba_cfg["status"] = "pending"
        BATCH_FILE.write_text(json.dumps(batch_data, indent=2))

    print(f"Appended {new_narratives} new NBA narratives.")
    print(f"Total narrative entries: {len(existing_narratives)}")
    print(f"Total meta entries: {len(existing_meta)}")


if __name__ == "__main__":
    main()

