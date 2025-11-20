#!/usr/bin/env python3
"""
Collect ESPN NBA scoreboards across multiple seasons for downstream modeling.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "nba_season"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_ROOT / "espn_scoreboard_complete.json"
SUMMARY_PATH = DATA_ROOT / "espn_scoreboard_summary.json"

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
REQUEST_LIMIT = 1000
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 6, 30)
REQUEST_SLEEP = 0.2


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_day(session: requests.Session, target_date: date) -> Optional[Dict[str, object]]:
    params = {"dates": target_date.strftime("%Y%m%d"), "limit": REQUEST_LIMIT}
    for attempt in range(3):
        try:
            resp = session.get(SCOREBOARD_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = REQUEST_SLEEP * (attempt + 1)
            print(f"[WARN] NBA fetch {target_date} failed ({exc}); retrying in {wait:.1f}s")
            time.sleep(wait)
    print(f"[ERROR] NBA fetch {target_date} exhausted retries")
    return None


def season_label(year: int, season_type: int) -> str:
    start_year = year - 1
    label = f"{start_year}-{str(year)[-2:]}"
    if season_type == 3:
        return f"{label} (post)"
    return label


def collect_scoreboards() -> Dict[str, List[Dict[str, object]]]:
    session = requests.Session()
    seasons: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    seen_ids = set()

    for current_date in daterange(START_DATE, END_DATE):
        payload = fetch_day(session, current_date)
        if not payload:
            continue

        events = payload.get("events", [])
        if not events:
            continue

        for event in events:
            event_id = event.get("id")
            if not event_id or event_id in seen_ids:
                continue
            seen_ids.add(event_id)

            season = event.get("season") or {}
            season_year = season.get("year")
            season_type = season.get("type", 2)
            if not season_year:
                continue

            label = season_label(int(season_year), int(season_type))
            seasons[label].append(event)

        if len(events) >= REQUEST_LIMIT:
            print(f"[WARN] NBA {current_date} returned {len(events)} events (limit={REQUEST_LIMIT})")

        time.sleep(REQUEST_SLEEP)

    if not seasons:
        raise RuntimeError("No NBA scoreboard data collected.")

    sorted_seasons = []
    for key in sorted(seasons.keys()):
        games = sorted(seasons[key], key=lambda e: e.get("date"))
        sorted_seasons.append({"season": key, "games": games})

    payload = {
        "collection_range": [START_DATE.isoformat(), END_DATE.isoformat()],
        "seasons": sorted_seasons,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "start_date": START_DATE.isoformat(),
                "end_date": END_DATE.isoformat(),
                "days": (END_DATE - START_DATE).days + 1,
                "unique_games": len(seen_ids),
                "season_counts": {season["season"]: len(season["games"]) for season in sorted_seasons},
                "output_path": str(OUTPUT_PATH),
            },
            indent=2,
        )
    )
    print(f"Collected {len(seen_ids)} NBA games across {len(sorted_seasons)} season buckets.")


def main() -> None:
    collect_scoreboards()


if __name__ == "__main__":
    main()

