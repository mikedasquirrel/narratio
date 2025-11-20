#!/usr/bin/env python3
"""
Background scraper to pull additional years of historical odds for major US sports.

This is a trimmed-down version of the maximum scraper that focuses on NBA, NFL,
MLB, NHL, NCAAB, and NCAAF so we can deepen the datasets without burning the
entire API budget. Run it in the background; it checkpoints every 50 days.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import requests

API_KEY = "2e330948334c9505ed5542a82fcfa3b9"
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS: List[Tuple[str, str, int]] = [
    ("basketball_nba", "NBA", 365 * 6),       # ~6 seasons
    ("icehockey_nhl", "NHL", 365 * 6),
    ("americanfootball_nfl", "NFL", 365 * 6),
    ("baseball_mlb", "MLB", 365 * 4),
    ("basketball_ncaab", "NCAAB", 365 * 3),
    ("americanfootball_ncaaf", "NCAAF", 365 * 3),
]

OUTPUT_DIR = Path("data/historical_odds_extra")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def scrape_sport(sport_key: str, sport_name: str, max_days: int) -> Dict[str, object]:
    all_games: List[dict] = []
    current_date = datetime.now()
    days_scraped = 0
    consecutive_empty = 0
    total_requests = 0
    output_file = OUTPUT_DIR / f"{sport_name.lower()}_extra.json"

    if output_file.exists():
        data = json.loads(output_file.read_text())
        existing_games = data.get("games", [])
        if existing_games:
            all_games = existing_games
            current_date = datetime.fromisoformat(data["end_date"]) - timedelta(days=1)
            print(f"[{sport_name}] Resuming from {current_date.date()} with {len(all_games)} games.")

    start_date = current_date

    while days_scraped < max_days and consecutive_empty < 30:
        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "date": current_date.strftime("%Y-%m-%dT12:00:00Z"),
        }
        url = f"{BASE_URL}/historical/sports/{sport_key}/odds"
        try:
            resp = requests.get(url, params=params, timeout=20)
            total_requests += 1
            if resp.status_code == 200:
                data = resp.json()
                games = data.get("data", [])
                if games:
                    all_games.extend(games)
                    consecutive_empty = 0
                    if days_scraped % 50 == 0:
                        remaining = resp.headers.get("x-requests-remaining", "N/A")
                        print(
                            f"[{sport_name}] {days_scraped:>4} days | {current_date.date()} | "
                            f"{len(games):>3} games (total {len(all_games):>6}) remaining={remaining}"
                        )
                else:
                    consecutive_empty += 1
            elif resp.status_code == 401:
                print(f"[{sport_name}] API limit reached; stopping early.")
                break
            else:
                consecutive_empty += 1
        except Exception as exc:
            consecutive_empty += 1
            print(f"[{sport_name}] Error on {current_date.date()}: {exc}")

        if days_scraped % 100 == 0 and days_scraped > 0:
            save_checkpoint(
                output_file,
                sport_name,
                sport_key,
                start_date,
                current_date,
                days_scraped,
                total_requests,
                all_games,
            )

        time.sleep(0.12)  # ~8 req/s to stay within limits
        current_date -= timedelta(days=1)
        days_scraped += 1

    save_checkpoint(
        output_file,
        sport_name,
        sport_key,
        start_date,
        current_date,
        days_scraped,
        total_requests,
        all_games,
    )
    print(
        f"[{sport_name}] COMPLETE — Days: {days_scraped}, Games: {len(all_games)}, "
        f"Requests: {total_requests}"
    )
    return {
        "sport": sport_name,
        "sport_key": sport_key,
        "days_scraped": days_scraped,
        "total_games": len(all_games),
        "total_requests": total_requests,
    }


def save_checkpoint(
    output_file: Path,
    sport_name: str,
    sport_key: str,
    start_date: datetime,
    current_date: datetime,
    days_scraped: int,
    total_requests: int,
    games: List[dict],
) -> None:
    payload = {
        "sport": sport_name,
        "sport_key": sport_key,
        "start_date": start_date.isoformat(),
        "end_date": current_date.isoformat(),
        "days_scraped": days_scraped,
        "total_games": len(games),
        "total_requests": total_requests,
        "games": games,
    }
    output_file.write_text(json.dumps(payload, indent=2))
    print(f"[{sport_name}] Checkpoint saved ({len(games)} games).")


def main() -> None:
    summary = []
    for sport_key, sport_name, max_days in SPORTS:
        print(f"\n{'='*80}\nCollecting {sport_name} ({max_days} days)...")
        stats = scrape_sport(sport_key, sport_name, max_days)
        summary.append(stats)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nAll major US sports scraping jobs queued/done.")


if __name__ == "__main__":
    main()

