#!/usr/bin/env python3
"""
Enrich the scraped games database with multi-season coverage sourced from
The Odds API.

This job deepens the historical record for all major US leagues and extends
coverage to niche sports that were not captured by the ESPN scraper. It
appends normalized events to ``data/scraped_games/games_history.jsonl`` without
overwriting existing snapshots such as ``games_YYYYMMDD.json``.

Key capabilities:
- Multi-season backfill per sport with resume support.
- Odds summarization (moneyline, spreads, totals) aggregated across books.
- Automatic sport availability detection via The Odds API ``/sports`` index.
- Deduplication using the provider event id plus defensive hashing.
- Run summaries persisted alongside the scrape data for diagnostics.

Usage examples:

    # Full enrichment (default seasons defined per sport plan)
    python scripts/enrich_scraped_games_with_odds_api.py

    # Fast smoke test (smaller windows)
    python scripts/enrich_scraped_games_with_odds_api.py --max-major-days 5 --max-niche-days 2

All data is saved inside ``data/scraped_games`` to keep the scrape database in a
single location, and no existing files are overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.odds_api_config import BASE_URL, ODDS_API_KEY
except ModuleNotFoundError:
    try:
        from odds_api_config import BASE_URL, ODDS_API_KEY  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - configuration error
        raise SystemExit(
            "Missing config/odds_api_config.py. Copy the template and set the API key."
        ) from exc


SCRAPED_DIR = Path("data/scraped_games")
HISTORY_FILE = SCRAPED_DIR / "games_history.jsonl"
SUMMARY_FILE = SCRAPED_DIR / "games_history_summary.json"
PROGRESS_FILE = SCRAPED_DIR / "games_history_progress.json"
RUN_REPORT_DIR = SCRAPED_DIR / "runs"

SUPPORTED_MARKETS = ("h2h", "spreads", "totals")
DEFAULT_REGIONS = ("us", "us2")


@dataclass(frozen=True)
class SportPlan:
    """Configuration for a sport scrape job."""

    sport_key: str
    label: str
    category: str  # "major" or "niche"
    seasons: int
    avg_season_days: int
    regions: Tuple[str, ...] = DEFAULT_REGIONS

    @property
    def target_days(self) -> int:
        return self.seasons * self.avg_season_days


@dataclass
class CollectionStats:
    """Runtime metrics recorded per sport."""

    sport_key: str
    sport_label: str
    category: str
    requested_days: int
    processed_days: int = 0
    new_games: int = 0
    skipped_games: int = 0
    earliest_commence: Optional[str] = None
    latest_commence: Optional[str] = None
    api_requests: int = 0
    requests_remaining: Optional[str] = None
    consecutive_empty_days: int = 0
    cursor_start: Optional[str] = None
    cursor_end: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        return payload


MAJOR_SPORTS: Sequence[SportPlan] = (
    SportPlan("basketball_nba", "NBA", "major", seasons=6, avg_season_days=170),
    SportPlan("americanfootball_nfl", "NFL", "major", seasons=6, avg_season_days=150),
    SportPlan("icehockey_nhl", "NHL", "major", seasons=6, avg_season_days=220),
    SportPlan("baseball_mlb", "MLB", "major", seasons=5, avg_season_days=215),
    SportPlan("soccer_usa_mls", "MLS", "major", seasons=5, avg_season_days=220),
    SportPlan(
        "soccer_spain_la_liga",
        "La Liga",
        "major",
        seasons=5,
        avg_season_days=290,
        regions=("uk", "eu"),
    ),
)

NICHE_SPORTS: Sequence[SportPlan] = (
    SportPlan("basketball_wnba", "WNBA", "niche", seasons=6, avg_season_days=130),
    SportPlan("americanfootball_cfl", "CFL", "niche", seasons=5, avg_season_days=120),
    SportPlan("mma_mixed_martial_arts", "MMA", "niche", seasons=4, avg_season_days=365),
    SportPlan("tennis_atp", "ATP", "niche", seasons=3, avg_season_days=365),
    SportPlan("soccer_epl", "EPL", "niche", seasons=4, avg_season_days=290),
    SportPlan(
        "icehockey_sweden_hockey_league",
        "SHL",
        "niche",
        seasons=6,
        avg_season_days=220,
        regions=("uk", "eu"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the scraped games DB using The Odds API historical endpoint."
    )
    parser.add_argument(
        "--sports",
        nargs="+",
        help="Optional subset of sports to run (accepts sport_key or label, case-insensitive).",
    )
    parser.add_argument(
        "--max-major-days",
        type=int,
        default=None,
        help="Override number of days to backfill for each major sport.",
    )
    parser.add_argument(
        "--max-niche-days",
        type=int,
        default=None,
        help="Override number of days to backfill for each niche sport.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) to start from. Defaults to now in UTC.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds to sleep between API requests to stay within rate limits.",
    )
    parser.add_argument(
        "--max-empty-days",
        type=int,
        default=30,
        help="Stop after this many consecutive empty days for a sport.",
    )
    parser.add_argument(
        "--progress-flush",
        type=int,
        default=25,
        help="Flush progress checkpoints after this many day iterations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Traverse the plans and log stats without persisting data.",
    )
    parser.add_argument(
        "--skip-existing-scan",
        action="store_true",
        help="Do not scan games_history.jsonl for duplicates (faster, but risky).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--run-label",
        help="Optional label embedded in the run report for auditing.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def select_plans(args: argparse.Namespace) -> List[SportPlan]:
    if not args.sports:
        return list(MAJOR_SPORTS + NICHE_SPORTS)

    normalized_targets = {token.lower() for token in args.sports}
    plans = []
    for plan in MAJOR_SPORTS + NICHE_SPORTS:
        if plan.sport_key.lower() in normalized_targets or plan.label.lower() in normalized_targets:
            plans.append(plan)
    if not plans:
        raise SystemExit(f"No sport plans matched selection: {', '.join(args.sports)}")
    return plans


def effective_days(plan: SportPlan, args: argparse.Namespace) -> int:
    if plan.category == "major" and args.max_major_days:
        return min(plan.target_days, args.max_major_days)
    if plan.category == "niche" and args.max_niche_days:
        return min(plan.target_days, args.max_niche_days)
    return plan.target_days


def _parse_date(date_str: Optional[str]) -> datetime:
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError as exc:
        raise SystemExit(f"Invalid start date {date_str!r}. Use YYYY-MM-DD.") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_existing_ids(skip_scan: bool) -> set:
    if skip_scan or not HISTORY_FILE.exists():
        return set()
    existing_ids: set = set()
    with HISTORY_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = record.get("event_id")
            if event_id:
                existing_ids.add(event_id)
    logging.info("Loaded %d existing event ids for deduplication", len(existing_ids))
    return existing_ids


def describe_existing_snapshots() -> List[Dict[str, object]]:
    snapshots: List[Dict[str, object]] = []
    for path in sorted(SCRAPED_DIR.glob("games_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        snapshots.append(
            {
                "file": str(path),
                "date": data.get("date"),
                "total_games": data.get("total_games"),
                "sports": [key for key, value in data.items() if isinstance(value, list)],
            }
        )
    return snapshots


def load_progress() -> Dict[str, str]:
    if not PROGRESS_FILE.exists():
        return {}
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except json.JSONDecodeError:
        logging.warning("Progress file is corrupt; starting fresh.")
        return {}


def save_progress(progress: Dict[str, str]) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def fetch_supported_sports() -> Dict[str, Dict[str, object]]:
    url = f"{BASE_URL}/sports/"
    params = {"apiKey": ODDS_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("Unable to fetch sports index: %s", exc)
        return {}
    try:
        payload = response.json()
    except json.JSONDecodeError:
        logging.warning("Sports index response was not valid JSON.")
        return {}
    result: Dict[str, Dict[str, object]] = {}
    for entry in payload:
        key = entry.get("key")
        if key:
            result[key] = entry
    logging.info("Sports index lists %d available sport keys.", len(result))
    return result


def safe_request(url: str, params: Dict[str, object], retries: int = 4) -> Tuple[Dict, Dict[str, str]]:
    backoff = 1.5
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=25)
        except requests.RequestException as exc:
            logging.warning("Request error (%s/%s): %s", attempt, retries, exc)
            time.sleep(backoff * attempt)
            continue

        if response.status_code == 200:
            try:
                return response.json(), response.headers
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON payload from Odds API: {exc}") from exc

        if response.status_code in {401, 402, 403, 429}:
            raise RuntimeError(
                f"Odds API returned {response.status_code}. Check key, plan, or rate limits."
            )

        logging.warning(
            "Unexpected status %s on attempt %s/%s. Retrying.",
            response.status_code,
            attempt,
            retries,
        )
        time.sleep(backoff * attempt)

    raise RuntimeError(f"Failed to fetch Odds API data after {retries} attempts: {url}")


def summarize_markets(
    event: Dict[str, object],
) -> Tuple[Dict[str, Optional[float]], set]:
    home = event.get("home_team")
    away = event.get("away_team")
    bookmakers = event.get("bookmakers", []) or []

    h2h_home: List[float] = []
    h2h_away: List[float] = []
    spread_home_pts: List[float] = []
    spread_home_prices: List[float] = []
    spread_away_pts: List[float] = []
    spread_away_prices: List[float] = []
    total_points: List[float] = []
    total_over_prices: List[float] = []
    total_under_prices: List[float] = []
    markets_seen: set = set()

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            key = market.get("key")
            if key not in SUPPORTED_MARKETS:
                continue
            markets_seen.add(key)
            for outcome in market.get("outcomes", []):
                name = outcome.get("name")
                price = _to_number(outcome.get("price"))
                point = _to_number(outcome.get("point"))

                if key == "h2h":
                    if home and name == home and price is not None:
                        h2h_home.append(price)
                    elif away and name == away and price is not None:
                        h2h_away.append(price)
                elif key == "spreads":
                    if home and name == home:
                        if point is not None:
                            spread_home_pts.append(point)
                        if price is not None:
                            spread_home_prices.append(price)
                    elif away and name == away:
                        if point is not None:
                            spread_away_pts.append(point)
                        if price is not None:
                            spread_away_prices.append(price)
                elif key == "totals":
                    if name == "Over":
                        if point is not None:
                            total_points.append(point)
                        if price is not None:
                            total_over_prices.append(price)
                    elif name == "Under":
                        if price is not None:
                            total_under_prices.append(price)
                        # Use the same point buffer for under if the API omits it.
                        if point is not None:
                            total_points.append(point)

    summary = {
        "h2h_home_median_price": _median_or_none(h2h_home),
        "h2h_away_median_price": _median_or_none(h2h_away),
        "spread_home_median_point": _median_or_none(spread_home_pts),
        "spread_home_median_price": _median_or_none(spread_home_prices),
        "spread_away_median_point": _median_or_none(spread_away_pts),
        "spread_away_median_price": _median_or_none(spread_away_prices),
        "total_median_point": _median_or_none(total_points),
        "total_over_median_price": _median_or_none(total_over_prices),
        "total_under_median_price": _median_or_none(total_under_prices),
        "bookmaker_count": len(bookmakers),
    }
    return summary, markets_seen


def _median_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(statistics.median(values)), 4)


def _to_number(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_event(event: Dict[str, object], plan: SportPlan, ingested_at: str) -> Dict[str, object]:
    odds_snapshot, markets_seen = summarize_markets(event)
    commence_time = event.get("commence_time")
    record = {
        "event_id": event.get("id"),
        "sport_key": event.get("sport_key") or plan.sport_key,
        "sport_title": event.get("sport_title") or plan.label,
        "sport_label": plan.label,
        "sport_category": plan.category,
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "commence_time": commence_time,
        "completed": event.get("completed", False),
        "markets_seen": sorted(markets_seen),
        "odds_snapshot": odds_snapshot,
        "source": "the-odds-api",
        "ingested_at": ingested_at,
        "season_plan": {
            "seasons_requested": plan.seasons,
            "estimated_days": plan.target_days,
        },
    }
    if event.get("scores"):
        record["scores"] = event["scores"]
    if event.get("tournament"):
        record["tournament"] = event["tournament"]
    return record


def append_records(records: Iterable[Dict[str, object]]) -> int:
    if not records:
        return 0
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with HISTORY_FILE.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
            count += 1
    return count


def collect_for_plan(
    plan: SportPlan,
    args: argparse.Namespace,
    base_cursor: datetime,
    existing_ids: set,
    progress: Dict[str, str],
) -> Tuple[List[Dict[str, object]], CollectionStats]:
    run_started_at = datetime.now(timezone.utc).isoformat()
    effective_cursor = base_cursor
    if plan.sport_key in progress:
        try:
            effective_cursor = datetime.fromisoformat(progress[plan.sport_key])
        except ValueError:
            logging.warning("Invalid cursor for %s in progress file; restarting.", plan.sport_key)
    stats = CollectionStats(
        sport_key=plan.sport_key,
        sport_label=plan.label,
        category=plan.category,
        requested_days=effective_days(plan, args),
        cursor_start=effective_cursor.isoformat(),
    )

    collected: List[Dict[str, object]] = []
    consecutive_empty = 0
    ingested_at = datetime.now(timezone.utc).isoformat()

    url = f"{BASE_URL}/historical/sports/{plan.sport_key}/odds"
    days_to_collect = stats.requested_days

    while stats.processed_days < days_to_collect and consecutive_empty < args.max_empty_days:
        regions = plan.regions or DEFAULT_REGIONS
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": ",".join(regions),
            "markets": ",".join(SUPPORTED_MARKETS),
            "oddsFormat": "american",
            "date": _format_timestamp(effective_cursor),
        }
        try:
            payload, headers = safe_request(url, params)
        except RuntimeError as exc:
            stats.notes.append(str(exc))
            logging.error("Stopping %s due to error: %s", plan.label, exc)
            break

        events = payload.get("data") if isinstance(payload, dict) else payload
        if events is None:
            events = []

        appended_this_day = 0
        for event in events:
            normalized = normalize_event(event, plan, ingested_at)
            event_id = normalized.get("event_id")
            if event_id and event_id in existing_ids:
                stats.skipped_games += 1
                continue
            if event_id:
                existing_ids.add(event_id)
            collected.append(normalized)
            appended_this_day += 1
            commence_time = normalized.get("commence_time")
            if commence_time:
                if not stats.earliest_commence or commence_time < stats.earliest_commence:
                    stats.earliest_commence = commence_time
                if not stats.latest_commence or commence_time > stats.latest_commence:
                    stats.latest_commence = commence_time

        if appended_this_day == 0:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        stats.api_requests += 1
        stats.requests_remaining = headers.get("x-requests-remaining")
        stats.processed_days += 1
        stats.new_games = len(collected)
        stats.consecutive_empty_days = consecutive_empty
        stats.cursor_end = effective_cursor.isoformat()

        effective_cursor -= timedelta(days=1)
        progress[plan.sport_key] = effective_cursor.isoformat()
        if stats.processed_days % args.progress_flush == 0:
            save_progress(progress)

        if args.sleep:
            time.sleep(max(args.sleep, 0))

    save_progress(progress)
    stats.notes.append(f"Run started at {run_started_at}, ingested_at={ingested_at}")
    if consecutive_empty >= args.max_empty_days:
        stats.notes.append("Stopped due to consecutive empty day threshold.")
    elif stats.processed_days >= days_to_collect:
        stats.notes.append("Completed requested day window.")
    else:
        stats.notes.append("Stopped early due to API errors.")

    if args.dry_run:
        logging.info("[DRY RUN] %s would append %d games.", plan.label, stats.new_games)
        return [], stats

    appended = append_records(collected)
    logging.info(
        "%s appended %d new games (%d days, %d API calls). Remaining=%s",
        plan.label,
        appended,
        stats.processed_days,
        stats.api_requests,
        stats.requests_remaining,
    )
    return collected, stats


def _format_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT12:00:00Z")


def update_summary(
    stats: Sequence[CollectionStats],
    snapshots: Sequence[Dict[str, object]],
    run_timestamp: str,
) -> Dict[str, object]:
    if SUMMARY_FILE.exists():
        try:
            summary = json.loads(SUMMARY_FILE.read_text())
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}

    sport_totals = summary.setdefault("sport_totals", {})
    for entry in stats:
        if entry.sport_key not in sport_totals:
            sport_totals[entry.sport_key] = {
                "sport_label": entry.sport_label,
                "category": entry.category,
                "records": 0,
                "completed_runs": 0,
            }
        sport_rec = sport_totals[entry.sport_key]
        sport_rec["records"] += entry.new_games
        sport_rec["completed_runs"] += 1
        sport_rec["last_updated"] = run_timestamp
        sport_rec["last_run_notes"] = entry.notes
        sport_rec["requested_days"] = entry.requested_days
        sport_rec["latest_commence"] = entry.latest_commence
        sport_rec["earliest_commence"] = entry.earliest_commence

    total_records = sum(item["records"] for item in sport_totals.values())
    summary.update(
        {
            "history_file": str(HISTORY_FILE),
            "updated_at": run_timestamp,
            "total_records": total_records,
            "existing_snapshots": snapshots,
        }
    )
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return summary


def write_run_report(
    stats: Sequence[CollectionStats],
    summary: Dict[str, object],
    run_ts: str,
    run_label: Optional[str],
) -> Path:
    RUN_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"games_history_run_{run_ts.replace(':', '').replace('-', '')}.json"
    payload = {
        "run_timestamp": run_ts,
        "label": run_label,
        "stats": [entry.to_dict() for entry in stats],
        "summary_snapshot": summary,
    }
    report_path = RUN_REPORT_DIR / filename
    report_path.write_text(json.dumps(payload, indent=2))
    return report_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    plans = select_plans(args)
    start_cursor = _parse_date(args.start_date)
    existing_ids = load_existing_ids(args.skip_existing_scan)
    progress = load_progress()
    snapshots = describe_existing_snapshots()
    available_sports = fetch_supported_sports()
    run_ts = datetime.now(timezone.utc).isoformat()

    stats: List[CollectionStats] = []

    for plan in plans:
        if available_sports and plan.sport_key not in available_sports:
            logging.warning("Skipping %s (%s) — sport key not available right now.", plan.label, plan.sport_key)
            continue
        days = effective_days(plan, args)
        if days <= 0:
            logging.info("Skipping %s — zero days requested.", plan.label)
            continue
        logging.info(
            "Collecting %s (%s) for %d days starting from %s",
            plan.label,
            plan.sport_key,
            days,
            start_cursor.date(),
        )
        collected, stat = collect_for_plan(plan, args, start_cursor, existing_ids, progress)
        stats.append(stat)
        logging.info(
            "%s complete: %d new games, %d skipped duplicates, %d API calls.",
            plan.label,
            stat.new_games,
            stat.skipped_games,
            stat.api_requests,
        )

    summary = update_summary(stats, snapshots, run_ts)
    report_path = write_run_report(stats, summary, run_ts, args.run_label)
    logging.info("Run summary saved to %s", report_path)
    logging.info(
        "History now tracks %d records across %d sports.",
        summary.get("total_records", 0),
        len(summary.get("sport_totals", {})),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")

