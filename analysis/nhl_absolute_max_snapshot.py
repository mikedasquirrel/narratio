#!/usr/bin/env python3
"""
NHL Absolute Maximum Snapshot Utility
=====================================

Creates an immutable snapshot of the Absolute Maximum NHL scrape while the
primary scraper continues to run. The script:

1. Waits for the live JSON file to remain stable for a configurable window.
2. Copies the raw JSON into a timestamped directory under analysis/snapshots/.
3. Streams the games payload into a columnar (Parquet) mirror with aggregated
   bookmaker / market statistics for downstream modeling.
4. Emits metadata (counts, bookmaker coverage, date distribution) for quick QA.
5. Updates the toplevel analysis/nhl_absolute_max_snapshot.{json,parquet,meta}
   pointers so consumers always have a "latest" artifact.

Usage (from repo root):
    python analysis/nhl_absolute_max_snapshot.py

Advanced options (see --help):
    --max-games        limit processed games for dry runs
    --batch-size       Parquet batch size (default: 5,000 rows)
    --skip-stability   bypass file stability checks (not recommended)
    --snapshot-root    custom destination directory

The script is read-only with respect to the live scrape outputs and performs all
heavy processing on the copied snapshot to avoid interfering with the ongoing
collection pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import ijson
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'ijson'. Install project requirements first."
    ) from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'pyarrow'. Install project requirements first."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "historical_odds_complete" / "icehockey_nhl_complete.json"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_SNAPSHOT_ROOT = ANALYSIS_DIR / "snapshots" / "nhl_absolute_max"
LATEST_JSON = ANALYSIS_DIR / "nhl_absolute_max_snapshot.json"
LATEST_PARQUET = ANALYSIS_DIR / "nhl_absolute_max_snapshot.parquet"
LATEST_META = ANALYSIS_DIR / "nhl_absolute_max_snapshot.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stabilized snapshot of the Absolute Maximum NHL scrape.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the live Absolute Maximum NHL JSON feed.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=DEFAULT_SNAPSHOT_ROOT,
        help="Directory where timestamped snapshots will be stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of normalized games per Parquet batch.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap for processed games (useful for dry runs).",
    )
    parser.add_argument(
        "--stability-interval",
        type=float,
        default=3.0,
        help="Seconds to wait between file stability checks.",
    )
    parser.add_argument(
        "--stability-checks",
        type=int,
        default=2,
        help="How many consecutive identical file samples are required.",
    )
    parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Bypass stability checks (only if the scraper is paused).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch diagnostics.",
    )
    return parser.parse_args()


def wait_for_file_stability(path: Path, interval: float, checks: int) -> None:
    """
    Ensure the file is not actively being rewritten by sampling size + mtime.
    """
    if checks <= 1:
        return

    last_sample: Optional[Tuple[int, int]] = None
    stable_readings = 0

    while stable_readings < checks:
        stat = path.stat()
        sample = (stat.st_size, stat.st_mtime_ns)
        if sample == last_sample:
            stable_readings += 1
        else:
            stable_readings = 1
            last_sample = sample
        if stable_readings < checks:
            time.sleep(interval)


def atomic_copy(src: Path, dest: Path) -> Path:
    """
    Copy a file to destination atomically (write temp + rename).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp_dest)
    tmp_dest.replace(dest)
    return dest


def summarize_prices(values: List[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "avg": None,
            "median": None,
            "best": None,
            "worst": None,
        }
    return {
        "avg": float(sum(clean) / len(clean)),
        "median": float(statistics.median(clean)),
        "best": float(max(clean)),
        "worst": float(min(clean)),
    }


def summarize_points(values: List[float]) -> Dict[str, Optional[float]]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {
            "avg": None,
            "median": None,
        }
    return {
        "avg": float(sum(clean) / len(clean)),
        "median": float(statistics.median(clean)),
    }


def flatten_game(game: Dict) -> Tuple[Dict, Dict]:
    """
    Transform a raw Odds-API game payload into a modeling-friendly record.
    Returns (row_dict, telemetry_dict).
    """
    event_id = game.get("id") or game.get("event_id")
    commence_time = game.get("commence_time")
    home_team = game.get("home_team")
    away_team = game.get("away_team")
    sport_key = game.get("sport_key")
    sport_title = game.get("sport_title")
    completed = game.get("completed")

    bookmakers = game.get("bookmakers", []) or []
    bookmaker_keys = []
    market_keys = set()
    last_updates: List[str] = []

    moneyline_home = []
    moneyline_away = []

    spread_home_points = []
    spread_home_prices = []
    spread_away_points = []
    spread_away_prices = []

    total_points = []
    total_over_prices = []
    total_under_prices = []

    for bookmaker in bookmakers:
        bkey = bookmaker.get("key")
        if bkey:
            bookmaker_keys.append(bkey)
        for market in bookmaker.get("markets", []) or []:
            mkey = market.get("key")
            if not mkey:
                continue
            market_keys.add(mkey)
            lu = market.get("last_update")
            if lu:
                last_updates.append(lu)
            for outcome in market.get("outcomes", []) or []:
                name = outcome.get("name")
                price = outcome.get("price")
                point = outcome.get("point")

                if mkey == "h2h":
                    if home_team and name == home_team:
                        moneyline_home.append(price)
                    elif away_team and name == away_team:
                        moneyline_away.append(price)
                elif mkey == "spreads":
                    if home_team and name == home_team:
                        spread_home_points.append(point)
                        spread_home_prices.append(price)
                    elif away_team and name == away_team:
                        spread_away_points.append(point)
                        spread_away_prices.append(price)
                elif mkey == "totals":
                    if name and name.lower().startswith("over"):
                        total_points.append(point)
                        total_over_prices.append(price)
                    elif name and name.lower().startswith("under"):
                        # Under shares the same point as over; append only price.
                        total_under_prices.append(price)

    moneyline_summary_home = summarize_prices(moneyline_home)
    moneyline_summary_away = summarize_prices(moneyline_away)
    spread_points_home = summarize_points(spread_home_points)
    spread_points_away = summarize_points(spread_away_points)
    spread_prices_home = summarize_prices(spread_home_prices)
    spread_prices_away = summarize_prices(spread_away_prices)
    total_points_summary = summarize_points(total_points)
    total_over_summary = summarize_prices(total_over_prices)
    total_under_summary = summarize_prices(total_under_prices)

    row = {
        "event_id": event_id,
        "commence_time": commence_time,
        "sport_key": sport_key,
        "sport_title": sport_title,
        "home_team": home_team,
        "away_team": away_team,
        "is_completed": completed,
        "bookmaker_count": len(bookmakers),
        "bookmaker_keys": sorted(set(bookmaker_keys)),
        "market_keys": sorted(market_keys),
        "moneyline_home_avg": moneyline_summary_home["avg"],
        "moneyline_home_median": moneyline_summary_home["median"],
        "moneyline_home_best": moneyline_summary_home["best"],
        "moneyline_home_worst": moneyline_summary_home["worst"],
        "moneyline_away_avg": moneyline_summary_away["avg"],
        "moneyline_away_median": moneyline_summary_away["median"],
        "moneyline_away_best": moneyline_summary_away["best"],
        "moneyline_away_worst": moneyline_summary_away["worst"],
        "spread_home_points_avg": spread_points_home["avg"],
        "spread_home_points_median": spread_points_home["median"],
        "spread_home_price_avg": spread_prices_home["avg"],
        "spread_home_price_best": spread_prices_home["best"],
        "spread_home_price_worst": spread_prices_home["worst"],
        "spread_away_points_avg": spread_points_away["avg"],
        "spread_away_points_median": spread_points_away["median"],
        "spread_away_price_avg": spread_prices_away["avg"],
        "spread_away_price_best": spread_prices_away["best"],
        "spread_away_price_worst": spread_prices_away["worst"],
        "total_points_avg": total_points_summary["avg"],
        "total_points_median": total_points_summary["median"],
        "total_over_price_avg": total_over_summary["avg"],
        "total_over_price_best": total_over_summary["best"],
        "total_over_price_worst": total_over_summary["worst"],
        "total_under_price_avg": total_under_summary["avg"],
        "total_under_price_best": total_under_summary["best"],
        "total_under_price_worst": total_under_summary["worst"],
        "last_market_update": max(last_updates) if last_updates else None,
    }

    telemetry = {
        "bookmakers": set(bookmaker_keys),
        "markets": set(market_keys),
    }

    return row, telemetry


def parse_commence_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def stream_games_to_parquet(
    snapshot_file: Path,
    parquet_path: Path,
    batch_size: int,
    max_games: Optional[int],
    verbose: bool = False,
) -> Dict:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    batch: List[Dict] = []
    stats = {
        "games_processed": 0,
        "bookmaker_counter": Counter(),
        "market_counter": Counter(),
        "date_counter": Counter(),
        "earliest_commence": None,
        "latest_commence": None,
        "sport_keys": Counter(),
    }

    def flush_batch(current_batch: List[Dict], parquet_writer: Optional[pq.ParquetWriter]) -> pq.ParquetWriter:
        if not current_batch:
            return parquet_writer
        table = pa.Table.from_pylist(current_batch)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
        parquet_writer.write_table(table)
        current_batch.clear()
        return parquet_writer

    start = time.time()
    with snapshot_file.open("rb") as handle:
        game_iter = ijson.items(handle, "games.item")
        for idx, game in enumerate(game_iter, start=1):
            row, telemetry = flatten_game(game)
            batch.append(row)

            commence_dt = parse_commence_datetime(row.get("commence_time"))
            if commence_dt:
                stats["earliest_commence"] = (
                    commence_dt if not stats["earliest_commence"] else min(stats["earliest_commence"], commence_dt)
                )
                stats["latest_commence"] = (
                    commence_dt if not stats["latest_commence"] else max(stats["latest_commence"], commence_dt)
                )
                stats["date_counter"][commence_dt.date().isoformat()] += 1

            for bookmaker in telemetry["bookmakers"]:
                stats["bookmaker_counter"][bookmaker] += 1
            for market in telemetry["markets"]:
                stats["market_counter"][market] += 1

            sport_key = row.get("sport_key")
            if sport_key:
                stats["sport_keys"][sport_key] += 1

            stats["games_processed"] += 1

            if len(batch) >= batch_size:
                writer = flush_batch(batch, writer)
                if verbose:
                    elapsed = time.time() - start
                    rate = stats["games_processed"] / elapsed if elapsed else 0
                    print(
                        f"[Parquet] {stats['games_processed']:,} games | "
                        f"{rate:,.0f} games/s | file={parquet_path.name}"
                    )

            if max_games and stats["games_processed"] >= max_games:
                break

    writer = flush_batch(batch, writer)
    if writer:
        writer.close()

    return stats


def build_metadata(
    source: Path,
    snapshot_path: Path,
    parquet_path: Path,
    stats: Dict,
    duration_seconds: float,
    max_games: Optional[int],
    batch_size: int,
) -> Dict:
    source_stat = source.stat()
    snapshot_stat = snapshot_path.stat()
    parquet_size = parquet_path.stat().st_size if parquet_path.exists() else 0

    metadata = {
        "snapshot_created": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_modified": datetime.fromtimestamp(source_stat.st_mtime, tz=timezone.utc).isoformat(),
        "source_size_bytes": source_stat.st_size,
        "snapshot_path": str(snapshot_path),
        "snapshot_size_bytes": snapshot_stat.st_size,
        "parquet_path": str(parquet_path),
        "parquet_size_bytes": parquet_size,
        "games_processed": stats["games_processed"],
        "earliest_commence_time": stats["earliest_commence"].isoformat() if stats["earliest_commence"] else None,
        "latest_commence_time": stats["latest_commence"].isoformat() if stats["latest_commence"] else None,
        "unique_bookmakers": sorted(stats["bookmaker_counter"]),
        "bookmaker_frequency": [
            {"bookmaker": name, "games": count}
            for name, count in stats["bookmaker_counter"].most_common()
        ],
        "market_frequency": [
            {"market": name, "games": count}
            for name, count in stats["market_counter"].most_common()
        ],
        "games_by_date": dict(stats["date_counter"]),
        "sport_key_breakdown": dict(stats["sport_keys"]),
        "duration_seconds": duration_seconds,
        "max_games_limit": max_games,
        "batch_size": batch_size,
    }
    return metadata


def write_metadata(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(meta, handle, indent=2)


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve()

    if not source_path.exists():
        raise SystemExit(f"Source file not found: {source_path}")

    if not args.skip_stability:
        print(f"Waiting for {source_path} to stabilize ({args.stability_checks} checks, {args.stability_interval}s interval)...")
        wait_for_file_stability(source_path, args.stability_interval, args.stability_checks)

    run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    snapshot_root = args.snapshot_root.expanduser().resolve()
    snapshot_dir = snapshot_root / run_timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_json = snapshot_dir / f"nhl_absolute_max_{run_timestamp}.json"
    parquet_path = snapshot_dir / f"nhl_absolute_max_{run_timestamp}.parquet"
    meta_path = snapshot_dir / f"nhl_absolute_max_{run_timestamp}.meta.json"

    print(f"Copying source JSON to {snapshot_json} ...")
    atomic_copy(source_path, snapshot_json)

    print(f"Streaming games into Parquet {parquet_path} ...")
    start = time.time()
    stats = stream_games_to_parquet(
        snapshot_json,
        parquet_path,
        batch_size=args.batch_size,
        max_games=args.max_games,
        verbose=args.verbose,
    )
    duration = time.time() - start
    print(f"Processed {stats['games_processed']:,} games in {duration:.1f}s")

    metadata = build_metadata(
        source=source_path,
        snapshot_path=snapshot_json,
        parquet_path=parquet_path,
        stats=stats,
        duration_seconds=duration,
        max_games=args.max_games,
        batch_size=args.batch_size,
    )
    write_metadata(metadata, meta_path)

    print("Updating latest snapshot pointers ...")
    atomic_copy(snapshot_json, LATEST_JSON)
    atomic_copy(parquet_path, LATEST_PARQUET)
    write_metadata(metadata, LATEST_META)

    print("\nSnapshot complete ✅")
    print(json.dumps(
        {
            "games_processed": metadata["games_processed"],
            "snapshot_path": metadata["snapshot_path"],
            "parquet_path": metadata["parquet_path"],
            "earliest": metadata["earliest_commence_time"],
            "latest": metadata["latest_commence_time"],
            "duration_seconds": metadata["duration_seconds"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nSnapshot interrupted by user")

