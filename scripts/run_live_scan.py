#!/usr/bin/env python3
"""
Unified Live Scan CLI
=====================

Usage:
    python scripts/run_live_scan.py --sport nhl --mode intermission
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.live_feed import LiveFeedService
from narrative_optimization.domains.nhl.live_betting_engine import LiveBettingEngine
from narrative_optimization.domains.nba.live_betting_engine import NBALiveBettingEngine
from narrative_optimization.domains.nfl.live_betting_engine import NFLLiveBettingEngine

ENGINE_MAP = {
    "nhl": LiveBettingEngine,
    "nba": NBALiveBettingEngine,
    "nfl": NFLLiveBettingEngine,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan live games for edges.")
    parser.add_argument("--sport", required=True, help="e.g., nhl, nba, nfl")
    parser.add_argument("--mode", default="intermission", help="informational flag (intermission, halftime, etc.)")
    parser.add_argument("--output", type=Path, default=Path("analysis/live_scan_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feed = LiveFeedService(bookmaker="fanduel")
    try:
        states = feed.fetch_game_states(args.sport)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch {args.sport} game states: {exc}")
        return

    if not states:
        print(f"No live {args.sport.upper()} games at the moment.")
        return

    engine_cls = ENGINE_MAP.get(args.sport, LiveBettingEngine)
    engine = engine_cls()

    recs = []
    for state in states:
        payload = {
            "home_team": state.home_team,
            "away_team": state.away_team,
            "current_period": state.period,
            "time_remaining": state.time_remaining,
            "home_score": state.home_score,
            "away_score": state.away_score,
            "period_scores": state.period_scores,
            "live_odds": state.live_odds,
            "props": state.props,
            "meta": state.meta,
        }
        rec = engine.analyze_live_game(payload)
        if rec:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat()
            recs.append(rec)

    if not recs:
        print("No recommendations at this scan.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(args.output.read_text())
    except Exception:
        existing = []
    existing.extend(recs)
    args.output.write_text(json.dumps(existing, indent=2))
    print(f"{len(recs)} recommendations written to {args.output}")


if __name__ == "__main__":
    main()
