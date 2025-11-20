#!/usr/bin/env python3
"""
Append today's high-confidence NHL predictions to the forward-testing log.

Usage:
    python3 scripts/log_nhl_forward_predictions.py [--min-prob 0.65]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILE = PROJECT_ROOT / "analysis" / "nhl_upcoming_predictions.json"
LOG_FILE = PROJECT_ROOT / "data" / "paper_trading" / "nhl_forward_log.jsonl"
SUMMARY_FILE = PROJECT_ROOT / "data" / "paper_trading" / "nhl_forward_summary.json"


def load_predictions() -> List[Dict]:
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDICTION_FILE}")
    return json.loads(PREDICTION_FILE.read_text())


def append_log(entries: Iterable[Dict]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def update_summary() -> Dict:
    stats: Dict[str, Dict[str, int]] = {}
    total = 0
    if LOG_FILE.exists():
        with LOG_FILE.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                model = record["model"]
                stats.setdefault(model, {"bets": 0})
                stats[model]["bets"] += 1
                total += 1
    payload = {"updated_at": datetime.utcnow().isoformat() + "Z", "total_bets": total, "by_model": stats}
    SUMMARY_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Log NHL forward predictions for live tracking.")
    parser.add_argument("--min-prob", type=float, default=0.65, help="Minimum model probability to log a bet.")
    args = parser.parse_args()

    predictions = load_predictions()
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_entries: List[Dict] = []

    for game in predictions:
        for rec in game.get("recommendations", []):
            if rec.get("prob", 0) < args.min_prob:
                continue
            log_entries.append(
                {
                    "logged_at": timestamp,
                    "game_id": game.get("game_id"),
                    "game_date": game.get("date"),
                    "matchup": game.get("matchup"),
                    "model": rec.get("model"),
                    "side": rec.get("side"),
                    "prob": rec.get("prob"),
                    "edge": rec.get("edge"),
                    "moneyline": rec.get("moneyline"),
                    "implied_prob": rec.get("implied_prob"),
                }
            )

    if not log_entries:
        print("No recommendations met the probability threshold.")
        return

    append_log(log_entries)
    summary = update_summary()

    print(f"Logged {len(log_entries)} bets to {LOG_FILE.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

