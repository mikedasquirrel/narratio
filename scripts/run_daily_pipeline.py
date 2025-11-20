#!/usr/bin/env python3
"""
Run the end-to-end NHL investor refresh pipeline with logging and manifest output.

Usage:
    python3 scripts/run_daily_pipeline.py [--skip-models]

Outputs:
    logs/daily_runs/YYYYMMDD_HHMMSS/run_manifest.json
    logs/daily_runs/.../*.log (per step stdout/stderr)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs" / "daily_runs"


def build_steps(skip_models: bool) -> List[Sequence[str]]:
    steps: List[Sequence[str]] = [
        ["python3", "scripts/process_historical_odds.py", "--sports", "icehockey_nhl"],
        ["python3", "scripts/build_nhl_modeling_dataset.py"],
        ["python3", "scripts/merge_nhl_features_with_odds.py"],
        [
            "python3",
            "narrative_optimization/domains/nhl/build_narrative_betting_dataset.py",
            "--use-absolute-max-odds",
        ],
    ]

    if not skip_models:
        steps.extend(
            [
                ["python3", "narrative_optimization/domains/nhl/train_focused_temporal.py"],
                ["python3", "narrative_optimization/domains/nhl/train_temporal_models.py"],
                ["python3", "narrative_optimization/domains/nhl/train_narrative_models.py"],
                ["python3", "narrative_optimization/domains/nhl/nhl_complete_analysis.py"],
            ]
        )

    steps.extend(
        [
            ["python3", "scripts/generate_investor_dashboard.py"],
            ["python3", "scripts/update_investor_doc.py"],
        ]
    )

    return steps


def run_step(cmd: Sequence[str], log_dir: Path) -> dict:
    name = Path(cmd[1]).stem if len(cmd) > 1 else "_".join(cmd)
    safe_name = name.replace(".", "_")
    log_file = log_dir / f"{safe_name}.log"
    start = datetime.utcnow()
    with log_file.open("w") as handle:
        handle.write(f"$ {' '.join(cmd)}\n\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end = datetime.utcnow()
    return {
        "command": cmd,
        "log_file": str(log_file.relative_to(PROJECT_ROOT)),
        "return_code": process.returncode,
        "start": start.isoformat() + "Z",
        "end": end.isoformat() + "Z",
        "duration_seconds": (end - start).total_seconds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily investor refresh pipeline.")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model retraining steps for faster data-only refreshes.",
    )
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    steps = build_steps(skip_models=args.skip_models)
    manifest = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "steps": [],
    }

    for cmd in steps:
        result = run_step(cmd, log_dir)
        manifest["steps"].append(result)
        if result["return_code"] != 0:
            manifest["failed_step"] = result
            break

    manifest["ended_at"] = datetime.utcnow().isoformat() + "Z"
    manifest_path = log_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if manifest.get("failed_step"):
        print("❌ Pipeline halted. See manifest for details.")
        sys.exit(1)

    print(f"✅ Pipeline complete. Logs written to {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

