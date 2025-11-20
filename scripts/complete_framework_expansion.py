#!/usr/bin/env python3
"""
Complete Framework Expansion Runner
===================================

Wraps all Phase 1 deliverables (context discovery + nominative enrichment) into
a single pipeline that can be applied to any domain list via one command:

    python3 scripts/complete_framework_expansion.py --domains nhl nfl supreme_court
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "narrative_optimization" / "results"
CONTEXT_DIR = RESULTS_DIR / "context_stratification"
NOMINATIVE_DIR = RESULTS_DIR / "nominative_enrichment"


def _run_script(script_path: Path, extra_args: List[str]) -> None:
    env = os.environ.copy()
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{current_path}" if current_path else str(PROJECT_ROOT)
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n▶ Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, check=True)


def run_context_discovery(domains: List[str]) -> None:
    script = PROJECT_ROOT / "scripts" / "discover_context_patterns.py"
    args = ["--domains", *domains]
    _run_script(script, args)


def run_nominative_enrichment(domains: List[str]) -> None:
    script = PROJECT_ROOT / "scripts" / "enrich_nominative_features.py"
    args = ["--domains", *domains]
    _run_script(script, args)


@dataclass
class ContextSummary:
    domain: str
    pattern_count: int
    best_edge: float
    baseline: float


@dataclass
class NominativeSummary:
    domain: str
    metric: str
    baseline: float
    enriched: float
    lift: float


def load_context_summary(domain: str) -> ContextSummary:
    path = CONTEXT_DIR / f"{domain}_contexts.json"
    if not path.exists():
        return ContextSummary(domain, 0, 0.0, 0.0)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patterns = data.get("patterns", [])
    best_edge = max((p.get("edge_vs_baseline") or 0.0) for p in patterns) if patterns else 0.0
    return ContextSummary(
        domain=domain,
        pattern_count=len(patterns),
        best_edge=float(best_edge),
        baseline=float(data.get("baseline_win_rate") or 0.0),
    )


def load_nominative_summary(domain: str) -> NominativeSummary:
    path = NOMINATIVE_DIR / f"{domain}_nominative_enrichment.json"
    if not path.exists():
        return NominativeSummary(domain, "n/a", 0.0, 0.0, 0.0)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return NominativeSummary(
        domain=domain,
        metric=data.get("metric", "accuracy"),
        baseline=float(data.get("baseline", 0.0)),
        enriched=float(data.get("enriched", 0.0)),
        lift=float(data.get("lift", 0.0)),
    )


def print_unified_report(domains: List[str]) -> None:
    print("\n=== Unified Expansion Summary ===")
    print(f"{'Domain':<15} {'Contexts':<10} {'Best Edge':<12} {'Baseline':<10} {'Nom Metric':<12} {'Baseline':<12} {'Enriched':<12} {'Lift':<10}")
    print("-" * 95)
    for domain in domains:
        ctx = load_context_summary(domain)
        nom = load_nominative_summary(domain)
        print(
            f"{domain:<15} "
            f"{ctx.pattern_count:<10} "
            f"{ctx.best_edge * 100:>6.1f}%     "
            f"{ctx.baseline * 100:>6.1f}%   "
            f"{nom.metric:<12} "
            f"{nom.baseline * 100:>6.1f}%   "
            f"{nom.enriched * 100:>6.1f}%   "
            f"{nom.lift * 100:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Run complete framework expansion pipeline.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["nhl", "nfl", "supreme_court"],
        help="Domains to process.",
    )
    parser.add_argument(
        "--skip-contexts",
        action="store_true",
        help="Skip context stratification stage.",
    )
    parser.add_argument(
        "--skip-nominative",
        action="store_true",
        help="Skip nominative enrichment stage.",
    )
    args = parser.parse_args()

    if not args.skip_contexts:
        run_context_discovery(args.domains)
    if not args.skip_nominative:
        run_nominative_enrichment(args.domains)

    print_unified_report(args.domains)


if __name__ == "__main__":
    main()


