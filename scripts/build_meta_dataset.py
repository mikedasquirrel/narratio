#!/usr/bin/env python3
"""
Build Meta Dataset
==================

Aggregates per-domain context + nominative diagnostics into a unified dataset
for the universal predictor.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from narrative_optimization.domain_registry import get_domain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = PROJECT_ROOT / "narrative_optimization" / "results" / "context_stratification"
NOMINATIVE_DIR = PROJECT_ROOT / "narrative_optimization" / "results" / "nominative_enrichment"
OUTPUT_PATH = PROJECT_ROOT / "narrative_optimization" / "results" / "meta_framework" / "meta_dataset.json"


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_context_metrics(data: Dict) -> Dict[str, float]:
    patterns = data.get("patterns") or []
    if not patterns:
        return {"theta": 0.0, "context_strength": 0.0, "effect_size": 0.0}
    best = max(patterns, key=lambda p: p.get("edge_vs_baseline", 0.0))
    return {
        "theta": float(best.get("accuracy") or 0.0),
        "context_strength": float(best.get("edge_vs_baseline") or 0.0),
        "effect_size": float(best.get("edge_vs_baseline") or 0.0),
    }


def _extract_nominative_metrics(data: Dict) -> Dict[str, float]:
    if not data:
        return {"lambda": 0.0, "nominative_richness": 0.0}
    return {
        "lambda": float(data.get("enriched") or 0.0),
        "nominative_richness": float(data.get("lift") or 0.0),
    }


def build_meta_dataset(domains: Optional[List[str]] = None, output_path: Path = OUTPUT_PATH) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if domains is None:
        domains = sorted(
            {path.stem.replace("_contexts", "") for path in CONTEXT_DIR.glob("*_contexts.json")}
        )

    entries = []
    for domain in domains:
        context_data = _load_json(CONTEXT_DIR / f"{domain}_contexts.json") or {}
        nominative_data = _load_json(NOMINATIVE_DIR / f"{domain}_nominative_enrichment.json") or {}
        pi = (get_domain(domain).estimated_pi if get_domain(domain) else 0.5)

        ctx_metrics = _extract_context_metrics(context_data)
        nom_metrics = _extract_nominative_metrics(nominative_data)

        entry = {
            "name": domain,
            "pi": float(pi),
            "theta": ctx_metrics["theta"],
            "lambda": nom_metrics["lambda"],
            "context_strength": ctx_metrics["context_strength"],
            "nominative_richness": nom_metrics["nominative_richness"],
            "effect_size": ctx_metrics["effect_size"],
        }
        entries.append(entry)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "domains": entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✓ Saved meta dataset for {len(entries)} domains to {output_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build meta dataset for universal predictor.")
    parser.add_argument("--domains", nargs="+", help="Subset of domains to include.")
    parser.add_argument("--output", type=str, help="Custom output path.")
    args = parser.parse_args()

    build_meta_dataset(domains=args.domains, output_path=Path(args.output) if args.output else OUTPUT_PATH)

