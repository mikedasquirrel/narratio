#!/usr/bin/env python3
"""
Run the supervised feature extraction pipeline for any registered domain.

This script:
1. Loads narratives + outcomes via the domain registry.
2. Uses TransformerSelector to determine the full transformer suite.
3. Executes SupervisedFeatureExtractionPipeline (labels + genome-aware).
4. Saves `data/features/{domain}_all_features.npz` plus a processing report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.domain_registry import load_domain_safe  # noqa: E402
from narrative_optimization.src.pipelines import SupervisedFeatureExtractionPipeline  # noqa: E402
from narrative_optimization.src.transformers.transformer_selector import (  # noqa: E402
    TransformerSelector,
)


def _infer_domain_type(config) -> Optional[str]:
    """Best-effort domain type extraction from config metadata."""
    if config is None:
        return None
    candidate = getattr(config, "domain_type", None)
    if candidate:
        return candidate
    config_type = getattr(config, "type", None)
    if config_type is None:
        return None
    return getattr(config_type, "value", str(config_type))


def _resolve_pi(config, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    if config is not None:
        for attr in ("estimated_pi", "pi"):
            value = getattr(config, attr, None)
            if value is not None:
                return float(value)
    raise ValueError("Unable to determine narrativity π for the requested domain.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the supervised narrative feature pipeline for a domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--domain", required=True, help="Domain name (e.g., nba)")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional cap on narratives processed (default = full dataset)",
    )
    parser.add_argument(
        "--domain-type",
        help="Override domain type hint (sports, entertainment, business, nominative, ...)",
    )
    parser.add_argument(
        "--pi",
        type=float,
        help="Override narrativity π (otherwise uses config estimate)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use the selector's fast subset of transformers.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable pipeline caching (always recompute features).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Custom cache directory for pipeline artifacts.",
    )
    parser.add_argument(
        "--output",
        help="Path for the .npz feature file "
        "(default narrative_optimization/data/features/{domain}_all_features.npz).",
    )
    parser.add_argument(
        "--results-json",
        help="Path for the processing summary JSON "
        "(default narrative_optimization/data/features/{domain}_processing_results.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose pipeline logging.",
    )
    parser.add_argument(
        "--text-sample",
        type=int,
        default=100,
        help="Number of narratives to store alongside the feature matrix for reference.",
    )
    args = parser.parse_args()

    narratives, outcomes, config = load_domain_safe(args.domain)
    if narratives is None or outcomes is None:
        raise RuntimeError(f"Failed to load domain '{args.domain}'. See logs for details.")

    sample_size = args.sample_size or len(narratives)
    narratives = narratives[:sample_size]
    labels = np.asarray(outcomes[:sample_size])

    if labels.size == 0:
        raise ValueError(
            f"Domain '{args.domain}' does not provide outcomes. "
            "Supervised pipeline requires labels."
        )

    domain_pi = _resolve_pi(config, args.pi)
    domain_type = args.domain_type or _infer_domain_type(config)

    selector = TransformerSelector()
    transformer_names = selector.select_transformers(
        domain_name=args.domain,
        pi_value=domain_pi,
        domain_type=domain_type,
    )
    if args.fast_mode:
        transformer_names = selector.get_fast_subset(transformer_names)

    pipeline = SupervisedFeatureExtractionPipeline(
        transformer_names=transformer_names,
        domain_name=args.domain,
        domain_narrativity=domain_pi,
        cache_dir=args.cache_dir,
        enable_caching=not args.disable_cache,
        verbose=not args.quiet,
    )

    start_time = time.time()
    features = pipeline.fit_transform(narratives, labels)
    elapsed = time.time() - start_time

    feature_names = pipeline.all_feature_names
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    output_dir = PROJECT_ROOT / "narrative_optimization" / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else output_dir / f"{args.domain}_all_features.npz"
    results_path = (
        Path(args.results_json)
        if args.results_json
        else output_dir / f"{args.domain}_processing_results.json"
    )

    text_sample = min(args.text_sample, len(narratives))
    sample_texts = np.array(narratives[:text_sample], dtype=object)

    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        feature_names=np.array(feature_names, dtype=object),
        sample_texts=sample_texts,
        pi=domain_pi,
        transformers=np.array(transformer_names, dtype=object),
    )

    report = pipeline.get_extraction_report()
    summary = {
        "domain": args.domain,
        "status": "success",
        "pi": domain_pi,
        "domain_type": domain_type,
        "sample_size": len(narratives),
        "transformers_completed": report.get("successful"),
        "transformers_failed": report.get("failed"),
        "transformers_skipped": report.get("skipped"),
        "total_features": int(features.shape[1]),
        "duration_seconds": elapsed,
        "output_file": str(output_path),
        "pipeline_mode": report.get("pipeline_mode"),
        "transformer_stats": report.get("extraction_stats", []),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n✓ Supervised features built for {args.domain}: "
        f"{features.shape[0]:,} rows × {features.shape[1]:,} columns "
        f"({elapsed/60:.1f} min)"
    )
    print(f"  → Features: {output_path}")
    print(f"  → Summary:  {results_path}")


if __name__ == "__main__":
    main()


