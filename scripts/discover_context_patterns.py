#!/usr/bin/env python3
"""
Discover Context Patterns
=========================

Phase 1 deliverable: exhaustively test contextual combinations for validated
domains (NHL, NFL, Supreme Court) and export production-ready filters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from narrative_optimization.analysis.causal_pattern_analyzer import (
    CausalPatternAnalyzer,
)
from narrative_optimization.analysis.dynamic_pi_calculator import DynamicPiCalculator
from narrative_optimization.analysis.network_genome_builder import NetworkGenomeBuilder
from narrative_optimization.analysis.pattern_interaction_tester import (
    PatternInteractionTester,
)
from narrative_optimization.analysis.literary_alignment import (
    LiteraryAlignmentCalibrator,
)
from narrative_optimization.domain_registry import get_domain
from narrative_optimization.domain_targets import (
    build_nhl_game_win_row,
    build_nfl_game_win_row,
    build_nba_game_win_row,
    build_mlb_game_win_row,
    build_golf_win_row,
    build_startup_success_row,
    build_supreme_court_outcome_row,
    build_wikiplots_impact_row,
    build_stereotropes_impact_row,
    build_ml_research_impact_row,
    build_cmu_movies_revenue_row,
)
from narrative_optimization.transformers.multi_scale_enhanced import (
    MultiScaleEnhancedTransformer,
)
from narrative_optimization.src.transformers.context_pattern import (
    ContextPatternTransformer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT / "narrative_optimization" / "results" / "context_stratification"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LITERARY_CALIBRATOR = LiteraryAlignmentCalibrator()


def _to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_conditions(conditions: Dict[str, Dict]) -> Dict[str, Dict]:
    sanitized = {}
    for feature, cond in conditions.items():
        if isinstance(cond, dict):
            sanitized[feature] = {k: _to_native(v) for k, v in cond.items()}
        else:
            sanitized[feature] = _to_native(cond)
    return sanitized


FEATURE_BUILDERS = {
    "nhl": build_nhl_game_win_row,
    "nfl": build_nfl_game_win_row,
    "nba": build_nba_game_win_row,
    "mlb": build_mlb_game_win_row,
    "golf": build_golf_win_row,
    "startups": build_startup_success_row,
    "supreme_court": build_supreme_court_outcome_row,
    "wikiplots": build_wikiplots_impact_row,
    "stereotropes": build_stereotropes_impact_row,
    "ml_research": build_ml_research_impact_row,
    "cmu_movies": build_cmu_movies_revenue_row,
}


def load_records(domain: str) -> Tuple[List[Dict], float]:
    config = get_domain(domain)
    if not config:
        raise ValueError(f"Domain '{domain}' is not registered.")
    data = config.get_raw_records()
    if isinstance(data, dict):
        if "games" in data:
            data = data["games"]
        else:
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unexpected structure for domain {domain}: {type(data)}")
    return data, config.estimated_pi


def build_dataframe(
    domain: str,
    builder: Callable[[Dict], Optional[Dict]],
    records: List[Dict],
    default_pi: float,
) -> Tuple[pd.DataFrame, str, List[Dict]]:
    if not builder:
        raise ValueError(f"No feature builder provided for '{domain}'")

    rows = []
    retained_records = []
    for rec in records:
        row = builder(rec)
        if not row or "target" not in row:
            continue
        row.update(LITERARY_CALIBRATOR.score(rec))
        pi_meta = rec.get("pi_metadata") or {}
        row["pi_effective"] = float(
            pi_meta.get("pi_effective") or pi_meta.get("pi_base") or default_pi or 0.5
        )
        network = rec.get("network_features") or {}
        for feature in ("home_centrality", "away_centrality", "rivalry_strength"):
            if feature in network and feature not in row:
                row[feature] = float(network[feature])
        rows.append(row)
        retained_records.append(rec)

    if not rows:
        return pd.DataFrame(), "target", []

    df = pd.DataFrame(rows).dropna()
    retained_records = [retained_records[i] for i in df.index]
    df = df.reset_index(drop=True)
    return df, "target", retained_records


def _resolve_builder(config, target, domain_name: str):
    if target and target.builder:
        return target.builder
    return FEATURE_BUILDERS.get(domain_name)


def _target_output_path(domain: str, target_name: str) -> Path:
    return OUTPUT_DIR / f"{domain}_{target_name}_contexts.json"


def _run_target_analysis(
    domain: str,
    target,
    builder: Callable[[Dict], Optional[Dict]],
    records: List[Dict],
    default_pi: float,
    top_k: int,
):
    target_label = target.name
    print(f"\n[Context Discovery] >>> Target '{target_label}' ({target.scope})", flush=True)
    print(f"[Context Discovery] Building feature dataframe for target '{target_label}'...", flush=True)
    df, target_col, filtered_records = build_dataframe(domain, builder, records, default_pi)
    if df.empty:
        print(f"[Context Discovery] No data available for target '{target_label}'.", flush=True)
        return

    print(f"[Context Discovery] Extracting multi-scale features...", flush=True)
    ms_transformer = MultiScaleEnhancedTransformer()
    multi_features = ms_transformer.transform(filtered_records)
    multi_df = pd.DataFrame(
        multi_features, columns=ms_transformer.get_feature_names_out()
    )
    df = pd.concat([df.reset_index(drop=True), multi_df], axis=1)
    print(f"[Context Discovery] Multi-scale features added. Final shape: {df.shape}", flush=True)

    print(f"[Context Discovery] Initializing pattern transformer for target '{target_label}'...", flush=True)
    transformer = ContextPatternTransformer(
        min_accuracy=0.6 if domain != "supreme_court" else 0.55,
        min_samples=200 if domain == "nba" else 40 if domain != "supreme_court" else 25,
        max_patterns=top_k,
        feature_combinations=3,
    )
    feature_cols = [col for col in df.columns if col != target_col]
    print(f"[Context Discovery] Fitting transformer on {len(feature_cols)} features for '{target_label}'...", flush=True)
    transformer.fit(df[feature_cols], df[target_col])
    print(f"[Context Discovery] Transformer fit complete. Found {len(transformer.patterns_)} patterns", flush=True)

    print(f"[Context Discovery] Processing {len(transformer.patterns_)} discovered patterns...", flush=True)
    baseline = df[target_col].mean()
    patterns_payload = []
    pattern_flags: Dict[str, np.ndarray] = {}
    for idx, pattern in enumerate(transformer.patterns_):
        mask = pattern.matches(df[feature_cols])
        pi_slice = df.loc[mask, "pi_effective"] if "pi_effective" in df else None
        payload = {
            "pattern_name": f"{domain}_{target_label}_context_{idx+1}",
            "conditions": _sanitize_conditions(pattern.conditions),
            "accuracy": float(pattern.accuracy),
            "sample_size": int(pattern.sample_size),
            "effect_size": float(pattern.effect_size),
            "edge_vs_baseline": float(pattern.accuracy - baseline),
            "pi_mean": float(pi_slice.mean()) if pi_slice is not None else None,
            "pi_std": float(pi_slice.std()) if pi_slice is not None else None,
        }
        patterns_payload.append(payload)
        pattern_flags[payload["pattern_name"]] = mask.astype(int)
        if (idx + 1) % 3 == 0:
            print(f"[Context Discovery] Processed {idx + 1}/{len(transformer.patterns_)} patterns...", flush=True)

    # Causal + interaction diagnostics
    print(f"[Context Discovery] Running interaction and causal analysis...", flush=True)
    interaction_results = []
    causal_results = []
    if len(pattern_flags) >= 2:
        pattern_df = df.copy()
        for name, values in pattern_flags.items():
            pattern_df[name] = values
        tester = PatternInteractionTester(target_col, min_samples=20)
        interaction_results = tester.evaluate(pattern_df, list(pattern_flags.keys()))
        print(f"[Context Discovery] Interaction analysis complete", flush=True)

        analyzer = CausalPatternAnalyzer(target_col)
        causal_reports = analyzer.analyze(
            pattern_df,
            treatment_cols=list(pattern_flags.keys())[: min(5, len(pattern_flags))],
            control_features=feature_cols,
            tag=f"{domain}_contexts",
        )
        causal_results = [report.to_dict() for report in causal_reports]
        print(f"[Context Discovery] Causal analysis complete", flush=True)

    print(f"[Context Discovery] Saving results to JSON...", flush=True)
    output = {
        "domain": domain,
        "target": target_label,
        "target_scope": target.scope,
        "target_outcome_type": target.outcome_type,
        "baseline_value": baseline,
        "patterns": patterns_payload,
        "interactions": interaction_results,
        "causal": causal_results,
    }
    output_path = _target_output_path(domain, target_label)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved {len(patterns_payload)} contexts to {output_path}", flush=True)


def run_domain(domain: str, top_k: int, target_filter: Optional[List[str]] = None):
    print(f"\n=== {domain.upper()} :: Loading records ===", flush=True)
    print(f"[Context Discovery] Loading records for {domain}...", flush=True)
    records, default_pi = load_records(domain)
    print(f"[Context Discovery] Loaded {len(records)} records, default π={default_pi:.3f}", flush=True)
    config = get_domain(domain)

    print(f"[Context Discovery] Calculating dynamic π...", flush=True)
    pi_calc = DynamicPiCalculator(domain)
    annotated_records = pi_calc.annotate(records)["records"]
    print(f"[Context Discovery] Dynamic π annotation complete", flush=True)

    print(f"[Context Discovery] Building network genome...", flush=True)
    network_builder = NetworkGenomeBuilder(domain)
    annotated_records = network_builder.annotate_records(annotated_records)["records"]
    print(f"[Context Discovery] Network genome annotation complete", flush=True)

    targets = config.get_targets()
    processed = False
    for target in targets:
        if target_filter and target.name not in target_filter:
            continue
        builder = _resolve_builder(config, target, domain)
        if not builder:
            print(f"[Context Discovery] No builder available for target '{target.name}' on domain '{domain}'. Skipping.", flush=True)
            continue
        _run_target_analysis(
            domain=domain,
            target=target,
            builder=builder,
            records=annotated_records,
            default_pi=default_pi,
            top_k=top_k,
        )
        processed = True

    if not processed:
        msg = "No targets processed (check --targets filter)." if target_filter else "No enabled targets."
        print(f"[Context Discovery] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Discover high-edge contexts.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["nhl", "nfl", "nba", "mlb", "golf", "startups", "supreme_court"],
        help="Domains to process",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Max patterns per domain/target")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional target names to process (applies to all domains).",
    )
    args = parser.parse_args()
    print(f"[Context Discovery] Arguments parsed: domains={args.domains}, targets={args.targets}", flush=True)

    for domain in args.domains:
        run_domain(domain, args.top_k, args.targets)


if __name__ == "__main__":
    print("[Context Discovery] CLI entrypoint engaged.", flush=True)
    main()


