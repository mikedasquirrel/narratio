#!/usr/bin/env python3
"""
Complete Framework Expansion Runner
===================================

Unified entry point that executes every pillar of the expansion plan (Phase 1):
- Context stratification (dynamic π + network + multi-scale features)
- Nominative enrichment benchmark

Usage:
    PYTHONPATH=. python3 scripts/run_framework_expansion.py \
        --domains nhl nfl nba mlb golf startups supreme_court
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure we can import the sibling scripts when executed directly
if __package__ is None:
    sys.path.append(".")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.discover_context_patterns import run_domain as run_context_discovery
from scripts.enrich_nominative_features import run_domain as run_nominative_enrichment
from scripts.build_meta_dataset import build_meta_dataset
from scripts.train_universal_predictor import train_universal_predictor
from scripts.ingest_stereotropes import ingest_stereotropes
from scripts.ingest_research_corpus import ingest_research_corpus
from scripts.ingest_wikiplots import ingest_wikiplots
from scripts.ingest_cmu_movies import ingest_cmu_movies


def run_for_domain(domain: str):
    print(f"\n===== {domain.upper()} :: Context Stratification =====", flush=True)
    print(f"[Progress] Starting context discovery for {domain}...", flush=True)
    run_context_discovery(domain, top_k=12)
    print(f"[Progress] Context discovery complete for {domain}", flush=True)
    print(f"\n===== {domain.upper()} :: Nominative Enrichment =====", flush=True)
    print(f"[Progress] Starting nominative enrichment for {domain}...", flush=True)
    run_nominative_enrichment(domain)
    print(f"[Progress] Nominative enrichment complete for {domain}", flush=True)


def main(domains: List[str]):
    print("\n=== Framework Expansion :: Boot Sequence ===", flush=True)
    print(f"[Framework] Domains queued: {', '.join(domains)}", flush=True)
    print(f"[Framework] Total domains to process: {len(domains)}", flush=True)
    
    # Ensure literary corpora exist before running domain loops
    print("\n[Framework] Checking literary corpus sources...", flush=True)
    stereo_source = PROJECT_ROOT / "external" / "stereotropes-data-public"
    if stereo_source.exists():
        print("\n===== Ingesting Stereotropes Corpus =====", flush=True)
        ingest_stereotropes(stereo_source)
        print("[Framework] Stereotropes ingestion complete", flush=True)
    else:
        print("[Framework] Stereotropes source not found, skipping", flush=True)
        
    research_source = PROJECT_ROOT / "external" / "Papers-Literature-ML-DL-RL-AI"
    if research_source.exists():
        print("\n===== Ingesting Research Corpus =====", flush=True)
        ingest_research_corpus(research_source)
        print("[Framework] Research corpus ingestion complete", flush=True)
    else:
        print("[Framework] Research corpus source not found, skipping", flush=True)
        
    wiki_dir = PROJECT_ROOT / "external" / "WikiPlots"
    if wiki_dir.exists():
        plots = wiki_dir / "plots"
        titles = wiki_dir / "titles"
        if plots.exists() and titles.exists():
            print("\n===== Ingesting WikiPlots Corpus =====", flush=True)
            ingest_wikiplots(plots, titles)
            print("[Framework] WikiPlots ingestion complete", flush=True)
        else:
            print("[Framework] WikiPlots files not found, skipping", flush=True)
    else:
        print("[Framework] WikiPlots directory not found, skipping", flush=True)
    
    cmu_source = PROJECT_ROOT / "data" / "MovieSummaries"
    if cmu_source.exists():
        print("\n===== Ingesting CMU Movie Corpus =====", flush=True)
        ingest_cmu_movies(cmu_source)
        print("[Framework] CMU Movies ingestion complete", flush=True)
    else:
        print("[Framework] CMU Movies source not found, skipping", flush=True)

    print(f"\n[Framework] Starting domain processing loop ({len(domains)} domains)...", flush=True)
    for idx, domain in enumerate(domains, 1):
        print(f"\n[Framework] Processing domain {idx}/{len(domains)}: {domain}", flush=True)
        run_for_domain(domain)
        print(f"[Framework] Completed domain {idx}/{len(domains)}: {domain}", flush=True)
        
    print("\n===== Building Meta Dataset =====", flush=True)
    print("[Framework] Aggregating results across domains...", flush=True)
    build_meta_dataset(domains)
    print("[Framework] Meta dataset built successfully", flush=True)
    
    print("\n===== Training Universal Predictor =====", flush=True)
    print("[Framework] Training meta-model...", flush=True)
    train_universal_predictor()
    print("[Framework] Universal predictor training complete", flush=True)
    print("\n=== Framework Expansion :: Complete ===", flush=True)


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("FRAMEWORK EXPANSION PIPELINE :: STARTING", flush=True)
    print("=" * 70, flush=True)
    
    parser = argparse.ArgumentParser(description="Run full framework expansion pipeline.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=[
            "nhl",
            "nfl",
            "nba",
            "mlb",
            "golf",
            "startups",
            "stereotropes",
            "ml_research",
            "wikiplots",
            "cmu_movies",
            "supreme_court",
        ],
        help="Domains to process in order.",
    )
    args = parser.parse_args()
    print(f"[Framework] Command-line args parsed. Domains: {args.domains}", flush=True)
    main(args.domains)

