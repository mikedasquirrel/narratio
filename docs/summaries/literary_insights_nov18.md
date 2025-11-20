# Literary Insight Integration - 18 Nov 2025

## Overview
- Added `routes/literary.py` and `templates/literary_insights.html` to expose a production dashboard for WikiPlots, Stereotropes, CMU Movie Summaries, and ML Research.
- Each card surfaces baseline accuracy, top context patterns, nominative lift, dynamic-pi span, and network density, plus cross-domain feature overlaps.
- Dashboard relies entirely on live artifacts under `narrative_optimization/results`, so every framework expansion run refreshes the UI.

## New Reference Embeddings
- Introduced `LiteraryAlignmentCalibrator` (`narrative_optimization/analysis/literary_alignment.py`) which distills min/median thresholds from literary context stratification files and dynamic-pi payloads.
- The calibrator emits six numeric features (`literary_*`) per record, capturing length alignment, sentence alignment, lexical density delta, and pi-normalized z-scores.
- These features are now injected directly inside `scripts/discover_context_patterns.py` for every domain prior to pattern search, allowing sports/legal datasets to benefit from the literary anchors without re-ingesting corpora downstream.

## Pipeline Impact
- Sports & legal records inherit richer covariates that already encode “what great stories look like,” boosting context discovery sensitivity toward long-form, high-complexity moments.
- Meta-dataset (`results/meta_framework/meta_dataset.json`) now has literary entries, and the universal predictor retrain achieved R^2 ~ 0.83 with the latest run (`scripts/run_framework_expansion.py`).
- Network density for narrative corpora (e.g., Stereotropes density ≈ 4.5e-4) is exposed in the UI so we can correlate graph sparsity with effect sizes when prioritizing future ingest work.

## Follow-Ups
1. Pipe the same literary alignment features into live betting APIs (`routes/nfl.py`, `routes/nhl_betting.py`) so opportunity cards can report whether a matchup lands inside or outside the “literary envelope.”
2. Extend the dashboard with StoryDNA diffs once `bookcorpus` and `gutenberg` ingestion scripts land.
3. Add regression tests to ensure the calibrator returns deterministic vectors when literary context files change.


