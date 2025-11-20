## NHL Evidence Readiness Tracker

Goal: convert the NHL narrative edge into an investable, audit-ready program by tightening factual evidence across data quality, model validation, and forward-testing.

### 1. Coverage & Data Reliability
- **Absolute Max odds fusion:** `build_nhl_modeling_dataset.py --use-absolute-max-odds` now writes `nhl_games_with_closing_odds.jsonl` (6,251 matched games) plus `nhl_games_without_closing_odds.json` (567 cases) for transparency.
- **Diagnostics:** `analysis/nhl_odds_coverage_diagnostics.json` (generated 2025-11-20) quantifies coverage by season and flags that 94% of the remaining gaps are regular-season 2020–2025 games. Action: enrich alias map + secondary odds feed to eliminate the 13% of 2024–25 games still unmatched.
- **Next:** add automated CI check that fails if coverage <90% or if new unmatched spikes occur.

### 2. Holdout & Stress Verification
- **Temporal holdout:** `train_temporal_models.py` already splits on 2023-11-04; need to export those predictions + ROI tables into `/docs/investor/verification`.
- **Independent kit:** `notebooks/investor_verification/nhl_holdout_verification.ipynb` now emits `docs/investor/verification/nhl_holdout_metrics.json` (≥0.55–0.70 tiers). Next: bundle a SHA256 manifest under `docs/investor/verification/MANIFEST.json`.
- **Stress tests:** incorporate the new feature genome into `docs/investor/TECHNICAL_VALIDATION_REPORT.md` stress appendix (limit throttles, Kelly overlays, bookmaker liquidity).

### 3. Live Forward Testing
- **Daily scoring:** `narrative_optimization/domains/nhl/score_upcoming_games.py --edge-threshold 0.02` now supports the embedded feature genome; warnings resolved via transformer metadata fallback.
- **Logging:** `scripts/log_nhl_forward_predictions.py --min-prob 0.60` appends to `data/paper_trading/nhl_forward_log.jsonl` (15 logged bets, updated 2025-11-20T10:46Z). Action: add result reconciliation (join with ESPN game outcomes) and publish rolling ROI chart inside the investor dashboard.
- **Target:** ≥30 paper trades with realized ROI + drawdown stats before pitching capital deployment.

### 4. Model Transparency
- **Universal transformer inventory:** `nhl_features_metadata.json` mirrors the 39 successful universal transformers; `score_upcoming_games.py` now falls back to this list when the structured dataset embeds narrative features.
- **Meta-ensemble audit:** `nhl_complete_analysis.py` runs with convergent `saga` logistic head (max_iter=20k) and produces pathed outputs (`nhl_complete_analysis.json`). Action: diff the top-10 patterns vs. older investors’ decks and highlight the new invisible-context uplift using `analysis/transformer_uplift.md`.

### 5. Automation Backbone
- **Daily pipeline:** `scripts/run_daily_pipeline.py` orchestrates scrape → merge → retrain → docs in `logs/daily_runs/YYYYMMDD_HHMMSS/`. Action: schedule via cron/GitHub Actions and attach manifest hashes to investor packages.

Use this tracker as the canonical scoreboard; update whenever a milestone ships so investors can see the march toward “best possible evidence.”

