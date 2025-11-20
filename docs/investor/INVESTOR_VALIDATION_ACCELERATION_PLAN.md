## Investor Validation Acceleration Plan

Goal: convert the demonstrated NHL edge into an investable, audit-ready package. Focus areas are ordered by investor impact.

### 1. Reproducible Daily Pipeline
- **Owner:** Data Infra
- **Scope:** single orchestration script (or Make target) that runs scrape monitoring → odds normalization → dataset merges → model retrains → dashboard regeneration.
- **Outputs:** timestamped artifacts under `logs/daily_runs/YYYYMMDD/` plus hash manifest.
- **Status:** Not started. Current work was ad hoc CLI.

### 2. Odds Coverage Expansion
- **Owner:** Data Engineering
- **Scope:** raise join coverage from 39% to 70%+ via better team alias normalization, multi-source odds, and enhanced matching heuristics.
- **Tasks:**
  - Build coverage diagnostics notebook.
  - Add alias map + fuzzy match fallback.
  - Optionally ingest Betstamp/SpankOdds snapshots for missing games.
- **Outputs:** refreshed `nhl_feature_matrix_summary.json` showing ≥0.70 coverage.

### 3. Independent Verification Kit
- **Owner:** Research
- **Scope:** bundle minimal datasets + notebook that reproduces 69% tier from scratch.
- **Tasks:** export limited Parquet/JSON slices, write `investor_repro.ipynb` with train/test + ROI calc, ship SHA256 manifest.
- **Outputs:** `/docs/investor/verification/README.md` + zipped artifacts.

### 4. Live Forward Testing Log
- **Owner:** Trading Ops
- **Scope:** daily prediction/paper-trade log with realized ROI and confidence tiers.
- **Implementation:** script writing to `data/paper_trading/nhl_forward_{date}.json`, aggregated weekly chart fed into dashboard.
- **Success Metric:** ≥30 new live bets with tracked ROI.

### 5. Transformer Expansion Roadmap
- **Owner:** Narrative Modeling
- **Scope:** benchmark upcoming transformer upgrades; quantify contribution to ROC/ROI and explainability.
- **Outputs:** `analysis/transformer_uplift.md` with before/after stats and gating criteria.

### 6. Risk & Stress Testing Upgrade
- **Owner:** Quant Research
- **Scope:** extend TECHNICAL_VALIDATION_REPORT with bookmaker limit assumptions, Kelly throttles, edge decay simulations.
- **Outputs:** new “Stress Tests” section + Monte Carlo charts in dashboard.

### Sequencing
1. Automate reproducible pipeline (provides backbone for all downstream proofs).
2. Improve odds coverage + diagnostics.
3. Ship verification kit + forward log in parallel.
4. Layer transformer roadmap and risk appendix once the data/methodology story is fully auditable.

### Tracking
- Add each initiative to project board with owner + ETA.
- Require runbooks + manifests checked into repo for every automated process.

