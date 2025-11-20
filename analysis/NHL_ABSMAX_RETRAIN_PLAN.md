# NHL Absolute Maximum Retrain Plan

This document captures the end-to-end steps required to leverage the Absolute Maximum scrape for NHL model refreshes without interrupting the live collection jobs.

## 1. Data Snapshot + Validation
- Run `python analysis/nhl_absolute_max_snapshot.py` off the main scraping machine once per day (ideally during API quiet hours).
- Confirm the CLI summary reports stable counts and copy the generated metadata (`analysis/nhl_absolute_max_snapshot.meta.json`) into the nightly run artifacts.
- Track snapshot coverage deltas (games processed, bookmaker distribution) inside the existing `analysis/multi_market_validation_status.json`.

## 2. Dataset Rebuild Workflow
- Execute `python narrative_optimization/domains/nhl/build_narrative_betting_dataset.py --use-absolute-max-odds` to infuse bookmaker consensus odds into the modeling dataset.
- If experimentation is needed, pass `--games-path` or `--output-dir` to isolate dry-run datasets.
- Capture the Absolute Max coverage stats (logged in `nhl_narrative_betting_metadata.json`) and fail the run when coverage <80% to avoid training on stale/imputed prices.

## 3. Feature Drift + QA
- Compare new Parquet columns for odds-derived features against the previous baseline using `analysis/cross_league_pattern_validation.py` (add NHL mode to that script if necessary).
- Report drift metrics (mean delta, KS statistic, missing ratio) inside `analysis/nhl_data_quality.md`.
- Flag any teams/dates missing Absolute Max odds and push them back into the scraping backlog if needed.

## 4. Model Retraining Sequence
1. `python narrative_optimization/domains/nhl/train_focused_temporal.py`
2. `python narrative_optimization/domains/nhl/train_temporal_models.py`
3. `python narrative_optimization/domains/nhl/train_narrative_models.py`
4. `python narrative_optimization/domains/nhl/nhl_complete_analysis.py`

Each script should reference the refreshed dataset artifacts automatically. Capture CV stats + ROI shifts in `analysis/RECENT_SEASON_BACKTEST_REPORT.md`.

## 5. Reporting + Dashboards
- Publish a concise executive summary to `analysis/EXECUTIVE_SUMMARY_BACKTEST.md` with:
  - Snapshot date
  - Odds coverage
  - Model accuracy / ROI deltas
  - Top new patterns discovered
- Refresh the NHL dashboards (`templates/nhl_unified.html`, `templates/nhl_results.html`) by re-running the Flask app or regenerating static exports if needed.

## 6. Automation Hook
- Extend `scripts/nhl_automated_daily.sh` to:
  1. Invoke the snapshot utility
  2. Rebuild the dataset with `--use-absolute-max-odds`
  3. Trigger the four-model retrain chain
  4. Sync metadata + summaries to S3 / long-term storage
- Add runtime guards that skip retraining if the snapshot coverage drops below the configured threshold or if the live scraper is mid-write (stability checks already included in the snapshot script).

Following this loop keeps the NHL stack in lockstep with the massive Absolute Maximum scrape while preserving the “no interruptions” constraint on the live collector. Once validated here, replicate the workflow for NBA/NFL/MLB by swapping the sport keys in the snapshot allocator.***

