# NHL Checkpoint Narrative Playbook

## Purpose

Establish deterministic post-period updates for NHL games using the existing
`nhl_games_with_odds.json` dataset so the narrative optimization pipeline can
ship “after P1/P2/Final” recommendations before full live feeds arrive.

## Architecture Overview

- **Domain registry**: `DomainConfig` now supports `checkpoint_schema`,
  `checkpoint_fields`, and a `checkpoint_builder`. The NHL entry declares
  `['P1', 'P2', 'FINAL']` and routes to
  `narrative_optimization/domains/nhl/checkpoint_narratives.py`.
- **Checkpoint builder**: deterministically splits final scores across periods
  using rest/context/odds signals, emits snapshots containing narrative text,
  score, and metrics (`win_probability_home`, `rest_advantage`, etc.).
- **Access API**: `load_domain_checkpoints('nhl', checkpoint='P2')` loads
  structured payloads ready for modeling or UI rendering.

## Data Flow

1. Load NHL games via `DomainConfig._load_raw_records()`.
2. `build_nhl_checkpoint_snapshots` iterates games and produces three snapshots
   per game.
3. Downstream consumers (models, routers, UI) request snapshots per checkpoint
   and either:
   - Aggregate into per-period training sets.
   - Render real-time timeline cards (pregame + checkpoint snapshots).

## Backtesting Strategy

1. **Historical replay**: iterate snapshots sorted by `sequence`. Feed each
   snapshot into the betting engine using only data up to that checkpoint.
2. **Calibration**: compare `win_probability_home` vs actual outcomes for P1 and
   P2 checkpoints (reliability curves per season).
3. **ROI simulation**: reuse existing staking logic but gate trades to when
   checkpoint probability deviates from market by ≥X%.
4. **Stress tests**: measure sensitivity to the deterministic goal-splitter by
   perturbing rest/record/odds inputs ±10% and confirming ranking robustness.

## Rollout Path

1. **Phase 1 – Analytics**: expose `load_domain_checkpoints` output in notebooks
   to validate hit rates and user-facing narratives.
2. **Phase 2 – Product**: add “Period Timeline” to the NHL experience using the
   snapshot narratives + metrics.
3. **Phase 3 – Automation**: schedule nightly job to regenerate snapshots when
   new games land, persist to `data/live/nhl_period_snapshots.parquet`.
4. **Phase 4 – Cross-sport template**: replicate builder pattern for NBA/NFL
   (quarters) using the same registry hooks once domain-specific logic exists.

## Operational Notes

- Builder uses only existing fields (scores, temporal context, betting odds);
  no new upstream feeds required.
- Outputs are deterministic; identical between runs to simplify testing.
- Narratives intentionally capped at 420 chars to preserve UI fidelity.
- Guardrails: if domain lacks checkpoint support, `load_domain_checkpoints`
  raises a descriptive error for early failure.
- Batch export: run `python narrative_optimization/domains/nhl/build_checkpoint_dataset.py`
  (optionally pass `--checkpoint P2`) to generate flattened Parquet +
  metadata files for analytics/model training.

## Next Steps

- Embed snapshots into model training scripts to learn checkpoint-specific win
  probabilities.
- Extend builder once true per-period stats are sourced (replace deterministic
  split with real scoring progression while keeping interface identical).

