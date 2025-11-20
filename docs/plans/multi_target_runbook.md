# Multi-Target Modeling Runbook

Date: 18 Nov 2025  
Owner: Narrative Optimization Framework

## 1. Objectives
- Elevate every domain from a single hard-coded `target` to a **target family** (game, checkpoint, season, financial ROI, trait-level KPIs).
- Provide deterministic cache semantics so repeated runs only recompute when inputs change.
- Preserve backwards compatibility: if a domain doesn’t define explicit targets, fall back to the legacy `outcome_field`.

## 2. Target Taxonomy by Domain
| Domain | Primary Targets | Secondary Targets | Notes |
| --- | --- | --- | --- |
| NHL / NFL / NBA | `game_win` (binary), `ats_cover` (binary), `roi_pct` (continuous) | `moneyline_edge`, `period_win_prob` (per checkpoint) | Checkpoint builders already emit period snapshots; ATS + ROI computed from odds. |
| MLB | `home_win`, `runline_cover` | `team_total_over` | Data currently sparse → mark ATS targets as optional until dataset restored. |
| Golf | `won_tournament` | `top10_finish`, `strokes_gained_delta` | Requires enrichment from round data; top10 is binary, strokes gained is continuous. |
| Startups | `successful` | `funding_amount`, `survival_years` | Funding/survival act as continuous targets for regression contexts. |
| Supreme Court | `petitioner_win` | `citation_count`, `vote_margin` | Already logged; map to continuous, allow per-term grouping later. |
| Stereotropes / WikiPlots / CMU Movies | `impact_score`, `revenue` | `award_score`, `sentiment_shift`, `character_diversity_index` | Additional continuous signals derived from metadata; to be added progressively. |
| ML Research | `impact_score` | `citation_count`, `readership_rank` | Small sample, but schema should allow future scaling. |

## 3. Target Schema (Proposed)
Each domain entry in `domain_registry.py` gains:

```python
targets=[
    TargetConfig(
        name="game_win",
        scope="game",
        outcome_type="binary",
        builder=build_nhl_game_win_row,
        description="Home team wins the game.",
        enabled=True,
    ),
    TargetConfig(
        name="ats_cover",
        scope="game",
        outcome_type="binary",
        builder=build_nhl_ats_row,
        requires_odds=True,
    ),
    TargetConfig(
        name="roi_pct",
        scope="bet",
        outcome_type="continuous",
        builder=build_nhl_roi_row,
    ),
]
```

- `scope`: `"game"`, `"checkpoint"`, `"season"`, `"bet"`, `"trait"`.
- `builder`: returns `(features_dict, target_value)` for that scope.
- `enabled`: allows toggling without removing config.
- `requires_checkpoints`: when true, builder expects checkpoint snapshots.

## 4. Pipeline Touchpoints
1. **Domain registry**: add `TargetConfig` dataclass, helper methods (`get_targets()`, `has_targets()`).
2. **Context Discovery** (`scripts/discover_context_patterns.py`):
   - Accept `--targets` CLI argument (default: all registered).
   - Iterate over targets: reuse shared features (multi-scale, literary alignment) and persist to `context_stratification/{domain}_{target}.json`.
   - Embed target metadata into result payload (`target_name`, `scope`, `outcome_type`).
3. **Nominative Enrichment**:
   - Mirror the target loop; output one JSON per domain containing an array of target metrics.
4. **Meta Dataset + Universal Predictor**:
   - Each `(domain, target)` combination becomes a separate entry with `target_scope` and `target_type` features.
5. **Caching layer**:
   - Before ingestion, check for `.cache_stamp` file storing source checksum & timestamp.
   - Feature-building stage writes intermediate parquet/feather caches keyed by `(domain, target)` to `results/cache/`.
   - Provide `--force-refresh` CLI flag to bypass caches.

## 5. Execution Plan Snapshot
- **Phase 1 (current)**: finalize schema + doc (this file), identify domain targets, design caching semantics.
- **Phase 2**: implement `TargetConfig` + domain entries; add loader utilities.
- **Phase 3**: refactor discovery/enrichment scripts to loop across targets and emit target-tagged artifacts.
- **Phase 4**: update meta framework + UI dashboards.
- **Phase 5**: regression tests, runtime tuning, finalize docs.

## 6. Open Questions
1. For literary domains, should `impact_score` remain the default baseline target even if additional scores are missing? (Proposed: yes; mark extra targets as disabled until data exists.)
2. How aggressively should we cache? (Proposal: auto-skip ingestion when destination JSON exists and is newer than source; add `--refresh-ingest` flag.)
3. Should checkpoint targets be stored as separate domains? (Proposal: no—treat them as `scope="checkpoint"` targets under the parent domain to keep meta-analysis coherent.)

Feedback welcome before proceeding to schema implementation.

