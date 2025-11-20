## Supervised Transformer Integration – Handoff Notes (for next bot)

We now have several transformers that are **intentionally skipped** by the unsupervised feature pipeline because they require labels (`y`) or “genome”-style inputs. To complete the build-out, a separate bot should follow the checklist below.

### Transformers waiting on supervised support

| Transformer | Reason skipped historically | Expected inputs |
|-------------|---------------------------|-----------------|
| `AlphaTransformer` | Needs `y` during fit (label-aware) | Canonical feature matrix `X`, outcome vector `y` |
| `GoldenNarratioTransformer` | Needs `y` | Raw narratives `X`, labels `y` |
| `ContextPatternTransformer` | Performs pattern discovery with `y` | Canonical feature matrix `X`, `y` |
| `MetaFeatureInteractionTransformer` | Expects 2D `X` + `y` for supervised interactions | Canonical feature matrix + labels |
| `EnsembleMetaTransformer` | Designed for stacked ensembles, needs `y` | Feature blocks from upstream transformers + labels |
| `CrossDomainEmbeddingTransformer` | Requires precomputed “genome” dict or ndarray | `{'genome_features': np.ndarray}` per narrative |

### Tasks for the follow-up bot

1. **Build a supervised extraction pipeline** ✅
   - `feature_extraction_pipeline_supervised.py` now wraps the base pipeline, handles canonical feature assembly, and dispatches labels/genome payloads to the supervised set.
   - The pipeline tags cache metadata with `pipeline_mode: supervised` and includes hashes of the label vector so supervised/unsupervised runs never collide.

2. **Genome feature adapter** ✅
   - `genome_feature_adapter.py` groups features by provenance (nominative/archetypal/historial/uniquity) and emits the dictionaries required by `CrossDomainEmbeddingTransformer`.

3. **Selector integration**
   - `TransformerSelector` continues to enumerate the full catalog; consumers should call the supervised pipeline whenever the selected set includes any transformer from the table above.
   - `README.md` now documents both pipelines so future agents know when to switch modes.

4. **NBA-specific runbook**
   - Next operational step: rerun `universal_domain_processor.py --domain nba --use_transformers --save_results --mode supervised` (or the equivalent domain script) so the new supervised features populate `narrative_optimization/data/features/nba_all_features.npz`.
   - After regeneration, rerun `scripts/merge_nba_features_with_odds.py` and downstream training notebooks to bake the supervised columns into the modeling datasets.

5. **Docs + logging** ✅
   - This file plus the README describe the supervised flow.
   - The supervised pipeline logs every transformer outcome (success/error/skip) with the same structure as the base pipeline for parity.

### Usage snapshot

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import (
    SupervisedFeatureExtractionPipeline,
)

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=selected_transformers,
    domain_name='nba',
    domain_narrativity=0.49,
    enable_caching=True,
)

features = pipeline.fit_transform(narratives, labels)
report = pipeline.get_extraction_report()
```

* Use the same transformer list returned by `TransformerSelector`. The supervised pipeline automatically routes unsupervised transformers through the base pipeline and activates the label-aware/genome-aware subset.
* Cache keys now encode a hash of the label vector plus a supervised flag so inference datasets (no labels) continue using the unsupervised cache.

### Input locations

- Narratives: `data/domains/nba_pregame_narratives.json`
- Labels/outcomes: `data/modeling_datasets/nba_games_with_closing_odds.jsonl` (contains moneyline results)
- Genome components: produced during `merge_nba_features_with_odds.py` (can be repurposed)

Once these steps are finished, rerun `universal_domain_processor.py --domain nba --use_transformers --save_results` to regenerate the caches and ensure no transformer is skipped unless it explicitly requires a future capability.

---

### NHL parity & historical rebuild

Run the same supervised pipeline (once built) against NHL assets:

1. Refresh scoreboards + narratives  
   ```bash
   python scripts/fetch_nhl_scoreboard.py
   ```
2. Rebuild the domain dataset with Absolute-Max odds  
   ```bash
   python narrative_optimization/domains/nhl/build_narrative_betting_dataset.py --use-absolute-max-odds
   ```
3. Merge features with closing odds and retrain all NHL models  
   ```bash
   python scripts/merge_nhl_features_with_odds.py
   python narrative_optimization/domains/nhl/train_focused_temporal.py
   python narrative_optimization/domains/nhl/train_temporal_models.py
   python narrative_optimization/domains/nhl/train_narrative_models.py
   ```
4. Re-run the transformer pass to populate caches  
   ```bash
   python narrative_optimization/universal_domain_processor.py --domain nhl --use_transformers --save_results
   ```

Mirror the NBA supervised steps (provide `y` + genome dict) so every NHL transformer is exercised.

---

### Live odds → website integration

Use the unified live scan CLI to pipe Odds API + league scoreboards into website storage.

#### Command pattern

```bash
python scripts/run_live_scan.py --sport nhl --mode intermission --output data/live/nhl_recommendations.json
python scripts/run_live_scan.py --sport nba --mode halftime --output data/live/nba_recommendations.json
```

This script already:
- Pulls live game states via `services/live_feed.LiveFeedService`
- Hits The Odds API (`config/odds_api_config.py`) for moneyline + props
- Runs the sport-specific live betting engine (see `narrative_optimization/domains/<sport>/live_betting_engine.py`)
- Writes JSON payloads ready for the website

#### Website hook

1. Point the web tier (Flask or Next.js API route) at the JSON output path (`data/live/*.json`).
2. Add a cron/worker entry every 60 seconds:
   ```bash
   */1 * * * * cd /path/to/repo && python scripts/run_live_scan.py --sport nhl --mode intermission --output data/live/nhl_recommendations.json >> logs/live_scan.log 2>&1
   ```
3. If the site needs WebSocket pushes, tail the JSON and broadcast via the existing `LiveFeedService`.

Repeat for any sport key supported by `ENGINE_MAP` (`nhl`, `nba`, `nfl`). Add new engines by updating `ENGINE_MAP` and implementing a `live_betting_engine.py` in the target domain.

