# Checkpoint Feed Template

Reusable conventions for exposing per-period/quarter/half narratives across sports domains.

## Why this exists

- Normalize how the product surfaces checkpoint snapshots so new domains can launch quickly.
- Share the same UI vocabulary (hero metrics, timeline rail, live odds panel) across leagues.
- Keep the pipeline contract explicit: what a `DomainConfig` must expose, how snapshots are hydrated, and how optional live odds bolt on.

## Required domain plumbing

1. **Domain registry fields**
   - `checkpoint_schema`: ordered checkpoints (`["P1", "P2", "FINAL"]`, etc.).
   - `checkpoint_builder`: callable `(records, checkpoint, limit) -> List[Dict]`.
   - `checkpoint_fields`: metrics (score, differential, win prob, etc.).
   - `supports_checkpoints()` must return `True`.

2. **Deterministic snapshot builder**
   - Lives under `narrative_optimization/domains/<league>/`.
   - Deterministically splits scoring, computes derived metrics, emits `metadata`, `metrics`, `score`, and `narrative` per checkpoint.

3. **Raw record accessor**
   - `DomainConfig.get_raw_records()` already wraps JSON flattening; re-use it for the feed so UI + modeling stay in sync.

## Shared feed blueprint

`routes/checkpoint_feeds.py` now provides:

- `build_domain_checkpoint_payload(domain, limit, checkpoint, live_odds_loader=None)` – generic snapshot compiler.
- `FEED_REGISTRY` – register each league with minimal metadata:
  ```python
  FEED_REGISTRY["nhl"] = {
      "domain": "nhl",
      "theme": {...},        # icon, hero copy, CTA link
      "live_odds_loader": _load_nhl_live_odds,  # optional
  }
  ```
- Blueprint routes:
  - `GET /sports/checkpoints/<league>` → renders `templates/checkpoints/feed.html`.
  - `GET /sports/checkpoints/<league>/api` → JSON payload (same structure as UI expects).
  - `GET /sports/checkpoints` → Directory of available feeds.

### Theme payload

Each feed registers a `theme` dict consumed by the shared template:

```python
{
    "league": "NHL",
    "icon": "🏒",
    "page_title": "NHL Checkpoint Feed",
    "hero_title": "Period-by-period NHL betting intelligence",
    "hero_body": "...",
    "cta_label": "View NHL patterns",
    "cta_href": "/nhl/betting/patterns",
    "empty_state": "Run NHL ingestion to populate this feed."
}
```

## UI contract (`templates/checkpoints/feed.html`)

Context keys required by the template:

- `theme`: dict above.
- `games`: list of snapshot payloads (see below).
- `summary`: string used in hero + pills.
- `checkpoint_total`: integer.
- `generated_at`: ISO timestamp.
- `last_update_label`: human-readable refresh string.
- `limit_games`: int (used to annotate data attribute).
- `checkpoint_filter`: string (e.g., `"all"`, `"P2"`).
- `error`: optional error code.
- `live_odds_count`, `live_odds_attached`, `live_odds_label`.
- `api_endpoint`: URL the auto-refresh script should call.

### Game payload structure

Each entry in `games` should include:

```json
{
  "game_id": "2014010077",
  "matchup": "WSH @ BUF",
  "home_team": "BUF",
  "away_team": "WSH",
  "date_label": "Oct 01, 2014",
  "venue": "KeyBank Center",
  "status": "End of 2nd period",
  "score": {"home": 3, "away": 1},
  "momentum_summary": "BUF converted their pregame edge ...",
  "context_badges": [{"label": "Rest edge", "value": "+1 days"}, ...],
  "synthetic_narrative": "Pregame storyline ...",
  "result_trend": "home" | "away" | "even",
  "snapshots": [
    {
      "label": "End of 1st period",
      "score_display": "BUF 2 - 0 WSH",
      "narrative": "...",
      "win_probability_pct": "68.5%",
      "edge_delta_pct": 12.3,
      "leverage": 0.82,
      "minutes_label": "20 min",
      "progress_pct": 33.3,
      "trend_class": "trend-home"
    },
    ...
  ],
  "live_odds": {
    "bookmaker": "The Odds API",
    "moneyline_home_display": "-150",
    "moneyline_away_display": "+130",
    "puck_line_home_display": "-1.5 (+165)",
    "puck_line_away_display": "+1.5 (-185)",
    "over_display": "6.0 (-110)",
    "under_display": "6.0 (-110)"
  },
  "live_odds_label": "Nov 16 · 06:41 PM UTC"
}
```

## Optional live odds hook

- Implement `*_live_odds.json` dump (e.g., `data/live/nhl_live_odds.json`) via scraper or API script.
- Loader signature: `() -> Tuple[Optional[str], Dict[str, Dict]]`, returning source timestamp + mapping keyed by `HOME::AWAY`.
- Register loader inside `FEED_REGISTRY` entry; the shared payload builder will automatically stitch odds into the UI.

## Onboarding checklist for a new sport

1. Ensure domain JSON has fields required for checkpoint builder (scores, context, odds).
2. Add `checkpoint_schema`, `checkpoint_builder`, etc. to `DomainConfig`.
3. Author deterministic builder under `narrative_optimization/domains/<league>/`.
4. (Optional) Add live odds fetcher writing to `data/live/<league>_live_odds.json`.
5. Register feed in `routes/checkpoint_feeds.FEED_REGISTRY` with theme + loader.
6. (Optional) Link to `/sports/checkpoints/<league>` from nav or league-specific betting pages.
7. Document any league-specific nuances in `docs/domain_specific/<league>_checkpoint_playbook.md`.

Once these steps are complete, the shared UI/API will automatically serve the new league, and other pages can either link to `/sports/checkpoints/<league>` or render the shared template directly (as `nhl_live_betting.html` now does).

