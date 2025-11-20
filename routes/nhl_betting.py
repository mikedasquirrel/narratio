"""
NHL Betting Routes

Web interface for NHL betting patterns and live opportunities.
Displays validated patterns with ROI statistics and provides API endpoints.

Author: Narrative Integration System
Date: November 16, 2025
"""

from flask import Blueprint, render_template, jsonify, request
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from narrative_optimization.domain_registry import get_domain
from narrative_optimization.analysis.dynamic_pi_calculator import DynamicPiCalculator
from utils.market_edge_calculator import MarketEdgeCalculator

nhl_betting_bp = Blueprint('nhl_betting', __name__, url_prefix='/nhl/betting')

# Paths
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data' / 'domains'
live_data_dir = project_root / 'data' / 'live'
LIVE_ODDS_PATH = live_data_dir / 'nhl_live_odds.json'
CONTEXT_RESULTS_DIR = (
    project_root / 'narrative_optimization' / 'results' / 'context_stratification'
)

EDGE_CALCULATOR = MarketEdgeCalculator(min_edge=0.04)

DEFAULT_GAME_LIMIT = 6

NHL_THEME = {
    "league": "NHL",
    "icon": "🏒",
    "page_title": "NHL Checkpoint Feed",
    "hero_title": "Period-by-period NHL betting intelligence",
    "hero_body": (
        "Synthetic checkpoint snapshots merge rest/record context, odds, and deterministic scoring splits "
        "so you can act after every period without waiting for full live data ingestion."
    ),
    "cta_label": "View NHL patterns",
    "cta_href": "/nhl/betting/patterns",
    "empty_state": "Run NHL ingestion to populate this feed.",
}

def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_date_label(date_str: Optional[str]) -> str:
    if not date_str:
        return "Date TBD"
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%b %d, %Y")
        except ValueError:
            continue
    return date_str


def _format_timestamp_label(timestamp: Optional[str]) -> str:
    if not timestamp:
        return "—"
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%b %d · %I:%M %p UTC")
    except ValueError:
        return timestamp


def _trend_class(score_diff: float) -> str:
    if score_diff > 1:
        return "trend-home"
    if score_diff < -1:
        return "trend-away"
    return "trend-balanced"


def _odds_key(home: str, away: str) -> str:
    return f"{(home or '').upper()}::{(away or '').upper()}"


def _compose_momentum_summary(home: str, away: str, win_shift_pct: float, final_score: Dict[str, int]) -> str:
    scoreline = f"{home} {final_score.get('home', 0)} - {final_score.get('away', 0)} {away}"
    if win_shift_pct >= 8:
        return f"{home} converted their pregame edge (+{win_shift_pct:.1f} pts) and close out {scoreline}."
    if win_shift_pct <= -8:
        return f"{away} flipped expectation ({win_shift_pct:.1f} pts swing) en route to {scoreline}."
    return f"{scoreline} stays close to the pregame thesis."


def _format_snapshot_payload(
    snapshot: Dict,
    home: str,
    away: str,
    pre_prob: float,
) -> Dict:
    metrics = snapshot.get("metrics") or {}
    metadata = snapshot.get("metadata") or {}
    score = snapshot.get("score") or {}
    win_prob = _safe_float(metrics.get("win_probability_home"), pre_prob)
    score_diff = _safe_float(metrics.get("score_differential"), 0.0)
    leverage = _safe_float(metrics.get("leverage_index"), 0.0)
    progress = _safe_float(metadata.get("game_progress"), 0.0)
    minutes_elapsed = metadata.get("minutes_elapsed")
    return {
        "checkpoint_id": snapshot.get("checkpoint_id"),
        "label": metadata.get("checkpoint_label", snapshot.get("checkpoint_id", "")),
        "narrative": snapshot.get("narrative", ""),
        "score_display": f"{home} {int(score.get('home', 0))} - {int(score.get('away', 0))} {away}",
        "home_score": int(score.get("home", 0)),
        "away_score": int(score.get("away", 0)),
        "win_probability_pct": f"{win_prob * 100:.1f}%",
        "win_probability_value": round(win_prob, 4),
        "leverage": round(leverage, 2),
        "score_differential": score_diff,
        "trend_class": _trend_class(score_diff),
        "edge_delta_pct": round((win_prob - pre_prob) * 100, 1),
        "minutes_elapsed": minutes_elapsed,
        "minutes_label": f"{int(minutes_elapsed)} min" if minutes_elapsed is not None else "—",
        "progress_pct": round(progress * 100, 1),
        "progress_label": f"{round(progress * 100)}%",
    }


def _build_context_badges(rest_advantage, record_diff, implied_prob) -> List[Dict[str, str]]:
    badges = []
    if rest_advantage is not None:
        badges.append({"label": "Rest edge", "value": f"{rest_advantage:+.0f} days"})
    if record_diff is not None:
        badges.append({"label": "Record diff", "value": f"{record_diff:+.1f}"})
    badges.append({"label": "Pregame win", "value": f"{implied_prob * 100:.1f}%"})
    return badges


def _format_live_odds_entry(raw_odds: Optional[Dict]) -> Optional[Dict]:
    if not raw_odds:
        return None

    odds = raw_odds.get("odds") or {}

    def fmt_american(value: Optional[int]) -> str:
        if value is None:
            return "—"
        sign = "+" if value > 0 else ""
        return f"{sign}{value}"

    def fmt_line(point: Optional[float], price: Optional[int]) -> str:
        if point is None and price is None:
            return "—"
        point_str = f"{point:+.1f}" if point is not None else "n/a"
        price_str = fmt_american(price)
        return f"{point_str} ({price_str})"

    def fmt_total(total_point: Optional[float], price: Optional[int]) -> str:
        if total_point is None and price is None:
            return "—"
        price_str = fmt_american(price)
        total_str = f"{total_point:.1f}" if isinstance(total_point, (int, float)) else "n/a"
        return f"{total_str} ({price_str})"

    return {
        "bookmaker": raw_odds.get("bookmaker", "Live Feed"),
        "commence_time": raw_odds.get("commence_time"),
        "commence_label": _format_date_label(
            raw_odds.get("commence_time", "").split("T")[0] if raw_odds.get("commence_time") else ""
        ),
        "moneyline_home": _safe_int(odds.get("moneyline_home")),
        "moneyline_away": _safe_int(odds.get("moneyline_away")),
        "moneyline_home_display": fmt_american(_safe_int(odds.get("moneyline_home"))),
        "moneyline_away_display": fmt_american(_safe_int(odds.get("moneyline_away"))),
        "puck_line_home": _safe_float(odds.get("puck_line_home"), None),
        "puck_line_away": _safe_float(odds.get("puck_line_away"), None),
        "puck_line_home_odds": _safe_int(odds.get("puck_line_home_odds")),
        "puck_line_away_odds": _safe_int(odds.get("puck_line_away_odds")),
        "puck_line_home_display": fmt_line(
            _safe_float(odds.get("puck_line_home"), None),
            _safe_int(odds.get("puck_line_home_odds")),
        ),
        "puck_line_away_display": fmt_line(
            _safe_float(odds.get("puck_line_away"), None),
            _safe_int(odds.get("puck_line_away_odds")),
        ),
        "total": _safe_float(odds.get("total"), None),
        "over_odds": _safe_int(odds.get("over_odds")),
        "under_odds": _safe_int(odds.get("under_odds")),
        "over_display": fmt_total(_safe_float(odds.get("total"), None), _safe_int(odds.get("over_odds"))),
        "under_display": fmt_total(_safe_float(odds.get("total"), None), _safe_int(odds.get("under_odds"))),
    }


def _hydrate_game_payload(
    game: Dict,
    snapshots: List[Dict],
    live_odds_entry: Optional[Dict],
    live_odds_label: Optional[str],
    context_patterns: Optional[List[Dict]] = None,
) -> Dict:
    home = game.get("home_team", "HOME")
    away = game.get("away_team", "AWAY")
    gid = str(game.get("game_id"))
    pregame_odds = game.get("betting_odds") or {}
    context = game.get("temporal_context") or {}

    implied_prob = _safe_float(pregame_odds.get("implied_prob_home"), 0.5)
    rest_adv = context.get("rest_advantage")
    record_diff = context.get("record_differential")

    snapshots_sorted = sorted(snapshots, key=lambda snap: snap.get("sequence", 0))
    final_snapshot = snapshots_sorted[-1]
    final_score = final_snapshot.get("score") or {}
    status = (
        "Final"
        if (final_snapshot.get("checkpoint_id", "").upper() == "FINAL")
        else final_snapshot.get("metadata", {}).get("checkpoint_label", "In-progress")
    )
    final_prob = _safe_float(final_snapshot.get("metrics", {}).get("win_probability_home"), implied_prob)
    win_shift_pct = round((final_prob - implied_prob) * 100, 1)

    snapshots_payload = [
        _format_snapshot_payload(snapshot, home, away, implied_prob) for snapshot in snapshots_sorted
    ]

    formatted_live_odds = _format_live_odds_entry(live_odds_entry)

    context_matches = _evaluate_context_matches(game, context_patterns or [])
    pi_value = (game.get("pi_metadata") or {}).get("pi_effective")

    return {
        "game_id": gid,
        "matchup": f"{away} @ {home}",
        "home_team": home,
        "away_team": away,
        "date_label": _format_date_label(game.get("date")),
        "season": game.get("season"),
        "venue": game.get("venue", "Venue TBA"),
        "status": status,
        "score": {
            "home": int(final_score.get("home", game.get("home_score", 0))),
            "away": int(final_score.get("away", game.get("away_score", 0))),
        },
        "scoreline": f"{home} {int(final_score.get('home', 0))} - {int(final_score.get('away', 0))} {away}",
        "pregame": {
            "implied_home_pct": f"{implied_prob * 100:.1f}%",
            "moneyline_home": pregame_odds.get("moneyline_home") or pregame_odds.get("home_moneyline"),
            "moneyline_away": pregame_odds.get("moneyline_away") or pregame_odds.get("away_moneyline"),
            "rest_advantage": rest_adv,
            "record_differential": record_diff,
        },
        "context_badges": _build_context_badges(rest_adv, record_diff, implied_prob),
        "snapshots": snapshots_payload,
        "checkpoint_count": len(snapshots_payload),
        "momentum_summary": _compose_momentum_summary(home, away, win_shift_pct, final_score),
        "win_shift_pct": win_shift_pct,
        "synthetic_narrative": game.get("synthetic_narrative"),
        "result_trend": (
            "home"
            if final_score.get("home", 0) > final_score.get("away", 0)
            else "away"
            if final_score.get("home", 0) < final_score.get("away", 0)
            else "even"
        ),
        "live_odds": formatted_live_odds,
        "live_odds_label": live_odds_label if formatted_live_odds else None,
        "context_matches": context_matches,
        "pi_effective": pi_value,
    }


def _load_live_odds_map() -> Tuple[Optional[str], Dict[str, Dict]]:
    if not LIVE_ODDS_PATH.exists():
        return None, {}
    try:
        with open(LIVE_ODDS_PATH) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, {}

    odds_map: Dict[str, Dict] = {}
    for game in payload.get("games", []):
        key = _odds_key(game.get("home_team"), game.get("away_team"))
        if key:
            odds_map[key] = game
    return payload.get("timestamp"), odds_map


def _load_context_patterns() -> List[Dict]:
    cache_key = "nhl_context_patterns"
    if cache_key in _cache:
        return _cache[cache_key]
    path = CONTEXT_RESULTS_DIR / "nhl_contexts.json"
    if not path.exists():
        _cache[cache_key] = []
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patterns = data.get("patterns", [])
    _cache[cache_key] = patterns
    _cache["nhl_context_summary"] = data
    return patterns


def _get_annotated_records(config) -> List[Dict]:
    cache_key = "nhl_annotated_records"
    if cache_key in _cache:
        return _cache[cache_key]
    raw_records = config.get_raw_records()
    try:
        annotated = DynamicPiCalculator("nhl").annotate(raw_records)["records"]
    except Exception:
        annotated = raw_records
    _cache[cache_key] = annotated
    return annotated


def _extract_context_features(record: Dict) -> Dict[str, float]:
    context = record.get("temporal_context") or {}
    pi_meta = record.get("pi_metadata") or {}
    return {
        "is_playoff": int(record.get("is_playoff") or 0),
        "is_rivalry": int(record.get("is_rivalry") or 0),
        "rest_advantage": float(context.get("rest_advantage") or 0.0),
        "record_differential": float(context.get("record_differential") or 0.0),
        "pi_effective": float(pi_meta.get("pi_effective") or pi_meta.get("pi_base") or 0.5),
    }


def _match_conditions(features: Dict[str, float], conditions: Dict[str, Dict]) -> bool:
    for feature, condition in conditions.items():
        value = features.get(feature)
        if value is None:
            return False
        if isinstance(condition, dict):
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
            if "eq" in condition and float(value) != float(condition["eq"]):
                return False
        else:
            if value != condition:
                return False
    return True


def _evaluate_context_matches(record: Dict, context_patterns: List[Dict]) -> List[Dict]:
    if not context_patterns:
        return []
    features = _extract_context_features(record)
    matches = []
    for pattern in context_patterns:
        conditions = pattern.get("conditions") or {}
        if _match_conditions(features, conditions):
            matches.append(
                {
                    "pattern_name": pattern.get("pattern_name"),
                    "edge": pattern.get("edge_vs_baseline"),
                    "sample_size": pattern.get("sample_size"),
                    "pi_mean": pattern.get("pi_mean"),
                }
            )
    return matches[:3]


def _build_checkpoint_payload(limit_games: int = DEFAULT_GAME_LIMIT, checkpoint: Optional[str] = None) -> Dict:
    limit_games = max(1, min(limit_games, 12))
    payload = {
        "games": [],
        "summary": "Checkpoint data unavailable",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "limit": limit_games,
        "checkpoint": checkpoint or "all",
        "checkpoint_total": 0,
    }

    config = get_domain("nhl")
    if config is None:
        payload["summary"] = "NHL domain not registered."
        payload["error"] = "nhl_domain_missing"
        return payload

    if not config.supports_checkpoints():
        payload["summary"] = "NHL checkpoints not enabled."
        payload["error"] = "checkpoints_disabled"
        return payload

    try:
        raw_records = _get_annotated_records(config)
    except FileNotFoundError:
        payload["summary"] = "NHL dataset missing. Run data ingestion."
        payload["error"] = "data_missing"
        return payload

    if not raw_records:
        payload["summary"] = "NHL dataset empty."
        payload["error"] = "data_empty"
        return payload

    odds_timestamp, odds_map = _load_live_odds_map()
    odds_label = _format_timestamp_label(odds_timestamp)

    sorted_records = sorted(
        [record for record in raw_records if record.get("game_id")],
        key=lambda rec: rec.get("date") or "",
        reverse=True,
    )

    from collections import defaultdict

    records_by_matchup: Dict[str, List[Dict]] = defaultdict(list)
    for record in raw_records:
        matchup_key = _odds_key(record.get("home_team"), record.get("away_team"))
        if matchup_key and record.get("game_id"):
            records_by_matchup[matchup_key].append(record)

    for record_list in records_by_matchup.values():
        record_list.sort(key=lambda rec: rec.get("date") or "", reverse=True)

    selected_records: List[Dict] = []
    used_ids = set()

    if odds_map:
        for key in odds_map.keys():
            if len(selected_records) >= limit_games:
                break
            record_options = records_by_matchup.get(key)
            if not record_options:
                continue
            chosen = next((rec for rec in record_options if rec.get("game_id") not in used_ids), record_options[0])
            selected_records.append(chosen)
            used_ids.add(chosen.get("game_id"))

    for record in sorted_records:
        if len(selected_records) >= limit_games:
            break
        if record.get("game_id") in used_ids:
            continue
        selected_records.append(record)
        used_ids.add(record.get("game_id"))

    builder = config.checkpoint_builder
    snapshots = builder(selected_records, checkpoint, None) if builder else []

    grouped_snapshots: Dict[str, List[Dict]] = {}
    for snapshot in snapshots:
        gid = str(snapshot.get("game_id"))
        grouped_snapshots.setdefault(gid, []).append(snapshot)

    context_patterns = _load_context_patterns()
    games_payload = []
    for record in selected_records:
        gid = str(record.get("game_id"))
        snaps = grouped_snapshots.get(gid)
        if not snaps:
            continue
        odds_entry = odds_map.get(_odds_key(record.get("home_team"), record.get("away_team")))
        games_payload.append(
            _hydrate_game_payload(record, snaps, odds_entry, odds_label, context_patterns)
        )

    payload["games"] = games_payload
    checkpoint_total = sum(game["checkpoint_count"] for game in games_payload)
    payload["checkpoint_total"] = checkpoint_total
    payload["summary"] = f"{len(games_payload)} games · {checkpoint_total} checkpoints"
    payload["live_odds_timestamp"] = odds_timestamp
    payload["live_odds_label"] = odds_label
    payload["live_odds_game_count"] = len(odds_map)
    payload["live_odds_attached"] = sum(1 for game in games_payload if game.get("live_odds"))
    context_summary = _cache.get("nhl_context_summary", {})
    if context_summary:
        payload["context_summary"] = {
            "baseline": context_summary.get("baseline_win_rate"),
            "pattern_count": len(context_patterns),
        }
    return payload


def _build_edge_rows(limit_games: int = DEFAULT_GAME_LIMIT) -> List[Dict]:
    payload = _build_checkpoint_payload(limit_games=limit_games)
    rows: List[Dict] = []
    for game in payload.get("games", []):
        snapshots = game.get("snapshots") or []
        if not snapshots:
            continue
        latest = snapshots[-1]
        model_prob = latest.get("win_probability_value") or 0.5
        pregame = game.get("pregame") or {}
        edges = EDGE_CALCULATOR.evaluate_matchup(
            model_prob,
            _safe_int(pregame.get("moneyline_home")),
            _safe_int(pregame.get("moneyline_away")),
            context_label=", ".join(match["pattern_name"] for match in game.get("context_matches", [])),
            pi_effective=game.get("pi_effective"),
        )
        rows.append(
            {
                "matchup": game.get("matchup"),
                "game_id": game.get("game_id"),
                "summary": game.get("momentum_summary"),
                "home": edges["home"],
                "away": edges["away"],
                "context_matches": game.get("context_matches", []),
                "pi_effective": game.get("pi_effective"),
                "win_shift": game.get("win_shift_pct"),
            }
        )
    return rows


@nhl_betting_bp.route('/patterns')
def betting_patterns():
    """Display validated betting patterns"""
    
    patterns_path = data_dir / 'nhl_betting_patterns_validated.json'
    
    if patterns_path.exists():
        with open(patterns_path, 'r') as f:
            data = json.load(f)
            patterns = data.get('patterns', [])
            summary = data.get('summary', {})
    else:
        # Try non-validated patterns
        alt_path = data_dir / 'nhl_betting_patterns.json'
        if alt_path.exists():
            with open(alt_path, 'r') as f:
                patterns = json.load(f)
            summary = {'note': 'Patterns not yet validated'}
        else:
            patterns = []
            summary = {'note': 'No patterns found. Run discover_nhl_patterns.py'}
    
    context_cards = _load_context_patterns()[:6]
    context_summary = _cache.get("nhl_context_summary", {})
    return render_template(
        'nhl_betting_patterns.html',
                         patterns=patterns, 
        summary=summary,
        context_cards=context_cards,
        context_summary=context_summary,
    )


@nhl_betting_bp.route('/live')
def live_betting():
    """Surface checkpoint narratives inside the NHL betting UI."""
    payload = _build_checkpoint_payload()
    payload_update = {
        'games': payload['games'],
        'summary': payload['summary'],
        'checkpoint_total': payload['checkpoint_total'],
        'generated_at': payload['generated_at'],
        'last_update_label': datetime.utcnow().strftime("%I:%M %p UTC"),
        'limit_games': payload['limit'],
        'checkpoint_filter': payload['checkpoint'],
        'error': payload.get('error'),
        'live_odds_count': payload.get('live_odds_game_count', 0),
        'live_odds_label': payload.get('live_odds_label', '—'),
        'live_odds_attached': payload.get('live_odds_attached', 0),
        'theme': NHL_THEME,
        'api_endpoint': '/nhl/betting/api/opportunities',
    }
    return render_template('nhl_live_betting.html', **payload_update)


@nhl_betting_bp.route('/edges')
def edge_dashboard():
    context_data = _cache.get("nhl_context_summary", {})
    edge_rows = _build_edge_rows(limit_games=8)
    return render_template(
        'betting_edge_dashboard.html',
        league='NHL',
        edge_rows=edge_rows,
        context_patterns=_load_context_patterns(),
        context_summary=context_data,
    )


@nhl_betting_bp.route('/api/patterns')
def api_patterns():
    """API endpoint for betting patterns"""
    
    patterns_path = data_dir / 'nhl_betting_patterns_validated.json'
    
    if patterns_path.exists():
        with open(patterns_path, 'r') as f:
            return jsonify(json.load(f))
    else:
        alt_path = data_dir / 'nhl_betting_patterns.json'
        if alt_path.exists():
            with open(alt_path, 'r') as f:
                patterns = json.load(f)
            return jsonify({'patterns': patterns, 'validated': False})
        else:
            return jsonify({'error': 'No patterns found'}), 404


@nhl_betting_bp.route('/api/opportunities')
def api_opportunities():
    """API endpoint for checkpoint-based live insights."""
    limit = request.args.get('limit', default=DEFAULT_GAME_LIMIT, type=int)
    checkpoint = request.args.get('checkpoint')
    payload = _build_checkpoint_payload(limit_games=limit or DEFAULT_GAME_LIMIT, checkpoint=checkpoint)
    payload['timestamp'] = payload['generated_at']
    return jsonify(payload)


@nhl_betting_bp.route('/health')
def health_check():
    """Health check endpoint"""
    
    return jsonify({
        'status': 'operational',
        'domain': 'nhl',
        'timestamp': datetime.now().isoformat()
    })

