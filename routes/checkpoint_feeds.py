"""
Shared Checkpoint Feed Blueprint
--------------------------------

Provides a reusable blueprint + helper utilities so any sport domain with
checkpoint narratives can expose a consistent UI/API without bespoke plumbing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from flask import Blueprint, abort, jsonify, render_template, request

from narrative_optimization.domain_registry import get_domain

project_root = Path(__file__).parent.parent
live_data_dir = project_root / "data" / "live"

DEFAULT_GAME_LIMIT = 6


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
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


def _odds_key(home: Optional[str], away: Optional[str]) -> str:
    return f"{(home or '').upper()}::{(away or '').upper()}"


def _compose_momentum_summary(
    home: str, away: str, win_shift_pct: float, final_score: Dict[str, int]
) -> str:
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
        point_str = f"{point:+.1f}" if isinstance(point, (int, float)) else "n/a"
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
        "moneyline_home": _safe_int(odds.get("moneyline_home")),
        "moneyline_away": _safe_int(odds.get("moneyline_away")),
        "moneyline_home_display": fmt_american(_safe_int(odds.get("moneyline_home"))),
        "moneyline_away_display": fmt_american(_safe_int(odds.get("moneyline_away"))),
        "puck_line_home_display": fmt_line(
            _safe_float(odds.get("puck_line_home"), None),
            _safe_int(odds.get("puck_line_home_odds")),
        ),
        "puck_line_away_display": fmt_line(
            _safe_float(odds.get("puck_line_away"), None),
            _safe_int(odds.get("puck_line_away_odds")),
        ),
        "over_display": fmt_total(_safe_float(odds.get("total"), None), _safe_int(odds.get("over_odds"))),
        "under_display": fmt_total(_safe_float(odds.get("total"), None), _safe_int(odds.get("under_odds"))),
    }


def _hydrate_game_payload(
    game: Dict,
    snapshots: List[Dict],
    live_odds_entry: Optional[Dict],
    live_odds_label: Optional[str],
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
    }


def _load_live_odds_file(file_path: Path) -> Tuple[Optional[str], Dict[str, Dict]]:
    if not file_path.exists():
        return None, {}
    try:
        with open(file_path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, {}

    odds_map: Dict[str, Dict] = {}
    for game in payload.get("games", []):
        key = _odds_key(game.get("home_team"), game.get("away_team"))
        if key:
            odds_map[key] = game
    return payload.get("timestamp"), odds_map


def _load_nhl_live_odds() -> Tuple[Optional[str], Dict[str, Dict]]:
    return _load_live_odds_file(live_data_dir / "nhl_live_odds.json")


def build_domain_checkpoint_payload(
    domain_name: str,
    limit_games: int = DEFAULT_GAME_LIMIT,
    checkpoint: Optional[str] = None,
    live_odds_loader: Optional[Callable[[], Tuple[Optional[str], Dict[str, Dict]]]] = None,
) -> Dict:
    limit_games = max(1, min(limit_games, 12))
    payload = {
        "games": [],
        "summary": "Checkpoint data unavailable",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "limit": limit_games,
        "checkpoint": checkpoint or "all",
        "checkpoint_total": 0,
    }

    config = get_domain(domain_name)
    if config is None or not config.supports_checkpoints():
        payload["summary"] = f"{domain_name.upper()} checkpoints not available."
        payload["error"] = "checkpoints_disabled"
        return payload

    try:
        raw_records = config.get_raw_records()
    except FileNotFoundError:
        payload["summary"] = f"{domain_name.upper()} dataset missing. Run data ingestion."
        payload["error"] = "data_missing"
        return payload

    if not raw_records:
        payload["summary"] = f"{domain_name.upper()} dataset empty."
        payload["error"] = "data_empty"
        return payload

    odds_timestamp = None
    odds_map: Dict[str, Dict] = {}
    if live_odds_loader:
        odds_timestamp, odds_map = live_odds_loader()

    odds_label = _format_timestamp_label(odds_timestamp)

    sorted_records = sorted(
        [record for record in raw_records if record.get("game_id")],
        key=lambda rec: rec.get("date") or "",
        reverse=True,
    )

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

    games_payload = []
    for record in selected_records:
        gid = str(record.get("game_id"))
        snaps = grouped_snapshots.get(gid)
        if not snaps:
            continue
        odds_entry = odds_map.get(_odds_key(record.get("home_team"), record.get("away_team")))
        games_payload.append(_hydrate_game_payload(record, snaps, odds_entry, odds_label))

    payload["games"] = games_payload
    checkpoint_total = sum(game["checkpoint_count"] for game in games_payload)
    payload["checkpoint_total"] = checkpoint_total
    payload["summary"] = f"{len(games_payload)} games · {checkpoint_total} checkpoints"
    payload["live_odds_timestamp"] = odds_timestamp
    payload["live_odds_label"] = odds_label
    payload["live_odds_game_count"] = len(odds_map)
    payload["live_odds_attached"] = sum(1 for game in games_payload if game.get("live_odds"))
    return payload


def _nhl_theme() -> Dict:
    return {
        "league": "NHL",
        "icon": "🏒",
        "page_title": "NHL Checkpoint Feed",
        "hero_title": "Period-by-period NHL betting intelligence",
        "hero_body": (
            "Synthetic checkpoint snapshots merge rest/record context, implied odds, "
            "and deterministic scoring splits so you can act after each period without full live data feeds."
        ),
        "cta_label": "View NHL patterns",
        "cta_href": "/nhl/betting/patterns",
        "empty_state": "Run NHL ingestion or replay builder to populate this feed.",
    }


FEED_REGISTRY: Dict[str, Dict] = {
    "nhl": {
        "domain": "nhl",
        "theme": _nhl_theme(),
        "live_odds_loader": _load_nhl_live_odds,
    },
}


def list_available_feeds() -> List[Dict]:
    feeds = []
    for key, meta in FEED_REGISTRY.items():
        feeds.append(
            {
                "key": key,
                "league": meta["theme"]["league"],
                "title": meta["theme"]["page_title"],
                "description": meta["theme"]["hero_body"],
            }
        )
    return feeds


def build_feed_context(feed_key: str, limit: int, checkpoint: Optional[str]) -> Dict:
    config = FEED_REGISTRY[feed_key]
    payload = build_domain_checkpoint_payload(
        config["domain"],
        limit_games=limit,
        checkpoint=checkpoint,
        live_odds_loader=config.get("live_odds_loader"),
    )
    payload["theme"] = config["theme"]
    payload["feed_key"] = feed_key
    payload["api_endpoint"] = f"/sports/checkpoints/{feed_key}/api"
    return payload


checkpoint_feeds_bp = Blueprint("checkpoint_feeds", __name__, url_prefix="/sports/checkpoints")


@checkpoint_feeds_bp.route("/")
def checkpoint_index():
    feeds = list_available_feeds()
    return render_template("checkpoints/index.html", feeds=feeds)


@checkpoint_feeds_bp.route("/<feed_key>")
def checkpoint_feed_page(feed_key: str):
    feed_key = feed_key.lower()
    if feed_key not in FEED_REGISTRY:
        abort(404)
    limit = request.args.get("limit", default=DEFAULT_GAME_LIMIT, type=int) or DEFAULT_GAME_LIMIT
    checkpoint = request.args.get("checkpoint")
    context = build_feed_context(feed_key, limit, checkpoint)
    context.setdefault("last_update_label", datetime.utcnow().strftime("%I:%M %p UTC"))
    return render_template("checkpoints/feed.html", **context)


@checkpoint_feeds_bp.route("/<feed_key>/api")
def checkpoint_feed_api(feed_key: str):
    feed_key = feed_key.lower()
    if feed_key not in FEED_REGISTRY:
        abort(404)
    limit = request.args.get("limit", default=DEFAULT_GAME_LIMIT, type=int) or DEFAULT_GAME_LIMIT
    checkpoint = request.args.get("checkpoint")
    config = FEED_REGISTRY[feed_key]
    payload = build_domain_checkpoint_payload(
        config["domain"],
        limit_games=limit,
        checkpoint=checkpoint,
        live_odds_loader=config.get("live_odds_loader"),
    )
    payload["timestamp"] = payload["generated_at"]
    return jsonify(payload)

