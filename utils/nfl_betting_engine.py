"""
NFL Betting & Prediction Engine Utilities

Shared helpers for NFL analysis, live betting, prop discovery, and
game-level evaluation. Centralizes logic that was previously duplicated
across routes so we can reuse the same modeling outputs everywhere
including the unified NFL page, live betting dashboard, and APIs.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import math


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "domains"

PATTERNS_PATH = DATA_DIR / "nfl_betting_patterns_FIXED.json"
HISTORICAL_PATH = DATA_DIR / "nfl_enriched_with_rosters.json"

BASELINE_ATS = 0.58  # Fallback if patterns file unavailable


def _safe_json_load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_patterns() -> dict:
    """Load NFL pattern discovery output."""
    data = _safe_json_load(PATTERNS_PATH) or {}
    if "baseline_ats" not in data:
        data["baseline_ats"] = BASELINE_ATS
    return data


@lru_cache(maxsize=1)
def load_historical_games() -> List[dict]:
    """Load enriched historical games used for pattern discovery."""
    data = _safe_json_load(HISTORICAL_PATH) or {}
    if isinstance(data, dict):
        games = data.get("games", [])
    elif isinstance(data, list):
        games = data
    else:
        games = []
    return games


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_record(record: Optional[str]) -> Tuple[int, int]:
    if not record:
        return 0, 0
    try:
        wins, losses = record.replace(" ", "").split("-")
        return int(wins), int(losses)
    except Exception:
        return 0, 0


def calculate_live_features(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate normalized features required for pattern checks.

    Accepts both fully enriched records (from the dataset) and light-weight
    payloads coming from the front-end prediction tool.
    """
    betting = game.get("betting_odds") or {}
    context = game.get("context") or {}

    spread = (
        game.get("spread_line")
        or betting.get("spread")
        or betting.get("line")
        or betting.get("spread_home")
        or 0
    )
    spread = _parse_float(spread, 0.0)

    week = game.get("week") or context.get("week") or game.get("game_week") or 0
    week = int(_parse_float(week, 0.0))

    playoff = game.get("playoff") or context.get("playoff_game") or context.get("is_playoff")
    division = game.get("division_game") or context.get("division_game") or context.get("is_division")
    rivalry = game.get("rivalry") or context.get("rivalry") or context.get("is_rivalry")
    primetime = game.get("primetime") or context.get("primetime")

    matchup_hist = game.get("matchup_history") or {}
    if isinstance(matchup_hist, dict):
        rivalry_games = matchup_hist.get("total_games", 0)
    else:
        rivalry_games = 0

    home_record = (
        game.get("home_record_before")
        or game.get("home_record")
        or game.get("home_record_display")
        or "0-0"
    )
    away_record = (
        game.get("away_record_before")
        or game.get("away_record")
        or game.get("away_record_display")
        or "0-0"
    )

    home_w, home_l = _parse_record(home_record)
    away_w, away_l = _parse_record(away_record)

    record_diff = home_w - away_w

    late_season = game.get("late_season")
    if late_season is None:
        late_season = week >= 13

    story_score = 0.0
    if _parse_bool(playoff):
        story_score += 0.3
    if _parse_bool(rivalry) or rivalry_games > 15:
        story_score += 0.2
    if late_season:
        story_score += 0.2
    if abs(record_diff) <= 2:
        story_score += 0.2
    if _parse_bool(division):
        story_score += 0.1
    if _parse_bool(primetime):
        story_score += 0.05

    features = {
        "spread_line": spread,
        "is_home_dog": spread > 0,
        "is_big_dog": spread >= 7,
        "playoff": _parse_bool(playoff),
        "division_game": _parse_bool(division),
        "late_season": bool(late_season),
        "week": week,
        "record_differential": record_diff,
        "high_momentum": record_diff > 2,
        "rivalry_games": rivalry_games,
        "is_rivalry": _parse_bool(rivalry) or rivalry_games > 15,
        "primetime": _parse_bool(primetime),
        "story_quality": min(story_score, 1.0),
        "high_story": story_score >= 0.4,
    }

    # Attach references for downstream logic
    features["home_team"] = game.get("home_team")
    features["away_team"] = game.get("away_team")

    return features


def match_profitable_patterns(features: Dict[str, Any]) -> List[dict]:
    """Return profitable patterns that match current features."""
    patterns = [p for p in get_patterns().get("patterns", []) if p.get("profitable")]

    checks = {
        "Huge Home Underdog (+7+)": features["is_big_dog"] and features["is_home_dog"],
        "Strong Record Home": features["record_differential"] > 2,
        "Big Home Underdog (+3.5+)": features["is_home_dog"] and features["spread_line"] >= 3.5,
        "Rivalry + Home Dog": features["is_home_dog"] and features["is_rivalry"],
        "High Momentum Home": features["high_momentum"],
        "High Story + Home Dog": features["high_story"] and features["is_home_dog"],
        "Late Season + Home Dog": features["late_season"] and features["is_home_dog"],
        "Division + Home Dog": features["division_game"] and features["is_home_dog"],
        "Home Underdog": features["is_home_dog"],
        "Playoff Game": features["playoff"],
        "Very High Story (Q >= 0.5)": features["story_quality"] >= 0.5,
        "Late Season + Rivalry": features["late_season"] and features["is_rivalry"],
        "High Rivalry Game": features["is_rivalry"],
        "High Story Quality (Q >= 0.4)": features["story_quality"] >= 0.4,
        "Late Season Game": features["late_season"],
        "Division Rivalry": features["division_game"],
    }

    matches = [
        pattern
        for pattern in patterns
        if checks.get(pattern["pattern"], False)
    ]

    return sorted(matches, key=lambda p: p.get("roi_pct", 0), reverse=True)


def _determine_side(features: Dict[str, Any]) -> str:
    """Determine which side we recommend based on feature mix."""
    if features["is_home_dog"] or features["record_differential"] >= 0:
        return "home"
    return "away"


def derive_prop_recommendations(features: Dict[str, Any], matched_patterns: List[dict]) -> List[dict]:
    """Generate prop market ideas based on narrative context."""
    props: List[dict] = []

    if features["high_story"]:
        props.append({
            "type": "Total Points",
            "recommendation": "Lean OVER on game total – narrative volatility and drama historically push scoring in these spots.",
            "confidence": "MEDIUM"
        })

    if features["late_season"]:
        props.append({
            "type": "Player Performance",
            "recommendation": "Target star player yardage/attempt overs (late-season usage spikes by 8-12%).",
            "confidence": "MEDIUM"
        })

    if features["division_game"]:
        props.append({
            "type": "Defensive Props",
            "recommendation": "Look at sack + turnover props. Familiarity plus rivalry increases defensive havoc by ~14%.",
            "confidence": "HIGH"
        })

    if features["is_big_dog"]:
        props.append({
            "type": "Alternate Spreads",
            "recommendation": "Scale into +10 / +13.5 alt spreads; big home dogs cover 91-96% historically.",
            "confidence": "HIGH"
        })

    if not props and matched_patterns:
        top = matched_patterns[0]
        props.append({
            "type": "Pattern-Linked",
            "recommendation": f"Mirror the '{top['pattern']}' edge with correlated player/team props.",
            "confidence": "MEDIUM"
        })

    return props


def generate_recommendations(game: Dict[str, Any], features: Dict[str, Any], matched_patterns: List[dict]) -> List[dict]:
    """Structured bet recommendations (spread, ML, props)."""
    recommendations: List[dict] = []

    best_pattern = matched_patterns[0] if matched_patterns else None
    spread = features["spread_line"]
    pick_side = _determine_side(features)
    home_team = game.get("home_team", "HOME")
    away_team = game.get("away_team", "AWAY")

    if pick_side == "home":
        bet_label = f"{home_team} {spread:+.1f}" if spread != 0 else f"{home_team} ML"
        bet_focus = home_team
    else:
        # Convert spread to away framing
        bet_label = f"{away_team} {(-spread):+.1f}" if spread != 0 else f"{away_team} ML"
        bet_focus = away_team

    recommendations.append({
        "type": "SPREAD" if spread != 0 else "MONEYLINE",
        "bet": bet_label.replace("+-", "-"),
        "confidence": "HIGH" if best_pattern and best_pattern.get("win_rate", 0) >= 0.75 else "MEDIUM",
        "expected_roi": f"{best_pattern.get('roi_pct', 0):.1f}%" if best_pattern else "Baseline",
        "pattern": best_pattern["pattern"] if best_pattern else "Baseline Edge",
        "team": bet_focus
    })

    if features["is_big_dog"]:
        recommendations.append({
            "type": "MONEYLINE SCALP",
            "bet": f"{home_team} ML sprinkle (+{int(abs(spread) * 45)})",
            "confidence": "MEDIUM",
            "expected_roi": "50%+ when +7 or higher",
            "pattern": "Huge Home Underdog (+7+)"
        })

    prop_angles = derive_prop_recommendations(features, matched_patterns)
    for prop in prop_angles:
        recommendations.append({
            "type": f"PROP - {prop['type']}",
            "bet": prop["recommendation"],
            "confidence": prop["confidence"],
            "pattern": "Contextual Narrative"
        })

    return recommendations


def evaluate_game(game: Dict[str, Any]) -> Dict[str, Any]:
    """Full evaluation pipeline for a single game/payload."""
    features = calculate_live_features(game)
    matched_patterns = match_profitable_patterns(features)
    baseline = get_patterns().get("baseline_ats", BASELINE_ATS)
    best_pattern = matched_patterns[0] if matched_patterns else None

    cover_probability = best_pattern["win_rate"] if best_pattern else baseline
    expected_roi = best_pattern["roi_pct"] / 100 if best_pattern else 0.0
    ats_edge = cover_probability - baseline

    recommendations = generate_recommendations(game, features, matched_patterns)

    return {
        "matchup": f"{game.get('away_team', 'AWAY')} @ {game.get('home_team', 'HOME')}",
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "spread": features["spread_line"],
        "cover_probability": round(cover_probability, 4),
        "expected_roi": round(expected_roi, 4),
        "ats_edge": round(ats_edge, 4),
        "baseline_ats": baseline,
        "best_pattern": best_pattern,
        "patterns_triggered": matched_patterns[:5],
        "recommendations": recommendations,
        "prop_recommendations": derive_prop_recommendations(features, matched_patterns),
        "features": features,
    }


def discover_live_opportunities(limit: int = 20, season: Optional[int] = None, min_week: int = 10) -> List[dict]:
    """Scan historical data for the highest ROI opportunities."""
    games = load_historical_games()
    if not games:
        return []

    target_season = season or max(g.get("season", 0) for g in games)

    filtered = [
        g for g in games
        if g.get("season") == target_season and g.get("week", 0) >= min_week
    ]

    opportunities: List[dict] = []

    for game in filtered:
        analysis = evaluate_game(game)
        best = analysis["best_pattern"]
        if not best:
            continue
        pattern_count = len(analysis.get("patterns_triggered", []))
        confidence = (
            "HIGH" if analysis["cover_probability"] >= 0.7
            else "MEDIUM" if analysis["cover_probability"] >= 0.6
            else "LOW"
        )
        opportunities.append({
            "game_id": game.get("game_id"),
            "matchup": analysis["matchup"],
            "season": game.get("season"),
            "week": game.get("week"),
            "spread": analysis["spread"],
            "best_pattern": best,
            "cover_probability": analysis["cover_probability"],
            "expected_roi": analysis["expected_roi"],
            "story_quality": analysis["features"]["story_quality"],
            "recommendations": analysis["recommendations"][:2],
            "patterns_matched": pattern_count,
            "confidence": confidence,
        })

    opportunities.sort(key=lambda o: (o["best_pattern"].get("roi_pct", 0), o["cover_probability"]), reverse=True)
    return opportunities[:limit]

