"""
Domain Target Builders
----------------------

Shared feature/target row builders used by multi-target modeling.
Extracted from the legacy context discovery script so they can be referenced
from `TargetConfig` definitions and other pipelines.
"""

from __future__ import annotations

from typing import Dict, Optional

def _hash_to_unit(value: Optional[str]) -> float:
    if not value:
        return 0.0
    return (hash(value) % 1000) / 1000.0


def _compute_moneyline_roi(odds: Optional[int], outcome: Optional[int], stake: float = 100.0) -> Optional[float]:
    if odds is None or outcome is None:
        return None
    payout = stake * (100 / abs(odds)) if odds < 0 else stake * (odds / 100)
    profit = payout if outcome else -stake
    return profit / stake


def build_nhl_game_win_row(record: Dict) -> Optional[Dict]:
    context = record.get("temporal_context") or {}
    odds = record.get("betting_odds") or {}
    if record.get("home_won") is None:
        return None
    return {
        "is_playoff": int(record.get("is_playoff") or 0),
        "is_rivalry": int(record.get("is_rivalry") or 0),
        "rest_advantage": float(context.get("rest_advantage") or 0.0),
        "record_differential": float(context.get("record_differential") or 0.0),
        "spread_line": float(odds.get("puck_line_home") or 0.0),
        "implied_prob": float(odds.get("implied_prob_home") or 0.5),
        "target": int(record.get("home_won")),
    }


def build_nhl_roi_row(record: Dict) -> Optional[Dict]:
    base = build_nhl_game_win_row(record)
    if not base:
        return None
    odds = (record.get("betting_odds") or {}).get("moneyline_home")
    roi = _compute_moneyline_roi(odds, record.get("home_won"))
    if roi is None:
        return None
    base["target"] = float(roi)
    return base


def build_nfl_game_win_row(record: Dict) -> Optional[Dict]:
    context = record.get("temporal_context") or {}
    if record.get("home_won") is None:
        return None
    return {
        "is_playoff": int(record.get("playoff") or record.get("is_playoff") or 0),
        "is_rivalry": int(record.get("div_game") or record.get("is_rivalry") or 0),
        "rest_advantage": float(context.get("rest_advantage") or record.get("rest_advantage") or 0.0),
        "record_differential": float(context.get("record_differential") or 0.0),
        "spread_line": float(record.get("spread_line") or 0.0),
        "prime_time": int(record.get("prime_time") or 0),
        "week": int(record.get("week") or 0),
        "target": int(record.get("home_won")),
    }


def build_nfl_roi_row(record: Dict) -> Optional[Dict]:
    base = build_nfl_game_win_row(record)
    if not base:
        return None
    odds = record.get("moneyline_home") or (record.get("betting_odds") or {}).get("moneyline_home")
    roi = _compute_moneyline_roi(odds, record.get("home_won"))
    if roi is None:
        return None
    base["target"] = float(roi)
    return base


def build_nba_game_win_row(record: Dict) -> Optional[Dict]:
    if record.get("won") is None:
        return None
    context = record.get("temporal_context") or {}
    return {
        "season_win_pct": float(context.get("season_win_pct") or 0.0),
        "l10_win_pct": float(context.get("l10_win_pct") or 0.0),
        "games_played": float(context.get("games_played") or 0.0),
        "home_game": int(record.get("home_game") or 0),
        "points": float(record.get("points") or 0.0),
        "plus_minus": float(record.get("plus_minus") or 0.0),
        "target": int(record.get("won")),
    }


def build_nba_margin_row(record: Dict) -> Optional[Dict]:
    base = build_nba_game_win_row(record)
    if not base:
        return None
    margin = record.get("plus_minus")
    if margin is None:
        return None
    base["target"] = float(margin)
    return base


def build_mlb_game_win_row(record: Dict) -> Optional[Dict]:
    winner = record.get("winner")
    if winner not in {"home", "away"}:
        return None
    home = record.get("home_team") or {}
    away = record.get("away_team") or {}
    home_record = home.get("record") or {}
    away_record = away.get("record") or {}
    home_games = max(1, (home_record.get("wins") or 0) + (home_record.get("losses") or 0))
    away_games = max(1, (away_record.get("wins") or 0) + (away_record.get("losses") or 0))
    home_win_pct = (home_record.get("wins") or 0) / home_games
    away_win_pct = (away_record.get("wins") or 0) / away_games
    return {
        "home_win_pct": float(home_win_pct),
        "away_win_pct": float(away_win_pct),
        "score_differential": float(record.get("home_score", 0) - record.get("away_score", 0)),
        "venue_hash": _hash_to_unit((record.get("venue") or {}).get("name")),
        "target": int(winner == "home"),
    }


def build_golf_win_row(record: Dict) -> Optional[Dict]:
    won = record.get("won_tournament")
    if won is None:
        return None
    rounds = record.get("rounds") or []
    final_round = rounds[-1] if rounds else 0
    return {
        "is_major": int(record.get("is_major") or 0),
        "player_prestige": float(record.get("player_prestige") or 0.0),
        "player_majors": float(record.get("player_majors") or 0.0),
        "course_difficulty": float(record.get("course_difficulty") or 0.0),
        "world_ranking": float(record.get("world_ranking_before") or 0.0),
        "final_round": float(final_round),
        "made_cut": int(record.get("made_cut") or 0),
        "target": int(bool(won)),
    }


def build_startup_success_row(record: Dict) -> Optional[Dict]:
    if record.get("successful") is None:
        return None
    yc_batch = record.get("yc_batch") or ""
    yc_year = "".join(filter(str.isdigit, yc_batch)) or "0"
    return {
        "founder_count": float(record.get("founder_count") or 0),
        "total_funding": float(record.get("total_funding_usd") or 0.0),
        "years_active": float(record.get("years_active") or 0.0),
        "market_hash": _hash_to_unit(record.get("market_category")),
        "location_hash": _hash_to_unit(record.get("location")),
        "yc_year": float(yc_year),
        "target": int(record.get("successful")),
    }


def build_startup_funding_row(record: Dict) -> Optional[Dict]:
    base = build_startup_success_row(record)
    if not base:
        return None
    funding = record.get("total_funding_usd")
    if funding is None:
        return None
    base["target"] = float(funding)
    return base


def build_supreme_court_outcome_row(record: Dict) -> Optional[Dict]:
    outcome = record.get("outcome") or {}
    metadata = record.get("metadata") or {}
    if outcome.get("winner") is None:
        return None
    return {
        "vote_margin": float(outcome.get("vote_margin") or 0.0),
        "unanimous": int(outcome.get("unanimous") or 0),
        "precedent_setting": int(outcome.get("precedent_setting") or 0),
        "overturned": int(outcome.get("overturned") or 0),
        "citation_count": float(outcome.get("citation_count") or metadata.get("citation_count") or 0),
        "target": int(outcome.get("winner") == "petitioner"),
    }


def build_supreme_court_citation_row(record: Dict) -> Optional[Dict]:
    base = build_supreme_court_outcome_row(record)
    if not base:
        return None
    citation = record.get("outcome", {}).get("citation_count") or (record.get("metadata") or {}).get("citation_count")
    if citation is None:
        return None
    base["target"] = float(citation)
    return base


def build_wikiplots_impact_row(record: Dict) -> Optional[Dict]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    narrative = record.get("narrative", "")
    return {
        "narrative_length": float(len(narrative)),
        "word_count": float(len(narrative.split())),
        "sentence_count": float(narrative.count(".") + narrative.count("!") + narrative.count("?")),
        "title_hash": _hash_to_unit(record.get("title", "")),
        "target": float(impact),
    }


def build_stereotropes_impact_row(record: Dict) -> Optional[Dict]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    narrative = record.get("narrative", "")
    category = record.get("category", "")
    return {
        "narrative_length": float(len(narrative)),
        "category_hash": _hash_to_unit(category),
        "has_labels": int(bool(record.get("labels"))),
        "target": float(impact),
    }


def build_ml_research_impact_row(record: Dict) -> Optional[Dict]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    narrative = record.get("narrative", "")
    return {
        "narrative_length": float(len(narrative)),
        "word_count": float(len(narrative.split())),
        "citation_density": float(narrative.count("[")),
        "link_density": float(narrative.count("http")),
        "target": float(impact),
    }


def build_cmu_movies_revenue_row(record: Dict) -> Optional[Dict]:
    impact = record.get("impact_score")
    narrative = record.get("narrative", "")
    revenue = record.get("revenue") or record.get("box_office")
    if impact is None and revenue is None:
        return None
    return {
        "narrative_length": float(len(narrative)),
        "word_count": float(len(narrative.split())),
        "sentence_count": float(narrative.count(".") + narrative.count("!") + narrative.count("?")),
        "title_hash": _hash_to_unit(record.get("title", "")),
        "target": float(revenue if revenue is not None else impact),
    }

