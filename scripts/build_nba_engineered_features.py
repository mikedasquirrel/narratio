#!/usr/bin/env python3
"""
Engineer scoreboard-based NBA features aligned with closing odds.

Input:  data/modeling_datasets/nba_games_with_closing_odds.jsonl
Output: data/modeling_datasets/nba_engineered_features.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
ODDS_FILE = DATA_ROOT / "modeling_datasets" / "nba_games_with_closing_odds.jsonl"
OUTPUT_FILE = DATA_ROOT / "modeling_datasets" / "nba_engineered_features.jsonl"


def parse_time(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(cleaned).astimezone(timezone.utc)


BASE_ELO = 1500.0
ELO_K = 18.0
ELO_REGRESSION = 0.8


def ensure_team_elo(
    team: str,
    season_year: int,
    elo_ratings: Dict[str, float],
    last_seen: Dict[str, int],
) -> float:
    rating = elo_ratings.get(team, BASE_ELO)
    prev_year = last_seen.get(team)
    if prev_year is None or season_year == prev_year:
        last_seen[team] = season_year
        elo_ratings[team] = rating
        return rating
    if season_year > prev_year:
        rating = rating * ELO_REGRESSION + BASE_ELO * (1 - ELO_REGRESSION)
        elo_ratings[team] = rating
        last_seen[team] = season_year
    return rating


def update_elo_ratings(
    elo_ratings: Dict[str, float],
    home_team: str,
    away_team: str,
    home_score: Optional[int],
    away_score: Optional[int],
    home_rating: float,
    away_rating: float,
) -> None:
    if home_score is None or away_score is None:
        return
    if home_score == away_score:
        result = 0.5
    else:
        result = 1.0 if home_score > away_score else 0.0
    expected_home = 1.0 / (1.0 + 10 ** ((away_rating - home_rating) / 400.0))
    margin = abs(home_score - away_score)
    margin_mult = max(1.0, (margin + 3) ** 0.7)
    delta = ELO_K * margin_mult * (result - expected_home)
    elo_ratings[home_team] = home_rating + delta
    elo_ratings[away_team] = away_rating - delta


def result_to_bool(value: Optional[float]) -> Optional[bool]:
    if value is None:
        return None
    if abs(value - 0.5) < 1e-6:
        return None
    return value > 0.5


def invert_result(value: Optional[bool]) -> Optional[bool]:
    if value is None:
        return None
    return not value


@dataclass
class GameHistoryEntry:
    event_time: datetime
    points_for: int
    points_against: int
    won: bool
    cover: Optional[bool]


@dataclass
class TeamHistory:
    games: List[GameHistoryEntry] = field(default_factory=list)

    def add_game(
        self,
        event_time: datetime,
        points_for: Optional[int],
        points_against: Optional[int],
        cover: Optional[bool],
    ) -> None:
        if points_for is None or points_against is None:
            return
        self.games.append(
            GameHistoryEntry(
                event_time=event_time,
                points_for=points_for,
                points_against=points_against,
                won=points_for > points_against,
                cover=cover,
            )
        )

    def games_played(self) -> int:
        return len(self.games)

    def rest_days(self, event_time: datetime) -> Optional[float]:
        if not self.games:
            return None
        delta = event_time - self.games[-1].event_time
        return delta.total_seconds() / 86400.0

    def streak(self) -> int:
        streak = 0
        for entry in reversed(self.games[-10:]):
            if entry.won:
                streak = streak + 1 if streak >= 0 else 1
            else:
                streak = streak - 1 if streak <= 0 else -1
        return streak

    def avg_margin(self, window: int) -> Optional[float]:
        subset = self.games[-window:]
        if not subset:
            return None
        margins = [g.points_for - g.points_against for g in subset]
        return sum(margins) / len(margins)

    def win_pct(self, window: int) -> Optional[float]:
        subset = self.games[-window:]
        if not subset:
            return None
        wins = sum(1 for g in subset if g.won)
        return wins / len(subset)

    def cover_pct(self, window: int) -> Optional[float]:
        subset = [g for g in self.games[-window:] if g.cover is not None]
        if not subset:
            return None
        covers = sum(1 for g in subset if g.cover)
        return covers / len(subset)

    def points_avg(self, window: int) -> Optional[float]:
        subset = self.games[-window:]
        if not subset:
            return None
        return sum(g.points_for for g in subset) / len(subset)

    def points_allowed_avg(self, window: int) -> Optional[float]:
        subset = self.games[-window:]
        if not subset:
            return None
        return sum(g.points_against for g in subset) / len(subset)


def safe_float(value: Optional[float]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_feature_rows(games: List[Dict[str, object]]) -> List[Dict[str, object]]:
    teams: Dict[str, TeamHistory] = {}
    elo_ratings: Dict[str, float] = {}
    last_season_seen: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    for game in games:
        event_time = game["event_dt"]
        home = game["home_team"]
        away = game["away_team"]
        season_year = game["season_year"]

        home_hist = teams.setdefault(home, TeamHistory())
        away_hist = teams.setdefault(away, TeamHistory())

        home_elo = ensure_team_elo(home, season_year, elo_ratings, last_season_seen)
        away_elo = ensure_team_elo(away, season_year, elo_ratings, last_season_seen)

        row = {
            "event_time": game["event_time"],
            "season_year": season_year,
            "neutral_site": game["neutral_site"],
            "closing_moneyline_home": safe_float(game["closing_moneyline_home"]),
            "closing_moneyline_home_implied": safe_float(game["closing_moneyline_home_implied"]),
            "closing_moneyline_away": safe_float(game["closing_moneyline_away"]),
            "closing_spread_home": safe_float(game["closing_spread_home"]),
            "closing_spread_home_price": safe_float(game["closing_spread_home_price"]),
            "closing_spread_away_price": safe_float(game["closing_spread_away_price"]),
            "closing_total": safe_float(game["closing_total"]),
            "closing_total_price": safe_float(game["closing_total_price"]),
            "total_under_price": safe_float(game["total_under_price"]),
            "home_games_played": home_hist.games_played(),
            "away_games_played": away_hist.games_played(),
            "home_rest_days": home_hist.rest_days(event_time),
            "away_rest_days": away_hist.rest_days(event_time),
            "home_avg_margin_last5": home_hist.avg_margin(5),
            "away_avg_margin_last5": away_hist.avg_margin(5),
            "home_win_pct_last5": home_hist.win_pct(5),
            "away_win_pct_last5": away_hist.win_pct(5),
            "home_win_pct_last10": home_hist.win_pct(10),
            "away_win_pct_last10": away_hist.win_pct(10),
            "home_points_avg_last5": home_hist.points_avg(5),
            "away_points_avg_last5": away_hist.points_avg(5),
            "home_points_allowed_avg_last5": home_hist.points_allowed_avg(5),
            "away_points_allowed_avg_last5": away_hist.points_allowed_avg(5),
            "home_streak": home_hist.streak(),
            "away_streak": away_hist.streak(),
            "home_cover_pct_last5": home_hist.cover_pct(5),
            "away_cover_pct_last5": away_hist.cover_pct(5),
            "home_cover_pct_last10": home_hist.cover_pct(10),
            "away_cover_pct_last10": away_hist.cover_pct(10),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "home_margin": game["home_margin"],
            "moneyline_result": game["moneyline_result"],
            "spread_result": game["spread_result"],
            "total_result": game["total_result"],
            "home_team": home,
            "away_team": away,
            "espn_game_id": game["espn_game_id"],
        }
        rows.append(row)

        home_cover_flag = result_to_bool(game.get("spread_result"))
        away_cover_flag = invert_result(home_cover_flag)

        home_hist.add_game(event_time, game["home_score"], game["away_score"], home_cover_flag)
        away_hist.add_game(event_time, game["away_score"], game["home_score"], away_cover_flag)

        update_elo_ratings(
            elo_ratings,
            home,
            away,
            game.get("home_score"),
            game.get("away_score"),
            home_elo,
            away_elo,
        )

    return rows


def load_games() -> List[Dict[str, object]]:
    if not ODDS_FILE.exists():
        raise FileNotFoundError("Run build_nba_modeling_dataset.py first.")
    games: List[Dict[str, object]] = []
    with ODDS_FILE.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("home_team") == "Los Angeles Clippers":
                entry["home_team"] = "LA Clippers"
            if entry.get("away_team") == "Los Angeles Clippers":
                entry["away_team"] = "LA Clippers"
            entry["event_dt"] = parse_time(entry["event_time"])
            games.append(entry)
    games.sort(key=lambda g: g["event_dt"])
    return games


def main() -> None:
    games = load_games()
    rows = build_feature_rows(games)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        for row in rows:
            serializable = {k: (v if not isinstance(v, float) or not (v != v) else None) for k, v in row.items()}
            f.write(json.dumps(serializable) + "\n")
    print(f"Wrote {len(rows)} engineered NBA feature rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

