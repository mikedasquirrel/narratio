"""
Unified Live Feed Service
=========================

Provides a single interface to gather live game states + bookmaker odds
across multiple sports. Each sport registers a fetcher that knows how to
pull the scoreboard (from league APIs) and normalize team names, while
the odds fetcher hits The Odds API (FanDuel by default) for both moneyline
and prop markets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import requests
import re
from datetime import datetime

from config.odds_api_config import BASE_URL, ODDS_API_KEY, SPORTS


@dataclass
class LiveGameState:
    sport: str
    home_team: str
    away_team: str
    period: str
    time_remaining: float  # minutes
    home_score: int
    away_score: int
    period_scores: List[Dict[str, int]] = field(default_factory=list)
    live_odds: Dict[str, Optional[float]] = field(default_factory=dict)
    props: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


class LiveFeedService:
    def __init__(self, bookmaker: str = "fanduel"):
        self.bookmaker = bookmaker
        self.session = requests.Session()

    def fetch_odds(self, sport_key: str, markets: str = "h2h") -> Dict[str, Dict]:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": markets,
            "oddsFormat": "american",
            "bookmakers": self.bookmaker,
            "dateFormat": "iso",
        }
        resp = self.session.get(f"{BASE_URL}/sports/{sport_key}/odds", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        lookup: Dict[str, Dict] = {}
        for event in data:
            key = f"{event.get('away_team')}@{event.get('home_team')}"
            prices = {}
            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != self.bookmaker:
                    continue
                for market in bookmaker.get("markets", []):
                    mk = market.get("key")
                    if mk not in prices:
                        prices[mk] = {}
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        prices[mk][name] = outcome.get("price")
            lookup[key] = prices
        return lookup

    def fetch_game_states(self, sport: str) -> List[LiveGameState]:
        fetcher = SPORT_FETCHERS.get(sport)
        if not fetcher:
            raise ValueError(f"No live fetcher registered for sport '{sport}'")
        return fetcher(self)


def _clock_to_minutes(clock: str) -> float:
    if not clock:
        return 0.0
    clock = clock.strip()
    if clock.upper() in {"END", "FINAL"}:
        return 0.0
    if clock.startswith("PT"):
        minutes = 0.0
        match = re.match(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", clock)
        if match:
            mins = match.group(1)
            secs = match.group(2)
            if mins:
                minutes += float(mins)
            if secs:
                minutes += float(secs) / 60.0
        return minutes
    if ":" in clock:
        mins, secs = clock.split(":")
        try:
            return float(mins) + float(secs) / 60.0
        except ValueError:
            return 0.0
    return 0.0


def fetch_nhl_states(service: LiveFeedService) -> List[LiveGameState]:
    date_str = time.strftime("%Y-%m-%d")
    schedule = requests.get(
        "https://statsapi.web.nhl.com/api/v1/schedule", params={"date": date_str}
    ).json()
    games = schedule.get("dates", [{}])[0].get("games", [])
    odds_lookup = service.fetch_odds(SPORTS["nhl"], markets="h2h,player_shots")

    states = []
    for g in games:
        status = g.get("status", {}).get("detailedState", "")
        if not status.startswith("In Progress"):
            continue
        game_pk = g.get("gamePk")
        linescore = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_pk}/linescore").json()
        home = linescore["teams"]["home"]["team"]["name"]
        away = linescore["teams"]["away"]["team"]["name"]
        period_scores = [
            {"home": p.get("home", {}).get("goals", 0), "away": p.get("away", {}).get("goals", 0)}
            for p in linescore.get("periods", [])
        ]
        clock_str = linescore.get("currentPeriodTimeRemaining", "")
        time_remaining = _clock_to_minutes(clock_str)
        h2h = odds_lookup.get(key, {}).get("h2h", {})
        state = LiveGameState(
            sport="nhl",
            home_team=home,
            away_team=away,
            period=str(linescore.get("currentPeriod", 1)),
            time_remaining=time_remaining,
            home_score=linescore["teams"]["home"]["goals"],
            away_score=linescore["teams"]["away"]["goals"],
            period_scores=period_scores,
            live_odds={
                "home_moneyline": h2h.get(home),
                "away_moneyline": h2h.get(away),
            },
            meta={"gamePk": game_pk},
        )
        states.append(state)
    return states


def fetch_nba_states(service: LiveFeedService) -> List[LiveGameState]:
    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    data = requests.get(url, timeout=15).json()
    games = data.get("scoreboard", {}).get("games", [])
    odds_lookup = service.fetch_odds(SPORTS["nba"], markets="h2h")
    states = []
    for game in games:
        if game.get("gameStatus") != 2:
            continue
        home = game["homeTeam"]["teamName"]
        away = game["awayTeam"]["teamName"]
        clock = game.get("gameClock")
        time_remaining = _clock_to_minutes(clock)
        period = game.get("period")
        key = f"{away}@{home}"
        h2h = odds_lookup.get(key, {}).get("h2h", {})
        state = LiveGameState(
            sport="nba",
            home_team=home,
            away_team=away,
            period=str(period),
            time_remaining=time_remaining,
            home_score=int(game["homeTeam"]["score"]),
            away_score=int(game["awayTeam"]["score"]),
            live_odds={
                "home_moneyline": h2h.get(home),
                "away_moneyline": h2h.get(away),
            },
            meta={"gameId": game.get("gameId"), "clock": clock},
        )
        states.append(state)
    return states


def fetch_nfl_states(service: LiveFeedService) -> List[LiveGameState]:
    date_str = datetime.utcnow().strftime("%Y%m%d")
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    data = requests.get(url, params={"dates": date_str}, timeout=15).json()
    events = data.get("events", [])
    odds_lookup = service.fetch_odds(SPORTS["nfl"], markets="h2h")
    states = []
    for event in events:
        status = event.get("status", {}).get("type", {}).get("state")
        if status != "in":
            continue
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home_comp or not away_comp:
            continue
        home = home_comp["team"]["displayName"]
        away = away_comp["team"]["displayName"]
        clock = comp.get("status", {}).get("displayClock", "")
        time_remaining = _clock_to_minutes(clock)
        period = comp.get("status", {}).get("period", 1)
        key = f"{away}@{home}"
        h2h = odds_lookup.get(key, {}).get("h2h", {})
        state = LiveGameState(
            sport="nfl",
            home_team=home,
            away_team=away,
            period=str(period),
            time_remaining=time_remaining,
            home_score=int(home_comp.get("score", 0)),
            away_score=int(away_comp.get("score", 0)),
            live_odds={
                "home_moneyline": h2h.get(home),
                "away_moneyline": h2h.get(away),
            },
            meta={"id": event.get("id"), "clock": clock},
        )
        states.append(state)
    return states


SPORT_FETCHERS = {
    "nhl": fetch_nhl_states,
    "nba": fetch_nba_states,
    "nfl": fetch_nfl_states,
}

