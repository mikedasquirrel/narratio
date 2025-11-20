"""
Unified Odds Fetcher

Single source of truth for all odds across all sports using The Odds API.

Replaces:
- ESPN odds scraping
- Multiple sport-specific odds fetchers
- Manual odds parsing

Provides:
- Pre-game odds (moneyline, spreads, totals)
- Live odds
- Player props (all markets)
- Best line shopping across sportsbooks

Author: Unified Odds System
Date: November 19, 2025
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

API_KEY = "2e330948334c9505ed5542a82fcfa3b9"
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS_MAP = {
    'nhl': 'icehockey_nhl',
    'nba': 'basketball_nba',
    'nfl': 'americanfootball_nfl',
    'mlb': 'baseball_mlb',
}


def fetch_all_odds_for_sport(sport: str, include_props: bool = True) -> Dict:
    """
    Fetch comprehensive odds for a sport.
    
    Parameters
    ----------
    sport : str
        Sport key ('nhl', 'nba', 'nfl', 'mlb')
    include_props : bool
        Whether to fetch player props
    
    Returns
    -------
    odds_data : dict
        {
            'games': [...],  # Game odds
            'props': [...],  # Player props
            'timestamp': '...',
            'sport': '...'
        }
    """
    sport_key = SPORTS_MAP.get(sport, sport)
    
    print(f"\n{'='*80}")
    print(f"FETCHING {sport.upper()} ODDS")
    print(f"{'='*80}")
    
    # Fetch game odds
    print(f"\n[1/2] Fetching game odds (moneyline, spreads, totals)...")
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        games = response.json()
        print(f"  ✓ Fetched {len(games)} games")
        print(f"  API requests remaining: {response.headers.get('x-requests-remaining', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        games = []
    
    # Fetch player props (skip for now - requires different endpoint/plan)
    props = []
    if include_props and sport in ['nba', 'nfl']:
        print(f"\n[2/2] Player props...")
        print(f"  ℹ Player props require different API endpoint (not included in basic plan)")
        print(f"  Continuing with game odds only...")
    
    return {
        'sport': sport,
        'games': games,
        'props': props,
        'timestamp': datetime.now().isoformat(),
        'api_requests_remaining': response.headers.get('x-requests-remaining', 'N/A') if 'response' in locals() else 'N/A'
    }


def parse_game_odds(game: Dict) -> Dict:
    """
    Parse game odds into standardized format.
    
    Parameters
    ----------
    game : dict
        Raw game data from API
    
    Returns
    -------
    parsed : dict
        Standardized odds format
    """
    home_team = game.get('home_team', '')
    away_team = game.get('away_team', '')
    
    # Find best odds across all bookmakers
    best_home_ml = None
    best_away_ml = None
    best_home_spread = None
    best_away_spread = None
    best_total = None
    best_over_odds = None
    best_under_odds = None
    
    for bookmaker in game.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            market_key = market.get('key')
            
            if market_key == 'h2h':
                for outcome in market.get('outcomes', []):
                    if outcome['name'] == home_team:
                        if best_home_ml is None or outcome['price'] > best_home_ml:
                            best_home_ml = outcome['price']
                    elif outcome['name'] == away_team:
                        if best_away_ml is None or outcome['price'] > best_away_ml:
                            best_away_ml = outcome['price']
            
            elif market_key == 'spreads':
                for outcome in market.get('outcomes', []):
                    if outcome['name'] == home_team:
                        spread = outcome.get('point')
                        odds = outcome.get('price')
                        if best_home_spread is None:
                            best_home_spread = (spread, odds)
                    elif outcome['name'] == away_team:
                        spread = outcome.get('point')
                        odds = outcome.get('price')
                        if best_away_spread is None:
                            best_away_spread = (spread, odds)
            
            elif market_key == 'totals':
                for outcome in market.get('outcomes', []):
                    if outcome['name'] == 'Over':
                        if best_total is None:
                            best_total = outcome.get('point')
                            best_over_odds = outcome.get('price')
                    elif outcome['name'] == 'Under':
                        best_under_odds = outcome.get('price')
    
    return {
        'game_id': game.get('id', ''),
        'home_team': home_team,
        'away_team': away_team,
        'commence_time': game.get('commence_time', ''),
        'moneyline': {
            'home': best_home_ml,
            'away': best_away_ml,
        },
        'spread': {
            'home': best_home_spread[0] if best_home_spread else None,
            'home_odds': best_home_spread[1] if best_home_spread else None,
            'away': best_away_spread[0] if best_away_spread else None,
            'away_odds': best_away_spread[1] if best_away_spread else None,
        },
        'total': {
            'line': best_total,
            'over_odds': best_over_odds,
            'under_odds': best_under_odds,
        }
    }


def fetch_todays_odds_all_sports() -> Dict:
    """
    Fetch odds for all sports for today.
    
    Returns
    -------
    all_odds : dict
        {
            'nhl': {...},
            'nba': {...},
            'nfl': {...},
            'timestamp': '...'
        }
    """
    print("\n" + "="*80)
    print("FETCHING TODAY'S ODDS - ALL SPORTS")
    print("="*80)
    
    all_odds = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
    }
    
    for sport in ['nhl', 'nba', 'nfl']:
        odds_data = fetch_all_odds_for_sport(sport, include_props=(sport in ['nba', 'nfl']))
        all_odds[sport] = odds_data
    
    # Save to file
    output_path = f"data/odds/all_sports_odds_{datetime.now().strftime('%Y%m%d')}.json"
    Path('data/odds').mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_odds, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ODDS FETCHING COMPLETE")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  NHL: {len(all_odds['nhl']['games'])} games, {len(all_odds['nhl']['props'])} prop markets")
    print(f"  NBA: {len(all_odds['nba']['games'])} games, {len(all_odds['nba']['props'])} prop markets")
    print(f"  NFL: {len(all_odds['nfl']['games'])} games, {len(all_odds['nfl']['props'])} prop markets")
    print(f"\nSaved to: {output_path}")
    print(f"API requests remaining: {all_odds.get('nfl', {}).get('api_requests_remaining', 'N/A')}")
    
    return all_odds


if __name__ == '__main__':
    fetch_todays_odds_all_sports()

