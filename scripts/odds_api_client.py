"""
The Odds API Client

Comprehensive odds fetching for all sports using official paid API.

Features:
- Pre-game odds (moneyline, spreads, totals)
- Live in-game odds
- Player props (all markets)
- Multiple sportsbooks (best line shopping)
- Caching to minimize API calls
- Rate limit management

Author: Odds API Integration
Date: November 19, 2025
"""

import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.odds_api_config import (
    ODDS_API_KEY, BASE_URL, SPORTS, MARKETS, REGIONS, BOOKMAKERS, CACHE_DURATION
)


class OddsAPIClient:
    """Client for The Odds API"""
    
    def __init__(self, api_key: str = ODDS_API_KEY):
        """Initialize client"""
        self.api_key = api_key
        self.base_url = BASE_URL
        self.cache = {}
        self.request_count = 0
        self.last_request_time = 0
        
    def get_odds(self, sport: str, markets: List[str] = ['h2h'], 
                 regions: List[str] = ['us'], bookmakers: Optional[List[str]] = None) -> List[Dict]:
        """
        Get current odds for a sport.
        
        Parameters
        ----------
        sport : str
            Sport key (e.g., 'nhl', 'nba', 'nfl')
        markets : list of str
            Markets to fetch (e.g., ['h2h', 'spreads', 'totals'])
        regions : list of str
            Regions for odds
        bookmakers : list of str, optional
            Specific bookmakers to fetch
        
        Returns
        -------
        odds : list of dict
            Odds for all upcoming games
        """
        # Check cache
        cache_key = f"{sport}_{','.join(markets)}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CACHE_DURATION['pre_game']:
                print(f"  [Cache] Using cached odds for {sport}")
                return cached_data
        
        # Rate limit
        self._rate_limit()
        
        # Build request
        sport_key = SPORTS.get(sport, sport)
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': ','.join(regions),
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        print(f"\n[Odds API] Fetching {sport.upper()} odds...")
        print(f"  Markets: {', '.join(markets)}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.request_count += 1
            
            print(f"  ✓ Fetched odds for {len(data)} games")
            print(f"  API requests used: {self.request_count}")
            
            # Cache
            self.cache[cache_key] = (datetime.now(), data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching odds: {e}")
            return []
    
    def get_live_odds(self, sport: str) -> List[Dict]:
        """
        Get live in-game odds.
        
        Parameters
        ----------
        sport : str
            Sport key
        
        Returns
        -------
        live_odds : list of dict
            Live odds for games in progress
        """
        # Check cache (shorter duration for live)
        cache_key = f"{sport}_live"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CACHE_DURATION['live']:
                return cached_data
        
        self._rate_limit()
        
        sport_key = SPORTS.get(sport, sport)
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        }
        
        print(f"\n[Odds API] Fetching LIVE {sport.upper()} odds...")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.request_count += 1
            
            # Filter to only in-progress games
            live_games = [g for g in data if self._is_game_live(g)]
            
            print(f"  ✓ Found {len(live_games)} live games")
            
            self.cache[cache_key] = (datetime.now(), live_games)
            
            return live_games
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching live odds: {e}")
            return []
    
    def get_player_props(self, sport: str, markets: List[str] = ['player_points']) -> List[Dict]:
        """
        Get player prop odds.
        
        Parameters
        ----------
        sport : str
            Sport key
        markets : list of str
            Prop markets to fetch
        
        Returns
        -------
        props : list of dict
            Player prop odds
        """
        cache_key = f"{sport}_props_{','.join(markets)}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < CACHE_DURATION['props']:
                print(f"  [Cache] Using cached props for {sport}")
                return cached_data
        
        self._rate_limit()
        
        sport_key = SPORTS.get(sport, sport)
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso',
        }
        
        print(f"\n[Odds API] Fetching {sport.upper()} player props...")
        print(f"  Markets: {', '.join(markets)}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.request_count += 1
            
            print(f"  ✓ Fetched props for {len(data)} games")
            
            self.cache[cache_key] = (datetime.now(), data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error fetching props: {e}")
            return []
    
    def get_best_odds(self, game_odds: Dict, side: str) -> Tuple[float, str]:
        """
        Find best odds across all sportsbooks.
        
        Parameters
        ----------
        game_odds : dict
            Odds data for a single game
        side : str
            'home' or 'away'
        
        Returns
        -------
        best_odds : float
            Best available odds
        best_book : str
            Sportsbook offering best odds
        """
        bookmakers = game_odds.get('bookmakers', [])
        
        best_odds = None
        best_book = None
        
        for book in bookmakers:
            book_name = book.get('key', '')
            markets = book.get('markets', [])
            
            for market in markets:
                if market.get('key') != 'h2h':
                    continue
                
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name', '').lower() == game_odds.get(f'{side}_team', '').lower():
                        odds = outcome.get('price')
                        
                        if best_odds is None or self._is_better_odds(odds, best_odds):
                            best_odds = odds
                            best_book = book_name
        
        return best_odds, best_book
    
    def _is_better_odds(self, odds1: float, odds2: float) -> bool:
        """Check if odds1 is better than odds2"""
        # For positive odds, higher is better
        # For negative odds, closer to 0 is better
        if odds1 > 0 and odds2 > 0:
            return odds1 > odds2
        elif odds1 < 0 and odds2 < 0:
            return odds1 > odds2
        else:
            # Mixed signs - convert to decimal and compare
            dec1 = self._american_to_decimal(odds1)
            dec2 = self._american_to_decimal(odds2)
            return dec1 > dec2
    
    def _american_to_decimal(self, american: float) -> float:
        """Convert American odds to decimal"""
        if american > 0:
            return (american / 100) + 1
        else:
            return (100 / abs(american)) + 1
    
    def _is_game_live(self, game: Dict) -> bool:
        """Check if game is currently in progress"""
        commence_time = game.get('commence_time', '')
        if not commence_time:
            return False
        
        game_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
        now = datetime.now(game_time.tzinfo)
        
        # Game is live if it started 0-3 hours ago
        time_since_start = (now - game_time).total_seconds() / 3600
        return 0 <= time_since_start <= 3
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        # Max 10 requests per second
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < 0.1:  # 100ms between requests
            time.sleep(0.1 - time_since_last)
        
        self.last_request_time = time.time()
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'requests_this_session': self.request_count,
            'cache_entries': len(self.cache),
            'api_key': self.api_key[:8] + '...' + self.api_key[-4:],
        }


def test_api_connection():
    """Test API connection and display available data"""
    print("\n" + "="*80)
    print("THE ODDS API - CONNECTION TEST")
    print("="*80)
    
    client = OddsAPIClient()
    
    # Test NHL
    print("\n[Test 1] NHL Pre-Game Odds")
    nhl_odds = client.get_odds('nhl', markets=['h2h', 'spreads', 'totals'])
    
    if nhl_odds:
        game = nhl_odds[0]
        print(f"\nSample game:")
        print(f"  {game.get('away_team')} @ {game.get('home_team')}")
        print(f"  Commence: {game.get('commence_time')}")
        print(f"  Bookmakers: {len(game.get('bookmakers', []))}")
        
        # Show best odds
        home_odds, home_book = client.get_best_odds(game, 'home')
        away_odds, away_book = client.get_best_odds(game, 'away')
        print(f"  Best odds: {game.get('home_team')} {home_odds} ({home_book}), {game.get('away_team')} {away_odds} ({away_book})")
    
    # Test NBA
    print("\n[Test 2] NBA Pre-Game Odds")
    nba_odds = client.get_odds('nba', markets=['h2h', 'spreads'])
    print(f"  Found {len(nba_odds)} NBA games")
    
    # Test NBA Props
    print("\n[Test 3] NBA Player Props")
    nba_props = client.get_player_props('nba', markets=['player_points', 'player_rebounds', 'player_assists'])
    print(f"  Found props for {len(nba_props)} NBA games")
    
    # Usage stats
    print("\n[Usage Stats]")
    stats = client.get_usage_stats()
    print(f"  Requests this session: {stats['requests_this_session']}")
    print(f"  Cache entries: {stats['cache_entries']}")
    print(f"  API Key: {stats['api_key']}")
    
    print("\n" + "="*80)
    print("CONNECTION TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_api_connection()

