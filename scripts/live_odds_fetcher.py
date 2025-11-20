"""
Live Odds Fetcher
==================

Fetches real-time odds from The Odds API for NBA and NFL games.
Supports line shopping, arbitrage detection, and odds movement tracking.

Requires: THE_ODDS_API_KEY environment variable

Author: AI Coding Assistant
Date: November 16, 2025
"""

import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import time

class LiveOddsFetcher:
    """Fetches and manages live betting odds."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds fetcher.
        
        Args:
            api_key: The Odds API key (or uses THE_ODDS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('THE_ODDS_API_KEY')
        if not self.api_key:
            print("WARNING: No API key found. Set THE_ODDS_API_KEY environment variable.")
            print("Get a key at: https://the-odds-api.com/")
        
        self.base_url = "https://api.the-odds-api.com/v4"
        self.data_dir = Path(__file__).parent.parent / 'data' / 'live'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_nba_odds(
        self,
        markets: List[str] = ['h2h', 'spreads', 'totals'],
        regions: str = 'us'
    ) -> Dict:
        """
        Fetch current NBA odds.
        
        Args:
            markets: List of markets ('h2h', 'spreads', 'totals', 'player_props')
            regions: Betting regions ('us', 'uk', 'eu', 'au')
            
        Returns:
            Dict with odds data
        """
        if not self.api_key:
            return self._mock_nba_odds()
        
        url = f"{self.base_url}/sports/basketball_nba/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to file
            timestamp = datetime.now().isoformat()
            save_path = self.data_dir / f'nba_odds_{timestamp.replace(":", "-")}.json'
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also save as "latest"
            latest_path = self.data_dir / 'nba_odds_latest.json'
            with open(latest_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Fetched odds for {len(data)} NBA games")
            return data
            
        except Exception as e:
            print(f"Error fetching NBA odds: {e}")
            return self._mock_nba_odds()
    
    def fetch_nfl_odds(
        self,
        markets: List[str] = ['h2h', 'spreads', 'totals'],
        regions: str = 'us'
    ) -> Dict:
        """
        Fetch current NFL odds.
        
        Args:
            markets: List of markets
            regions: Betting regions
            
        Returns:
            Dict with odds data
        """
        if not self.api_key:
            return self._mock_nfl_odds()
        
        url = f"{self.base_url}/sports/americanfootball_nfl/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Save
            timestamp = datetime.now().isoformat()
            save_path = self.data_dir / f'nfl_odds_{timestamp.replace(":", "-")}.json'
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            latest_path = self.data_dir / 'nfl_odds_latest.json'
            with open(latest_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Fetched odds for {len(data)} NFL games")
            return data
            
        except Exception as e:
            print(f"Error fetching NFL odds: {e}")
            return self._mock_nfl_odds()
    
    def find_best_odds(self, game_odds: Dict, market: str = 'h2h', side: str = 'home') -> Dict:
        """
        Find best available odds across all sportsbooks.
        
        Args:
            game_odds: Odds data for single game
            market: Market type
            side: 'home' or 'away'
            
        Returns:
            Dict with best odds and sportsbook
        """
        best_odds = None
        best_book = None
        
        for bookmaker in game_odds.get('bookmakers', []):
            for market_data in bookmaker.get('markets', []):
                if market_data['key'] == market:
                    for outcome in market_data['outcomes']:
                        if outcome['name'].lower() == side.lower() or \
                           (side == 'home' and outcome['name'] == game_odds.get('home_team')) or \
                           (side == 'away' and outcome['name'] == game_odds.get('away_team')):
                            
                            odds = outcome.get('price')
                            if best_odds is None or odds > best_odds:
                                best_odds = odds
                                best_book = bookmaker['title']
        
        return {
            'odds': best_odds,
            'sportsbook': best_book,
            'market': market,
            'side': side
        }
    
    def detect_arbitrage(self, game_odds: Dict) -> Optional[Dict]:
        """
        Detect arbitrage opportunities.
        
        Returns:
            Dict with arbitrage info if found, None otherwise
        """
        # Find best odds for both sides
        best_home = self.find_best_odds(game_odds, 'h2h', 'home')
        best_away = self.find_best_odds(game_odds, 'h2h', 'away')
        
        if not best_home['odds'] or not best_away['odds']:
            return None
        
        # Convert to decimal for calculation
        def american_to_decimal(odds):
            if odds > 0:
                return (odds / 100.0) + 1.0
            else:
                return (100.0 / abs(odds)) + 1.0
        
        home_decimal = american_to_decimal(best_home['odds'])
        away_decimal = american_to_decimal(best_away['odds'])
        
        # Calculate arbitrage percentage
        arb_pct = (1 / home_decimal + 1 / away_decimal) * 100
        
        if arb_pct < 100:  # Arbitrage exists!
            profit_pct = 100 - arb_pct
            
            return {
                'game': f"{game_odds.get('away_team')} @ {game_odds.get('home_team')}",
                'home_bet': best_home,
                'away_bet': best_away,
                'arb_percentage': arb_pct,
                'profit_percentage': profit_pct,
                'exists': True
            }
        
        return None
    
    def _mock_nba_odds(self) -> List[Dict]:
        """Return mock NBA odds for testing without API key."""
        return [
            {
                'id': 'mock_game_1',
                'sport_key': 'basketball_nba',
                'sport_title': 'NBA',
                'commence_time': datetime.now().isoformat(),
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'bookmakers': [
                    {
                        'key': 'draftkings',
                        'title': 'DraftKings',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Los Angeles Lakers', 'price': -150},
                                    {'name': 'Golden State Warriors', 'price': +130}
                                ]
                            },
                            {
                                'key': 'spreads',
                                'outcomes': [
                                    {'name': 'Los Angeles Lakers', 'price': -110, 'point': -3.5},
                                    {'name': 'Golden State Warriors', 'price': -110, 'point': 3.5}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    
    def _mock_nfl_odds(self) -> List[Dict]:
        """Return mock NFL odds for testing."""
        return [
            {
                'id': 'mock_game_1',
                'sport_key': 'americanfootball_nfl',
                'sport_title': 'NFL',
                'commence_time': datetime.now().isoformat(),
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Buffalo Bills',
                'bookmakers': [
                    {
                        'key': 'fanduel',
                        'title': 'FanDuel',
                        'markets': [
                            {
                                'key': 'h2h',
                                'outcomes': [
                                    {'name': 'Kansas City Chiefs', 'price': -200},
                                    {'name': 'Buffalo Bills', 'price': +170}
                                ]
                            },
                            {
                                'key': 'spreads',
                                'outcomes': [
                                    {'name': 'Kansas City Chiefs', 'price': -110, 'point': -4.5},
                                    {'name': 'Buffalo Bills', 'price': -110, 'point': 4.5}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]


def main():
    """Test odds fetcher."""
    print("=" * 80)
    print("LIVE ODDS FETCHER TEST")
    print("=" * 80)
    
    fetcher = LiveOddsFetcher()
    
    # Fetch NBA odds
    print("\nFetching NBA odds...")
    nba_odds = fetcher.fetch_nba_odds()
    
    if nba_odds:
        print(f"\nFound {len(nba_odds)} NBA games")
        
        if len(nba_odds) > 0:
            game = nba_odds[0]
            print(f"\nExample game: {game.get('away_team')} @ {game.get('home_team')}")
            
            # Find best odds
            best = fetcher.find_best_odds(game, 'h2h', 'home')
            print(f"Best home odds: {best['odds']} ({best['sportsbook']})")
            
            # Check for arbitrage
            arb = fetcher.detect_arbitrage(game)
            if arb and arb.get('exists'):
                print(f"\n🎰 ARBITRAGE FOUND! Profit: {arb['profit_percentage']:.2f}%")
            else:
                print("\nNo arbitrage opportunities detected")
    
    # Fetch NFL odds
    print("\n" + "-" * 80)
    print("\nFetching NFL odds...")
    nfl_odds = fetcher.fetch_nfl_odds()
    
    if nfl_odds:
        print(f"Found {len(nfl_odds)} NFL games")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nTo use with real API:")
    print("  1. Get API key from https://the-odds-api.com/")
    print("  2. Set environment variable: export THE_ODDS_API_KEY='your_key'")
    print("  3. Run this script again")


if __name__ == '__main__':
    main()

