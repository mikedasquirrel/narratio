"""
NHL Live Odds Fetcher

Fetches current NHL betting odds from The Odds API.
Integrates with pattern matching system for daily recommendations.

The Odds API: https://the-odds-api.com/
- Free tier: 500 requests/month
- NHL markets: moneyline (h2h), puck_line (spreads), totals

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


class NHLOddsFetcher:
    """Fetch live NHL odds"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds fetcher.
        
        Parameters
        ----------
        api_key : str, optional
            The Odds API key (get from https://the-odds-api.com/)
            If not provided, looks for THE_ODDS_API_KEY environment variable
        """
        self.api_key = api_key or os.environ.get('THE_ODDS_API_KEY')
        
        if not self.api_key:
            print("⚠️  No API key found. Set THE_ODDS_API_KEY environment variable")
            print("   Get a key at: https://the-odds-api.com/")
        
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "icehockey_nhl"
    
    def fetch_upcoming_games(self) -> List[Dict]:
        """
        Fetch upcoming NHL games with odds.
        
        Returns
        -------
        games : list
            List of games with odds
        """
        if not self.api_key:
            return self._mock_games()
        
        print("\n📡 FETCHING LIVE NHL ODDS")
        print("-"*80)
        
        url = f"{self.base_url}/sports/{self.sport}/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',  # Moneyline, puck line, totals
            'oddsFormat': 'american',
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            print(f"   ✓ Fetched {len(data)} games")
            
            # Parse games
            games = []
            for game in data:
                parsed = self._parse_game(game)
                if parsed:
                    games.append(parsed)
            
            return games
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ API error: {e}")
            return self._mock_games()
    
    def _parse_game(self, api_game: Dict) -> Optional[Dict]:
        """Parse game from API response"""
        try:
            game_id = api_game.get('id', '')
            commence_time = api_game.get('commence_time', '')
            
            teams = api_game.get('home_team', ''), api_game.get('away_team', '')
            home_team_full = api_game.get('home_team', '')
            away_team_full = api_game.get('away_team', '')
            
            # Convert full names to abbreviations
            home_team = self._team_name_to_abbrev(home_team_full)
            away_team = self._team_name_to_abbrev(away_team_full)
            
            # Extract odds from bookmakers
            bookmakers = api_game.get('bookmakers', [])
            
            if not bookmakers:
                return None
            
            # Use first bookmaker (or average multiple)
            odds = bookmakers[0]
            markets = {m['key']: m for m in odds.get('markets', [])}
            
            # Moneyline (h2h)
            h2h = markets.get('h2h', {})
            h2h_outcomes = {o['name']: o['price'] for o in h2h.get('outcomes', [])}
            
            # Spreads (puck line)
            spreads = markets.get('spreads', {})
            spread_outcomes = {o['name']: {'point': o.get('point'), 'price': o.get('price')} 
                              for o in spreads.get('outcomes', [])}
            
            # Totals
            totals = markets.get('totals', {})
            total_outcomes = {o['name']: {'point': o.get('point'), 'price': o.get('price')}
                             for o in totals.get('outcomes', [])}
            
            game = {
                'game_id': game_id,
                'commence_time': commence_time,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_full': home_team_full,
                'away_team_full': away_team_full,
                'odds': {
                    'moneyline_home': h2h_outcomes.get(home_team_full),
                    'moneyline_away': h2h_outcomes.get(away_team_full),
                    'puck_line_home': spread_outcomes.get(home_team_full, {}).get('point'),
                    'puck_line_home_odds': spread_outcomes.get(home_team_full, {}).get('price'),
                    'puck_line_away': spread_outcomes.get(away_team_full, {}).get('point'),
                    'puck_line_away_odds': spread_outcomes.get(away_team_full, {}).get('price'),
                    'total': total_outcomes.get('Over', {}).get('point'),
                    'over_odds': total_outcomes.get('Over', {}).get('price'),
                    'under_odds': total_outcomes.get('Under', {}).get('price'),
                },
                'bookmaker': odds.get('title', 'Unknown'),
            }
            
            return game
            
        except Exception as e:
            return None
    
    def _team_name_to_abbrev(self, full_name: str) -> str:
        """Convert full team name to abbreviation"""
        mapping = {
            'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
            'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
            'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
            'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
            'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
            'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
            'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
            'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
            'Seattle Kraken': 'SEA', 'St Louis Blues': 'STL', 'St. Louis Blues': 'STL',
            'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR',
            'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
            'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
        }
        
        return mapping.get(full_name, full_name[:3].upper())
    
    def _mock_games(self) -> List[Dict]:
        """Return mock games for testing without API key"""
        print("   📝 Using mock data (no API key)")
        
        return [
            {
                'game_id': 'mock_001',
                'commence_time': '2025-11-17T00:00:00Z',
                'home_team': 'TOR',
                'away_team': 'BOS',
                'home_team_full': 'Toronto Maple Leafs',
                'away_team_full': 'Boston Bruins',
                'odds': {
                    'moneyline_home': -150,
                    'moneyline_away': +130,
                    'puck_line_home': -1.5,
                    'puck_line_home_odds': +165,
                    'puck_line_away': +1.5,
                    'puck_line_away_odds': -185,
                    'total': 6.0,
                    'over_odds': -110,
                    'under_odds': -110,
                },
                'bookmaker': 'Mock',
            }
        ]


def main():
    """Main execution"""
    
    fetcher = NHLOddsFetcher()
    games = fetcher.fetch_upcoming_games()
    
    # Save
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'data' / 'live' / 'nhl_live_odds.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'games': games,
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
    print(f"✅ {len(games)} games with odds")


if __name__ == "__main__":
    main()

