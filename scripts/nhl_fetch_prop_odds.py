"""
NHL Prop Odds Fetcher

Fetches player prop betting lines from The Odds API:
- Goals (over/under 0.5, 1.5)
- Assists (over/under 0.5, 1.5)
- Shots on goal (over/under lines)
- Points (over/under 0.5, 1.5, 2.5)
- Goalie saves (over/under lines)

Markets available:
- player_points_over_under
- player_assists_over_under
- player_shots_on_goal_over_under
- player_goals_over_under
- player_saves_over_under

Author: Prop Betting System
Date: November 20, 2024
"""

import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.odds_api_config import ODDS_API_KEY, BASE_URL, SPORTS


class NHLPropOddsFetcher:
    """Fetch NHL player prop odds from The Odds API"""
    
    def __init__(self, api_key: str = ODDS_API_KEY):
        """Initialize with API key"""
        self.api_key = api_key
        self.base_url = BASE_URL
        self.sport = SPORTS['nhl']
        
        # Prop markets we care about (canonical key -> fallback variants)
        self.prop_market_variants = {
            'player_points': [
                'player_points',  # Current Odds API key
                'player_points_over_under',  # Legacy key (Nov 2024)
            ],
            'player_assists': [
                'player_assists',
                'player_assists_over_under',
            ],
            'player_shots_on_goal': [
                'player_shots_on_goal',
                'player_shots_on_goal_over_under',
            ],
            'player_goals': [
                'player_goals',
                'player_goals_over_under',
            ],
            'player_saves': [
                'player_saves',
                'player_saves_over_under',
            ],
        }
        
    def fetch_games_with_props(self) -> List[Dict]:
        """
        Fetch all upcoming NHL games with prop odds.
        
        Returns
        -------
        games : list of dict
            Games with available prop markets
        """
        print("\n📡 FETCHING NHL GAMES WITH PROP ODDS")
        print("-" * 80)
        
        # First get list of games
        games_url = f"{self.base_url}/sports/{self.sport}/events"
        
        params = {
            'apiKey': self.api_key,
            'dateFormat': 'iso',
        }
        
        try:
            response = requests.get(games_url, params=params, timeout=10)
            response.raise_for_status()
            
            games = response.json()
            print(f"   ✓ Found {len(games)} upcoming NHL games")
            
            # Check API usage
            remaining = response.headers.get('x-requests-remaining', 'N/A')
            used = response.headers.get('x-requests-used', 'N/A') 
            print(f"   📊 API usage: {used} used, {remaining} remaining")
            
            return games
            
        except Exception as e:
            print(f"   ✗ Error fetching games: {e}")
            return []
            
    def fetch_props_for_game(self, event_id: str) -> Dict:
        """
        Fetch prop odds for a specific game.
        
        Parameters
        ----------
        event_id : str
            The Odds API event ID
            
        Returns
        -------
        props : dict
            All prop markets for the game
        """
        print(f"\n   Fetching props for event {event_id[:8]}...")
        
        all_props = {}
        
        for canonical_market, variants in self.prop_market_variants.items():
            market_data = None
            
            for market_key in variants:
                market_data = self._request_market(event_id, market_key)
                
                if market_data:
                    break  # Stop after first successful variant
            
            if market_data:
                parsed = self._parse_prop_odds(market_data, canonical_market)
                
                if parsed:
                    all_props[canonical_market] = parsed
                    print(f"      ✓ {canonical_market}: {len(parsed)} player props")
            else:
                print(f"      ✗ No odds returned for {canonical_market} (checked {', '.join(variants)})")
                
        return all_props
    
    def _request_market(self, event_id: str, market: str) -> Optional[Dict]:
        """Request a single market, handling legacy keys gracefully."""
        url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': market,
            'oddsFormat': 'american',
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('bookmakers'):
                return data
        except requests.HTTPError as http_err:  # type: ignore
            status = getattr(http_err.response, 'status_code', None)
            if status == 422:
                print(f"      • Market '{market}' not supported by API (422)")
            else:
                print(f"      ✗ HTTP error for market '{market}': {http_err}")
        except Exception as e:
            print(f"      ✗ Error fetching market '{market}': {e}")
        
        return None
        
    def _parse_prop_odds(self, data: Dict, market: str) -> List[Dict]:
        """Parse prop odds from API response"""
        props = []
        
        # Get best odds across all bookmakers
        for bookmaker in data.get('bookmakers', []):
            book_name = bookmaker['title']
            
            for market_data in bookmaker.get('markets', []):
                if market_data['key'] == market:
                    for outcome in market_data.get('outcomes', []):
                        player_name = outcome.get('description', '')
                        
                        # Parse player name and line
                        if ' - ' in player_name:
                            name, line_str = player_name.split(' - ', 1)
                            
                            # Extract line value (e.g., "Over 0.5" -> 0.5)
                            try:
                                if 'Over' in line_str:
                                    line = float(line_str.split()[-1])
                                    side = 'over'
                                elif 'Under' in line_str:
                                    line = float(line_str.split()[-1])
                                    side = 'under'
                                else:
                                    continue
                                    
                                props.append({
                                    'player_name': name.strip(),
                                    'market': market,
                                    'line': line,
                                    'side': side,
                                    'odds': outcome['price'],
                                    'bookmaker': book_name,
                                    'point': outcome.get('point'),  # Some markets have point spreads
                                })
                                
                            except (ValueError, IndexError):
                                continue
                                
        return props
        
    def get_best_props(self, props_by_market: Dict) -> List[Dict]:
        """
        Aggregate props to find best odds for each player/market.
        
        Returns
        -------
        best_props : list
            Best available odds for each unique prop
        """
        # Group by player + market + line
        prop_groups = {}
        
        for market, props in props_by_market.items():
            for prop in props:
                key = (prop['player_name'], market, prop['line'], prop['side'])
                
                if key not in prop_groups:
                    prop_groups[key] = []
                    
                prop_groups[key].append(prop)
                
        # Find best odds for each prop
        best_props = []
        
        for key, props in prop_groups.items():
            # Sort by odds (best first)
            if key[3] == 'over':  # For overs, higher positive odds are better
                props.sort(key=lambda x: x['odds'], reverse=True)
            else:  # For unders
                props.sort(key=lambda x: x['odds'], reverse=True)
                
            best = props[0]
            best['all_odds'] = [p['odds'] for p in props]
            best['all_books'] = [p['bookmaker'] for p in props]
            
            best_props.append(best)
            
        return best_props
        
    def fetch_all_nhl_props(self) -> Dict:
        """
        Fetch props for all upcoming NHL games.
        
        Returns
        -------
        all_props : dict
            {game_id: {props}}
        """
        games = self.fetch_games_with_props()
        
        all_game_props = {}
        
        for i, game in enumerate(games[:10]):  # Limit to 10 games to save API calls
            event_id = game['id']
            home = game['home_team']
            away = game['away_team']
            commence = game['commence_time']
            
            print(f"\n[{i+1}/{min(len(games), 10)}] {away} @ {home} ({commence})")
            
            # Fetch props
            props = self.fetch_props_for_game(event_id)
            
            if props:
                # Get best odds
                best_props = self.get_best_props(props)
                
                all_game_props[event_id] = {
                    'game_id': event_id,
                    'home_team': home,
                    'away_team': away, 
                    'commence_time': commence,
                    'props': best_props,
                    'prop_count': len(best_props),
                }
                
                # Show summary
                prop_types = {}
                for prop in best_props:
                    market = prop['market'].replace('player_', '').replace('_over_under', '')
                    prop_types[market] = prop_types.get(market, 0) + 1
                    
                print(f"   📊 Found {len(best_props)} props: {prop_types}")
                
        return all_game_props
        
    def save_props(self, props: Dict, output_file: Path):
        """Save props to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(props, f, indent=2)
            
        print(f"\n✓ Saved props to {output_file}")
        
    def american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal"""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
            
    def american_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


def main():
    """Example usage"""
    print("NHL PROP ODDS FETCHER")
    print("=" * 80)
    
    fetcher = NHLPropOddsFetcher()
    
    # Fetch all props
    all_props = fetcher.fetch_all_nhl_props()
    
    # Save to file
    output_dir = Path("data/processed_odds")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"nhl_props_{timestamp}.json"
    
    fetcher.save_props(all_props, output_file)
    
    # Show sample props
    print("\n" + "="*80)
    print("SAMPLE PLAYER PROPS")
    print("="*80)
    
    for game_id, game_props in list(all_props.items())[:1]:  # First game
        print(f"\n{game_props['away_team']} @ {game_props['home_team']}")
        print(f"Props available: {game_props['prop_count']}")
        
        # Group by player
        by_player = {}
        for prop in game_props['props'][:20]:  # First 20 props
            player = prop['player_name']
            if player not in by_player:
                by_player[player] = []
            by_player[player].append(prop)
            
        # Show top players
        for player, props in list(by_player.items())[:5]:
            print(f"\n  {player}:")
            for prop in props:
                market = prop['market'].replace('player_', '').replace('_over_under', '')
                implied_prob = fetcher.american_to_probability(prop['odds'])
                print(f"    {market.title()} {prop['side']} {prop['line']}: "
                      f"{prop['odds']:+d} ({implied_prob:.1%}) @ {prop['bookmaker']}")
                      

if __name__ == "__main__":
    main()
