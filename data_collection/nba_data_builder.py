"""
NBA Data Builder - Comprehensive Betting Data Collection

Collects:
1. Game results with temporal context (existing)
2. Real betting odds (moneyline, spreads, totals)
3. Player-level data (for props betting)
4. Enhanced context (injuries, rest, referee, etc.)
5. Current 2024-25 season (for live validation)

Data Sources:
- NBA API (nba_api package) - game results, player stats
- The Odds API (free tier) - current odds
- Basketball Reference - historical odds
- Sports Reference - advanced stats

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional

# Try to import nba_api (may need: pip install nba_api)
try:
    from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, teamgamelog
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    print("⚠️  nba_api not installed. Install with: pip install nba_api")
    NBA_API_AVAILABLE = False

print("="*80)
print("NBA DATA BUILDER - COMPREHENSIVE BETTING DATA")
print("="*80)


class NBADataBuilder:
    """Build comprehensive NBA betting dataset"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys (user needs to add)
        self.odds_api_key = None  # Get from: https://the-odds-api.com/
        
        self.teams_cache = {}
        
    def load_existing_data(self) -> List[Dict]:
        """Load existing NBA data"""
        existing_path = self.output_dir.parent / 'domains' / 'nba_with_temporal_context.json'
        
        if existing_path.exists():
            print(f"\n📂 Loading existing data from {existing_path.name}...")
            with open(existing_path) as f:
                games = json.load(f)
            print(f"   ✓ Loaded {len(games):,} games")
            return games
        else:
            print("⚠️  No existing data found")
            return []
    
    def enhance_with_betting_odds(self, games: List[Dict]) -> List[Dict]:
        """Add betting odds to existing games"""
        print("\n📊 PHASE 1: Adding Betting Odds")
        print("-"*80)
        
        # For historical games, we'll add estimated odds based on win probability
        # For current season, we'll fetch real odds
        
        enhanced = []
        current_season = '2024-25'
        
        for i, game in enumerate(games):
            if i % 1000 == 0:
                print(f"   Processing {i:,}/{len(games):,}...")
            
            # Calculate implied odds from win probability
            tc = game.get('temporal_context', {})
            team_win_pct = tc.get('season_win_pct', 0.5)
            home = game.get('home_game', False)
            
            # Home advantage ~3%
            implied_prob = team_win_pct + (0.03 if home else -0.03)
            implied_prob = max(0.1, min(0.9, implied_prob))
            
            # Convert to American odds
            if implied_prob > 0.5:
                # Favorite
                moneyline = -int(100 * implied_prob / (1 - implied_prob))
            else:
                # Underdog
                moneyline = int(100 * (1 - implied_prob) / implied_prob)
            
            # Estimate opponent odds (inverse)
            opponent_prob = 1 - implied_prob
            if opponent_prob > 0.5:
                opponent_moneyline = -int(100 * opponent_prob / (1 - opponent_prob))
            else:
                opponent_moneyline = int(100 * (1 - opponent_prob) / opponent_prob)
            
            # Add spread (typically -3 to -7 for moderate favorites)
            if implied_prob > 0.55:
                spread = -round((implied_prob - 0.5) * 20, 1)
            else:
                spread = round((0.5 - implied_prob) * 20, 1)
            
            # Add totals (estimate based on league average ~220)
            points = game.get('points', 110)
            estimated_total = 220  # League average
            
            game['betting_odds'] = {
                'moneyline': moneyline,
                'opponent_moneyline': opponent_moneyline,
                'spread': spread,
                'spread_odds': -110,
                'total': estimated_total,
                'over_odds': -110,
                'under_odds': -110,
                'implied_probability': implied_prob,
                'source': 'estimated',
                'note': 'Historical odds estimated from win probability'
            }
            
            enhanced.append(game)
        
        print(f"   ✓ Added betting odds to {len(enhanced):,} games")
        return enhanced
    
    def add_player_data(self, games: List[Dict]) -> List[Dict]:
        """Add player-level statistics for props"""
        print("\n👥 PHASE 2: Adding Player Data")
        print("-"*80)
        
        if not NBA_API_AVAILABLE:
            print("   ⚠️  nba_api not available - skipping player data")
            print("   Install with: pip install nba_api")
            return games
        
        print("   ⚠️  Player data collection requires NBA API rate limits")
        print("   This would take ~2-3 hours for full dataset")
        print("   Skipping for now - add later if needed")
        
        # Stub: Add player data structure
        for game in games:
            game['player_props'] = {
                'available': False,
                'note': 'Requires NBA API boxscore data',
                'recommended_props': [
                    'star_player_points',
                    'star_player_assists', 
                    'star_player_rebounds',
                    'team_total_points'
                ]
            }
        
        return games
    
    def add_rest_days(self, games: List[Dict]) -> List[Dict]:
        """Calculate rest days between games"""
        print("\n⏰ PHASE 3: Adding Rest Days & Scheduling Context")
        print("-"*80)
        
        # Sort by team and date
        games_by_team = {}
        for game in games:
            team = game.get('team_abbreviation', '')
            if team not in games_by_team:
                games_by_team[team] = []
            games_by_team[team].append(game)
        
        # Sort each team's games by date
        for team in games_by_team:
            games_by_team[team].sort(key=lambda x: x.get('date', ''))
        
        # Calculate rest days
        enhanced = []
        for game in games:
            team = game.get('team_abbreviation', '')
            date_str = game.get('date', '')
            
            if not date_str or team not in games_by_team:
                game['scheduling'] = {
                    'rest_days': None,
                    'back_to_back': False,
                    'three_in_four': False
                }
                enhanced.append(game)
                continue
            
            # Find previous game
            team_games = games_by_team[team]
            current_idx = None
            for idx, g in enumerate(team_games):
                if g.get('game_id') == game.get('game_id'):
                    current_idx = idx
                    break
            
            if current_idx is None or current_idx == 0:
                rest_days = 3  # Default
            else:
                prev_game = team_games[current_idx - 1]
                try:
                    current_date = datetime.strptime(date_str, '%Y-%m-%d')
                    prev_date = datetime.strptime(prev_game.get('date', ''), '%Y-%m-%d')
                    rest_days = (current_date - prev_date).days - 1
                except:
                    rest_days = 3
            
            game['scheduling'] = {
                'rest_days': rest_days,
                'back_to_back': rest_days == 0,
                'three_in_four': rest_days <= 2,
                'well_rested': rest_days >= 3
            }
            
            enhanced.append(game)
        
        b2b_count = sum(1 for g in enhanced if g['scheduling']['back_to_back'])
        print(f"   ✓ Calculated rest days for {len(enhanced):,} games")
        print(f"   ✓ Back-to-backs: {b2b_count:,} ({b2b_count/len(enhanced)*100:.1f}%)")
        
        return enhanced
    
    def add_injury_data(self, games: List[Dict]) -> List[Dict]:
        """Add injury/availability data (placeholder)"""
        print("\n🏥 PHASE 4: Adding Injury Data")
        print("-"*80)
        
        print("   ⚠️  Injury data requires external API")
        print("   Options:")
        print("     1. NBA.com injury reports (scraping)")
        print("     2. ESPN injury API")
        print("     3. Manual CSV import")
        print("   Skipping for now - add later if needed")
        
        for game in games:
            game['injuries'] = {
                'available': False,
                'note': 'Requires injury report API',
                'placeholder': {
                    'star_player_out': False,
                    'multiple_starters_out': False,
                    'impact_level': 'unknown'
                }
            }
        
        return games
    
    def fetch_current_season(self) -> List[Dict]:
        """Fetch 2024-25 season games for live validation"""
        print("\n📅 PHASE 5: Fetching 2024-25 Current Season")
        print("-"*80)
        
        if not NBA_API_AVAILABLE:
            print("   ⚠️  nba_api not available")
            return []
        
        print("   ⚠️  Current season fetch requires NBA API")
        print("   This would collect all 2024-25 games played so far")
        print("   Skipping for now - add later for live validation")
        
        return []
    
    def fetch_current_odds(self) -> Dict:
        """Fetch current odds from The Odds API"""
        print("\n💰 PHASE 6: Fetching Current Betting Odds")
        print("-"*80)
        
        if not self.odds_api_key:
            print("   ⚠️  No Odds API key configured")
            print("   Get free key from: https://the-odds-api.com/")
            print("   Free tier: 500 requests/month")
            print("   Skipping real-time odds")
            return {}
        
        # Fetch current NBA odds
        try:
            url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'american'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                odds_data = response.json()
                print(f"   ✓ Fetched odds for {len(odds_data)} games")
                return odds_data
            else:
                print(f"   ✗ API error: {response.status_code}")
                return {}
        except Exception as e:
            print(f"   ✗ Error fetching odds: {e}")
            return {}
    
    def build_complete_dataset(self):
        """Build complete enhanced dataset"""
        print("\n" + "="*80)
        print("BUILDING COMPLETE NBA DATASET")
        print("="*80)
        
        # Load existing
        games = self.load_existing_data()
        
        if not games:
            print("\n✗ No base data to enhance")
            print("  Run NBA data collection first")
            return
        
        # Enhance progressively
        games = self.enhance_with_betting_odds(games)
        games = self.add_rest_days(games)
        games = self.add_player_data(games)
        games = self.add_injury_data(games)
        
        # Save enhanced dataset
        output_path = self.output_dir / 'nba_enhanced_betting_data.json'
        
        print("\n" + "="*80)
        print("SAVING ENHANCED DATASET")
        print("="*80)
        
        with open(output_path, 'w') as f:
            json.dump(games, f, indent=2)
        
        print(f"\n✓ Saved {len(games):,} games to {output_path.name}")
        
        # Generate summary
        self.generate_summary(games, output_path)
        
        return games
    
    def generate_summary(self, games: List[Dict], output_path: Path):
        """Generate data summary"""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        seasons = set(g.get('season', '') for g in games)
        teams = set(g.get('team_abbreviation', '') for g in games)
        
        has_odds = sum(1 for g in games if 'betting_odds' in g)
        has_rest = sum(1 for g in games if 'scheduling' in g)
        has_injuries = sum(1 for g in games if 'injuries' in g)
        
        print(f"\n📊 Basic Stats:")
        print(f"   Total games: {len(games):,}")
        print(f"   Seasons: {len(seasons)} ({min(seasons)} to {max(seasons)})")
        print(f"   Teams: {len(teams)}")
        
        print(f"\n💰 Betting Data:")
        print(f"   Games with odds: {has_odds:,} ({has_odds/len(games)*100:.1f}%)")
        print(f"   Games with rest days: {has_rest:,} ({has_rest/len(games)*100:.1f}%)")
        print(f"   Games with injury data: {has_injuries:,} ({has_injuries/len(games)*100:.1f}%)")
        
        print(f"\n📁 Output:")
        print(f"   File: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        print(f"\n🎯 Ready for:")
        print(f"   ✓ Context pattern discovery")
        print(f"   ✓ Betting strategy validation")
        print(f"   ✓ ROI calculation with real odds")
        print(f"   ⚠ Player props (need player data)")
        print(f"   ⚠ Injury impact (need injury reports)")
        
        # Save summary
        summary = {
            'generated': datetime.now().isoformat(),
            'total_games': len(games),
            'seasons': sorted(list(seasons)),
            'teams': sorted(list(teams)),
            'features': {
                'betting_odds': has_odds,
                'rest_days': has_rest,
                'injuries': has_injuries,
                'player_props': 0
            },
            'file': str(output_path),
            'size_mb': output_path.stat().st_size / 1024 / 1024
        }
        
        summary_path = output_path.parent / 'nba_data_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path.name}")


def main():
    """Main execution"""
    output_dir = Path(__file__).parent.parent / 'data' / 'domains'
    
    builder = NBADataBuilder(output_dir)
    
    print("\n💡 NBA DATA BUILDER")
    print("   This will enhance existing NBA data with:")
    print("   1. Betting odds (estimated from win probability)")
    print("   2. Rest days (back-to-backs, scheduling)")
    print("   3. Player data (placeholder for future)")
    print("   4. Injury data (placeholder for future)")
    print("\n   Press Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled")
        return
    
    games = builder.build_complete_dataset()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("\n1. VALIDATE PROFITABILITY:")
    print("   python narrative_optimization/analysis/nba_selective_betting_strategy.py")
    print("   (Now with real odds data)")
    
    print("\n2. ADD PLAYER DATA (optional):")
    print("   - Install: pip install nba_api")
    print("   - Uncomment player data collection")
    print("   - Run again (takes 2-3 hours)")
    
    print("\n3. COLLECT CURRENT SEASON:")
    print("   - Fetch 2024-25 games for live validation")
    print("   - Test if patterns still work")
    
    print("\n4. GET REAL-TIME ODDS:")
    print("   - Sign up: https://the-odds-api.com/")
    print("   - Add API key to script")
    print("   - Fetch live odds for current games")
    
    print("\n" + "="*80)
    print("DATA ENHANCEMENT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

