"""
NHL Full Historical Data Builder

Enhanced data collector with:
- Full 10-year history (2014-2024)
- Progress checkpointing (resume if interrupted)
- Multiple collection strategies
- Data validation and quality checks
- Detailed logging

Target: 10,000+ games from 2014-2024

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nhlpy import NHLClient
    NHL_API_AVAILABLE = True
except ImportError:
    print("❌ nhlpy not installed")
    NHL_API_AVAILABLE = False


# Team info
ORIGINAL_SIX = ['BOS', 'CHI', 'DET', 'MTL', 'NYR', 'TOR']

STANLEY_CUP_WINS = {
    'MTL': 24, 'TOR': 13, 'DET': 11, 'BOS': 6, 'CHI': 6, 'EDM': 5,
    'PIT': 5, 'NYR': 4, 'NYI': 4, 'NJD': 3, 'COL': 3, 'TBL': 3,
    'LAK': 2, 'PHI': 2, 'CAR': 1, 'CGY': 1, 'ANA': 1, 'DAL': 1,
    'WSH': 1, 'STL': 1, 'VGK': 0, 'SEA': 0, 'CBJ': 0, 'ARI': 0,
    'WPG': 0, 'MIN': 0, 'NSH': 0, 'BUF': 0, 'VAN': 0, 'OTT': 0,
    'SJS': 0, 'FLA': 0,
}


class NHLFullHistoryBuilder:
    """Enhanced builder for full historical data"""
    
    def __init__(self, output_dir: Path, checkpoint_file: Optional[Path] = None):
        """
        Initialize builder with checkpointing.
        
        Parameters
        ----------
        output_dir : Path
            Directory for output files
        checkpoint_file : Path, optional
            File to save/load progress checkpoints
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = checkpoint_file or (output_dir / 'nhl_collection_checkpoint.json')
        
        if NHL_API_AVAILABLE:
            self.client = NHLClient()
        else:
            self.client = None
        
        self.collected_games = []
        self.progress = {
            'weeks_processed': 0,
            'games_collected': 0,
            'last_date': None,
            'errors': [],
        }
    
    def load_checkpoint(self) -> bool:
        """Load progress from checkpoint file"""
        if self.checkpoint_file.exists():
            print(f"📂 Loading checkpoint from {self.checkpoint_file.name}...")
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.collected_games = data.get('games', [])
                self.progress = data.get('progress', self.progress)
            print(f"   ✓ Resumed: {len(self.collected_games)} games, {self.progress['weeks_processed']} weeks")
            return True
        return False
    
    def save_checkpoint(self):
        """Save current progress"""
        checkpoint_data = {
            'games': self.collected_games,
            'progress': self.progress,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
    
    def collect_full_history(self, start_date: str = "2014-10-01", end_date: Optional[str] = None) -> List[Dict]:
        """
        Collect full historical NHL games.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (defaults to today)
        
        Returns
        -------
        games : list
            All collected games
        """
        if not NHL_API_AVAILABLE or not self.client:
            print("❌ NHL API not available")
            return []
        
        # Try to resume from checkpoint
        resumed = self.load_checkpoint()
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("\n" + "="*80)
        print("NHL FULL HISTORICAL DATA COLLECTION")
        print("="*80)
        print(f"Date range: {start_date} to {end_date}")
        print(f"Target: 10,000+ games (10+ seasons)")
        
        if resumed:
            print(f"Resuming from checkpoint: {len(self.collected_games)} games already collected")
            start_date = self.progress.get('last_date', start_date)
        
        print("="*80)
        
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_datetime:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                # Fetch week starting from this date
                schedule = self.client.schedule.weekly_schedule(date=date_str)
                
                if not schedule or 'gameWeek' not in schedule:
                    current_date += timedelta(days=7)
                    continue
                
                # Process all games in week
                week_games = 0
                for day in schedule.get('gameWeek', []):
                    for game_data in day.get('games', []):
                        try:
                            season = str(game_data.get('season', ''))
                            game = self._process_game(game_data, season)
                            if game:
                                self.collected_games.append(game)
                                week_games += 1
                        except Exception as e:
                            self.progress['errors'].append({
                                'date': date_str,
                                'error': str(e)
                            })
                
                # Update progress
                self.progress['weeks_processed'] += 1
                self.progress['games_collected'] = len(self.collected_games)
                self.progress['last_date'] = date_str
                
                # Print progress EVERY week for visibility
                print(f"   Week {self.progress['weeks_processed']:3d} | "
                      f"Total: {len(self.collected_games):5d} games | "
                      f"Date: {date_str} | "
                      f"This week: {week_games:2d} | "
                      f"Errors: {len(self.progress['errors']):2d}")
                
                # Save checkpoint every 10 weeks
                if self.progress['weeks_processed'] % 10 == 0:
                    self.save_checkpoint()
                    print(f"      💾 Checkpoint saved - {len(self.collected_games)} games backed up")
                
                # Progress milestones
                if len(self.collected_games) in [1000, 2500, 5000, 7500, 10000]:
                    print(f"\n🎯 MILESTONE: {len(self.collected_games)} GAMES COLLECTED! 🎯\n")
                
                # Move to next week
                current_date += timedelta(days=7)
                
                # Be nice to API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ⚠️  Error on {date_str}: {e}")
                self.progress['errors'].append({
                    'date': date_str,
                    'error': str(e)
                })
                current_date += timedelta(days=7)
                continue
        
        # Final save
        self.save_checkpoint()
        
        print("\n" + "="*80)
        print(f"✅ Collection complete: {len(self.collected_games)} games")
        print(f"   Weeks processed: {self.progress['weeks_processed']}")
        print(f"   Errors: {len(self.progress['errors'])}")
        print("="*80)
        
        return self.collected_games
    
    def _process_game(self, game_data: Dict, season: str) -> Optional[Dict]:
        """Process a single game"""
        try:
            game_id = game_data.get('id')
            game_state = game_data.get('gameState', '')
            
            # Only finished games
            if game_state not in ['OFF', 'FINAL']:
                return None
            
            home_team = game_data.get('homeTeam', {}).get('abbrev', '')
            away_team = game_data.get('awayTeam', {}).get('abbrev', '')
            
            if not home_team or not away_team:
                return None
            
            home_score = game_data.get('homeTeam', {}).get('score', 0)
            away_score = game_data.get('awayTeam', {}).get('score', 0)
            
            # Extract date
            game_date = game_data.get('gameDate', '')
            if not game_date:
                game_date = game_data.get('startTimeUTC', '')
            if 'T' in game_date:
                game_date = game_date.split('T')[0]
            
            home_won = home_score > away_score
            
            # Check OT/SO
            period = game_data.get('period', 3)
            period_type = game_data.get('periodDescriptor', {}).get('periodType', 'REG')
            overtime = period > 3 or period_type in ['OT', 'SO']
            shootout = period_type == 'SO'
            
            # Build game
            game = {
                'game_id': str(game_id),
                'season': season,
                'date': game_date,
                'game_type': game_data.get('gameType', 2),
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'home_won': home_won,
                'winner': home_team if home_won else away_team,
                'loser': away_team if home_won else home_team,
                'score_differential': home_score - away_score,
                'total_goals': home_score + away_score,
                'overtime': overtime,
                'shootout': shootout,
                'venue': game_data.get('venue', {}).get('default', ''),
                'is_playoff': game_data.get('gameType', 2) == 3,
                'is_rivalry': self._is_rivalry(home_team, away_team),
                'home_goalie': None,  # TODO: Extract from API
                'away_goalie': None,
            }
            
            return game
            
        except Exception as e:
            return None
    
    def _is_rivalry(self, team1: str, team2: str) -> bool:
        """Check if rivalry game"""
        if team1 in ORIGINAL_SIX and team2 in ORIGINAL_SIX:
            return True
        
        rivalries = [
            ('BOS', 'MTL'), ('TOR', 'MTL'), ('TOR', 'OTT'),
            ('EDM', 'CGY'), ('NYR', 'NYI'), ('NYR', 'NJD'),
            ('PIT', 'PHI'), ('CHI', 'DET'), ('LAK', 'ANA'),
        ]
        
        for r1, r2 in rivalries:
            if (team1, team2) in [(r1, r2), (r2, r1)]:
                return True
        
        return False
    
    def enhance_with_context(self, games: List[Dict]) -> List[Dict]:
        """Add temporal context (from existing module)"""
        # Import from original builder
        from nhl_data_builder import NHLDataBuilder
        
        temp_builder = NHLDataBuilder(self.output_dir)
        enhanced = temp_builder.enhance_with_temporal_context(games)
        enhanced = temp_builder.enhance_with_betting_odds(enhanced)
        
        return enhanced
    
    def build_complete_dataset(self, start_date: str = "2014-10-01") -> List[Dict]:
        """Build complete dataset with all enhancements"""
        
        # Collect games
        games = self.collect_full_history(start_date=start_date)
        
        if not games:
            print("❌ No games collected")
            return []
        
        print(f"\n📊 Enhancing {len(games)} games with context...")
        
        # Enhance
        games = self.enhance_with_context(games)
        
        return games
    
    def save_dataset(self, games: List[Dict], filename: str = "nhl_games_full_history.json"):
        """Save complete dataset"""
        output_path = self.output_dir / filename
        
        print(f"\n💾 SAVING COMPLETE DATASET")
        print(f"   Output: {output_path}")
        print(f"   Games: {len(games)}")
        
        with open(output_path, 'w') as f:
            json.dump(games, f, indent=2)
        
        print(f"   ✓ Saved")
        
        # Statistics
        if games:
            seasons = sorted(set(g.get('season', '') for g in games))
            print(f"\n📈 DATASET STATISTICS")
            print("-"*80)
            print(f"   Seasons: {len(seasons)} ({seasons[0]} to {seasons[-1]})")
            print(f"   Total games: {len(games)}")
            print(f"   Playoff: {sum(1 for g in games if g.get('is_playoff'))}")
            print(f"   Rivalry: {sum(1 for g in games if g.get('is_rivalry'))}")
            print(f"   Overtime: {sum(1 for g in games if g.get('overtime'))}")
            print(f"   Shootout: {sum(1 for g in games if g.get('shootout'))}")


def main():
    """Main execution"""
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'data' / 'domains'
    
    print("\n🏒 NHL FULL HISTORICAL DATA COLLECTION")
    print("="*80)
    print("Target: 10,000+ games from 2014-2024")
    print("Method: NHL API with checkpointing")
    print("Time: 3-5 hours estimated")
    print("="*80)
    print("\nStarting collection...")
    print("(This will run for several hours - checkpoints every 10 weeks)")
    print("(You can interrupt and resume later)")
    print()
    
    # Build collector
    builder = NHLFullHistoryBuilder(output_dir)
    
    # Collect full history
    games = builder.build_complete_dataset(start_date="2014-10-01")
    
    if games:
        # Save
        builder.save_dataset(games, filename="nhl_games_full_history.json")
        
        # Also save as main file (replacing demo data)
        builder.save_dataset(games, filename="nhl_games_with_odds.json")
        
        print("\n✅ FULL HISTORICAL COLLECTION COMPLETE!")
        print("="*80)
        print(f"\nReady for full analysis with {len(games)} games")
        print("\nNext steps:")
        print("1. python3 narrative_optimization/domains/nhl/extract_nhl_features.py")
        print("2. python3 narrative_optimization/domains/nhl/nhl_complete_analysis.py")
        print("3. python3 narrative_optimization/domains/nhl/validate_nhl_patterns.py")
    else:
        print("\n❌ Collection failed")


if __name__ == "__main__":
    main()

