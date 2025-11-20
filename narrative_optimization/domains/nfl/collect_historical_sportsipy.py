"""
Collect 2000-2013 NFL Data using sportsipy
"""

import json
from pathlib import Path
from datetime import datetime
import time

try:
    from sportsipy.nfl.schedule import Schedule
    from sportsipy.nfl.teams import Teams
except ImportError:
    print("sportsipy not available. Install with: pip install sportsipy")
    exit(1)

print("="*80)
print("COLLECTING NFL 2000-2013 DATA via sportsipy")
print("="*80)

# Collect seasons
collected_games = []

for year in range(2000, 2014):
    print(f"\n[{year}] Collecting season {year}...")
    
    try:
        # Get all teams for this year
        teams = Teams(year)
        
        season_games = []
        
        for team in teams:
            print(f"  {team.abbreviation}...", end=" ", flush=True)
            
            try:
                # Get schedule for team
                schedule = Schedule(team.abbreviation, year)
                
                for game in schedule:
                    # Only add each game once (home team)
                    if game.location == 'Home':
                        game_data = {
                            'season': year,
                            'week': game.week,
                            'gameday': str(game.datetime),
                            'home_team': team.abbreviation,
                            'away_team': game.opponent_abbr,
                            'home_score': game.points_scored if game.points_scored else 0,
                            'away_score': game.points_allowed if game.points_allowed else 0,
                            'home_won': game.result == 'Win',
                        }
                        season_games.append(game_data)
                
                print(f"✓", end=" ")
                time.sleep(0.5)  # Be respectful to server
                
            except Exception as e:
                print(f"✗ ({str(e)[:30]})", end=" ")
        
        print(f"\n  ✓ {len(season_games)} games collected for {year}")
        collected_games.extend(season_games)
        
    except Exception as e:
        print(f"  ✗ Failed to collect {year}: {e}")

print(f"\n✓ Total collected: {len(collected_games)} games")

# Save
output_path = Path(__file__).parent / 'historical_data' / 'nfl_2000_2013_basic.json'
output_path.parent.mkdir(exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(collected_games, f, indent=2)

print(f"✓ Saved to: {output_path}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

print("\nNext: Process this data to add QB/Coach names and merge with existing")
print("="*80)

