"""
Collect Current NFL Season Data (2024)

Fetches games from the current season for real-time predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime

def fetch_current_season_nfl():
    """
    Fetch 2024 NFL season games.
    
    Note: This requires nfl_data_py package.
    Install with: pip install nfl_data_py
    """
    
    try:
        import nfl_data_py as nfl
    except ImportError:
        print("⚠️  nfl_data_py not installed")
        print("   Install with: pip install nfl_data_py")
        print("\nCreating placeholder structure...")
        
        # Create placeholder file
        placeholder = {
            'season': 2024,
            'games': [],
            'note': 'Install nfl_data_py to fetch real data: pip install nfl_data_py',
            'last_updated': datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_2024_season.json'
        with open(output_path, 'w') as f:
            json.dump(placeholder, f, indent=2)
        
        print(f"✓ Placeholder created at: {output_path}")
        return placeholder
    
    print(f"\n{'='*60}")
    print("COLLECTING NFL 2024 SEASON DATA")
    print(f"{'='*60}\n")
    
    current_year = 2024
    
    try:
        # Import schedule
        print("Fetching schedule...")
        schedule = nfl.import_schedules([current_year])
        
        print(f"✓ Fetched {len(schedule)} games\n")
        
        # Import weekly data (player stats, etc.)
        print("Fetching weekly data...")
        try:
            weekly = nfl.import_weekly_data([current_year])
            print(f"✓ Fetched weekly data\n")
        except Exception as e:
            print(f"⚠️  Could not fetch weekly data: {e}\n")
            weekly = None
        
        # Convert schedule to list of dicts
        games = []
        for _, game in schedule.iterrows():
            game_dict = game.to_dict()
            
            # Convert any timestamps to strings
            for key, value in game_dict.items():
                if hasattr(value, 'isoformat'):
                    game_dict[key] = value.isoformat()
                elif isinstance(value, (float,)) and (value != value):  # NaN check
                    game_dict[key] = None
            
            games.append(game_dict)
        
        print(f"✓ Processed {len(games)} games")
        
        # Save to file
        output_data = {
            'season': current_year,
            'games': games,
            'total_games': len(games),
            'last_updated': datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_2024_season.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved to: {output_path}")
        
        print(f"\n{'='*60}")
        print(f"NFL {current_year} DATA COLLECTION COMPLETE")
        print(f"{'='*60}\n")
        
        return output_data
        
    except Exception as e:
        print(f"✗ Error fetching NFL data: {e}")
        
        # Create error placeholder
        error_data = {
            'season': current_year,
            'games': [],
            'error': str(e),
            'last_updated': datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_2024_season.json'
        with open(output_path, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        return error_data

if __name__ == '__main__':
    data = fetch_current_season_nfl()

