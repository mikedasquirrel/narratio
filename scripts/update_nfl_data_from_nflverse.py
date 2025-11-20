#!/usr/bin/env python3
"""
NFL Data Updater - Using nflverse Data
Downloads latest NFL data from nflverse and integrates into our system

Source: https://github.com/nflverse/nflverse-data.git
Documentation: https://nflreadr.nflverse.com/

This script:
1. Downloads play-by-play data from nflverse
2. Processes games with scores and outcomes
3. Enriches with team/player information
4. Saves in our standard format
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# nflverse data URLs
# Data is hosted at: https://github.com/nflverse/nflverse-data/releases
# We'll use direct CSV URLs from their CDN
NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Main data endpoints - try multiple possible locations
URLS = {
    # Try both release tags
    'schedules_options': [
        "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
        f"{NFLVERSE_BASE}/schedules/schedules.parquet",
        f"{NFLVERSE_BASE}/schedules/schedules.csv",
    ],
    'pbp': f"{NFLVERSE_BASE}/pbp/play_by_play_{{}}.parquet",
    'rosters': f"{NFLVERSE_BASE}/rosters/roster_{{}}.parquet",
}

def download_csv(url, description="data"):
    """Download CSV from URL"""
    print(f"📥 Downloading {description}...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save to temp file and read with pandas
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        print(f"✓ Downloaded {len(df):,} rows")
        return df
    except Exception as e:
        print(f"✗ Failed to download {description}: {e}")
        return None

def download_pbp_year(year):
    """Download play-by-play data for specific year"""
    url = f"{NFLVERSE_BASE}/pbp/play_by_play_{year}.csv.gz"
    print(f"📥 Downloading play-by-play {year}...")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Read gzipped CSV
        import gzip
        from io import BytesIO
        df = pd.read_csv(BytesIO(response.content), compression='gzip', low_memory=False)
        print(f"✓ Downloaded {len(df):,} plays for {year}")
        return df
    except Exception as e:
        print(f"✗ Failed to download {year}: {e}")
        return None

def aggregate_games_from_pbp(pbp_df):
    """Aggregate play-by-play data into game-level data"""
    if pbp_df is None or len(pbp_df) == 0:
        return []
    
    print(f"🔄 Aggregating {len(pbp_df):,} plays into games...")
    
    # Group by game
    game_groups = pbp_df.groupby('game_id')
    
    games = []
    for game_id, game_plays in game_groups:
        try:
            # Get first play for game info
            first_play = game_plays.iloc[0]
            last_play = game_plays.iloc[-1]
            
            # Extract game information
            game = {
                'game_id': game_id,
                'season': int(first_play['season']),
                'week': int(first_play['week']) if pd.notna(first_play['week']) else None,
                'game_type': first_play.get('season_type', 'REG'),
                'gameday': first_play.get('game_date', ''),
                'gametime': '',
                
                # Teams
                'home_team': first_play['home_team'],
                'away_team': first_play['away_team'],
                
                # Final scores (from last play)
                'home_score': int(last_play['total_home_score']) if pd.notna(last_play['total_home_score']) else 0,
                'away_score': int(last_play['total_away_score']) if pd.notna(last_play['total_away_score']) else 0,
                
                # Outcome
                'home_won': last_play['total_home_score'] > last_play['total_away_score'],
                'winner': first_play['home_team'] if last_play['total_home_score'] > last_play['total_away_score'] else first_play['away_team'],
                'loser': first_play['away_team'] if last_play['total_home_score'] > last_play['total_away_score'] else first_play['home_team'],
                'result': int(last_play['total_home_score'] - last_play['total_away_score']),
                
                # Stadium and weather
                'stadium': first_play.get('stadium', ''),
                'location': first_play.get('location', ''),
                'roof': first_play.get('roof', ''),
                'surface': first_play.get('surface', ''),
                'temp': first_play.get('temp', None),
                'wind': first_play.get('wind', None),
                
                # Spread info
                'spread_line': first_play.get('spread_line', None),
                'total_line': first_play.get('total_line', None),
                
                # Coaches (if available)
                'home_coach': first_play.get('home_coach', ''),
                'away_coach': first_play.get('away_coach', ''),
                
                # Game context
                'div_game': first_play.get('div_game', 0) == 1,
                'playoff': first_play.get('season_type', 'REG') == 'POST',
                'overtime': game_plays['qtr'].max() > 4 if 'qtr' in game_plays.columns else False,
                
                # Play count
                'total_plays': len(game_plays),
            }
            
            games.append(game)
            
        except Exception as e:
            print(f"⚠ Error processing game {game_id}: {e}")
            continue
    
    print(f"✓ Aggregated {len(games):,} games")
    return games

def download_rosters_year(year):
    """Download roster data for specific year"""
    url = URLS['rosters'].format(year)
    return download_csv(url, f"rosters {year}")

def download_team_info():
    """Download team logos and info"""
    return download_csv(URLS['team_logos'], "team information")

def process_games(schedules_df):
    """Process schedule data into our game format"""
    print("\n🔄 Processing games...")
    
    # Filter to completed games only
    completed = schedules_df[schedules_df['result'].notna()].copy()
    print(f"✓ Found {len(completed):,} completed games")
    
    games = []
    for idx, row in completed.iterrows():
        try:
            # Determine winner
            home_won = row['home_score'] > row['away_score']
            
            game = {
                'game_id': row['game_id'],
                'season': int(row['season']),
                'week': int(row['week']) if pd.notna(row['week']) else None,
                'game_type': row['game_type'],
                'gameday': row['gameday'],
                'gametime': row.get('gametime', ''),
                'weekday': row.get('weekday', ''),
                
                # Teams
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                
                # Scores
                'home_score': int(row['home_score']) if pd.notna(row['home_score']) else None,
                'away_score': int(row['away_score']) if pd.notna(row['away_score']) else None,
                'result': row['result'],
                
                # Outcome
                'home_won': home_won,
                'winner': row['home_team'] if home_won else row['away_team'],
                'loser': row['away_team'] if home_won else row['home_team'],
                
                # Spread info (if available)
                'spread_line': row.get('spread_line', None),
                'total_line': row.get('total_line', None),
                'home_spread_odds': row.get('home_spread_odds', None),
                'away_spread_odds': row.get('away_spread_odds', None),
                
                # Stadium info
                'stadium': row.get('stadium', ''),
                'location': row.get('location', ''),
                'roof': row.get('roof', ''),
                'surface': row.get('surface', ''),
                'temp': row.get('temp', None),
                'wind': row.get('wind', None),
                
                # Coaches
                'home_coach': row.get('home_coach', ''),
                'away_coach': row.get('away_coach', ''),
                
                # Metadata
                'div_game': row.get('div_game', 0) == 1,
                'playoff': row.get('game_type', '') == 'POST',
                'overtime': row.get('overtime', 0) == 1,
            }
            
            games.append(game)
            
        except Exception as e:
            print(f"⚠ Error processing game {row.get('game_id', 'unknown')}: {e}")
            continue
    
    print(f"✓ Processed {len(games):,} games")
    return games

def add_season_context(games):
    """Add season context (week number, playoff status, etc.)"""
    print("\n🔄 Adding season context...")
    
    # Sort by season and week
    games_df = pd.DataFrame(games)
    games_df = games_df.sort_values(['season', 'week', 'gameday'])
    
    # Add season progress (0-1)
    for season in games_df['season'].unique():
        season_mask = games_df['season'] == season
        season_games = games_df[season_mask]
        
        # Regular season games
        reg_season = season_games[season_games['game_type'] == 'REG']
        max_week = reg_season['week'].max() if len(reg_season) > 0 else 18
        
        # Calculate progress
        games_df.loc[season_mask, 'season_progress'] = games_df.loc[season_mask, 'week'] / max_week
    
    print(f"✓ Added season context")
    return games_df.to_dict('records')

def save_data(games, output_path):
    """Save processed games to JSON"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    data = {
        'metadata': {
            'source': 'nflverse-data',
            'url': 'https://github.com/nflverse/nflverse-data',
            'updated': datetime.now().isoformat(),
            'total_games': len(games),
            'seasons': sorted(list(set(g['season'] for g in games))),
            'description': 'NFL game data from nflverse, processed for narrative analysis'
        },
        'games': games
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved {len(games):,} games to {output_file}")
    return output_file

def create_summary_report(games, output_path):
    """Create summary report of the data"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    df = pd.DataFrame(games)
    
    print(f"\nTotal Games: {len(games):,}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"Total Weeks: {df['week'].nunique()}")
    
    print("\n--- Games by Season ---")
    season_counts = df['season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count:,} games")
    
    print("\n--- Game Types ---")
    game_type_counts = df['game_type'].value_counts()
    for game_type, count in game_type_counts.items():
        print(f"  {game_type}: {count:,} games")
    
    print("\n--- Teams ---")
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    print(f"  Total Teams: {len(all_teams)}")
    print(f"  Teams: {', '.join(sorted(all_teams))}")
    
    print("\n--- Stadium Info ---")
    print(f"  Unique Stadiums: {df['stadium'].nunique()}")
    print(f"  Games with Weather Data: {df['temp'].notna().sum():,}")
    print(f"  Games with Spread: {df['spread_line'].notna().sum():,}")
    
    print("\n--- Special Games ---")
    print(f"  Division Games: {df['div_game'].sum():,}")
    print(f"  Playoff Games: {df['playoff'].sum():,}")
    print(f"  Overtime Games: {df['overtime'].sum():,}")
    
    print("\n" + "="*60)
    print(f"✓ Data saved to: {output_path}")
    print("="*60)

def main():
    """Main execution"""
    print("="*60)
    print("NFL DATA UPDATER - NFLVERSE SOURCE")
    print("="*60)
    print("Source: https://github.com/nflverse/nflverse-data")
    print()
    
    # Download play-by-play data for recent seasons
    current_year = datetime.now().year
    start_year = 2014  # Go back to 2014 for comprehensive data
    
    all_games = []
    
    for year in range(start_year, current_year + 1):
        print(f"\n--- Processing {year} season ---")
        pbp_df = download_pbp_year(year)
        
        if pbp_df is not None:
            games = aggregate_games_from_pbp(pbp_df)
            all_games.extend(games)
            print(f"✓ Added {len(games):,} games from {year}")
        else:
            print(f"⚠ Skipping {year}")
    
    if not all_games:
        print("\n✗ No games collected")
        return 1
    
    print(f"\n✓ Total games collected: {len(all_games):,}")
    
    # Add season context
    all_games = add_season_context(all_games)
    
    # Save data
    output_path = Path(__file__).parent.parent / "data" / "domains" / "nfl_complete_nflverse.json"
    save_data(all_games, output_path)
    
    # Create summary
    create_summary_report(all_games, output_path)
    
    print("\n🎉 NFL data update complete!")
    print("\nNext steps:")
    print("1. Review the data: data/domains/nfl_complete_nflverse.json")
    print("2. Run narrative feature extraction on the new data")
    print("3. Update domain analysis with enriched dataset")
    print("\n📊 Database updated with complete nflverse data (2014-present)")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

