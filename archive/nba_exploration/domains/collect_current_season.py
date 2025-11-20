"""
Collect Current NBA Season Data (2024-2025)

Fetches games from the current season for real-time predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime

def fetch_current_season_nba():
    """
    Fetch 2024-25 NBA season games.
    
    Note: This requires nba_api package.
    Install with: pip install nba_api
    """
    
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams
    except ImportError:
        print("⚠️  nba_api not installed")
        print("   Install with: pip install nba_api")
        print("\nCreating placeholder structure...")
        
        # Create placeholder file
        placeholder = {
            'season': '2024-25',
            'games': [],
            'note': 'Install nba_api to fetch real data: pip install nba_api',
            'last_updated': datetime.now().isoformat()
        }
        
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_2024_2025_season.json'
        with open(output_path, 'w') as f:
            json.dump(placeholder, f, indent=2)
        
        print(f"✓ Placeholder created at: {output_path}")
        return placeholder
    
    print(f"\n{'='*60}")
    print("COLLECTING NBA 2024-25 SEASON DATA")
    print(f"{'='*60}\n")
    
    # Get all teams
    nba_teams = teams.get_teams()
    print(f"✓ Found {len(nba_teams)} NBA teams\n")
    
    # Fetch games for current season
    season = '2024-25'
    all_games = []
    
    print("Fetching games...")
    for i, team in enumerate(nba_teams):
        team_name = team['full_name']
        team_id = team['id']
        
        try:
            # Get games for this team
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                timeout=30
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            if not games_df.empty:
                games = games_df.to_dict('records')
                all_games.extend(games)
                print(f"  [{i+1}/{len(nba_teams)}] {team_name}: {len(games)} games")
            else:
                print(f"  [{i+1}/{len(nba_teams)}] {team_name}: No games yet")
            
        except Exception as e:
            print(f"  [{i+1}/{len(nba_teams)}] {team_name}: Error - {str(e)}")
            continue
    
    # Remove duplicates (each game appears twice - once for each team)
    unique_games = {}
    for game in all_games:
        game_id = game.get('GAME_ID')
        if game_id and game_id not in unique_games:
            unique_games[game_id] = game
    
    unique_games_list = list(unique_games.values())
    
    print(f"\n✓ Collected {len(unique_games_list)} unique games for {season}\n")
    
    # Save to file
    output_data = {
        'season': season,
        'games': unique_games_list,
        'total_games': len(unique_games_list),
        'last_updated': datetime.now().isoformat()
    }
    
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_2024_2025_season.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"NBA {season} DATA COLLECTION COMPLETE")
    print(f"{'='*60}\n")
    
    return output_data

if __name__ == '__main__':
    data = fetch_current_season_nba()

