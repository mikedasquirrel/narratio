#!/usr/bin/env python3
"""
NFL Data Enrichment - Add Rosters, Players, and Matchup Details
Downloads roster data, player stats, and team info from nflverse

Source: https://github.com/nflverse/nflverse-data
This adds:
- Season rosters (QB names, key players)
- Player stats and performance
- Team rankings and records
- Head-to-head matchup context
"""

import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

def download_rosters_year(year):
    """Download roster data for specific year"""
    url = f"{NFLVERSE_BASE}/weekly_rosters/roster_weekly_{year}.csv.gz"
    print(f"📥 Downloading rosters for {year}...")
    try:
        import gzip
        from io import BytesIO
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        df = pd.read_csv(BytesIO(response.content), compression='gzip', low_memory=False)
        print(f"✓ Downloaded {len(df):,} roster entries for {year}")
        return df
    except Exception as e:
        print(f"✗ Failed to download rosters for {year}: {e}")
        return None

def download_team_stats_year(year):
    """Download team-level stats for year"""
    url = f"{NFLVERSE_BASE}/stats_team/team_season_{year}.csv"
    print(f"📥 Downloading team stats for {year}...")
    try:
        df = pd.read_csv(url, low_memory=False)
        print(f"✓ Downloaded {len(df)} teams for {year}")
        return df
    except Exception as e:
        print(f"⚠ Team stats not available for {year}: {e}")
        return None

def download_player_stats_year(year):
    """Download player-level stats for year"""
    url = f"{NFLVERSE_BASE}/stats_player/player_stats_{year}.csv"
    print(f"📥 Downloading player stats for {year}...")
    try:
        df = pd.read_csv(url, low_memory=False)
        print(f"✓ Downloaded {len(df):,} player records for {year}")
        return df
    except Exception as e:
        print(f"⚠ Player stats not available for {year}: {e}")
        return None

def extract_qb_info(rosters_df, team, week):
    """Extract QB information for a team at a specific week"""
    if rosters_df is None:
        return None
    
    # Filter to team and week
    team_roster = rosters_df[
        (rosters_df['team'] == team) & 
        (rosters_df['week'] == week) &
        (rosters_df['position'] == 'QB') &
        (rosters_df['depth_chart_position'].isin(['QB', '1', 'Starter']))
    ]
    
    if len(team_roster) == 0:
        # Try without depth chart filter
        team_roster = rosters_df[
            (rosters_df['team'] == team) & 
            (rosters_df['week'] == week) &
            (rosters_df['position'] == 'QB')
        ]
    
    if len(team_roster) > 0:
        qb = team_roster.iloc[0]
        return {
            'qb_name': qb.get('full_name', qb.get('player_name', 'Unknown')),
            'qb_id': qb.get('gsis_id', ''),
            'qb_jersey': qb.get('jersey_number', None),
            'qb_status': qb.get('status', 'Active'),
        }
    return None

def extract_key_players(rosters_df, team, week, top_n=5):
    """Extract top players for a team"""
    if rosters_df is None:
        return []
    
    team_roster = rosters_df[
        (rosters_df['team'] == team) & 
        (rosters_df['week'] == week)
    ]
    
    if len(team_roster) == 0:
        return []
    
    # Priority positions for narrative analysis
    priority_positions = ['QB', 'RB', 'WR', 'TE', 'DE', 'LB', 'CB']
    
    key_players = []
    for pos in priority_positions:
        pos_players = team_roster[team_roster['position'] == pos]
        if len(pos_players) > 0:
            player = pos_players.iloc[0]
            key_players.append({
                'name': player.get('full_name', player.get('player_name', 'Unknown')),
                'position': pos,
                'jersey': player.get('jersey_number', None),
            })
        if len(key_players) >= top_n:
            break
    
    return key_players

def calculate_team_records(games, season):
    """Calculate win-loss records for teams through season"""
    season_games = [g for g in games if g['season'] == season]
    season_games = sorted(season_games, key=lambda x: (x.get('week', 99), x.get('gameday', '')))
    
    records = defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': []})
    
    for game in season_games:
        week = game.get('week', 0)
        home = game['home_team']
        away = game['away_team']
        
        # Store record BEFORE this game
        game['home_record_before'] = f"{records[home]['wins']}-{records[home]['losses']}"
        game['away_record_before'] = f"{records[away]['wins']}-{records[away]['losses']}"
        
        # Update records
        if game['home_won']:
            records[home]['wins'] += 1
            records[away]['losses'] += 1
        else:
            records[away]['wins'] += 1
            records[home]['losses'] += 1
        
        records[home]['games'].append(game['game_id'])
        records[away]['games'].append(game['game_id'])
    
    return season_games

def add_matchup_context(game, all_games):
    """Add head-to-head matchup context"""
    home = game['home_team']
    away = game['away_team']
    season = game['season']
    
    # Find previous matchups between these teams
    previous_matchups = [
        g for g in all_games
        if g['season'] < season and
        ((g['home_team'] == home and g['away_team'] == away) or
         (g['home_team'] == away and g['away_team'] == home))
    ]
    
    if previous_matchups:
        # Calculate head-to-head record
        home_wins = sum(1 for g in previous_matchups 
                       if (g['home_team'] == home and g['home_won']) or
                          (g['away_team'] == home and not g['home_won']))
        away_wins = len(previous_matchups) - home_wins
        
        game['matchup_history'] = {
            'total_games': len(previous_matchups),
            'home_wins': home_wins,
            'away_wins': away_wins,
            'last_meeting_season': previous_matchups[-1]['season'],
            'last_winner': previous_matchups[-1]['winner'],
        }
    
    return game

def enrich_games_with_rosters(games, start_year=2014, end_year=2025):
    """Enrich games with roster and matchup data"""
    print("\n" + "="*60)
    print("ENRICHING GAMES WITH ROSTERS & MATCHUPS")
    print("="*60)
    
    # Download all rosters
    rosters_by_year = {}
    for year in range(start_year, end_year + 1):
        rosters_df = download_rosters_year(year)
        if rosters_df is not None:
            rosters_by_year[year] = rosters_df
    
    # Calculate team records
    print("\n🔄 Calculating team records...")
    games_by_season = defaultdict(list)
    for game in games:
        games_by_season[game['season']].append(game)
    
    enriched_games = []
    for season in sorted(games_by_season.keys()):
        season_games = calculate_team_records(games_by_season[season], season)
        enriched_games.extend(season_games)
    
    # Enrich with roster data
    print("\n🔄 Adding roster data to games...")
    games_with_rosters = []
    
    for game in enriched_games:
        season = game['season']
        week = game.get('week', 1)
        
        if season in rosters_by_year:
            rosters_df = rosters_by_year[season]
            
            # Add QB info
            home_qb = extract_qb_info(rosters_df, game['home_team'], week)
            away_qb = extract_qb_info(rosters_df, game['away_team'], week)
            
            if home_qb:
                game['home_qb'] = home_qb
            if away_qb:
                game['away_qb'] = away_qb
            
            # Add key players
            game['home_key_players'] = extract_key_players(rosters_df, game['home_team'], week)
            game['away_key_players'] = extract_key_players(rosters_df, game['away_team'], week)
        
        # Add matchup context
        game = add_matchup_context(game, enriched_games)
        
        games_with_rosters.append(game)
    
    print(f"✓ Enriched {len(games_with_rosters):,} games")
    return games_with_rosters

def create_enriched_summary(games):
    """Create summary of enriched data"""
    print("\n" + "="*60)
    print("ENRICHED DATA SUMMARY")
    print("="*60)
    
    with_qbs = sum(1 for g in games if 'home_qb' in g and 'away_qb' in g)
    with_key_players = sum(1 for g in games if g.get('home_key_players') and g.get('away_key_players'))
    with_records = sum(1 for g in games if 'home_record_before' in g)
    with_matchup_history = sum(1 for g in games if 'matchup_history' in g)
    
    print(f"\nTotal Games: {len(games):,}")
    print(f"Games with QB data: {with_qbs:,} ({100*with_qbs/len(games):.1f}%)")
    print(f"Games with key players: {with_key_players:,} ({100*with_key_players/len(games):.1f}%)")
    print(f"Games with team records: {with_records:,} ({100*with_records/len(games):.1f}%)")
    print(f"Games with matchup history: {with_matchup_history:,} ({100*with_matchup_history/len(games):.1f}%)")
    
    # Sample enriched game
    enriched_game = next((g for g in games if 'home_qb' in g and 'matchup_history' in g), None)
    if enriched_game:
        print("\n--- Sample Enriched Game ---")
        print(f"Game: {enriched_game['away_team']} @ {enriched_game['home_team']}")
        print(f"Season: {enriched_game['season']}, Week {enriched_game.get('week')}")
        print(f"Records: {enriched_game.get('away_record_before')} @ {enriched_game.get('home_record_before')}")
        if 'home_qb' in enriched_game:
            print(f"Home QB: {enriched_game['home_qb']['qb_name']}")
        if 'away_qb' in enriched_game:
            print(f"Away QB: {enriched_game['away_qb']['qb_name']}")
        if 'matchup_history' in enriched_game:
            h = enriched_game['matchup_history']
            print(f"Matchup History: {h['home_wins']}-{h['away_wins']} (last: {h['last_meeting_season']})")
    
    print("="*60)

def main():
    """Main execution"""
    print("="*60)
    print("NFL DATA ENRICHMENT - ROSTERS & MATCHUPS")
    print("="*60)
    print("Source: https://github.com/nflverse/nflverse-data")
    print()
    
    # Load existing games
    games_file = Path(__file__).parent.parent / "data" / "domains" / "nfl_complete_nflverse.json"
    
    if not games_file.exists():
        print(f"✗ Games file not found: {games_file}")
        print("Run update_nfl_data_from_nflverse.py first")
        return 1
    
    print(f"📂 Loading games from {games_file.name}...")
    with open(games_file, 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"✓ Loaded {len(games):,} games")
    
    # Enrich with rosters and matchups
    enriched_games = enrich_games_with_rosters(games)
    
    # Save enriched data
    output_path = Path(__file__).parent.parent / "data" / "domains" / "nfl_enriched_with_rosters.json"
    
    enriched_data = {
        'metadata': {
            'source': 'nflverse-data (enriched)',
            'url': 'https://github.com/nflverse/nflverse-data',
            'updated': datetime.now().isoformat(),
            'total_games': len(enriched_games),
            'seasons': sorted(list(set(g['season'] for g in enriched_games))),
            'description': 'NFL games with rosters, QBs, key players, records, and matchup history',
            'enrichments': [
                'QB names and IDs',
                'Key players by position',
                'Team records before each game',
                'Head-to-head matchup history',
                'Season context',
            ]
        },
        'games': enriched_games
    }
    
    with open(output_path, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    print(f"\n✓ Saved enriched data to {output_path}")
    
    # Create summary
    create_enriched_summary(enriched_games)
    
    print("\n🎉 NFL data enrichment complete!")
    print("\nEnriched data includes:")
    print("  • QB names for home and away teams")
    print("  • Key players (RB, WR, TE, defensive stars)")
    print("  • Team win-loss records before each game")
    print("  • Head-to-head matchup history")
    print("  • Season progress and context")
    
    print(f"\n📊 File: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

