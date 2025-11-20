"""
Build Player Data from Play-by-Play NBA Data

This script processes the nba_data repository's play-by-play files to extract
player-level statistics (points, rebounds, assists, etc.) per game.

Source: https://github.com/shufinskiy/nba_data.git

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import json
import pandas as pd
import tarfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

print("="*80)
print("BUILD PLAYER DATA FROM PLAY-BY-PLAY")
print("="*80)
print()
print("Source: https://github.com/shufinskiy/nba_data")
print("Converting play-by-play data to player box scores")
print()

# Event types for parsing
EVENT_TYPES = {
    1: 'SHOT_MADE',
    2: 'SHOT_MISSED', 
    3: 'FREE_THROW',
    4: 'REBOUND',
    5: 'TURNOVER',
    6: 'FOUL',
    7: 'VIOLATION',
    8: 'SUBSTITUTION',
    10: 'JUMP_BALL',
    11: 'EJECTION',
    12: 'PERIOD_BEGIN',
    13: 'PERIOD_END'
}

def extract_player_stats_from_pbp(pbp_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Extract player box scores from play-by-play data
    
    Returns dict of {game_id: {player_id: stats}}
    """
    game_stats = defaultdict(lambda: defaultdict(lambda: {
        'player_id': None,
        'player_name': None,
        'team_id': None,
        'points': 0,
        'rebounds': 0,
        'assists': 0,
        'steals': 0,
        'blocks': 0,
        'turnovers': 0,
        'fouls': 0,
        'fg_made': 0,
        'fg_attempted': 0,
        'three_made': 0,
        'three_attempted': 0,
        'ft_made': 0,
        'ft_attempted': 0,
        'minutes': 0.0  # Will need to calculate from substitutions
    }))
    
    for _, event in pbp_df.iterrows():
        game_id = event['GAME_ID']
        event_type = event['EVENTMSGTYPE']
        action_type = event['EVENTMSGACTIONTYPE']
        
        player1_id = event.get('PLAYER1_ID')
        player2_id = event.get('PLAYER2_ID')
        
        # Skip if no player involved
        if pd.isna(player1_id):
            continue
            
        player1_id = int(player1_id)
        
        # Initialize player if new
        if game_stats[game_id][player1_id]['player_id'] is None:
            game_stats[game_id][player1_id]['player_id'] = player1_id
            game_stats[game_id][player1_id]['player_name'] = event.get('PLAYER1_NAME', '')
            game_stats[game_id][player1_id]['team_id'] = event.get('PLAYER1_TEAM_ID', '')
        
        # Made shot
        if event_type == 1:
            game_stats[game_id][player1_id]['fg_made'] += 1
            game_stats[game_id][player1_id]['fg_attempted'] += 1
            
            # Check if 3-pointer (action type 1 = 2PT, action type 2 = 3PT)
            description = str(event.get('HOMEDESCRIPTION', '')) + str(event.get('VISITORDESCRIPTION', ''))
            if '3PT' in description:
                game_stats[game_id][player1_id]['three_made'] += 1
                game_stats[game_id][player1_id]['three_attempted'] += 1
                game_stats[game_id][player1_id]['points'] += 3
            else:
                game_stats[game_id][player1_id]['points'] += 2
            
            # Check for assist (player2 is assister)
            if not pd.isna(player2_id):
                player2_id = int(player2_id)
                if game_stats[game_id][player2_id]['player_id'] is None:
                    game_stats[game_id][player2_id]['player_id'] = player2_id
                    game_stats[game_id][player2_id]['player_name'] = event.get('PLAYER2_NAME', '')
                    game_stats[game_id][player2_id]['team_id'] = event.get('PLAYER2_TEAM_ID', '')
                game_stats[game_id][player2_id]['assists'] += 1
        
        # Missed shot
        elif event_type == 2:
            game_stats[game_id][player1_id]['fg_attempted'] += 1
            description = str(event.get('HOMEDESCRIPTION', '')) + str(event.get('VISITORDESCRIPTION', ''))
            if '3PT' in description:
                game_stats[game_id][player1_id]['three_attempted'] += 1
        
        # Free throw
        elif event_type == 3:
            game_stats[game_id][player1_id]['ft_attempted'] += 1
            if action_type in [1, 10, 11, 12, 13, 15, 16]:  # Made free throw action types
                game_stats[game_id][player1_id]['ft_made'] += 1
                game_stats[game_id][player1_id]['points'] += 1
        
        # Rebound
        elif event_type == 4:
            game_stats[game_id][player1_id]['rebounds'] += 1
        
        # Turnover
        elif event_type == 5:
            game_stats[game_id][player1_id]['turnovers'] += 1
            # Check for steal (player2 is stealer)
            if not pd.isna(player2_id):
                player2_id = int(player2_id)
                if game_stats[game_id][player2_id]['player_id'] is None:
                    game_stats[game_id][player2_id]['player_id'] = player2_id
                    game_stats[game_id][player2_id]['player_name'] = event.get('PLAYER2_NAME', '')
                    game_stats[game_id][player2_id]['team_id'] = event.get('PLAYER2_TEAM_ID', '')
                game_stats[game_id][player2_id]['steals'] += 1
        
        # Foul
        elif event_type == 6:
            game_stats[game_id][player1_id]['fouls'] += 1
    
    return game_stats

def aggregate_team_stats(player_stats: List[Dict]) -> Dict:
    """Aggregate player stats to team level (same format as our collector)"""
    if not player_stats:
        return {}
    
    # Sort by points (proxy for importance without requiring minutes)
    sorted_by_points = sorted(player_stats, key=lambda x: x['points'], reverse=True)
    
    points_list = [p['points'] for p in player_stats]
    assists_list = [p['assists'] for p in player_stats]
    
    return {
        'players_used': len(player_stats),
        'players_20plus_pts': len([p for p in points_list if p >= 20]),
        'players_15plus_pts': len([p for p in points_list if p >= 15]),
        'players_10plus_pts': len([p for p in points_list if p >= 10]),
        'players_5plus_ast': len([p for p in assists_list if p >= 5]),
        
        # Top players by points
        'top1_points': sorted_by_points[0]['points'] if len(sorted_by_points) > 0 else 0,
        'top1_name': sorted_by_points[0]['player_name'] if len(sorted_by_points) > 0 else '',
        'top1_assists': sorted_by_points[0]['assists'] if len(sorted_by_points) > 0 else 0,
        
        'top2_points': sorted_by_points[1]['points'] if len(sorted_by_points) > 1 else 0,
        'top2_name': sorted_by_points[1]['player_name'] if len(sorted_by_points) > 1 else '',
        
        'top3_points': sorted_by_points[2]['points'] if len(sorted_by_points) > 2 else 0,
        'top3_name': sorted_by_points[2]['player_name'] if len(sorted_by_points) > 2 else '',
        
        # Scoring concentration
        'top1_scoring_share': sorted_by_points[0]['points'] / sum(points_list) if sum(points_list) > 0 else 0,
        'top3_scoring_share': sum(p['points'] for p in sorted_by_points[:3]) / sum(points_list) if sum(points_list) > 0 else 0,
        
        # Placeholder values (not available from play-by-play)
        'experienced_players': 0,
        'avg_experience': 0.0,
        'top1_minutes': 36.0,  # Estimate
        'top3_minutes_share': 0.5,  # Estimate
        'bench_points': sum(p['points'] for p in sorted_by_points[5:]) if len(sorted_by_points) > 5 else 0,
    }

# Load our existing NBA data
data_path = Path('data/domains/nba_enhanced_betting_data.json')
with open(data_path) as f:
    our_games = json.load(f)

print(f"✓ Loaded {len(our_games):,} games from our dataset")

# Create mapping of game_id from our format (0021401230) to match
game_id_mapping = {game['game_id']: game for game in our_games if 'game_id' in game}

print(f"✓ {len(game_id_mapping):,} games with game IDs")
print()

# Process nbastats files for seasons 2014-2023 (matching our data)
seasons = range(2014, 2024)
nba_data_path = Path('nba_data_repo/datasets')

enhanced_games = []
success_count = 0
fail_count = 0

for season in seasons:
    season_file = nba_data_path / f'nbastats_{season}.tar.xz'
    
    if not season_file.exists():
        print(f"⚠️  {season_file.name} not found, skipping")
        continue
    
    print(f"Processing {season}/{season+1} season...")
    
    # Extract and read CSV
    with tarfile.open(season_file, 'r:xz') as tar:
        csv_file = tar.extractfile(f'nbastats_{season}.csv')
        pbp_df = pd.read_csv(csv_file)
    
    print(f"  ✓ Loaded {len(pbp_df):,} play-by-play events")
    
    # Extract player stats per game
    game_stats = extract_player_stats_from_pbp(pbp_df)
    
    print(f"  ✓ Extracted stats for {len(game_stats)} games")
    
    # Match with our games and add player data
    for game_id_str, player_stats_dict in game_stats.items():
        # Convert to our format (add leading zero if needed)
        game_id = str(game_id_str).zfill(10) if len(str(game_id_str)) < 10 else str(game_id_str)
        
        if game_id in game_id_mapping:
            game = game_id_mapping[game_id].copy()
            
            # Convert player stats to list
            player_stats_list = list(player_stats_dict.values())
            
            # Aggregate to team level
            team_aggregates = aggregate_team_stats(player_stats_list)
            
            game['player_data'] = {
                'available': True,
                'source': 'nba_data_repo_pbp',
                'team_aggregates': team_aggregates,
                'individual_players': player_stats_list,
                'note': 'Stats derived from play-by-play data'
            }
            
            enhanced_games.append(game)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"  ✓ Matched {success_count} games so far")
    print()

print("="*80)
print(f"COMPLETE!")
print("="*80)
print(f"Total games processed: {success_count + fail_count}")
print(f"Successfully matched: {success_count} ({success_count/(success_count+fail_count)*100:.1f}%)")
print(f"Not matched: {fail_count}")
print()

# Save complete dataset
output_path = Path('data/domains/nba_complete_with_players.json')
with open(output_path, 'w') as f:
    json.dump(enhanced_games, f, indent=2)

print(f"✓ Saved to: {output_path}")
print(f"  Games with player data: {success_count:,}")
print()
print("Next step: python3 discover_player_patterns.py")
print("="*80)

