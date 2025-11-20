"""
NBA Props Data Collector
=========================

Collects historical player performance data for props betting:
- Points, rebounds, assists per game
- Player game logs
- Matchup-specific performance
- Over/under outcomes

Builds training dataset for prop models.

Author: AI Coding Assistant  
Date: November 16, 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_progress(text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {text}", flush=True)


def extract_player_props_from_pbp(games: List[Dict]) -> Dict:
    """
    Extract player-level statistics for props betting.
    
    From nba_complete_with_players.json which has aggregated player data.
    """
    
    print_progress("Extracting player props from game data...")
    
    player_game_logs = defaultdict(list)
    
    for game in games:
        if not game.get('player_data', {}).get('available'):
            continue
        
        team = game.get('team_name', 'Unknown')
        date = game.get('date', '')
        season = game.get('season', '')
        
        # Get player aggregates
        agg = game['player_data']['team_aggregates']
        tc = game.get('temporal_context', {})
        
        # Extract top scorers (we have their data)
        for i in range(1, 4):  # Top 3 players
            player_name = agg.get(f'top{i}_name')
            player_pts = agg.get(f'top{i}_points', 0)
            
            if player_name:
                player_game_logs[player_name].append({
                    'date': date,
                    'season': season,
                    'team': team,
                    'opponent': game.get('matchup', '').split('vs.')[-1].strip() if 'vs.' in game.get('matchup', '') else 'Unknown',
                    'home': game.get('home_game', False),
                    'points': player_pts,
                    'team_won': game.get('won', False),
                    'team_win_pct': tc.get('season_win_pct', 0.5),
                    'team_l10_pct': tc.get('l10_win_pct', 0.5)
                })
    
    print_progress(f"✓ Collected data for {len(player_game_logs)} players")
    
    return dict(player_game_logs)


def calculate_prop_lines(player_logs: Dict) -> Dict:
    """
    Calculate typical prop lines for each player.
    
    Line = player's average + adjustment for market
    """
    
    print_progress("Calculating typical prop lines...")
    
    player_props = {}
    
    for player, games in player_logs.items():
        if len(games) < 10:  # Need minimum games
            continue
        
        # Calculate averages
        points = [g['points'] for g in games]
        
        avg_points = np.mean(points)
        std_points = np.std(points)
        median_points = np.median(points)
        
        # Typical line is slightly above average (sportsbook edge)
        typical_line = avg_points + 0.5
        
        # Recent form (last 5 games)
        recent = points[-5:] if len(points) >= 5 else points
        recent_avg = np.mean(recent)
        
        player_props[player] = {
            'games_played': len(games),
            'avg_points': float(avg_points),
            'median_points': float(median_points),
            'std_points': float(std_points),
            'typical_line': float(typical_line),
            'recent_avg': float(recent_avg),
            'trend': 'up' if recent_avg > avg_points else 'down',
            'consistency': 'high' if std_points < 5 else 'medium' if std_points < 8 else 'low'
        }
    
    print_progress(f"✓ Calculated props for {len(player_props)} players")
    
    return player_props


def simulate_prop_outcomes(player_logs: Dict, player_props: Dict) -> List[Dict]:
    """
    Simulate historical prop bet outcomes.
    
    For each game, determine if player went over/under their typical line.
    """
    
    print_progress("Simulating historical prop outcomes...")
    
    prop_bets = []
    
    for player, games in player_logs.items():
        if player not in player_props:
            continue
        
        typical_line = player_props[player]['typical_line']
        
        for game in games:
            actual_points = game['points']
            
            # Outcome
            went_over = actual_points > typical_line
            
            prop_bets.append({
                'player': player,
                'date': game['date'],
                'opponent': game['opponent'],
                'home': game['home'],
                'line': typical_line,
                'actual': actual_points,
                'went_over': went_over,
                'diff': actual_points - typical_line,
                'team_won': game['team_won'],
                'team_win_pct': game['team_win_pct']
            })
    
    print_progress(f"✓ Simulated {len(prop_bets)} historical prop outcomes")
    
    return prop_bets


def main():
    """Collect props data"""
    
    print("\n" + "="*80)
    print("NBA PLAYER PROPS DATA COLLECTION")
    print("="*80)
    print()
    
    # Load NBA data
    print_progress("Loading NBA game data...")
    
    data_path = Path('data/domains/nba_complete_with_players.json')
    with open(data_path) as f:
        all_games = json.load(f)
    
    print_progress(f"✓ Loaded {len(all_games):,} games")
    
    # Extract player game logs
    player_logs = extract_player_props_from_pbp(all_games)
    
    # Calculate typical lines
    player_props = calculate_prop_lines(player_logs)
    
    # Simulate outcomes
    prop_outcomes = simulate_prop_outcomes(player_logs, player_props)
    
    # Calculate success rates
    print()
    print("="*80)
    print("PROPS ANALYSIS")
    print("="*80)
    print()
    
    # Overall
    total_overs = sum(1 for p in prop_outcomes if p['went_over'])
    over_pct = total_overs / len(prop_outcomes) * 100
    
    print(f"Total prop bets: {len(prop_outcomes):,}")
    print(f"Overs: {total_overs:,} ({over_pct:.1f}%)")
    print(f"Unders: {len(prop_outcomes) - total_overs:,} ({100-over_pct:.1f}%)")
    print()
    
    # Top players
    print("TOP 10 PLAYERS BY GAMES:")
    player_game_counts = [(p, len(logs)) for p, logs in player_logs.items()]
    player_game_counts.sort(key=lambda x: x[1], reverse=True)
    
    for i, (player, count) in enumerate(player_game_counts[:10], 1):
        props = player_props.get(player, {})
        print(f"  {i}. {player:<25} {count:>3} games | Avg: {props.get('avg_points', 0):.1f} pts | Line: {props.get('typical_line', 0):.1f}")
    
    # Save data
    output_data = {
        'collected_at': datetime.now().isoformat(),
        'total_players': len(player_logs),
        'total_prop_bets': len(prop_outcomes),
        'player_statistics': player_props,
        'player_game_logs': {k: v for k, v in player_logs.items()},
        'simulated_outcomes': prop_outcomes
    }
    
    output_path = Path('data/domains/nba_props_historical_data.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print_progress(f"✓ Props data saved to: {output_path}")
    
    print()
    print("="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\nReady for props model training!")
    print(f"Next: python narrative_optimization/betting/nba_props_model.py")
    print()


if __name__ == "__main__":
    main()

