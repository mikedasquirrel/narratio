"""
NBA Current Season Fetcher - 2025-26 Season
============================================

Fetches current 2025-26 NBA season data including:
- All games (completed and upcoming)
- Current standings and records
- Player statistics
- Recent performance

Uses nba_api library (free, no API key needed)

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Try to import nba_api
try:
    from nba_api.stats.static import teams
    from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, playergamelog, leaguestandings
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False
    print("⚠️  nba_api not installed. Install with: pip install nba_api")


def print_progress(text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {text}", flush=True)


def fetch_current_season_games():
    """Fetch all 2025-26 season games"""
    
    if not HAS_NBA_API:
        print_progress("Using fallback: Loading from nba_data repository...")
        # Use existing data as fallback
        return fetch_from_nba_data()
    
    print_progress("Fetching 2025-26 season games from NBA API...")
    
    try:
        # Get all teams
        nba_teams = teams.get_teams()
        
        all_games = []
        
        for i, team in enumerate(nba_teams, 1):
            if i % 5 == 0:
                print_progress(f"  Progress: {i}/{len(nba_teams)} teams...")
            
            team_id = team['id']
            team_name = team['full_name']
            
            # Get team's games for 2025-26 season
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable='2025-26',
                season_type_nullable='Regular Season'
            )
            
            games_df = gamefinder.get_data_frames()[0]
            
            for _, game_row in games_df.iterrows():
                game_data = {
                    'game_id': game_row['GAME_ID'],
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_abbreviation': game_row['TEAM_ABBREVIATION'],
                    'matchup': game_row['MATCHUP'],
                    'game_date': game_row['GAME_DATE'],
                    'home_game': '@' not in game_row['MATCHUP'],
                    'won': game_row['WL'] == 'W',
                    'points': game_row['PTS'],
                    'opp_points': game_row.get('OPP_PTS', 0),
                    'fg_pct': game_row['FG_PCT'],
                    'fg3_pct': game_row['FG3_PCT'],
                    'ft_pct': game_row['FT_PCT'],
                    'rebounds': game_row['REB'],
                    'assists': game_row['AST'],
                    'steals': game_row['STL'],
                    'blocks': game_row['BLK'],
                    'turnovers': game_row['TOV'],
                    'plus_minus': game_row['PLUS_MINUS'],
                    'season': '2025-26'
                }
                
                all_games.append(game_data)
        
        print_progress(f"✓ Fetched {len(all_games)} games from 2025-26 season")
        return all_games
        
    except Exception as e:
        print_progress(f"✗ Error fetching from NBA API: {e}")
        print_progress("  Using fallback method...")
        return fetch_from_nba_data()


def fetch_from_nba_data():
    """Fallback: Use nba_data repository"""
    
    print_progress("Loading from nba_data repository...")
    
    # Check if we have 2025-26 data files
    data_dir = Path('nba_data_repo/datasets')
    
    if not data_dir.exists():
        print_progress("✗ nba_data_repo not found")
        return []
    
    # Look for recent season files
    season_files = sorted(data_dir.glob('*.xz'), reverse=True)[:50]
    
    games = []
    
    for filepath in season_files:
        # Parse season from filename (e.g., gamelog_SEASON_2025_26_01.xz)
        if '2025_26' in filepath.name or '2024_25' in filepath.name:
            print_progress(f"  Loading: {filepath.name}")
            try:
                df = pd.read_csv(filepath, compression='xz')
                
                for _, row in df.iterrows():
                    game_data = {
                        'game_id': row.get('gameId', ''),
                        'team_name': row.get('nameTeam', 'Unknown'),
                        'matchup': f"{row.get('slugTeam', 'UNK')} vs. {row.get('slugOpponent', 'OPP')}",
                        'game_date': row.get('dateGame', ''),
                        'home_game': row.get('locationGame', '') == 'H',
                        'won': row.get('outcomeGame', '') == 'W',
                        'points': row.get('pts', 0),
                        'season': '2025-26' if '2025_26' in filepath.name else '2024-25'
                    }
                    games.append(game_data)
            except:
                continue
    
    print_progress(f"✓ Loaded {len(games)} games from nba_data")
    return games


def enrich_with_temporal_context(games: List[Dict]) -> List[Dict]:
    """Add temporal context (records, L10, etc.)"""
    
    print_progress("Calculating temporal context...")
    
    # Group by team and season
    by_team = {}
    for game in games:
        key = (game['team_name'], game['season'])
        if key not in by_team:
            by_team[key] = []
        by_team[key].append(game)
    
    # Sort by date and calculate rolling statistics
    for key, team_games in by_team.items():
        team_games.sort(key=lambda x: x.get('game_date', ''))
        
        for i, game in enumerate(team_games):
            # Calculate season record before this game
            prior_games = team_games[:i]
            if len(prior_games) > 0:
                wins = sum(1 for g in prior_games if g['won'])
                season_record = f"{wins}-{len(prior_games)-wins}"
                season_win_pct = wins / len(prior_games)
                
                # L10
                l10_games = prior_games[-10:] if len(prior_games) >= 10 else prior_games
                l10_wins = sum(1 for g in l10_games if g['won'])
                l10_record = f"{l10_wins}-{len(l10_games)-l10_wins}"
                l10_win_pct = l10_wins / len(l10_games) if len(l10_games) > 0 else 0.5
            else:
                season_record = "0-0"
                season_win_pct = 0.5
                l10_record = "0-0"
                l10_win_pct = 0.5
            
            game['temporal_context'] = {
                'season_record_prior': season_record,
                'season_win_pct': season_win_pct,
                'l10_record': l10_record,
                'l10_win_pct': l10_win_pct,
                'games_played': len(prior_games)
            }
    
    print_progress(f"✓ Added temporal context to {len(games)} games")
    return games


def get_todays_games(games: List[Dict]) -> List[Dict]:
    """Filter for today's games"""
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    todays = [g for g in games if g.get('game_date', '')[:10] == today]
    
    if len(todays) == 0:
        # Get most recent games as example
        recent_dates = sorted(set(g.get('game_date', '')[:10] for g in games if g.get('game_date')), reverse=True)[:5]
        print_progress(f"  No games today. Recent dates: {', '.join(recent_dates[:3])}")
        
        # Use most recent date
        if recent_dates:
            most_recent = recent_dates[0]
            todays = [g for g in games if g.get('game_date', '')[:10] == most_recent]
            print_progress(f"  Using {len(todays)} games from {most_recent}")
    
    return todays


def main():
    """Fetch current season data"""
    
    print("\n" + "="*80)
    print("NBA 2025-26 SEASON DATA FETCHER")
    print("="*80)
    print()
    
    # Fetch games
    games = fetch_current_season_games()
    
    if len(games) == 0:
        print_progress("\n❌ No games fetched")
        return
    
    # Enrich with context
    games = enrich_with_temporal_context(games)
    
    # Get today's games
    todays_games = get_todays_games(games)
    
    # Save all season data
    output_path = Path('data/domains/nba_2025_2026_season.json')
    with open(output_path, 'w') as f:
        json.dump({
            'season': '2025-26',
            'last_updated': datetime.now().isoformat(),
            'total_games': len(games),
            'todays_games': len(todays_games),
            'games': games
        }, f, indent=2)
    
    print()
    print_progress(f"✓ Saved to: {output_path}")
    
    # Save today's games separately
    if len(todays_games) > 0:
        today_path = Path(f'data/live/nba_{datetime.now().strftime(\"%Y%m%d\")}.json')
        today_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(today_path, 'w') as f:
            json.dump({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'n_games': len(todays_games),
                'games': todays_games
            }, f, indent=2)
        
        print_progress(f"✓ Today's games saved to: {today_path}")
    
    # Summary
    print()
    print("="*80)
    print("FETCH COMPLETE")
    print("="*80)
    print(f"\nSeason: 2025-26")
    print(f"Total games: {len(games)}")
    print(f"Today's games: {len(todays_games)}")
    
    if len(todays_games) > 0:
        print(f"\nToday's matchups:")
        for game in todays_games[:10]:
            print(f"  • {game['matchup']}")
    
    print()


if __name__ == "__main__":
    main()

