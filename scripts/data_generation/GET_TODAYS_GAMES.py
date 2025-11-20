#!/usr/bin/env python3
"""
Get Today's NBA Games - 2025-26 Season
=======================================

Fetches actual games for today from NBA API or displays sample from available data.

For LIVE games: Install nba_api with: pip install nba_api

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
from datetime import datetime
from pathlib import Path

print()
print("="*80)
print(f"TODAY'S NBA GAMES - {datetime.now().strftime('%A, %B %d, %Y')}")
print("="*80)
print()
print(f"Current Season: 2025-26")
print(f"Looking for games on: {datetime.now().strftime('%Y-%m-%d')}")
print()

# Try to import nba_api
try:
    from nba_api.live.nba.endpoints import scoreboard
    
    print("✓ nba_api available - fetching LIVE games...")
    print()
    
    # Get today's games
    board = scoreboard.ScoreBoard()
    games = board.get_dict()
    
    if games and 'scoreboard' in games and 'games' in games['scoreboard']:
        todays_games = games['scoreboard']['games']
        
        if len(todays_games) == 0:
            print("📅 No NBA games scheduled for today")
            print()
        else:
            print(f"🏀 {len(todays_games)} NBA GAMES TODAY:")
            print()
            
            for i, game in enumerate(todays_games, 1):
                home_team = game['homeTeam']['teamName']
                away_team = game['awayTeam']['teamName']
                game_time = game.get('gameTimeUTC', 'TBD')
                status = game.get('gameStatusText', 'Scheduled')
                
                print(f"GAME #{i}")
                print(f"  {away_team} @ {home_team}")
                print(f"  Time: {game_time}")
                print(f"  Status: {status}")
                
                # If game has started, show score
                if 'homeTeam' in game and 'score' in game['homeTeam']:
                    home_score = game['homeTeam'].get('score', 0)
                    away_score = game['awayTeam'].get('score', 0)
                    print(f"  Score: {away_team} {away_score} - {home_team} {home_score}")
                
                print()
            
            # Save for betting system
            output = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'n_games': len(todays_games),
                'games': todays_games
            }
            
            output_path = Path(f'data/live/nba_{datetime.now().strftime(\"%Y%m%d\")}.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"✓ Saved to: {output_path}")
            print()
            print("Now run betting predictions:")
            print("  python3 scripts/nba_ALL_MARKETS_predictions.py")
    else:
        print("✗ No game data returned from API")
    
except ImportError:
    print("⚠️  nba_api not installed")
    print()
    print("To get today's LIVE games, install:")
    print("  pip install nba_api")
    print()
    print("Or manually fetch from: https://www.nba.com/games")
    print()
    print("-"*80)
    print("SHOWING SAMPLE FROM AVAILABLE DATA (2024-25 Finals)")
    print("-"*80)
    print()
    
    # Load most recent available
    with open('data/domains/nba_2024_2025_season.json') as f:
        data = json.load(f)
    
    games = data.get('games', data) if isinstance(data, dict) else data
    games_sorted = sorted([g for g in games if g.get('GAME_DATE')], 
                          key=lambda x: x.get('GAME_DATE', ''), reverse=True)
    
    recent_date = games_sorted[0]['GAME_DATE']
    recent_games = [g for g in games_sorted if g['GAME_DATE'] == recent_date]
    
    print(f"Sample from: {recent_date} (2024-25 Finals)")
    print()
    
    for i, game in enumerate(recent_games[:5], 1):
        print(f"GAME #{i}: {game['MATCHUP']}")
        print(f"  {game['TEAM_NAME']}: {game['PTS']} pts")
        print(f"  Result: {game['WL']}")
        print()
    
    print("="*80)
    print("TO GET LIVE 2025-26 GAMES:")
    print("="*80)
    print()
    print("Option 1: Install nba_api")
    print("  pip install nba_api")
    print("  python3 GET_TODAYS_GAMES.py")
    print()
    print("Option 2: Fetch from nba_data repository")
    print("  python3 scripts/nba_fetch_current_season.py")
    print()
    print("Option 3: Use betting system with available data")
    print("  python3 scripts/nba_ALL_MARKETS_predictions.py")
    print("  (Works with 2024-25 data to show system capabilities)")
    print()

except Exception as e:
    print(f"✗ Error: {e}")
    print()

print("="*80)
print()

