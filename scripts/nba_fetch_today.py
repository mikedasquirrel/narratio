"""
NBA Live Data Fetcher
======================

Fetches today's NBA games, betting odds, and player information.
Saves to data/live/ for daily predictions.

Data sources:
- NBA games: nba_data repository or NBA API
- Betting odds: Mock data (integrate with odds API in production)
- Player stats: From nba_data play-by-play aggregation

Usage:
    python scripts/nba_fetch_today.py
    python scripts/nba_fetch_today.py --date 2024-11-16

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_progress(text):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {text}")


def fetch_nba_games_today(date: Optional[str] = None) -> List[Dict]:
    """
    Fetch today's NBA games.
    
    Parameters
    ----------
    date : str, optional
        Date in YYYY-MM-DD format
        
    Returns
    -------
    games : list of dict
        Today's games
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    print_progress(f"Fetching NBA games for {date}...")
    
    # For now, use the 2024-25 season data as example
    # In production, integrate with:
    # - nba_api library
    # - nba_data repository
    # - Official NBA stats API
    
    data_path = Path('data/domains/nba_2024_2025_season.json')
    
    if data_path.exists():
        print_progress(f"  Loading from: {data_path}")
        with open(data_path) as f:
            season_games = json.load(f)
        
        # Filter for today's date (if available in data)
        # For now, return sample games
        print_progress(f"  ✓ Loaded {len(season_games)} games from 2024-25 season")
        print_progress(f"  ⚠️  Using sample games (integrate live API in production)")
        
        # Return first 5 games as sample
        return season_games[:5]
    else:
        print_progress("  ⚠️  No 2024-25 season data found")
        print_progress("  Returning empty list")
        return []


def fetch_betting_odds(games: List[Dict]) -> List[Dict]:
    """
    Fetch current betting odds for games.
    
    Parameters
    ----------
    games : list of dict
        Games to fetch odds for
        
    Returns
    -------
    games_with_odds : list of dict
        Games enriched with current betting odds
    """
    print_progress("Fetching betting odds...")
    
    # For now, use existing odds or generate mock odds
    # In production, integrate with:
    # - The Odds API (theoddsapi.com)
    # - Sportsbook APIs
    # - Web scraping from bookmakers
    
    enriched_games = []
    
    for game in games:
        # Check if odds already exist
        if not game.get('betting_odds'):
            # Generate mock odds based on team quality
            # In production, fetch real odds
            tc = game.get('temporal_context', {})
            win_pct = tc.get('season_win_pct', 0.5)
            
            # Simple mock odds calculation
            if win_pct > 0.55:
                moneyline = -150
            elif win_pct > 0.50:
                moneyline = -110
            elif win_pct > 0.45:
                moneyline = 110
            else:
                moneyline = 150
            
            game['betting_odds'] = {
                'moneyline': moneyline,
                'spread': (win_pct - 0.5) * 20,  # Mock spread
                'implied_probability': abs(moneyline) / (abs(moneyline) + 100) if moneyline < 0 else 100 / (moneyline + 100),
                'source': 'MOCK',
                'timestamp': datetime.now().isoformat()
            }
            
        enriched_games.append(game)
    
    print_progress(f"  ✓ Enriched {len(enriched_games)} games with odds")
    print_progress(f"  ⚠️  NOTE: Using mock odds for demonstration")
    print_progress(f"  ⚠️  Integrate with odds API for production use")
    
    return enriched_games


def fetch_player_availability(games: List[Dict]) -> List[Dict]:
    """
    Fetch current player availability/lineups.
    
    Parameters
    ----------
    games : list of dict
        Games to fetch lineups for
        
    Returns
    -------
    games_with_players : list of dict
        Games with current player info
    """
    print_progress("Fetching player availability...")
    
    # For now, use existing player data
    # In production, fetch:
    # - Injury reports
    # - Starting lineups
    # - Recent player performance
    
    print_progress(f"  ✓ Player data available for {len([g for g in games if g.get('player_data')])} games")
    print_progress(f"  ⚠️  Using historical player data (integrate injury reports for production)")
    
    return games


def main():
    """Fetch today's data"""
    
    parser = argparse.ArgumentParser(description='Fetch today\'s NBA games and odds')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (default: today)')
    args = parser.parse_args()
    
    print()
    print("="*80)
    print("NBA LIVE DATA FETCHER")
    print("="*80)
    print()
    
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')
    print_progress(f"Fetching data for: {date_str}")
    
    # Fetch games
    games = fetch_nba_games_today(date=args.date)
    
    if len(games) == 0:
        print()
        print_progress("❌ No games found")
        return
    
    # Fetch odds
    games = fetch_betting_odds(games)
    
    # Fetch player info
    games = fetch_player_availability(games)
    
    # Save to live data directory
    output_date = date_str.replace('-', '')
    output_path = Path(f'data/live/nba_{output_date}.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'date': date_str,
            'fetched_at': datetime.now().isoformat(),
            'n_games': len(games),
            'games': games,
            'notes': [
                'MOCK odds used for demonstration',
                'Integrate with real odds API for production',
                'Integrate with injury reports for player availability'
            ]
        }, f, indent=2)
    
    print()
    print_progress(f"✓ Data saved to: {output_path}")
    
    # Summary
    print()
    print("="*80)
    print("FETCH SUMMARY")
    print("="*80)
    print(f"\nDate: {date_str}")
    print(f"Games fetched: {len(games)}")
    print(f"Games with odds: {len([g for g in games if g.get('betting_odds')])}")
    print(f"Games with players: {len([g for g in games if g.get('player_data')])}")
    print()
    print(f"Next step: Run predictions")
    print(f"  python scripts/nba_daily_predictions.py --dry-run")
    print()


if __name__ == "__main__":
    main()

