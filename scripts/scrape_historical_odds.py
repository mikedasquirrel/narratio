"""
Historical Odds Scraper - The Odds API

Scrapes historical odds data for model training.

Strategy:
- Pull odds for every day going back as far as API allows
- All sports (NHL, NBA, NFL, MLB, Soccer, UFC, Tennis, Golf, etc.)
- All markets (moneyline, spreads, totals)
- Multiple sportsbooks (best line data)

This data is GOLD for training - actual closing lines that our models will bet against.

Author: Historical Odds Collection
Date: November 19, 2025
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time

API_KEY = "2e330948334c9505ed5542a82fcfa3b9"
BASE_URL = "https://api.the-odds-api.com/v4"

SPORTS_TO_SCRAPE = [
    'icehockey_nhl',
    'basketball_nba',
    'americanfootball_nfl',
    'baseball_mlb',
    'soccer_epl',
    'soccer_uefa_champs_league',
    'mma_mixed_martial_arts',
    'boxing_boxing',
]


def scrape_historical_odds_for_date(sport: str, date: datetime) -> List[Dict]:
    """
    Scrape odds for a specific date.
    
    Parameters
    ----------
    sport : str
        Sport key
    date : datetime
        Date to scrape
    
    Returns
    -------
    games : list of dict
        All games with odds for that date
    """
    url = f"{BASE_URL}/historical/sports/{sport}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'american',
        'date': date.strftime('%Y-%m-%dT12:00:00Z'),
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('data', [])
            return games
        else:
            return []
    except Exception as e:
        print(f"    Error: {e}")
        return []


def scrape_historical_range(sport: str, start_date: datetime, end_date: datetime, 
                            output_dir: Path, progress_interval: int = 10):
    """
    Scrape historical odds for a date range.
    
    Parameters
    ----------
    sport : str
        Sport key
    start_date : datetime
        Start date
    end_date : datetime
        End date
    output_dir : Path
        Where to save data
    progress_interval : int
        Print progress every N days
    """
    print(f"\n{'='*80}")
    print(f"SCRAPING {sport.upper()} HISTORICAL ODDS")
    print(f"{'='*80}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_games = []
    current_date = start_date
    day_count = 0
    total_games = 0
    
    while current_date <= end_date:
        day_count += 1
        
        if day_count % progress_interval == 0 or day_count == 1:
            print(f"\n[Day {day_count}] {current_date.date()}")
        
        games = scrape_historical_odds_for_date(sport, current_date)
        
        if games:
            total_games += len(games)
            all_games.extend(games)
            
            if day_count % progress_interval == 0:
                print(f"  ✓ Found {len(games)} games (total: {total_games})")
        
        # Rate limit (10 requests/second max)
        time.sleep(0.15)
        
        current_date += timedelta(days=1)
    
    # Save to file
    output_file = output_dir / f"{sport}_historical_odds.json"
    with open(output_file, 'w') as f:
        json.dump({
            'sport': sport,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_games': len(all_games),
            'games': all_games
        }, f, indent=2)
    
    print(f"\n✓ Scraped {len(all_games)} total games")
    print(f"✓ Saved to: {output_file}")
    
    return all_games


def scrape_all_sports_historical(days_back: int = 365):
    """
    Scrape historical odds for all sports.
    
    Parameters
    ----------
    days_back : int
        How many days back to scrape
    """
    print("\n" + "="*80)
    print("HISTORICAL ODDS SCRAPING - ALL SPORTS")
    print("="*80)
    print(f"\nScraping {days_back} days of historical data")
    print(f"Sports: NHL, NBA, NFL, MLB, Soccer, UFC, Boxing")
    print(f"\nThis will use ~{days_back * len(SPORTS_TO_SCRAPE)} API requests")
    print(f"Estimated time: {days_back * len(SPORTS_TO_SCRAPE) * 0.15 / 60:.1f} minutes")
    
    input("\nPress Enter to begin scraping...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    output_dir = Path('data/historical_odds')
    
    summary = {}
    
    for sport in SPORTS_TO_SCRAPE:
        games = scrape_historical_range(sport, start_date, end_date, output_dir, progress_interval=30)
        summary[sport] = len(games)
    
    # Save summary
    summary_file = output_dir / 'scraping_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'scraped_date': datetime.now().isoformat(),
            'days_back': days_back,
            'sports': summary,
            'total_games': sum(summary.values())
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal games scraped: {sum(summary.values()):,}")
    for sport, count in summary.items():
        print(f"  {sport:<40} {count:>6,} games")
    print(f"\nAll data saved to: {output_dir}")


if __name__ == '__main__':
    # Start with 90 days to be safe with API limits
    scrape_all_sports_historical(days_back=90)

