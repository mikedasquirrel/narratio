"""
DEEP HISTORICAL ODDS SCRAPER - EVERYTHING

Scrapes MAXIMUM available history for ALL 75 sports.

Strategy:
- All 75 sports from The Odds API
- Go back as far as data exists (5-10 years for major sports)
- Use ALL available requests (~20k, will buy more if needed)
- Expected: 100,000-500,000+ games with closing odds

This is the COMPLETE training dataset for the universal pipeline.

Author: Deep Historical Collection
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


def get_all_sports() -> List[Dict]:
    """Get all available sports from API"""
    url = f"{BASE_URL}/sports"
    params = {'apiKey': API_KEY, 'all': 'true'}
    
    response = requests.get(url, params=params, timeout=10)
    sports = response.json()
    
    # Filter to active sports only
    active_sports = [s for s in sports if s.get('active')]
    
    print(f"✓ Found {len(active_sports)} active sports")
    return active_sports


def scrape_sport_deep(sport_key: str, sport_title: str, output_dir: Path, max_days: int = 2000) -> Dict:
    """
    Scrape as deep as possible for one sport.
    
    Goes back up to max_days or until API returns no data for 30 consecutive days.
    """
    print(f"\n{'='*80}")
    print(f"SCRAPING: {sport_title} ({sport_key})")
    print(f"{'='*80}")
    
    output_file = output_dir / f"{sport_key}_deep.json"
    
    # Check if already scraped
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
        if existing.get('total_games', 0) > 100:  # Already has substantial data
            print(f"  ✓ Already scraped: {existing['total_games']:,} games")
            print(f"  Skipping...")
            return existing
    
    all_games = []
    current_date = datetime.now()
    days_back = 0
    consecutive_empty = 0
    requests_used = 0
    last_game_date = None
    
    print(f"Starting from: {current_date.date()}")
    print(f"Going back up to: {max_days} days")
    
    while days_back < max_days and consecutive_empty < 30:
        url = f"{BASE_URL}/historical/sports/{sport_key}/odds"
        params = {
            'apiKey': API_KEY,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'date': current_date.strftime('%Y-%m-%dT12:00:00Z'),
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            requests_used += 1
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('data', [])
                
                if games:
                    all_games.extend(games)
                    consecutive_empty = 0
                    last_game_date = current_date.date()
                    
                    if days_back % 100 == 0 and days_back > 0:
                        remaining = response.headers.get('x-requests-remaining', 'N/A')
                        print(f"  [{days_back:>4} days] {current_date.date()} - {len(games):>3} games | Total: {len(all_games):>6,} | API: {remaining}")
                else:
                    consecutive_empty += 1
            
            elif response.status_code == 401 or response.status_code == 422:
                # Hit API limit or no historical data available
                print(f"  [API Limit] Stopped at {current_date.date()} after {days_back} days")
                break
            
            else:
                consecutive_empty += 1
        
        except Exception as e:
            if days_back % 100 == 0:
                print(f"  Error at {current_date.date()}: {e}")
            consecutive_empty += 1
        
        # Rate limit (8 req/sec to be safe)
        time.sleep(0.13)
        
        current_date -= timedelta(days=1)
        days_back += 1
        
        # Checkpoint every 200 days
        if days_back % 200 == 0 and len(all_games) > 0:
            checkpoint = {
                'sport': sport_title,
                'sport_key': sport_key,
                'total_games': len(all_games),
                'days_scraped': days_back,
                'requests_used': requests_used,
                'last_game_date': str(last_game_date),
                'games': all_games
            }
            with open(output_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"  [Checkpoint] Saved {len(all_games):,} games")
    
    # Final save
    final_data = {
        'sport': sport_title,
        'sport_key': sport_key,
        'total_games': len(all_games),
        'days_scraped': days_back,
        'requests_used': requests_used,
        'scraped_date': datetime.now().isoformat(),
        'earliest_game': str(last_game_date) if last_game_date else None,
        'games': all_games
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_data, f)
    
    print(f"\n✓ {sport_title}: {len(all_games):,} games ({days_back} days, {requests_used} requests)")
    
    return final_data


def scrape_everything_deep():
    """Scrape ALL sports as deep as possible"""
    print("\n" + "="*80)
    print("DEEP HISTORICAL SCRAPING - ALL 75 SPORTS")
    print("="*80)
    print(f"\nStrategy: Go back as far as API has data for EVERY sport")
    print(f"Expected: 100,000-500,000+ games")
    print(f"Time: 2-4 hours")
    print(f"API requests: Will use all available (~20k), buy more if needed")
    
    print(f"\n{'='*80}")
    input("Press Enter to begin DEEP scraping of ALL sports...")
    
    output_dir = Path('data/historical_odds_deep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all sports
    all_sports = get_all_sports()
    
    print(f"\nScraping {len(all_sports)} sports...")
    
    summary = {}
    total_games_collected = 0
    total_requests_used = 0
    
    for i, sport in enumerate(all_sports, 1):
        sport_key = sport['key']
        sport_title = sport['title']
        
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(all_sports)}] {sport_title}")
        print(f"{'#'*80}")
        
        # Determine max days based on sport type
        if 'championship_winner' in sport_key or 'presidential' in sport_key:
            max_days = 365  # Futures markets, less frequent
        elif 'mlb' in sport_key:
            max_days = 1000  # Baseball has tons of games
        else:
            max_days = 2000  # Go deep for everything else
        
        result = scrape_sport_deep(sport_key, sport_title, output_dir, max_days)
        
        games = result['total_games']
        requests = result['requests_used']
        
        summary[sport_title] = {
            'sport_key': sport_key,
            'games': games,
            'days': result['days_scraped'],
            'requests': requests
        }
        
        total_games_collected += games
        total_requests_used += requests
        
        print(f"\n  Running Total:")
        print(f"    Games: {total_games_collected:,}")
        print(f"    Requests: {total_requests_used:,}")
        print(f"    Remaining: ~{20000 - total_requests_used:,}")
        
        # Check if we're running low
        if total_requests_used > 19000:
            print(f"\n⚠ Approaching 20k request limit")
            print(f"  Collected {total_games_collected:,} games so far")
            print(f"  Continue scraping? (Will need to buy more requests)")
            response = input("  Continue? (y/n): ")
            if response.lower() != 'y':
                print(f"\n  Stopping scraping to preserve requests for daily use")
                break
    
    # Final summary
    summary_file = output_dir / 'deep_scraping_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'scraped_date': datetime.now().isoformat(),
            'total_games': total_games_collected,
            'total_requests': total_requests_used,
            'sports_scraped': len(summary),
            'sports': summary
        }, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("DEEP SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal Statistics:")
    print(f"  Total games: {total_games_collected:,}")
    print(f"  Sports scraped: {len(summary)}")
    print(f"  API requests used: {total_requests_used:,}")
    print(f"  API requests remaining: ~{20000 - total_requests_used:,}")
    
    print(f"\nTop 10 sports by game count:")
    sorted_sports = sorted(summary.items(), key=lambda x: x[1]['games'], reverse=True)
    for sport, stats in sorted_sports[:10]:
        print(f"  {sport:<30} {stats['games']:>7,} games")
    
    print(f"\nAll data saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    scrape_everything_deep()

