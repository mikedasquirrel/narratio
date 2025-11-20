"""
MAXIMUM Historical Odds Scraper

Scrapes as far back as The Odds API allows for ALL sports.

Strategy:
- Start from today, go backwards day by day
- Continue until API returns no data (hit the limit)
- Use ALL available API requests (~20,000)
- Expected: 3-7 years of data per sport

This will give us:
- 20,000-50,000+ games with actual closing odds
- Multiple sportsbooks per game (best line data)
- Complete training datasets for all sports

Author: Maximum Historical Collection
Date: November 19, 2025
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time

API_KEY = "2e330948334c9505ed5542a82fcfa3b9"
BASE_URL = "https://api.the-odds-api.com/v4"

# ALL sports to scrape (prioritized by betting volume)
SPORTS_PRIORITY = [
    ('icehockey_nhl', 'NHL', 1095),  # 3 years
    ('basketball_nba', 'NBA', 1095),
    ('americanfootball_nfl', 'NFL', 1095),
    ('baseball_mlb', 'MLB', 730),  # 2 years (huge volume)
    ('soccer_epl', 'EPL', 730),
    ('soccer_uefa_champs_league', 'UEFA_CL', 730),
    ('mma_mixed_martial_arts', 'UFC', 1095),
    ('boxing_boxing', 'Boxing', 1095),
    ('soccer_spain_la_liga', 'La_Liga', 730),
    ('soccer_germany_bundesliga', 'Bundesliga', 730),
    ('soccer_italy_serie_a', 'Serie_A', 730),
    ('soccer_france_ligue_one', 'Ligue_1', 730),
    ('basketball_ncaab', 'NCAAB', 365),
    ('americanfootball_ncaaf', 'NCAAF', 365),
]


def scrape_sport_maximum_history(sport_key: str, sport_name: str, max_days: int, output_dir: Path) -> Dict:
    """
    Scrape maximum available history for a sport.
    
    Goes back day by day until API returns no data.
    """
    print(f"\n{'='*80}")
    print(f"SCRAPING {sport_name} - MAXIMUM HISTORY")
    print(f"{'='*80}")
    print(f"Target: {max_days} days back (will stop if API limit hit)")
    
    all_games = []
    current_date = datetime.now()
    days_scraped = 0
    consecutive_empty = 0
    total_requests = 0
    
    output_file = output_dir / f"{sport_name.lower()}_historical_odds.json"
    
    # Check if we already have data
    if output_file.exists():
        print(f"\n✓ Found existing data at {output_file}")
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  Existing games: {len(existing.get('games', []))}")
        print(f"  Date range: {existing.get('start_date', 'N/A')} to {existing.get('end_date', 'N/A')}")
        
        # Resume from where we left off
        if existing.get('end_date'):
            current_date = datetime.fromisoformat(existing['end_date']) - timedelta(days=1)
            all_games = existing.get('games', [])
            print(f"  Resuming from {current_date.date()}")
    
    start_date = current_date
    
    while days_scraped < max_days and consecutive_empty < 30:
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
            total_requests += 1
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('data', [])
                
                if games:
                    all_games.extend(games)
                    consecutive_empty = 0
                    
                    if days_scraped % 50 == 0:
                        remaining = response.headers.get('x-requests-remaining', 'N/A')
                        print(f"  [{days_scraped:>4} days] {current_date.date()} - {len(games):>3} games (total: {len(all_games):>5}, API remaining: {remaining})")
                else:
                    consecutive_empty += 1
                    if consecutive_empty == 1:
                        print(f"  [No data at {current_date.date()}] - continuing...")
            
            elif response.status_code == 401:
                print(f"\n✗ API limit reached or historical data not available")
                break
            
            else:
                consecutive_empty += 1
        
        except Exception as e:
            print(f"  Error on {current_date.date()}: {e}")
            consecutive_empty += 1
        
        # Rate limit
        time.sleep(0.12)  # ~8 requests/second
        
        current_date -= timedelta(days=1)
        days_scraped += 1
        
        # Save checkpoint every 100 days
        if days_scraped % 100 == 0:
            checkpoint_data = {
                'sport': sport_name,
                'sport_key': sport_key,
                'start_date': start_date.isoformat(),
                'end_date': current_date.isoformat(),
                'days_scraped': days_scraped,
                'total_games': len(all_games),
                'total_requests': total_requests,
                'games': all_games
            }
            with open(output_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"  [Checkpoint] Saved {len(all_games)} games to {output_file}")
    
    # Final save
    final_data = {
        'sport': sport_name,
        'sport_key': sport_key,
        'start_date': start_date.isoformat(),
        'end_date': current_date.isoformat(),
        'days_scraped': days_scraped,
        'total_games': len(all_games),
        'total_requests': total_requests,
        'games': all_games
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"\n✓ COMPLETE: {sport_name}")
    print(f"  Days scraped: {days_scraped}")
    print(f"  Games collected: {len(all_games):,}")
    print(f"  API requests used: {total_requests}")
    print(f"  Saved to: {output_file}")
    
    return final_data


def scrape_all_sports_maximum():
    """Scrape maximum history for all sports"""
    print("\n" + "="*80)
    print("MAXIMUM HISTORICAL ODDS SCRAPING")
    print("="*80)
    print(f"\nStrategy: Go back as far as API allows for each sport")
    print(f"API Requests Available: ~20,000")
    print(f"Sports: {len(SPORTS_PRIORITY)}")
    print(f"\nEstimated collection:")
    print(f"  - Major sports (NHL/NBA/NFL): 3 years each")
    print(f"  - MLB: 2 years (high volume)")
    print(f"  - Soccer leagues: 2 years each")
    print(f"  - UFC/Boxing: 3 years")
    print(f"\nExpected total: 30,000-50,000 games")
    print(f"Estimated time: 45-60 minutes")
    
    print(f"\n{'='*80}")
    input("Press Enter to begin MAXIMUM scraping...")
    
    output_dir = Path('data/historical_odds')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    total_requests_used = 0
    
    for sport_key, sport_name, max_days in SPORTS_PRIORITY:
        print(f"\n\n{'#'*80}")
        print(f"# SPORT {len(summary) + 1}/{len(SPORTS_PRIORITY)}: {sport_name}")
        print(f"{'#'*80}")
        
        result = scrape_sport_maximum_history(sport_key, sport_name, max_days, output_dir)
        summary[sport_name] = {
            'games': result['total_games'],
            'days': result['days_scraped'],
            'requests': result['total_requests']
        }
        total_requests_used += result['total_requests']
        
        print(f"\nRunning total:")
        print(f"  Games collected: {sum(s['games'] for s in summary.values()):,}")
        print(f"  API requests used: {total_requests_used:,}")
        print(f"  API requests remaining: ~{20000 - total_requests_used:,}")
        
        # Stop if we're running low on requests
        if total_requests_used > 18000:
            print(f"\n⚠ Approaching API limit, stopping to preserve requests for daily use")
            break
    
    # Final summary
    summary_file = output_dir / 'scraping_summary_maximum.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'scraped_date': datetime.now().isoformat(),
            'total_requests_used': total_requests_used,
            'sports': summary,
            'total_games': sum(s['games'] for s in summary.values())
        }, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("MAXIMUM SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal Statistics:")
    print(f"  Total games: {sum(s['games'] for s in summary.values()):,}")
    print(f"  API requests used: {total_requests_used:,}")
    print(f"  API requests remaining: ~{20000 - total_requests_used:,}")
    print(f"\nBreakdown by sport:")
    for sport, stats in summary.items():
        print(f"  {sport:<15} {stats['games']:>6,} games ({stats['days']:>4} days, {stats['requests']:>4} requests)")
    print(f"\nAll data saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    scrape_all_sports_maximum()

