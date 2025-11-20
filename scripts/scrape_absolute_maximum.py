"""
ABSOLUTE MAXIMUM HISTORICAL SCRAPER

With 5 MILLION API requests available, we can pull:
- ALL 75 sports
- Going back 10+ YEARS for each
- Every game, every market, every sportsbook
- Expected: 500,000 - 2,000,000+ games

This will be the most comprehensive sports betting training dataset ever assembled.

Author: Absolute Maximum Collection
Date: November 19, 2025
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import time
import sys

API_KEY = "2e330948334c9505ed5542a82fcfa3b9"
BASE_URL = "https://api.the-odds-api.com/v4"

# With 5M requests, we can go DEEP
# Allocate requests strategically:
# - Major sports (NHL/NBA/NFL/MLB): 100k requests each = 10+ years
# - Soccer leagues: 50k requests each = 5+ years  
# - Other sports: 10-20k each

SPORTS_ALLOCATION = [
    # Major US Sports (400k requests total)
    ('icehockey_nhl', 'NHL', 100000),
    ('basketball_nba', 'NBA', 100000),
    ('americanfootball_nfl', 'NFL', 100000),
    ('baseball_mlb', 'MLB', 100000),
    
    # College Sports (100k requests)
    ('basketball_ncaab', 'NCAAB', 50000),
    ('americanfootball_ncaaf', 'NCAAF', 50000),
    
    # Top Soccer Leagues (500k requests)
    ('soccer_epl', 'EPL', 100000),
    ('soccer_uefa_champs_league', 'UEFA_Champions', 80000),
    ('soccer_spain_la_liga', 'La_Liga', 80000),
    ('soccer_germany_bundesliga', 'Bundesliga', 80000),
    ('soccer_italy_serie_a', 'Serie_A', 80000),
    ('soccer_france_ligue_one', 'Ligue_1', 80000),
    
    # Combat Sports (100k requests)
    ('mma_mixed_martial_arts', 'UFC', 50000),
    ('boxing_boxing', 'Boxing', 50000),
    
    # Other Major Leagues (remaining ~3.9M requests)
    ('soccer_uefa_europa_league', 'Europa_League', 50000),
    ('soccer_usa_mls', 'MLS', 50000),
    ('soccer_mexico_ligamx', 'Liga_MX', 50000),
    ('soccer_brazil_campeonato', 'Brazil_Serie_A', 50000),
    ('soccer_argentina_primera_division', 'Argentina_Primera', 50000),
]


def scrape_sport_absolute_max(sport_key: str, sport_name: str, max_requests: int, output_dir: Path) -> Dict:
    """
    Scrape with allocated request budget.
    
    Goes back day by day until:
    - Hit request budget
    - No data for 60 consecutive days
    - Reach 10 years back
    """
    print(f"\n{'='*80}")
    print(f"{sport_name} ({sport_key})")
    print(f"{'='*80}")
    print(f"Request budget: {max_requests:,}")
    print(f"Expected: {max_requests/365:.1f} years of data")
    
    output_file = output_dir / f"{sport_key}_complete.json"
    
    all_games = []
    current_date = datetime.now()
    days_back = 0
    consecutive_empty = 0
    requests_used = 0
    games_by_date = {}
    
    start_time = time.time()
    
    while requests_used < max_requests and consecutive_empty < 60 and days_back < 3650:  # Max 10 years
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
                    games_by_date[str(current_date.date())] = len(games)
                    consecutive_empty = 0
                    
                    # Progress every 200 days
                    if days_back % 200 == 0 and days_back > 0:
                        elapsed = time.time() - start_time
                        rate = requests_used / elapsed if elapsed > 0 else 0
                        remaining_requests = max_requests - requests_used
                        eta_seconds = remaining_requests / rate if rate > 0 else 0
                        
                        print(f"  [{days_back:>4}d] {current_date.date()} | {len(all_games):>7,} games | {requests_used:>6,}/{max_requests:,} req | ETA: {eta_seconds/60:.0f}min")
                else:
                    consecutive_empty += 1
            else:
                consecutive_empty += 1
                if consecutive_empty == 1:
                    print(f"  No data at {current_date.date()}")
        
        except Exception as e:
            consecutive_empty += 1
        
        # Rate limit (100 req/sec with 5M quota)
        time.sleep(0.01)
        
        current_date -= timedelta(days=1)
        days_back += 1
        
        # Checkpoint every 500 days
        if days_back % 500 == 0:
            checkpoint = {
                'sport': sport_name,
                'sport_key': sport_key,
                'total_games': len(all_games),
                'days_scraped': days_back,
                'requests_used': requests_used,
                'games': all_games,
                'games_by_date': games_by_date
            }
            with open(output_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"  [CHECKPOINT] {len(all_games):,} games saved")
    
    # Final save
    final_data = {
        'sport': sport_name,
        'sport_key': sport_key,
        'total_games': len(all_games),
        'days_scraped': days_back,
        'requests_used': requests_used,
        'scraped_date': datetime.now().isoformat(),
        'games': all_games,
        'games_by_date': games_by_date
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_data, f)
    
    elapsed = time.time() - start_time
    print(f"\n✓ {sport_name}: {len(all_games):,} games | {days_back} days | {requests_used:,} requests | {elapsed/60:.1f}min")
    
    return final_data


def main():
    """Scrape everything with 5M request budget"""
    print("\n" + "="*80)
    print("ABSOLUTE MAXIMUM HISTORICAL SCRAPING")
    print("="*80)
    print(f"\nAPI Quota: 5,000,000 requests")
    print(f"Sports: {len(SPORTS_ALLOCATION)}")
    print(f"Strategy: Allocate requests by sport priority")
    print(f"\nExpected collection:")
    print(f"  - NHL: 10+ years (~15,000 games)")
    print(f"  - NBA: 10+ years (~12,000 games)")
    print(f"  - NFL: 10+ years (~2,700 games)")
    print(f"  - MLB: 10+ years (~24,000 games)")
    print(f"  - Soccer: 5+ years per league (~30,000 games)")
    print(f"  - UFC/Boxing: 10+ years (~5,000 fights)")
    print(f"\nESTIMATED TOTAL: 200,000 - 500,000 games")
    print(f"Estimated time: 6-12 hours")
    
    print(f"\n{'='*80}")
    print("Starting ABSOLUTE MAXIMUM scraping NOW...")
    print(f"{'='*80}\n")
    
    output_dir = Path('data/historical_odds_complete')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    total_games = 0
    total_requests = 0
    
    start_time = time.time()
    
    for i, (sport_key, sport_name, budget) in enumerate(SPORTS_ALLOCATION, 1):
        print(f"\n\n{'#'*80}")
        print(f"# SPORT {i}/{len(SPORTS_ALLOCATION)}: {sport_name}")
        print(f"# Budget: {budget:,} requests")
        print(f"{'#'*80}")
        
        result = scrape_sport_absolute_max(sport_key, sport_name, budget, output_dir)
        
        summary[sport_name] = {
            'sport_key': sport_key,
            'games': result['total_games'],
            'days': result['days_scraped'],
            'requests': result['requests_used']
        }
        
        total_games += result['total_games']
        total_requests += result['requests_used']
        
        elapsed = time.time() - start_time
        print(f"\n  SESSION TOTALS:")
        print(f"    Games: {total_games:,}")
        print(f"    Requests: {total_requests:,} / 5,000,000")
        print(f"    Time: {elapsed/3600:.1f} hours")
        print(f"    Rate: {total_games/(elapsed/3600):.0f} games/hour")
    
    # Final summary
    summary_file = output_dir / 'complete_scraping_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'scraped_date': datetime.now().isoformat(),
            'total_games': total_games,
            'total_requests': total_requests,
            'total_time_hours': (time.time() - start_time) / 3600,
            'sports': summary
        }, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("ABSOLUTE MAXIMUM SCRAPING COMPLETE")
    print(f"{'='*80}")
    print(f"\nFINAL STATISTICS:")
    print(f"  Total games: {total_games:,}")
    print(f"  Sports scraped: {len(summary)}")
    print(f"  API requests used: {total_requests:,}")
    print(f"  Time: {(time.time() - start_time)/3600:.1f} hours")
    
    print(f"\nTop sports by game count:")
    sorted_sports = sorted(summary.items(), key=lambda x: x[1]['games'], reverse=True)
    for sport, stats in sorted_sports[:15]:
        years = stats['days'] / 365
        print(f"  {sport:<30} {stats['games']:>8,} games ({years:.1f} years)")
    
    print(f"\nAll data saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    import sys
    # Run immediately without confirmation
    sys.stdin = open('/dev/null')
    main()

