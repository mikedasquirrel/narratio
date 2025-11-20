"""
Real-Time Monitor for Absolute Maximum Scraping

Tracks the massive 5M-request historical collection.

Shows:
- Games collected per sport
- Years of data per sport
- API requests used/remaining
- Collection rate (games/hour)
- Estimated completion time
- Data quality metrics

Author: Absolute Scraping Monitor
Date: November 19, 2025
"""

import json
from pathlib import Path
from datetime import datetime
import time
import os

def format_large_number(n):
    """Format large numbers with K/M suffix"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(n)

def monitor_absolute_scraping():
    """Monitor the absolute maximum scraping in real-time"""
    output_dir = Path('data/historical_odds_complete')
    
    print("\n" + "="*80)
    print("ABSOLUTE MAXIMUM SCRAPING - LIVE MONITOR")
    print("="*80)
    print("\nAPI Quota: 5,000,000 requests")
    print("Target: 200,000 - 500,000+ games")
    print("\nPress Ctrl+C to stop monitoring (scraping continues)\n")
    
    start_monitor_time = time.time()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*80)
            print(f"ABSOLUTE MAXIMUM SCRAPING - {datetime.now().strftime('%H:%M:%S')}")
            print(f"Monitor running: {(time.time() - start_monitor_time)/60:.0f} minutes")
            print("="*80)
            
            if not output_dir.exists():
                print("\n⏳ Waiting for scraping to start...")
                time.sleep(10)
                continue
            
            # Collect stats
            total_games = 0
            total_requests = 0
            total_days = 0
            sports_data = []
            
            for file in sorted(output_dir.glob('*_complete.json')):
                try:
                    with open(file) as f:
                        data = json.load(f)
                    
                    sport = data.get('sport', file.stem)
                    games = data.get('total_games', 0)
                    days = data.get('days_scraped', 0)
                    requests = data.get('requests_used', 0)
                    
                    if games > 0:
                        sports_data.append({
                            'sport': sport,
                            'games': games,
                            'days': days,
                            'years': days / 365,
                            'requests': requests
                        })
                        
                        total_games += games
                        total_requests += requests
                        total_days += days
                
                except:
                    pass
            
            # Display table
            if sports_data:
                print(f"\n{'Sport':<25} {'Games':>10} {'Years':>7} {'Requests':>12} {'Games/Day':>10}")
                print("-"*80)
                
                for s in sorted(sports_data, key=lambda x: x['games'], reverse=True)[:20]:
                    games_per_day = s['games'] / s['days'] if s['days'] > 0 else 0
                    print(f"{s['sport']:<25} {s['games']:>10,} {s['years']:>7.1f} {format_large_number(s['requests']):>12} {games_per_day:>10.1f}")
                
                if len(sports_data) > 20:
                    print(f"  ... and {len(sports_data) - 20} more sports")
                
                print("-"*80)
                print(f"{'TOTAL':<25} {total_games:>10,} {total_days/365:>7.1f} {format_large_number(total_requests):>12}")
            else:
                print("\n⏳ No data collected yet...")
            
            # API usage
            print(f"\nAPI USAGE:")
            print(f"  Requests used: {format_large_number(total_requests)} / 5.0M ({total_requests/5_000_000*100:.1f}%)")
            print(f"  Requests remaining: {format_large_number(5_000_000 - total_requests)}")
            
            # Collection rate
            if total_requests > 0:
                elapsed_hours = (time.time() - start_monitor_time) / 3600
                if elapsed_hours > 0:
                    games_per_hour = total_games / elapsed_hours
                    requests_per_hour = total_requests / elapsed_hours
                    
                    print(f"\nCOLLECTION RATE:")
                    print(f"  Games/hour: {games_per_hour:,.0f}")
                    print(f"  Requests/hour: {requests_per_hour:,.0f}")
                    
                    # ETA
                    if total_requests < 4_000_000:  # Still scraping
                        remaining_requests = 4_000_000 - total_requests  # Use 4M to be safe
                        eta_hours = remaining_requests / requests_per_hour if requests_per_hour > 0 else 0
                        print(f"  ETA to completion: {eta_hours:.1f} hours")
            
            # Estimated final total
            if total_requests > 100000:
                estimated_final = total_games * (4_000_000 / total_requests)
                print(f"\nESTIMATED FINAL TOTAL: {estimated_final:,.0f} games")
            
            # Check if complete
            summary_file = output_dir / 'complete_scraping_summary.json'
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                
                print(f"\n{'='*80}")
                print("✓ SCRAPING COMPLETE")
                print(f"{'='*80}")
                print(f"Final total: {summary.get('total_games', 0):,} games")
                print(f"Time: {summary.get('total_time_hours', 0):.1f} hours")
                break
            
            print(f"\n⏳ Updating every 30 seconds... (iteration {iteration})")
            time.sleep(30)
    
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Scraping continues in background.")


if __name__ == '__main__':
    monitor_absolute_scraping()

