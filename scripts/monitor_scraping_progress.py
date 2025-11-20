"""
Monitor Historical Odds Scraping Progress

Real-time monitoring of the background scraping process.

Shows:
- Games collected per sport
- API requests used
- Estimated completion time
- Data quality metrics

Author: Scraping Monitor
Date: November 19, 2025
"""

import json
from pathlib import Path
from datetime import datetime
import time
import os

def monitor_progress():
    """Monitor scraping progress in real-time"""
    output_dir = Path('data/historical_odds')
    
    print("\n" + "="*80)
    print("HISTORICAL ODDS SCRAPING - LIVE MONITOR")
    print("="*80)
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("="*80)
            print(f"SCRAPING PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
            print("="*80)
            
            if not output_dir.exists():
                print("\n⏳ Waiting for scraping to start...")
                time.sleep(5)
                continue
            
            # Check each sport file
            total_games = 0
            total_requests = 0
            
            print(f"\n{'Sport':<15} {'Games':>8} {'Days':>6} {'Requests':>10} {'Status':<15}")
            print("-"*80)
            
            for file in sorted(output_dir.glob('*_historical_odds.json')):
                try:
                    with open(file) as f:
                        data = json.load(f)
                    
                    sport = data.get('sport', file.stem)
                    games = len(data.get('games', []))
                    days = data.get('days_scraped', 0)
                    requests = data.get('total_requests', 0)
                    
                    total_games += games
                    total_requests += requests
                    
                    # Determine status
                    if days >= 1000:
                        status = "COMPLETE"
                    elif games > 0:
                        status = "IN PROGRESS"
                    else:
                        status = "STARTING"
                    
                    print(f"{sport:<15} {games:>8,} {days:>6} {requests:>10} {status:<15}")
                
                except Exception as e:
                    print(f"{file.stem:<15} {'ERROR':<8} {str(e)[:30]}")
            
            print("-"*80)
            print(f"{'TOTAL':<15} {total_games:>8,} {'':<6} {total_requests:>10}")
            print(f"\nAPI Requests Remaining: ~{20000 - total_requests:,}")
            
            # Estimate completion
            if total_requests > 0:
                avg_games_per_request = total_games / total_requests if total_requests > 0 else 0
                estimated_total_games = avg_games_per_request * 18000  # Use 18k to be safe
                print(f"Estimated final total: ~{estimated_total_games:,.0f} games")
            
            # Check if complete
            summary_file = output_dir / 'scraping_summary_maximum.json'
            if summary_file.exists():
                print(f"\n✓ SCRAPING COMPLETE - Summary file found")
                with open(summary_file) as f:
                    summary = json.load(f)
                print(f"  Final total: {summary.get('total_games', 0):,} games")
                break
            
            print(f"\n⏳ Updating every 10 seconds...")
            time.sleep(10)
    
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped.")
        print(f"Scraping continues in background.")
        print(f"Run this script again to resume monitoring.")


if __name__ == '__main__':
    monitor_progress()

