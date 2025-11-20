#!/usr/bin/env python3
"""
Update Investor Document from Source Data

This script automatically updates INVESTOR_PRESENTATION.md with latest
backtest results, validation metrics, and performance data.

Usage:
    python scripts/update_investor_doc.py [--dry-run] [--check-only]
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVESTOR_DOC = PROJECT_ROOT / "docs" / "investor" / "INVESTOR_PRESENTATION.md"
BACKTEST_SUMMARY = PROJECT_ROOT / "docs" / "investor" / "data" / "backtest_summary.json"
PRODUCTION_BACKTEST = PROJECT_ROOT / "analysis" / "production_backtest_results.json"
EXECUTIVE_SUMMARY = PROJECT_ROOT / "analysis" / "EXECUTIVE_SUMMARY_BACKTEST.md"


def load_json_safe(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely, return None if not found."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Warning: {filepath} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Invalid JSON in {filepath}: {e}")
        return None


def extract_metrics_from_backtest(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from backtest data."""
    metrics = {}
    
    if 'nhl' in data.get('systems', {}):
        nhl = data['systems']['nhl']
        if 'results' in nhl:
            # Get best result (meta_ensemble_65)
            best = nhl['results'].get('meta_ensemble_65', {})
            metrics['nhl'] = {
                'win_rate': best.get('win_rate', 0) * 100,
                'roi': best.get('roi', 0) * 100,
                'bets': best.get('bets', 0),
                'wins': best.get('wins', 0),
                'losses': best.get('losses', 0),
            }
    
    if 'nfl' in data.get('systems', {}):
        nfl = data['systems']['nfl']
        if 'results' in nfl and 'testing' in nfl['results']:
            test = nfl['results']['testing']
            metrics['nfl'] = {
                'win_rate': test.get('win_rate', 0) * 100,
                'roi': test.get('roi', 0) * 100,
                'bets': test.get('games', 0),
                'wins': test.get('wins', 0),
                'losses': test.get('losses', 0),
            }
    
    if 'nba' in data.get('systems', {}):
        nba = data['systems']['nba']
        if 'results' in nba and 'testing' in nba['results']:
            test = nba['results']['testing']
            metrics['nba'] = {
                'win_rate': test.get('win_rate', 0) * 100,
                'roi': test.get('roi', 0) * 100,
                'bets': test.get('games', 0),
                'wins': test.get('wins', 0),
                'losses': test.get('losses', 0),
            }
    
    return metrics


def calculate_profit_1m_bankroll(roi_pct: float, bets: int, kelly_fraction: float = 0.01) -> float:
    """Calculate expected profit for $1M bankroll with Kelly compounding."""
    # Simplified: assumes average bet size of 1% of bankroll
    # More accurate calculation would simulate each bet
    initial_bet = 1_000_000 * kelly_fraction  # $10,000
    profit_per_bet = initial_bet * (roi_pct / 100)
    return profit_per_bet * bets


def update_metrics_table(content: str, metrics: Dict[str, Any]) -> str:
    """Update the Key Metrics table in the document."""
    # Find the metrics table
    pattern = r'(\| System \| Win Rate \| ROI \| Volume/Season \| Expected Profit\* \| Status \|\n\|[-\|]+\n)(.*?)(\n\n\*At.*?\n)'
    
    def replace_table(match):
        header = match.group(1)
        footer = match.group(3)
        
        # Build new table rows
        rows = []
        
        if 'nhl' in metrics:
            nhl = metrics['nhl']
            profit = calculate_profit_1m_bankroll(nhl['roi'], nhl['bets'])
            rows.append(f"| **NHL (Primary)** | **{nhl['win_rate']:.1f}%** | **+{nhl['roi']:.1f}%** | {nhl['bets']} bets | **${profit:,.0f}** | ✅ Deploy Ready |")
        
        if 'nfl' in metrics:
            nfl = metrics['nfl']
            profit = calculate_profit_1m_bankroll(nfl['roi'], nfl['bets'])
            rows.append(f"| **NFL** | **{nfl['win_rate']:.1f}%** | **+{nfl['roi']:.1f}%** | {nfl['bets']} bets | **${profit:,.0f}** | ✅ Deploy Ready |")
        
        if 'nba' in metrics:
            nba = metrics['nba']
            profit = calculate_profit_1m_bankroll(nba['roi'], nba['bets'])
            rows.append(f"| **NBA** | {nba['win_rate']:.1f}% | +{nba['roi']:.1f}% | {nba['bets']} bets | ${profit:,.0f} | ✅ Validated (Marginal) |")
        
        # Calculate combined
        if 'nhl' in metrics and 'nfl' in metrics:
            nhl = metrics['nhl']
            nfl = metrics['nfl']
            total_bets = nhl['bets'] + nfl['bets']
            combined_profit = calculate_profit_1m_bankroll(nhl['roi'], nhl['bets']) + \
                            calculate_profit_1m_bankroll(nfl['roi'], nfl['bets'])
            rows.append(f"| **Combined Portfolio** | - | - | {total_bets} bets | **${combined_profit:,.0f}** | ✅ Diversified |")
        
        return header + '\n'.join(rows) + footer
    
    return re.sub(pattern, replace_table, content, flags=re.DOTALL)


def update_date_stamp(content: str) -> str:
    """Update date stamp in document."""
    today = datetime.now().strftime("%B %Y")
    pattern = r'\*\*Date:\*\* .*?\n'
    replacement = f"**Date:** {today}\n"
    return re.sub(pattern, replacement, content)


def check_data_freshness() -> Dict[str, Any]:
    """Check if source data files are up to date."""
    status = {}
    
    files_to_check = {
        'backtest_summary': BACKTEST_SUMMARY,
        'production_backtest': PRODUCTION_BACKTEST,
        'executive_summary': EXECUTIVE_SUMMARY,
    }
    
    for name, filepath in files_to_check.items():
        if filepath.exists():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            status[name] = {
                'exists': True,
                'last_modified': mtime.isoformat(),
                'age_days': age_days,
                'fresh': age_days < 30  # Consider fresh if < 30 days old
            }
        else:
            status[name] = {'exists': False}
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Update investor document from source data")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--check-only', action='store_true', help='Only check data freshness, don\'t update')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Investor Document Update Script")
    print("=" * 80)
    print()
    
    # Check data freshness
    print("📊 Checking data freshness...")
    freshness = check_data_freshness()
    
    for name, status in freshness.items():
        if status.get('exists'):
            age = status['age_days']
            fresh = "✅" if status['fresh'] else "⚠️"
            print(f"  {fresh} {name}: {age} days old")
        else:
            print(f"  ❌ {name}: File not found")
    
    print()
    
    if args.check_only:
        return
    
    # Load source data
    print("📥 Loading source data...")
    backtest_data = load_json_safe(BACKTEST_SUMMARY)
    
    if not backtest_data:
        print("❌ No backtest data found. Cannot update document.")
        return
    
    # Extract metrics
    print("🔍 Extracting metrics...")
    metrics = extract_metrics_from_backtest(backtest_data)
    
    print(f"  Found metrics for: {', '.join(metrics.keys())}")
    print()
    
    # Load current document
    if not INVESTOR_DOC.exists():
        print(f"❌ Investor document not found: {INVESTOR_DOC}")
        return
    
    print("📄 Loading investor document...")
    with open(INVESTOR_DOC, 'r') as f:
        content = f.read()
    
    # Update document
    print("✏️  Updating document...")
    
    if args.dry_run:
        print("  [DRY RUN] Would update metrics table")
        print("  [DRY RUN] Would update date stamp")
    else:
        # Update metrics table
        content = update_metrics_table(content, metrics)
        
        # Update date stamp
        content = update_date_stamp(content)
        
        # Save updated document
        print("💾 Saving updated document...")
        with open(INVESTOR_DOC, 'w') as f:
            f.write(content)
        
        print("✅ Document updated successfully!")
    
    print()
    print("=" * 80)
    print("Update complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review updated metrics")
    print("  2. Check financial projections")
    print("  3. Verify statistical significance sections")
    print("  4. Commit changes to version control")


if __name__ == "__main__":
    main()

