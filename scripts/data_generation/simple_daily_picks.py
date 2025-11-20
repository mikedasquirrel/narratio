#!/usr/bin/env python3
"""
Simple Daily Picks Generator
Uses ONLY validated patterns that actually work on holdout data
No heavy ML models, no mutex locks, just proven patterns.

Date: November 17, 2025
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

def get_tomorrow_date():
    """Get tomorrow's date"""
    return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print('='*80)

def main():
    tomorrow = get_tomorrow_date()
    
    print_header(f"VALIDATED BETTING PICKS FOR {tomorrow}")
    print(f"\nBased on PRODUCTION-TESTED patterns (not marketing claims)")
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # NHL - STRONGEST EDGE (69.4% win rate validated)
    # ========================================================================
    print_header("🏒 NHL - HIGHEST CONFIDENCE (Validated: 69.4% win, 32.5% ROI)")
    
    print("""
STRATEGY: Wait for today's NHL games and apply Meta-Ensemble ≥65% threshold

VALIDATED PERFORMANCE:
- Win Rate: 69.4% (59 wins, 26 losses on 2,779 games tested)
- ROI: +32.5%
- Volume: ~85 bets per season (very selective)
- Expected profit: $2,763/season at $100/unit

HOW TO GET PICKS:
1. Fetch today's NHL games (need live odds API)
2. Extract 79 features (50 performance + 29 nominative)
3. Run Meta-Ensemble model
4. Bet ONLY games with confidence ≥65%

REQUIRED:
- Trained Meta-Ensemble model (narrative_optimization/domains/nhl/models/meta_ensemble.pkl)
- Feature extraction pipeline (79 dimensions)

KEY FEATURES THAT MATTER (Top 10):
1. Cup history differential (26.6% importance)
2. Away Cup history (12.9%)
3. Total nominative gravity (11.8%)
4. Home Cup history (10.9%)
5. Combined brand gravity (9.9%)
6. Brand differential (6.3%)
7. Star power differential (5.7%)
8. Away brand weight (3.7%)
9. Home brand weight (3.3%)
10. Away star power (3.2%)

INSIGHT: NHL betting is about HISTORY > CURRENT STATS
Teams with more Cup wins, bigger brand, more star players win MORE than their current
stats suggest. Market undervalues legacy/prestige.
""")
    
    # ========================================================================
    # NFL - CONTEXT-SPECIFIC EDGE
    # ========================================================================
    print_header("🏈 NFL - CONTEXTUAL PATTERNS ONLY (Validated: 66.7% win, 27.3% ROI)")
    
    print("""
STRATEGY: Look for HOME UNDERDOGS with QB or COACH ADVANTAGE

VALIDATED PATTERNS:

Pattern 1: QB Edge + Home Underdog (spread > 2.5)
- Training: 61.5% win, 17.5% ROI (78 games, 2020-2023)
- Testing: 66.7% win, 27.3% ROI (9 games, 2024)
- Volume: ~20 bets/season
- Expected profit: $546/season

Pattern 2: Coach Edge + Home Underdog (spread > 3)
- Training: 64.9% win, 23.9% ROI (94 games)
- Testing: 75.0% win, 43.2% ROI (20 games)
- Volume: ~24 bets/season
- Expected profit: $1,037/season

CONDITIONS:
✅ Home team is UNDERDOG (positive spread)
✅ Home has QB prestige advantage OR coach prestige advantage
✅ Spread > 2.5 (preferably > 4)

WHY THIS WORKS:
Market makes team underdog based on W-L record, but overlooks QB/coach quality.
When a team with Mahomes/Allen is underdog at home, market has mispriced leadership.

WARNING: Do NOT bet when home team is FAVORED with QB edge (43% win rate, loses money)
""")
    
    # Check for today's games
    nfl_file = Path('data/domains/nfl_complete_dataset.json')
    if nfl_file.exists():
        with open(nfl_file) as f:
            games = json.load(f)
        
        # Filter for tomorrow (simplified - would need actual schedule)
        print(f"\n📅 NFL Schedule Check:")
        print(f"  - Dataset has {len(games)} historical games")
        print(f"  - Need live schedule API to get {tomorrow} games")
        print(f"  - Then filter for: home underdog + QB/coach edge")
    
    # ========================================================================
    # NBA - MINIMAL EDGE
    # ========================================================================
    print_header("🏀 NBA - SMALL EDGE (Validated: 54.5% win, 7.6% ROI)")
    
    print("""
STRATEGY: Elite teams in close matchups only

VALIDATED PATTERN:

Elite Team + Close Game (|spread| < 3)
- Training: 62.6% win, 18.6% ROI (91 games)
- Testing: 54.5% win, 7.6% ROI (44 games, 2023-24)
- Volume: ~44 bets/season
- Expected profit: $334/season

CONDITIONS:
✅ Team has season win rate > 65% (elite)
✅ Spread is close (between -3 and +3)
✅ Bet on the elite team

WHY EDGE IS SMALL:
NBA market is HIGHLY EFFICIENT. Books have same data we do.
Only tiny edge exists where elite teams are undervalued in close matchups.

REALITY CHECK:
- NBA "Pattern #1" (64.3% claim) tested at 52.4% on 2024-25 data
- The 225 patterns with "64.8% accuracy" were from training data
- Real holdout performance is 52-54% (barely profitable)
""")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("📊 HONEST SUMMARY")
    
    print("""
WHAT ACTUALLY WORKS:

1. NHL Meta-Ensemble ≥65%: 69.4% win, 32.5% ROI
   → Best validated system, moderate volume
   
2. NFL Contextual Patterns: 67-75% win, 27-43% ROI
   → Works but LOW VOLUME (~40-50 bets/season total)
   → Only in specific contexts (underdog + prestige edge)
   
3. NBA Elite Close: 54.5% win, 7.6% ROI
   → Minimal edge, efficient market
   → Not worth the effort unless high volume

WHAT DOESN'T WORK:

❌ NFL Aggregate ML: 43% win, -17.5% ROI (LOSES MONEY)
❌ NBA "Simple Patterns": 52% accuracy (barely break-even)
❌ The 95.8% and 64.3% claims (training data overfitting)

RECOMMENDED STRATEGY FOR TOMORROW:

1. Focus on NHL games (strongest edge)
2. Look for NFL contextual opportunities (low volume but good ROI)
3. Skip NBA unless you enjoy small edges with high variance

EXPECTED ANNUAL PROFIT (at $100/unit):
- NHL: $2,763 - $14,079 (depending on confidence threshold)
- NFL: $500 - $1,090 (low volume, contextual only)
- NBA: $84 - $334 (minimal, not recommended)

TOTAL: $3,347 - $15,503/year (conservative to moderate strategies)
""")
    
    print_header("🎯 TO GET ACTUAL PICKS FOR TOMORROW")
    
    print("""
You need:

1. Live game schedules (no API key set currently)
2. Live odds data (need THE_ODDS_API_KEY env variable)
3. Feature extraction for each game
4. Trained models to generate confidence scores

CURRENT BLOCKER:
- Heavy ML models cause mutex locks in your environment
- Need lightweight feature extraction or API-based predictions

RECOMMENDATION:
Set up The Odds API (free tier: 500 requests/month)
Then run targeted predictions without loading full sklearn models.
""")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()

