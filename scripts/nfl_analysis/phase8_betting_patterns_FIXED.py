#!/usr/bin/env python3
"""
Phase 8: Betting Pattern Analysis - FIXED
Properly calculates ATS (Against The Spread) performance
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def calculate_ats_result(row):
    """Calculate if home team covered the spread"""
    spread = row.get('spread_line', 0)
    if spread == 0:
        return None
    
    # Point differential (positive = home won by more)
    actual_diff = row['home_score'] - row['away_score']
    
    # Home covers if: actual_diff + spread > 0
    # Example: Home is -7 favorite, wins by 10: 10 + (-7) = 3 > 0 = COVERS
    # Example: Home is +3 dog, loses by 2: -2 + 3 = 1 > 0 = COVERS
    covered = (actual_diff + spread) > 0
    
    return 1 if covered else 0

def analyze_pattern_ats(games_df, pattern_name, condition, min_games=20):
    """Analyze betting pattern using proper ATS calculation"""
    
    # Filter to pattern games
    pattern_games = games_df[condition].copy()
    
    if len(pattern_games) < min_games:
        return None
    
    # Calculate ATS results
    pattern_games['home_covered'] = pattern_games.apply(calculate_ats_result, axis=1)
    pattern_games = pattern_games[pattern_games['home_covered'].notna()]
    
    if len(pattern_games) < min_games:
        return None
    
    # Calculate metrics (betting on home team)
    total = len(pattern_games)
    wins = pattern_games['home_covered'].sum()
    win_rate = wins / total
    
    # Calculate ROI (assuming -110 odds)
    profit = (wins * 100) - ((total - wins) * 110)
    roi = profit / (total * 110) * 100
    
    return {
        'pattern': pattern_name,
        'games': int(total),
        'wins': int(wins),
        'losses': int(total - wins),
        'win_rate': float(win_rate),
        'profit': float(profit),
        'roi_pct': float(roi),
        'profitable': bool(roi > 5.0),  # 5%+ ROI threshold
    }

def main():
    print("="*60)
    print(f"PHASE 8: BETTING PATTERNS (FIXED) - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "domains"
    
    # Load story scores and features
    print("\n📂 Loading data...")
    with open(data_dir / "nfl_story_scores.json", 'r') as f:
        story_data = json.load(f)
    
    features = pd.read_csv(data_dir / "nfl_complete_features.csv")
    
    print(f"  ✓ {len(story_data['games']):,} games")
    print(f"  ✓ {features.shape[1]} features")
    
    # Create analysis DataFrame
    df = pd.DataFrame(story_data['games'])
    for col in features.columns:
        df[col] = features[col]
    
    # Filter to games with spread data
    df = df[df['spread_line'] != 0].copy()
    print(f"  ✓ {len(df):,} games with spreads")
    
    # Test baseline ATS
    print("\n🔍 Testing baseline ATS performance...")
    baseline_covered = df.apply(calculate_ats_result, axis=1)
    baseline_wins = baseline_covered.sum()
    baseline_rate = baseline_wins / len(df)
    print(f"  Baseline ATS: {baseline_rate:.1%} (should be ~50%)")
    
    # Test betting patterns
    print("\n🔄 Testing narrative-based betting patterns...")
    print("   Betting ON HOME TEAM when pattern matches\n")
    
    patterns_to_test = [
        ('High Story Quality (Q >= 0.4)', df['story_quality'] >= 0.4),
        ('Very High Story (Q >= 0.5)', df['story_quality'] >= 0.5),
        ('Elite Story (Q >= 0.6)', df['story_quality'] >= 0.6),
        
        ('QB Prestige: Home Advantage', df['qb_prestige_diff'] > 0.02),
        ('QB Prestige: Strong Home', df['qb_prestige_diff'] > 0.05),
        
        ('High Rivalry Game', df['rivalry_intensity'] > 0.3),
        ('Division Rivalry', df['division_game'] == 1),
        
        ('Late Season Game', df['week'] >= 13),
        ('Playoff Game', df['playoff'] == 1),
        
        ('Home Underdog', df['spread_line'] > 0),
        ('Big Home Underdog (+3.5+)', df['spread_line'] >= 3.5),
        ('Huge Home Underdog (+7+)', df['spread_line'] >= 7),
        
        ('Home Favorite', df['spread_line'] < 0),
        ('Big Home Favorite (-7+)', df['spread_line'] <= -7),
        
        # Combo patterns
        ('High Story + Home Dog', (df['story_quality'] >= 0.4) & (df['spread_line'] > 0)),
        ('Rivalry + Home Dog', (df['rivalry_intensity'] > 0.3) & (df['spread_line'] > 0)),
        ('Late Season + Home Dog', (df['week'] >= 13) & (df['spread_line'] > 0)),
        ('Division + Home Dog', (df['division_game'] == 1) & (df['spread_line'] > 0)),
        
        ('High Story + Favorite', (df['story_quality'] >= 0.4) & (df['spread_line'] < 0)),
        ('QB Edge + Favorite', (df['qb_prestige_diff'] > 0.02) & (df['spread_line'] < 0)),
        
        ('Playoff + Close Spread', (df['playoff'] == 1) & (abs(df['spread_line']) <= 3)),
        ('Late Season + Rivalry', (df['week'] >= 13) & (df['rivalry_intensity'] > 0.3)),
        
        # Advanced patterns
        ('High Momentum Home', df['momentum_differential'] > 0.2),
        ('Strong Record Home', df['record_differential'] > 2),
        ('Week 13-14 (Critical)', (df['week'] >= 13) & (df['week'] <= 14)),
    ]
    
    results = []
    
    for i, (pattern_name, condition) in enumerate(patterns_to_test, 1):
        print(f"  Pattern {i:2d}/{len(patterns_to_test)}: {pattern_name:40s}", end=" ")
        result = analyze_pattern_ats(df, pattern_name, condition, min_games=20)
        
        if result:
            results.append(result)
            status = "✓" if result['profitable'] else "•"
            print(f"{status} {result['games']:4d} games, {result['win_rate']:5.1%}, {result['roi_pct']:+6.1f}% ROI")
        else:
            print(f"  ⚠ < 20 games")
    
    # Sort by ROI
    results = sorted(results, key=lambda x: x['roi_pct'], reverse=True)
    
    # Summary
    profitable = [r for r in results if r['profitable']]
    
    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS SUMMARY (PROPER ATS)")
    print(f"{'='*60}")
    print(f"  Baseline ATS: {baseline_rate:.1%}")
    print(f"  Total patterns tested: {len(results)}")
    print(f"  Profitable patterns (ROI > 5%): {len(profitable)}")
    if results:
        print(f"  Best ROI: {results[0]['roi_pct']:+.1f}% ({results[0]['pattern']})")
        print(f"  Worst ROI: {results[-1]['roi_pct']:+.1f}% ({results[-1]['pattern']})")
    print(f"{'='*60}")
    
    # Save results
    betting_results = {
        'timestamp': datetime.now().isoformat(),
        'domain': 'NFL',
        'method': 'ATS (Against The Spread)',
        'baseline_ats': float(baseline_rate),
        'total_games': len(df),
        'patterns_tested': len(patterns_to_test),
        'profitable_patterns': len(profitable),
        'patterns': results,
        'summary': {
            'best_pattern': results[0] if results else None,
            'total_profitable': len(profitable),
            'avg_roi_profitable': float(np.mean([r['roi_pct'] for r in profitable])) if profitable else 0,
        }
    }
    
    output_path = data_dir / "nfl_betting_patterns_FIXED.json"
    with open(output_path, 'w') as f:
        json.dump(betting_results, f, indent=2)
    
    print(f"\n✓ Betting patterns saved: {output_path.name}")
    
    if profitable:
        print(f"\n🎯 PROFITABLE PATTERNS FOUND:")
        for i, pattern in enumerate(profitable[:5], 1):
            print(f"  {i}. {pattern['pattern']}")
            print(f"      {pattern['games']} games | {pattern['win_rate']:.1%} win | {pattern['roi_pct']:+.1f}% ROI | ${pattern['profit']:,.0f}")
    else:
        print(f"\n⚠ No profitable patterns found (still surprising)")
    
    print(f"\n{'='*60}")
    print("PHASE 8 COMPLETE (FIXED) ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

