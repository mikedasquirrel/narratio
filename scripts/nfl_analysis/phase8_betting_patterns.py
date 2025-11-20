#!/usr/bin/env python3
"""
Phase 8: Betting Pattern Analysis
Finds exploitable patterns in betting markets using narrative features
Even though narrative doesn't control outcomes, it may create market inefficiencies
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def analyze_pattern(games_df, pattern_name, condition, min_games=20):
    """Analyze a specific betting pattern"""
    
    # Filter to pattern games
    pattern_games = games_df[condition].copy()
    
    if len(pattern_games) < min_games:
        return None
    
    # Calculate metrics
    total = len(pattern_games)
    wins = (pattern_games['favorite_won'] == 1).sum()
    win_rate = wins / total
    
    # Calculate ROI (assuming -110 odds)
    # Win = +100, Loss = -110
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
        'profitable': bool(roi > 10.0),  # 10%+ ROI threshold
    }

def main():
    print("="*60)
    print(f"PHASE 8: BETTING PATTERNS - {datetime.now().strftime('%H:%M:%S')}")
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
    print(f"  ✓ {len(df):,} games with spreads for betting analysis")
    
    # Test betting patterns
    print("\n🔄 Testing betting patterns...")
    
    patterns_to_test = [
        ('High Story Quality (Q >= 0.4)', df['story_quality'] >= 0.4),
        ('Very High Story (Q >= 0.5)', df['story_quality'] >= 0.5),
        ('Elite Story (Q >= 0.6)', df['story_quality'] >= 0.6),
        
        ('QB Prestige Edge > 0.02', df['qb_prestige_diff'] > 0.02),
        ('QB Prestige Edge > 0.05', df['qb_prestige_diff'] > 0.05),
        
        ('High Rivalry (> 0.3)', df['rivalry_intensity'] > 0.3),
        ('Division Game', df['division_game'] == 1),
        
        ('Late Season (Week 13+)', df['week'] >= 13),
        ('Playoff Games', df['playoff'] == 1),
        
        ('Big Underdog (+7 or more)', df['spread_line'] >= 7),
        ('Big Favorite (-7 or more)', df['spread_line'] <= -7),
        
        ('Underdog + High Story', (df['spread_line'] > 3) & (df['story_quality'] >= 0.4)),
        ('Late Season + Rivalry', (df['week'] >= 13) & (df['rivalry_intensity'] > 0.3)),
        ('Division + High Momentum', (df['division_game'] == 1) & (abs(df['momentum_differential']) > 0.3)),
        
        ('Playoff + Close Matchup', (df['playoff'] == 1) & (abs(df['record_differential']) <= 2)),
    ]
    
    results = []
    
    for i, (pattern_name, condition) in enumerate(patterns_to_test, 1):
        print(f"\n  Pattern {i}/{len(patterns_to_test)}: {pattern_name}")
        result = analyze_pattern(df, pattern_name, condition)
        
        if result:
            results.append(result)
            status = "✓ PROFITABLE" if result['profitable'] else "  neutral"
            print(f"    {status}: {result['games']} games, {result['win_rate']:.1%} win, {result['roi_pct']:+.1f}% ROI")
        else:
            print(f"    ⚠ Insufficient data")
    
    # Sort by ROI
    results = sorted(results, key=lambda x: x['roi_pct'], reverse=True)
    
    # Summary
    profitable = [r for r in results if r['profitable']]
    
    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total patterns tested: {len(results)}")
    print(f"  Profitable patterns: {len(profitable)}")
    print(f"  Best ROI: {results[0]['roi_pct']:.1f}% ({results[0]['pattern']})" if results else "")
    print(f"{'='*60}")
    
    # Save results
    betting_results = {
        'timestamp': datetime.now().isoformat(),
        'domain': 'NFL',
        'total_games': len(df),
        'patterns_tested': len(patterns_to_test),
        'profitable_patterns': len(profitable),
        'patterns': results,
        'summary': {
            'best_pattern': results[0] if results else None,
            'total_profitable': len(profitable),
            'avg_roi_profitable': np.mean([r['roi_pct'] for r in profitable]) if profitable else 0,
        }
    }
    
    output_path = data_dir / "nfl_betting_patterns.json"
    with open(output_path, 'w') as f:
        json.dump(betting_results, f, indent=2)
    
    print(f"\n✓ Betting patterns saved: {output_path.name}")
    
    if profitable:
        print(f"\n🎯 Top 3 Profitable Patterns:")
        for i, pattern in enumerate(profitable[:3], 1):
            print(f"  {i}. {pattern['pattern']}")
            print(f"      {pattern['games']} games, {pattern['win_rate']:.1%} win, {pattern['roi_pct']:+.1f}% ROI")
    
    print(f"\n{'='*60}")
    print("PHASE 8 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

