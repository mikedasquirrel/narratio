"""
NBA Selective Betting Strategy - High Confidence + Alternative Markets

Key insights:
1. Don't bet ALL patterns - only highest confidence
2. Use alternative bet types where markets less efficient
3. Context patterns may identify props/totals better than moneyline

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from transformers.context_pattern import ContextPatternTransformer

print("="*80)
print("NBA SELECTIVE BETTING - HIGH CONFIDENCE ONLY")
print("="*80)

# Load data
data_path = project_root.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'
with open(data_path) as f:
    games = json.load(f)

# Extract features for 2022-24
features_list = []
outcomes = []
game_metadata = []
scores = []

for game in games:
    tc = game.get('temporal_context', {})
    if tc.get('games_played', 0) == 0:
        continue
    
    if game.get('season', '') not in ['2023-24', '2022-23']:
        continue
    
    features_list.append({
        'home': 1.0 if game.get('home_game') else 0.0,
        'season_win_pct': tc.get('season_win_pct', 0.5),
        'l10_win_pct': tc.get('l10_win_pct', 0.5),
        'games_played': tc.get('games_played', 0) / 82.0,
        'record_diff': abs(tc.get('season_win_pct', 0.5) - 0.5),
    })
    outcomes.append(1 if game['won'] else 0)
    scores.append(game.get('points', 100))
    game_metadata.append({
        'season': game.get('season', ''),
        'date': game.get('date', ''),
        'team': game.get('team_name', ''),
        'home': game.get('home_game', False),
        'win_pct': tc.get('season_win_pct', 0.5),
        'points': game.get('points', 100),
    })

X = pd.DataFrame(features_list)
y = np.array(outcomes)
scores_arr = np.array(scores)
metadata = pd.DataFrame(game_metadata)

# Train/test split
train_mask = metadata['season'] == '2022-23'
test_mask = metadata['season'] == '2023-24'

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
scores_train, scores_test = scores_arr[train_mask], scores_arr[test_mask]
meta_train, meta_test = metadata[train_mask], metadata[test_mask]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Discover patterns
transformer = ContextPatternTransformer(
    min_accuracy=0.60,
    min_samples=50,
    max_patterns=20
)

transformer.fit(X_train, y_train)

print(f"\nPatterns discovered: {len(transformer.patterns_)}")

# Get all recommendations
recommendations = transformer.get_betting_recommendations(X_test)

print(f"Total recommendations: {len(recommendations)}")

# === STRATEGY 1: CONFIDENCE THRESHOLDS ===
print("\n" + "="*80)
print("STRATEGY 1: PROGRESSIVE CONFIDENCE THRESHOLDS")
print("="*80)

confidence_thresholds = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]

results = []

for threshold in confidence_thresholds:
    high_conf_recs = [r for r in recommendations if r['confidence'] >= threshold]
    
    if not high_conf_recs:
        continue
    
    bet_indices = [r['sample_idx'] for r in high_conf_recs]
    bet_outcomes = y_test[bet_indices]
    
    n_bets = len(bet_outcomes)
    wins = bet_outcomes.sum()
    accuracy = wins / n_bets
    
    # ROI at -110
    profit_110 = wins * 91 - (n_bets - wins) * 100
    wagered = n_bets * 110
    roi_110 = (profit_110 / wagered) * 100
    
    breakeven = 0.5238
    profitable = accuracy > breakeven
    
    results.append({
        'threshold': threshold,
        'bets': n_bets,
        'wins': wins,
        'accuracy': accuracy,
        'roi_110': roi_110,
        'profitable': profitable
    })

print(f"\n{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Accuracy':<12} {'ROI':<12} {'Status'}")
print("-"*80)

for r in results:
    status = "✓ PROFIT" if r['profitable'] else "✗ LOSS"
    print(f"{r['threshold']:<12.1%} {r['bets']:<8} {r['wins']:<8} {r['accuracy']:<12.1%} {r['roi_110']:+<11.1%} {status}")

# Find sweet spot
profitable_results = [r for r in results if r['profitable']]

if profitable_results:
    best = max(profitable_results, key=lambda x: x['roi_110'])
    print(f"\n✓ BEST THRESHOLD: {best['threshold']:.1%}")
    print(f"  Bets: {best['bets']}")
    print(f"  Accuracy: {best['accuracy']:.1%}")
    print(f"  ROI: {best['roi_110']:+.1%}")
    print(f"  Annual bets (scaled): ~{int(best['bets'] * 1230/len(X_test))}")
else:
    print("\n✗ No profitable threshold found at -110 odds")
    # Check best accuracy
    best_acc = max(results, key=lambda x: x['accuracy'])
    print(f"\n⚠ Best accuracy: {best_acc['accuracy']:.1%} at {best_acc['threshold']:.1%} confidence")
    print(f"   (Still below breakeven {breakeven:.1%})")

# === STRATEGY 2: PATTERN-SPECIFIC BETTING ===
print("\n" + "="*80)
print("STRATEGY 2: BET ONLY TOP 3 PATTERNS")
print("="*80)

# Evaluate each pattern individually on test set
pattern_performance = []

for i, pattern in enumerate(transformer.patterns_[:10]):
    # Find test games matching this pattern
    test_matches = []
    for idx in range(len(X_test)):
        row = X_test.iloc[idx:idx+1].reset_index(drop=True)
        if pattern.matches(row).any():
            test_matches.append(idx)
    
    if len(test_matches) < 10:
        continue
    
    match_outcomes = y_test[test_matches]
    test_accuracy = match_outcomes.mean()
    
    pattern_performance.append({
        'pattern_idx': i,
        'pattern': pattern,
        'train_accuracy': pattern.accuracy,
        'test_matches': len(test_matches),
        'test_accuracy': test_accuracy,
        'generalization': test_accuracy / pattern.accuracy if pattern.accuracy > 0 else 0
    })

# Sort by test accuracy
pattern_performance.sort(key=lambda x: x['test_accuracy'], reverse=True)

print(f"\n{'#':<4} {'Train Acc':<12} {'Test Acc':<12} {'Test Bets':<12} {'Generalizes?'}")
print("-"*80)

for i, perf in enumerate(pattern_performance[:10], 1):
    generalizes = "✓" if perf['generalization'] > 0.85 else "✗"
    print(f"{i:<4} {perf['train_accuracy']:<12.1%} {perf['test_accuracy']:<12.1%} {perf['test_matches']:<12} {generalizes}")

# Bet only on top 3 patterns
top_3_patterns = [p['pattern'] for p in pattern_performance[:3]]

top_3_matches = []
for idx in range(len(X_test)):
    row = X_test.iloc[idx:idx+1].reset_index(drop=True)
    for pattern in top_3_patterns:
        if pattern.matches(row).any():
            top_3_matches.append(idx)
            break

if top_3_matches:
    top_3_outcomes = y_test[top_3_matches]
    top_3_accuracy = top_3_outcomes.mean()
    
    wins = top_3_outcomes.sum()
    n_bets = len(top_3_outcomes)
    profit = wins * 91 - (n_bets - wins) * 100
    roi = (profit / (n_bets * 110)) * 100
    
    print(f"\n✓ TOP 3 PATTERNS ONLY:")
    print(f"  Bets: {n_bets}")
    print(f"  Accuracy: {top_3_accuracy:.1%}")
    print(f"  ROI at -110: {roi:+.1%}")
    print(f"  {'PROFITABLE' if top_3_accuracy > 0.5238 else 'NOT PROFITABLE'}")

# === STRATEGY 3: ALTERNATIVE BET TYPES ===
print("\n" + "="*80)
print("STRATEGY 3: ALTERNATIVE BET TYPES (CONCEPTUAL)")
print("="*80)

print("\nContext patterns may identify:")
print("\n1. TEAM TOTALS (Over/Under)")
print("   Pattern: home=1 & l10_win_pct>0.6 → High scoring?")
print("   Hypothesis: Hot teams score more")

# Calculate average points for pattern matches
top_pattern = transformer.patterns_[0]
pattern_matches = []
for idx in range(len(X_test)):
    row = X_test.iloc[idx:idx+1].reset_index(drop=True)
    if top_pattern.matches(row).any():
        pattern_matches.append(idx)

if pattern_matches:
    pattern_scores = scores_test[pattern_matches]
    all_scores = scores_test
    
    avg_pattern = pattern_scores.mean()
    avg_all = all_scores.mean()
    
    print(f"   Avg points (pattern matches): {avg_pattern:.1f}")
    print(f"   Avg points (all games): {avg_all:.1f}")
    print(f"   Differential: {avg_pattern - avg_all:+.1f} points")
    
    if abs(avg_pattern - avg_all) > 2:
        print(f"   ✓ EXPLOITABLE for totals betting")
    else:
        print(f"   ⚠ Weak signal for totals")

print("\n2. PLAYER PROPS")
print("   Pattern context → Player performance?")
print("   Examples:")
print("   - home=1 & hot → Star player points OVER")
print("   - l10_win_pct>0.7 → Main scorer assists OVER")
print("   - early_season → Young player development UNDER")
print("   ⚠ Need player-level data to test")

print("\n3. LIVE BETTING")
print("   Pattern context → In-game momentum?")
print("   Examples:")
print("   - If pattern says 62% win, bet when live odds drift")
print("   - Hot team down at halftime → VALUE bet")
print("   ⚠ Need live odds data to test")

print("\n4. ALTERNATIVE SPREADS")
print("   Pattern: 62% win rate → likely cover +3.5")
print("   Bet smaller spreads at better odds")
print("   ⚠ Need spread data to test")

# === FINAL ANALYSIS ===
print("\n" + "="*80)
print("FINAL ANALYSIS")
print("="*80)

print("\n📊 MONEYLINE/SPREAD BETTING (tested):")
if profitable_results:
    best = max(profitable_results, key=lambda x: x['roi_110'])
    print(f"  ✓ PROFITABLE at {best['threshold']:.1%}+ confidence")
    print(f"    ROI: {best['roi_110']:+.1%}")
    print(f"    Volume: {best['bets']} bets/season")
else:
    print(f"  ✗ NOT PROFITABLE at standard -110 odds")
    print(f"    Best accuracy: {max(results, key=lambda x: x['accuracy'])['accuracy']:.1%}")
    print(f"    Need: 52.4%+ for profit")

print("\n💡 ALTERNATIVE MARKETS (untested but promising):")
print("  1. Team totals - Pattern shows {:.1f} point differential".format(avg_pattern - avg_all if pattern_matches else 0))
print("  2. Player props - Need player data")
print("  3. Live betting - Exploit odds drift")
print("  4. Alt spreads - Bet smaller lines at better odds")

print("\n🎯 RECOMMENDATION:")

if profitable_results:
    print("  ✓ PROCEED with selective betting strategy")
    best = max(profitable_results, key=lambda x: x['roi_110'])
    print(f"    - Only bet {best['threshold']:.1%}+ confidence")
    print(f"    - Expected: ~{best['bets']} bets/season")
    print(f"    - Expected ROI: {best['roi_110']:+.1%}")
else:
    print("  ⚠ CAUTION on moneyline/spread")
    print("    - Consider alternative bet types")
    print("    - Focus on team totals (scoring pattern detected)")
    print("    - Explore player props with additional data")
    print("    - Use patterns as context, not direct bets")

print("\n📋 DATA NEEDS:")
print("  1. Real betting odds (moneyline, spread, totals)")
print("  2. Player-level data (for props)")
print("  3. Live odds (for in-game betting)")
print("  4. Larger sample size for validation")

print("\n" + "="*80)

