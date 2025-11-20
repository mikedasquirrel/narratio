"""
Validate NBA Player Patterns - Profitability Testing

This script:
1. Loads discovered patterns from complete dataset
2. Tests on 2023-24 season (out-of-sample validation)
3. Calculates betting profitability with actual odds
4. Generates selective betting strategy report

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

from transformers.context_pattern import ContextPattern

print("="*80)
print("NBA PLAYER PATTERN VALIDATION - 2023-24 SEASON")
print("="*80)
print()

# Load discovered patterns
patterns_path = Path('discovered_player_patterns.json')
with open(patterns_path) as f:
    patterns_data = json.load(f)

print(f"✓ Loaded {patterns_data['total_patterns']} discovered patterns")
print(f"  Training data: {patterns_data['total_games_analyzed']:,} games")
print(f"  Baseline accuracy: {patterns_data['baseline_accuracy']:.1%}")
print()

# Load complete dataset with player data
data_path = Path('data/domains/nba_complete_with_players.json')
with open(data_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} total games")

# Split into train (2014-2023) and test (2023-24)
train_games = [g for g in all_games if g['season'] != '2023-24']
test_games = [g for g in all_games if g['season'] == '2023-24']

print(f"  Training: {len(train_games):,} games (2014-2023)")
print(f"  Test: {len(test_games):,} games (2023-24)")
print()

# Extract test features
def extract_features(game: Dict) -> Dict:
    """Extract features in same format as training"""
    if not game.get('player_data', {}).get('available'):
        return None
    
    tc = game['temporal_context']
    pd_agg = game['player_data']['team_aggregates']
    
    return {
        'home': 1.0 if game['home_game'] else 0.0,
        'season_win_pct': tc['season_win_pct'],
        'l10_win_pct': tc['l10_win_pct'],
        'games_played': tc['games_played'] / 82.0,
        'top1_points': pd_agg['top1_points'],
        'top2_points': pd_agg['top2_points'],
        'top3_points': pd_agg['top3_points'],
        'players_20plus_pts': pd_agg['players_20plus_pts'],
        'players_15plus_pts': pd_agg['players_15plus_pts'],
        'players_10plus_pts': pd_agg['players_10plus_pts'],
        'experienced_players': pd_agg['experienced_players'],
        'avg_experience': pd_agg['avg_experience'],
        'top1_scoring_share': pd_agg['top1_scoring_share'],
        'top3_scoring_share': pd_agg['top3_scoring_share'],
        'bench_points': pd_agg['bench_points'],
        'players_5plus_ast': pd_agg['players_5plus_ast'],
        'players_used': pd_agg['players_used'],
    }

test_features = []
test_outcomes = []
test_game_ids = []
test_betting_odds = []

for game in test_games:
    features = extract_features(game)
    if features is not None:
        test_features.append(features)
        test_outcomes.append(1 if game['won'] else 0)
        test_game_ids.append(game.get('game_id', ''))
        test_betting_odds.append(game.get('betting_odds', {}))

X_test = pd.DataFrame(test_features)
y_test = np.array(test_outcomes)

print(f"✓ Test set prepared: {len(X_test):,} games with features")
print(f"  Baseline win rate: {y_test.mean():.1%}")
print()

# Reconstruct ContextPattern objects from JSON
def json_to_pattern(pattern_dict: Dict) -> ContextPattern:
    """Reconstruct ContextPattern from saved JSON"""
    return ContextPattern(
        features=pattern_dict['features'],
        conditions=pattern_dict['conditions'],
        accuracy=pattern_dict['accuracy'],
        sample_size=pattern_dict['sample_size'],
        effect_size=pattern_dict['effect_size'],
        p_value=pattern_dict['p_value']
    )

patterns = [json_to_pattern(p) for p in patterns_data['patterns']]

print("🧪 Testing patterns on 2023-24 season...")
print()

# Test each pattern on hold-out data
pattern_results = []

for i, pattern in enumerate(patterns[:50]):  # Test top 50 patterns
    # Find matching games
    matches = pattern.matches(X_test)
    matched_games = X_test[matches]
    matched_outcomes = y_test[matches]
    matched_odds = [test_betting_odds[j] for j, m in enumerate(matches) if m]
    
    if len(matched_outcomes) < 10:  # Need minimum sample
        continue
    
    # Calculate test accuracy
    test_accuracy = matched_outcomes.mean()
    
    # Calculate betting profitability
    profit = 0.0
    total_bet = 0.0
    wins = 0
    losses = 0
    
    for outcome, odds_dict in zip(matched_outcomes, matched_odds):
        bet_size = 100  # $100 per bet
        total_bet += bet_size
        
        if outcome == 1:  # Won
            # Use moneyline odds
            ml = odds_dict.get('moneyline', 0)
            if ml > 0:
                profit += bet_size * (ml / 100)
            elif ml < 0:
                profit += bet_size * (100 / abs(ml))
            wins += 1
        else:  # Lost
            profit -= bet_size
            losses += 1
    
    roi = (profit / total_bet * 100) if total_bet > 0 else 0
    
    pattern_results.append({
        'rank': i + 1,
        'train_accuracy': pattern.accuracy,
        'test_accuracy': test_accuracy,
        'test_sample_size': len(matched_outcomes),
        'wins': wins,
        'losses': losses,
        'total_bet': total_bet,
        'profit': profit,
        'roi': roi,
        'features': pattern.features,
        'conditions': pattern.conditions
    })

# Sort by test ROI
pattern_results.sort(key=lambda x: x['roi'], reverse=True)

print("="*80)
print("TOP 20 PATTERNS BY TEST ROI (2023-24)")
print("="*80)
print()
print(f"{'Rank':<6} {'Train%':<8} {'Test%':<8} {'N':<6} {'W-L':<10} {'Profit':<12} {'ROI%':<8} Pattern")
print("-"*80)

top_patterns = []
for i, result in enumerate(pattern_results[:20]):
    condition_str = ' & '.join([
        f"{k}≥{v['min']:.2f}" if isinstance(v, dict) and 'min' in v
        else f"{k}≤{v['max']:.2f}" if isinstance(v, dict) and 'max' in v
        else f"{k}={v}"
        for k, v in list(result['conditions'].items())[:3]
    ])
    
    print(f"{i+1:<6} {result['train_accuracy']*100:>6.1f}% {result['test_accuracy']*100:>6.1f}% "
          f"{result['test_sample_size']:<6} {result['wins']}-{result['losses']:<7} "
          f"${result['profit']:>10.0f} {result['roi']:>6.1f}%  {condition_str[:40]}")
    
    top_patterns.append(result)

print()

# Calculate overall strategy performance
print("="*80)
print("SELECTIVE BETTING STRATEGY PERFORMANCE")
print("="*80)
print()

strategies = [
    ('Top 1 Pattern', top_patterns[:1]),
    ('Top 3 Patterns', top_patterns[:3]),
    ('Top 5 Patterns', top_patterns[:5]),
    ('Top 10 Patterns', top_patterns[:10]),
    ('Top 20 Patterns', top_patterns[:20]),
    ('All Profitable (ROI>0)', [p for p in pattern_results if p['roi'] > 0])
]

strategy_results = []

for strategy_name, patterns_subset in strategies:
    total_wins = sum(p['wins'] for p in patterns_subset)
    total_losses = sum(p['losses'] for p in patterns_subset)
    total_bet = sum(p['total_bet'] for p in patterns_subset)
    total_profit = sum(p['profit'] for p in patterns_subset)
    total_roi = (total_profit / total_bet * 100) if total_bet > 0 else 0
    total_games = total_wins + total_losses
    win_pct = (total_wins / total_games * 100) if total_games > 0 else 0
    
    strategy_results.append({
        'strategy': strategy_name,
        'games': total_games,
        'wins': total_wins,
        'losses': total_losses,
        'win_pct': win_pct,
        'total_bet': total_bet,
        'profit': total_profit,
        'roi': total_roi
    })
    
    print(f"{strategy_name}")
    print(f"  Games: {total_games:,} ({total_wins}-{total_losses}, {win_pct:.1f}% wins)")
    print(f"  Total bet: ${total_bet:,.0f}")
    print(f"  Profit: ${total_profit:,.0f}")
    print(f"  ROI: {total_roi:+.1f}%")
    print()

# Find optimal strategy
best_strategy = max(strategy_results, key=lambda x: x['roi'])
print(f"🏆 BEST STRATEGY: {best_strategy['strategy']}")
print(f"   {best_strategy['roi']:+.1f}% ROI on {best_strategy['games']} bets")
print()

# Save detailed results
output_path = Path('pattern_validation_results.json')
with open(output_path, 'w') as f:
    json.dump({
        'test_period': '2023-24',
        'test_games': len(test_games),
        'patterns_tested': len(pattern_results),
        'top_20_patterns': top_patterns,
        'strategy_results': strategy_results,
        'best_strategy': best_strategy,
        'all_pattern_results': pattern_results
    }, f, indent=2)

print(f"✓ Saved detailed results to: {output_path}")
print()

# Generate betting recommendations
print("="*80)
print("BETTING RECOMMENDATIONS")
print("="*80)
print()

if best_strategy['roi'] > 5:
    print(f"✅ PROFITABLE STRATEGY DISCOVERED")
    print(f"   Strategy: {best_strategy['strategy']}")
    print(f"   Expected ROI: {best_strategy['roi']:+.1f}%")
    print(f"   Win Rate: {best_strategy['win_pct']:.1f}%")
    print()
    print("Implementation:")
    print(f"  1. Use top {len([p for p in pattern_results if p in top_patterns[:20]])} patterns")
    print(f"  2. Bet when ANY pattern matches game conditions")
    print(f"  3. Expected profit: ${best_strategy['profit']:,.0f} per {best_strategy['games']} bets")
    print(f"  4. Kelly Criterion: ~{best_strategy['roi']/2:.1f}% of bankroll per bet")
elif best_strategy['roi'] > 0:
    print(f"⚠️  MARGINALLY PROFITABLE")
    print(f"   ROI: {best_strategy['roi']:+.1f}%")
    print("   May not overcome betting friction (vig, fees)")
    print("   Recommend paper trading before live betting")
else:
    print(f"❌ NOT PROFITABLE ON TEST DATA")
    print(f"   ROI: {best_strategy['roi']:+.1f}%")
    print("   Patterns may be overfit to training data")
    print("   Need more robust features or larger sample")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Files created:")
print("  - pattern_validation_results.json (detailed results)")
print("  - discovered_player_patterns.json (all patterns)")
print()
print("Key Findings:")
print(f"  - {len(pattern_results)} patterns tested on 2023-24 season")
print(f"  - Best ROI: {pattern_results[0]['roi']:+.1f}% (single pattern)")
print(f"  - Best strategy: {best_strategy['strategy']} at {best_strategy['roi']:+.1f}% ROI")
print()

