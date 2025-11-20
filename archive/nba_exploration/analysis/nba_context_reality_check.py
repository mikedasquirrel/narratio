"""
NBA Context Pattern Reality Check - Would This Actually Profit?

Critical question: Do discovered patterns beat the BETTING MARKET or just find favorites?

Reality: NBA data lacks real betting odds, so we must test both scenarios:
1. BEST CASE: Patterns identify market inefficiencies (profitable)
2. WORST CASE: Patterns just identify favorites already priced in (unprofitable)

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
print("NBA CONTEXT PATTERNS - REAL-WORLD PROFITABILITY CHECK")
print("="*80)

# Load NBA data
data_path = project_root.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'

with open(data_path) as f:
    games = json.load(f)

print(f"\nTotal games: {len(games):,}")

# Extract features for 2023-24 season (most recent full season)
features_list = []
outcomes = []
game_metadata = []

for game in games:
    tc = game.get('temporal_context', {})
    if tc.get('games_played', 0) == 0:
        continue
    
    # Focus on recent season for current relevance
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
    game_metadata.append({
        'season': game.get('season', ''),
        'date': game.get('date', ''),
        'team': game.get('team_name', ''),
        'home': game.get('home_game', False),
        'win_pct': tc.get('season_win_pct', 0.5)
    })

X = pd.DataFrame(features_list)
y = np.array(outcomes)
metadata = pd.DataFrame(game_metadata)

print(f"Recent seasons (2022-24): {len(X):,} games")
print(f"Overall win rate: {y.mean():.1%}")

# Temporal split - train on 2022-23, test on 2023-24
train_mask = metadata['season'] == '2022-23'
test_mask = metadata['season'] == '2023-24'

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
meta_train, meta_test = metadata[train_mask], metadata[test_mask]

print(f"\nTrain (2022-23): {len(X_train):,} games")
print(f"Test (2023-24): {len(X_test):,} games")

# Discover patterns
print("\n" + "="*80)
print("PATTERN DISCOVERY")
print("="*80)

transformer = ContextPatternTransformer(
    min_accuracy=0.60,
    min_samples=50,
    max_patterns=10
)

transformer.fit(X_train, y_train)

print(f"\nPatterns discovered: {len(transformer.patterns_)}")

if not transformer.patterns_:
    print("\n⚠ No patterns discovered - cannot evaluate profitability")
    sys.exit(0)

# Show patterns
for i, pattern in enumerate(transformer.patterns_[:5], 1):
    print(f"\nPattern {i}:")
    print(f"  Features: {pattern.features}")
    conditions_str = []
    for feat, cond in pattern.conditions.items():
        if isinstance(cond, dict):
            if 'min' in cond:
                conditions_str.append(f"{feat} ≥ {cond['min']:.3f}")
            if 'max' in cond:
                conditions_str.append(f"{feat} ≤ {cond['max']:.3f}")
            if 'eq' in cond:
                conditions_str.append(f"{feat} = {cond['eq']}")
    print(f"  Conditions: {' & '.join(conditions_str)}")
    print(f"  Train accuracy: {pattern.accuracy:.1%}")
    print(f"  Train sample size: {pattern.sample_size}")

# Test on 2023-24
print("\n" + "="*80)
print("TEST ON 2023-24 SEASON (OUT-OF-SAMPLE)")
print("="*80)

recommendations = transformer.get_betting_recommendations(X_test)
bet_recommendations = [r for r in recommendations if r['recommendation'] == 'BET']

print(f"\nTest games: {len(X_test):,}")
print(f"BET recommendations: {len(bet_recommendations)}")

if not bet_recommendations:
    print("\n⚠ No BET recommendations - patterns may not generalize to 2023-24")
    print("This suggests patterns are overfit to 2022-23")
    sys.exit(0)

# Evaluate recommendations
bet_indices = [r['sample_idx'] for r in bet_recommendations]
bet_outcomes = y_test[bet_indices]
bet_metadata = meta_test.iloc[bet_indices]

accuracy = bet_outcomes.mean()
print(f"\nRecommendation accuracy: {accuracy:.1%}")

# === SCENARIO 1: ASSUME -110 ODDS (STANDARD) ===
print("\n" + "="*80)
print("SCENARIO 1: STANDARD -110 ODDS (Best Case)")
print("="*80)
print("Assumption: We can bet at standard -110 odds on all games")

wins = bet_outcomes.sum()
losses = len(bet_outcomes) - wins
breakeven = 0.5238  # Need 52.38% to break even at -110

profit_110 = wins * 91 - losses * 100  # Win $91 on $100 bet (risk $110 to win $100)
wagered = len(bet_outcomes) * 110
roi_110 = (profit_110 / wagered) * 100

print(f"\nBets: {len(bet_outcomes)}")
print(f"Wins: {wins} ({accuracy:.1%})")
print(f"Breakeven needed: {breakeven:.1%}")
print(f"Total wagered: ${wagered:,}")
print(f"Net profit: ${profit_110:+,.0f}")
print(f"ROI: {roi_110:+.1%}")

if accuracy > breakeven:
    print(f"\n✓ PROFITABLE at -110 odds (+{roi_110:.1%} ROI)")
    print(f"  Edge over breakeven: {(accuracy - breakeven)*100:.1%}")
else:
    print(f"\n✗ NOT PROFITABLE at -110 odds ({roi_110:+.1%} ROI)")
    print(f"  Short of breakeven by: {(breakeven - accuracy)*100:.1%}")

# === SCENARIO 2: PATTERNS IDENTIFY FAVORITES ===
print("\n" + "="*80)
print("SCENARIO 2: PATTERNS FIND FAVORITES (Worst Case)")
print("="*80)
print("Question: Are we just identifying favorites that markets already priced?")

# Check if patterns correlate with quality
bet_win_pcts = bet_metadata['win_pct'].values
avg_win_pct = bet_win_pcts.mean()

print(f"\nAverage win% of recommended teams: {avg_win_pct:.1%}")
print(f"Overall average win%: {meta_test['win_pct'].mean():.1%}")

if avg_win_pct > 0.55:
    print("\n⚠ WARNING: Patterns heavily favor good teams")
    print("Sportsbooks likely price these as favorites with unfavorable odds")
    
    # Estimate typical odds for teams at this win%
    # Rough formula: -110 * (win% / (1-win%))
    implied_odds = -110 * (avg_win_pct / (1 - avg_win_pct))
    
    print(f"\nEstimated typical odds for {avg_win_pct:.1%} teams: {implied_odds:.0f}")
    
    # Calculate ROI with these unfavorable odds
    if implied_odds < 0:  # Favorite
        payout_per_100 = 100 / abs(implied_odds) * 100
        profit_realistic = wins * payout_per_100 - losses * 100
        roi_realistic = (profit_realistic / (len(bet_outcomes) * 100)) * 100
        
        print(f"ROI at {implied_odds:.0f} odds: {roi_realistic:+.1%}")
        
        if roi_realistic < 0:
            print(f"\n✗ LIKELY UNPROFITABLE")
            print(f"  Patterns identify favorites, but odds make it unprofitable")
            print(f"  Need {accuracy:.1%} accuracy but only breaking even requires {(abs(implied_odds)/(abs(implied_odds)+100)):.1%}")
else:
    print(f"\n✓ Patterns don't just favor good teams")
    print("  This suggests potential market inefficiency")

# === SCENARIO 3: WHAT ODDS DO WE NEED? ===
print("\n" + "="*80)
print("SCENARIO 3: REQUIRED ODDS FOR PROFITABILITY")
print("="*80)

# At our accuracy, what's the worst odds we can accept?
# Break-even: accuracy = odds/(odds+100)
# Rearrange: odds = 100 * accuracy / (1 - accuracy)

max_favorable_odds = -100 * accuracy / (1 - accuracy)
print(f"\nOur accuracy: {accuracy:.1%}")
print(f"Maximum unfavorable odds we can beat: {max_favorable_odds:.0f}")
print(f"(We profit at any odds better than {max_favorable_odds:.0f})")

# Examples
test_odds = [-300, -200, -150, -110, +100]
print(f"\n{'Odds':<10} {'Breakeven':<12} {'Our Edge':<12} {'Profitable?'}")
print("-"*50)

for odds in test_odds:
    if odds < 0:
        breakeven_needed = abs(odds) / (abs(odds) + 100)
    else:
        breakeven_needed = 100 / (odds + 100)
    
    edge = accuracy - breakeven_needed
    profitable = "✓ YES" if edge > 0 else "✗ NO"
    
    print(f"{odds:<10} {breakeven_needed:.1%}       {edge:+.1%}       {profitable}")

# === FINAL VERDICT ===
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

print(f"\n📊 PATTERN PERFORMANCE:")
print(f"  Recommendations: {len(bet_recommendations)} games")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  Sample: 2023-24 season (out-of-sample)")

print(f"\n💰 PROFITABILITY:")

if accuracy > breakeven:
    print(f"  ✓ PROFITABLE at standard -110 odds (+{roi_110:.1%} ROI)")
    print(f"  ✓ Can beat odds up to {max_favorable_odds:.0f}")
else:
    print(f"  ✗ NOT PROFITABLE at standard -110 odds ({roi_110:.1%} ROI)")
    print(f"  Accuracy ({accuracy:.1%}) below breakeven ({breakeven:.1%})")

print(f"\n⚠️  CRITICAL LIMITATIONS:")
print(f"  1. NO REAL BETTING ODDS in data - cannot verify actual profitability")
print(f"  2. If patterns identify favorites, sportsbooks likely priced them in")
print(f"  3. Need {avg_win_pct:.1%} win% teams at odds better than {max_favorable_odds:.0f}")
print(f"  4. Market efficiency may eliminate edge")

print(f"\n🎯 WHAT THIS MEANS:")

if avg_win_pct > 0.55 and accuracy < 0.60:
    print(f"  LIKELY UNPROFITABLE - Patterns find good teams, but not enough edge")
    print(f"  Markets already price these teams as favorites")
elif accuracy > 0.65 and avg_win_pct < 0.55:
    print(f"  POTENTIALLY PROFITABLE - High accuracy on balanced teams")
    print(f"  May find market inefficiencies")
elif accuracy > 0.70:
    print(f"  LIKELY PROFITABLE - Accuracy high enough to beat most odds")
else:
    print(f"  UNCERTAIN - Need real odds data to evaluate")

print(f"\n📋 NEXT STEPS TO VALIDATE:")
print(f"  1. Collect real betting odds for 2023-24 season")
print(f"  2. Calculate actual ROI per game with real odds")
print(f"  3. Compare pattern recommendations vs closing lines")
print(f"  4. Test if patterns identify undervalued teams (not just favorites)")

print("\n" + "="*80)

