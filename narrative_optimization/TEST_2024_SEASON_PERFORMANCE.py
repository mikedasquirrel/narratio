"""
Test Model on 2024-25 Season
See how our validated model performs on current season

MODEL: Trained on 2014-2019 (2,107 games)
TEST: 2024 season (285 games) - completely out-of-sample
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

print("="*80)
print("2024-25 SEASON PERFORMANCE TEST")
print("Out-of-Sample Validation on Current Season")
print("="*80)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n[1/4] Loading production model...")

model_path = Path(__file__).parent / 'nfl_production_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
qb_stats_train = model_data['qb_stats']
coach_stats_train = model_data['coach_stats']

print(f"✓ Model loaded (trained on 2014-2019)")

# ============================================================================
# LOAD 2024 SEASON DATA
# ============================================================================

print("\n[2/4] Loading 2024 season games...")

games_path = Path(__file__).parent.parent / 'data' / 'domains' / 'nfl_games_with_odds.json'
with open(games_path) as f:
    all_games = json.load(f)

season_2024 = [g for g in all_games if g.get('season') == 2024]

print(f"✓ {len(season_2024)} games from 2024 season")
print(f"  Weeks 1-{max(g.get('week', 0) for g in season_2024)}")
print(f"  Date range: {min(g.get('gameday','') for g in season_2024)} to {max(g.get('gameday','') for g in season_2024)}")

# ============================================================================
# LOAD 2024 FEATURES
# ============================================================================

print("\n[3/4] Loading features for 2024 games...")

# Load all features
nom_path = Path(__file__).parent / 'domains' / 'nfl' / 'nfl_specific_features.npz'
nom_data = np.load(nom_path, allow_pickle=True)
X_all = nom_data['features']
game_ids_all = nom_data['game_ids']

# Find 2024 games in feature matrix
season_2024_ids = set(g['game_id'] for g in season_2024)
season_2024_indices = [i for i, gid in enumerate(game_ids_all) if gid in season_2024_ids]

X_2024 = X_all[season_2024_indices]
y_2024 = np.array([int(g['home_won']) for g in season_2024])

print(f"✓ {len(X_2024)} games matched with features")

# ============================================================================
# PREDICT 2024 SEASON
# ============================================================================

print("\n[4/4] Generating predictions for 2024 season...")

X_2024_sc = scaler.transform(X_2024)
y_pred_2024 = model.predict(X_2024_sc)
y_proba_2024 = model.predict_proba(X_2024_sc)[:, 1]

overall_acc = (y_pred_2024 == y_2024).mean()

print(f"✓ Predictions generated")
print(f"  Overall accuracy: {overall_acc:.1%}")

# ============================================================================
# TEST VALIDATED PATTERNS ON 2024
# ============================================================================

print("\n" + "="*80)
print("TESTING VALIDATED PATTERNS ON 2024 SEASON")
print("="*80)

confidence = np.abs(y_proba_2024 - 0.5) * 2
weeks_2024 = np.array([g.get('week', 0) for g in season_2024])
spreads_2024 = np.array([g['betting_odds'].get('spread', 0) for g in season_2024])

# QB differential
qb_diff_2024 = X_2024[:, 2]  # QB prestige diff

def test_pattern_2024(mask, pattern_name):
    """Test pattern on 2024 data"""
    if mask.sum() < 5:
        return None
    
    win_rate = (y_pred_2024[mask] == y_2024[mask]).mean()
    n = mask.sum()
    
    wins = int(win_rate * n)
    losses = n - wins
    profit = wins * 100 - losses * 110
    wagered = n * 110
    roi = (profit / wagered) * 100
    
    return {
        'pattern': pattern_name,
        'games': n,
        'win_rate': win_rate,
        'roi': roi,
        'profit': profit,
        'profitable': win_rate > 0.5238
    }

print(f"\n{'Pattern':<45} {'Games':<7} {'Win%':<8} {'ROI':<10} {'2024 vs Train'}")
print("-" * 85)

results_2024 = []

# Top validated patterns
patterns_to_test = [
    (qb_diff_2024 > 0.2, "QB Edge >0.2", "80.0% → "),
    (confidence >= 0.5, "Confidence ≥0.5", "74.9% → "),
    ((weeks_2024 >= 13) & (weeks_2024 <= 14), "Weeks 13-14", "75.3% → "),
    (spreads_2024 <= -7, "Big Favorites -7", "74.7% → "),
    (confidence >= 0.6, "Confidence ≥0.6", "76.1% → "),
    ((weeks_2024 >= 14) & (weeks_2024 <= 18), "Late Season", "64.2% → "),
]

for mask, name, train_perf in patterns_to_test:
    r = test_pattern_2024(mask, name)
    if r:
        results_2024.append(r)
        status = '✓' if r['profitable'] else '✗'
        comparison = f"{train_perf}{r['win_rate']:.1%}"
        print(f"{r['pattern']:<45} {r['games']:<7} {r['win_rate']:>6.1%} {r['roi']:>+8.1f}% {comparison} {status}")

# ============================================================================
# WEEK-BY-WEEK PERFORMANCE
# ============================================================================

print("\n" + "="*80)
print("WEEK-BY-WEEK 2024 SEASON PERFORMANCE")
print("="*80)

print(f"\n{'Week':<8} {'Games':<8} {'Correct':<8} {'Accuracy':<10} {'ROI':<10}")
print("-" * 50)

weekly_results = []

for week in sorted(set(weeks_2024)):
    if week == 0:
        continue
    
    mask = weeks_2024 == week
    if mask.sum() < 5:
        continue
    
    correct = (y_pred_2024[mask] == y_2024[mask]).sum()
    total = mask.sum()
    acc = correct / total
    
    roi = ((correct * 100) - ((total - correct) * 110)) / (total * 110) * 100
    
    weekly_results.append({
        'week': week,
        'games': total,
        'accuracy': acc,
        'roi': roi
    })
    
    print(f"{week:<8} {total:<8} {correct:<8} {acc:>8.1%} {roi:>+8.1f}%")

# ============================================================================
# FINANCIAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("2024 SEASON FINANCIAL SUMMARY")
print("="*80)

# Calculate if we had bet on high confidence games
high_conf_mask = confidence >= 0.5
if high_conf_mask.sum() > 0:
    high_conf_correct = (y_pred_2024[high_conf_mask] == y_2024[high_conf_mask]).sum()
    high_conf_total = high_conf_mask.sum()
    high_conf_acc = high_conf_correct / high_conf_total
    
    profit_hc = high_conf_correct * 100 - (high_conf_total - high_conf_correct) * 110
    wagered_hc = high_conf_total * 110
    roi_hc = (profit_hc / wagered_hc) * 100
    
    print(f"\nHigh Confidence Strategy (≥0.5):")
    print(f"  Bets placed: {high_conf_total}")
    print(f"  Won: {high_conf_correct} ({high_conf_acc:.1%})")
    print(f"  Wagered: ${wagered_hc:,.0f}")
    print(f"  Profit: ${profit_hc:,.0f}")
    print(f"  ROI: {roi_hc:+.1f}%")
    print(f"  vs Training: 74.9% accuracy, +43.0% ROI")
    print(f"  Performance: {'✓ Met expectations' if high_conf_acc >= 0.70 else '✗ Below expectations' if high_conf_acc < 0.60 else '⚠ Close to expectations'}")

# QB Edge pattern
qb_edge_mask = qb_diff_2024 > 0.2
if qb_edge_mask.sum() > 0:
    qb_edge_correct = (y_pred_2024[qb_edge_mask] == y_2024[qb_edge_mask]).sum()
    qb_edge_total = qb_edge_mask.sum()
    qb_edge_acc = qb_edge_correct / qb_edge_total
    
    profit_qb = qb_edge_correct * 100 - (qb_edge_total - qb_edge_correct) * 110
    wagered_qb = qb_edge_total * 110
    roi_qb = (profit_qb / wagered_qb) * 100
    
    print(f"\nQB Prestige Edge >0.2:")
    print(f"  Bets placed: {qb_edge_total}")
    print(f"  Won: {qb_edge_correct} ({qb_edge_acc:.1%})")
    print(f"  Wagered: ${wagered_qb:,.0f}")
    print(f"  Profit: ${profit_qb:,.0f}")
    print(f"  ROI: {roi_qb:+.1f}%")
    print(f"  vs Training: 80.0% accuracy, +52.7% ROI")
    print(f"  Performance: {'✓ Met expectations' if qb_edge_acc >= 0.75 else '✗ Below expectations' if qb_edge_acc < 0.65 else '⚠ Close to expectations'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("2024 SEASON SUMMARY")
print("="*80)

profitable_2024 = [r for r in results_2024 if r['profitable']]

print(f"\nOVERALL PERFORMANCE:")
print(f"  Total games: {len(season_2024)}")
print(f"  Overall accuracy: {overall_acc:.1%}")
print(f"  Patterns tested: {len(results_2024)}")
print(f"  Patterns profitable: {len(profitable_2024)}/{len(results_2024)}")

if profitable_2024:
    best_2024 = max(profitable_2024, key=lambda x: x['roi'])
    print(f"\n  Best 2024 pattern: {best_2024['pattern']}")
    print(f"    Accuracy: {best_2024['win_rate']:.1%}")
    print(f"    ROI: {best_2024['roi']:+.1f}%")
    print(f"    Profit: ${best_2024['profit']:,.0f}")

# Overall betting if we followed system
total_bets_system = high_conf_total if high_conf_mask.sum() > 0 else 0
total_profit_system = profit_hc if high_conf_mask.sum() > 0 else 0

print(f"\nIF WE HAD USED THIS SYSTEM IN 2024:")
print(f"  Bets: {total_bets_system}")
print(f"  Profit: ${total_profit_system:,.0f}")
print(f"  ROI: {roi_hc:+.1f}%" if high_conf_mask.sum() > 0 else "  ROI: N/A")

print(f"\n{'✓ SYSTEM WORKS - profitable on 2024 season' if total_profit_system > 0 else '✗ SYSTEM FAILED - lost money on 2024 season'}")

print("="*80)

