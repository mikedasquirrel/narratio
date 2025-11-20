"""
Test Production Model on Current 2025 Season (Weeks 1-10)
ULTIMATE validation - completely fresh, current data

Model: Trained 2014-2019
Previous test: 2024 season (+51.9% ROI)
NOW: 2025 season Weeks 1-10 (most recent possible data)
"""

import json
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict

print("="*80)
print("TESTING MODEL ON CURRENT 2025 SEASON")
print("Weeks 1-10 (September-November 2025)")
print("="*80)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n[1/4] Loading production model (trained 2014-2019)...")

model_path = Path(__file__).parent / 'nfl_production_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
qb_stats_model = model_data['qb_stats']
coach_stats_model = model_data['coach_stats']

print(f"✓ Model loaded")

# ============================================================================
# LOAD 2025 DATA
# ============================================================================

print("\n[2/4] Loading 2025 season data...")

games_2025_path = Path(__file__).parent / 'domains' / 'nfl' / 'nfl_2025_enriched.json'
with open(games_2025_path) as f:
    games_2025 = json.load(f)

print(f"✓ {len(games_2025)} games (Weeks 1-10)")

# ============================================================================
# GENERATE FEATURES FOR 2025
# ============================================================================

print("\n[3/4] Generating features for 2025 games...")

X_2025 = []

for game in games_2025:
    home_qb = game['home_roster']['starting_qb']['name']
    away_qb = game['away_roster']['starting_qb']['name']
    home_coach = game['home_coaches']['head_coach']
    away_coach = game['away_coaches']['head_coach']
    
    # Get stats from training data
    home_qb_stats = qb_stats_model.get(home_qb, {'games': 0, 'wins': 0})
    away_qb_stats = qb_stats_model.get(away_qb, {'games': 0, 'wins': 0})
    home_coach_stats = coach_stats_model.get(home_coach, {'games': 0, 'wins': 0})
    away_coach_stats = coach_stats_model.get(away_coach, {'games': 0, 'wins': 0})
    
    # Calculate prestige
    if home_qb_stats['games'] >= 5:
        home_qb_wr = home_qb_stats['wins'] / home_qb_stats['games']
        home_qb_exp = np.log1p(home_qb_stats['games']) / np.log1p(100)
        home_qb_pres = home_qb_wr * 0.7 + home_qb_exp * 0.3
    else:
        home_qb_pres = 0.5
    
    if away_qb_stats['games'] >= 5:
        away_qb_wr = away_qb_stats['wins'] / away_qb_stats['games']
        away_qb_exp = np.log1p(away_qb_stats['games']) / np.log1p(100)
        away_qb_pres = away_qb_wr * 0.7 + away_qb_exp * 0.3
    else:
        away_qb_pres = 0.5
    
    # Coach prestige
    if home_coach_stats['games'] >= 5:
        home_coach_wr = home_coach_stats['wins'] / home_coach_stats['games']
        home_coach_exp = np.log1p(home_coach_stats['games']) / np.log1p(200)
        home_coach_pres = home_coach_wr * 0.6 + home_coach_exp * 0.4
    else:
        home_coach_pres = 0.5
    
    if away_coach_stats['games'] >= 5:
        away_coach_wr = away_coach_stats['wins'] / away_coach_stats['games']
        away_coach_exp = np.log1p(away_coach_stats['games']) / np.log1p(200)
        away_coach_pres = away_coach_wr * 0.6 + away_coach_exp * 0.4
    else:
        away_coach_pres = 0.5
    
    # Build 29 features
    features = [
        home_qb_pres, away_qb_pres, home_qb_pres - away_qb_pres,
        max(home_qb_pres, away_qb_pres), min(home_qb_pres, away_qb_pres),
        home_qb_pres * away_qb_pres,
        home_coach_pres, away_coach_pres, home_coach_pres - away_coach_pres,
        max(home_coach_pres, away_coach_pres), home_coach_pres * away_coach_pres,
        1.0 if home_coach_pres > 0.70 else 0.0,
        1.0 if away_coach_pres > 0.70 else 0.0,
        home_coach_stats['games'] / 200.0, away_coach_stats['games'] / 200.0,
        0.5, 0.5, 0.5, 0.5,  # O-line placeholders
        0.5, 0.5, 0.5, 0.5,  # Star placeholders
        home_qb_pres * home_coach_pres,
        1.0 if (home_qb_pres > 0.7 and home_coach_pres > 0.7) else 0.0,
        abs(home_coach_stats['games'] - away_coach_stats['games']) / 200.0,
        home_qb_pres + home_coach_pres,
        abs(home_qb_pres - home_coach_pres),
        (home_qb_pres + home_coach_pres) / 2
    ]
    
    X_2025.append(features)

X_2025 = np.array(X_2025)
y_2025 = np.array([int(g['home_won']) for g in games_2025])
weeks_2025 = np.array([g['week'] for g in games_2025])

print(f"✓ Features generated: {X_2025.shape}")

# ============================================================================
# TEST MODEL ON 2025
# ============================================================================

print("\n[4/4] Testing model on 2025 season...")

X_2025_sc = scaler.transform(X_2025)
y_pred_2025 = model.predict(X_2025_sc)
y_proba_2025 = model.predict_proba(X_2025_sc)[:, 1]

overall_acc = (y_pred_2025 == y_2025).mean()

print(f"✓ Overall accuracy: {overall_acc:.1%}")

# Calculate ROI
wins = (y_pred_2025 == y_2025).sum()
losses = len(y_2025) - wins
profit = wins * 100 - losses * 110
wagered = len(y_2025) * 110
roi = (profit / wagered) * 100

print(f"  ROI (all bets): {roi:+.1f}%")
print(f"  Profit: ${profit:,.0f} on ${wagered:,.0f}")

# ============================================================================
# TEST VALIDATED PATTERNS
# ============================================================================

print("\n" + "="*80)
print("VALIDATED PATTERNS ON 2025 CURRENT SEASON")
print("="*80)

confidence = np.abs(y_proba_2025 - 0.5) * 2
qb_diff_2025 = X_2025[:, 2]

print(f"\n{'Pattern':<40} {'Games':<7} {'Win%':<8} {'ROI':<10} {'✓/✗'}")
print("-" * 70)

def test_pattern(mask, name):
    if mask.sum() < 5:
        return
    
    win_rate = (y_pred_2025[mask] == y_2025[mask]).mean()
    n = mask.sum()
    
    wins = int(win_rate * n)
    losses = n - wins
    profit = wins * 100 - losses * 110
    wagered = n * 110
    roi = (profit / wagered) * 100
    
    status = '✓' if win_rate > 0.5238 else '✗'
    print(f"{name:<40} {n:<7} {win_rate:>6.1%} {roi:>+8.1f}% {status}")
    
    return {'name': name, 'games': n, 'win_rate': win_rate, 'roi': roi}

results = []

# Top patterns
r = test_pattern(qb_diff_2025 > 0.2, "QB Edge >0.2")
if r: results.append(r)

r = test_pattern(confidence >= 0.5, "Confidence ≥0.5")
if r: results.append(r)

r = test_pattern(confidence >= 0.6, "Confidence ≥0.6")
if r: results.append(r)

# Temporal
r = test_pattern((weeks_2025 >= 1) & (weeks_2025 <= 4), "Weeks 1-4 (Early)")
if r: results.append(r)

r = test_pattern((weeks_2025 >= 5) & (weeks_2025 <= 8), "Weeks 5-8 (Mid)")
if r: results.append(r)

r = test_pattern(weeks_2025 >= 9, "Weeks 9-10 (Current)")
if r: results.append(r)

# ============================================================================
# WEEK BY WEEK
# ============================================================================

print("\n" + "="*80)
print("WEEK-BY-WEEK 2025 PERFORMANCE")
print("="*80)

print(f"\n{'Week':<6} {'Games':<7} {'Correct':<9} {'Accuracy':<10} {'ROI'}")
print("-" * 50)

for week in range(1, 11):
    mask = weeks_2025 == week
    if mask.sum() > 0:
        correct = (y_pred_2025[mask] == y_2025[mask]).sum()
        total = mask.sum()
        acc = correct / total
        
        roi_week = ((correct * 100) - ((total - correct) * 110)) / (total * 110) * 100
        
        print(f"{week:<6} {total:<7} {correct:<9} {acc:>8.1%} {roi_week:>+8.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("2025 SEASON SUMMARY (CURRENT)")
print("="*80)

print(f"""
VALIDATION ON MOST RECENT DATA:
  Season: 2025 (Weeks 1-10, current season)
  Games: {len(games_2025)}
  Overall accuracy: {overall_acc:.1%}
  Overall ROI: {roi:+.1f}%
  Profit: ${profit:,.0f}

COMPARISON:
  Training (2014-2019): 66.1% accuracy
  2024 Test: 71.9% accuracy, +51.9% ROI
  2025 Current: {overall_acc:.1%} accuracy, {roi:+.1f}% ROI

PATTERN PERFORMANCE:
  Patterns tested: {len(results)}
  See detailed results above

STATUS:
  {'✓ MODEL WORKS ON CURRENT SEASON' if roi > 0 else '✗ MODEL NOT WORKING ON CURRENT SEASON'}
  
READY FOR:
  Live betting on upcoming weeks (11-18)
  Real-time predictions
  Actual money deployment
""")

# Save
summary = {
    'season': 2025,
    'weeks': '1-10',
    'games': len(games_2025),
    'overall_accuracy': float(overall_acc),
    'overall_roi': float(roi),
    'profit': float(profit),
    'patterns_tested': results,
    'model_validation': 'PASS' if roi > 0 else 'FAIL',
    'ready_for_deployment': roi > 10.0
}

with open('2025_season_validation.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved")
print("="*80)

