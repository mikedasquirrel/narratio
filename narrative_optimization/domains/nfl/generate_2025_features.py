"""
Generate Complete 29 Features for 2025 Season
Using all 32 team rosters (QBs, Coaches, O-line, Stars)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("GENERATING COMPLETE 2025 FEATURES")
print("="*80)

# Load complete rosters
with open('NFL_2025_COMPLETE_ROSTERS.json') as f:
    rosters_2025 = json.load(f)

print(f"✓ {len(rosters_2025['teams'])} team rosters loaded")

# Load 2025 games
with open('nfl_2025_enriched.json') as f:
    games_2025 = json.load(f)

print(f"✓ {len(games_2025)} games to process")

# Load historical stats from model
import pickle
model_path = Path(__file__).parent.parent.parent / 'nfl_production_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

qb_stats_historical = model_data['qb_stats']
coach_stats_historical = model_data['coach_stats']

print(f"✓ Historical stats loaded ({len(qb_stats_historical)} QBs, {len(coach_stats_historical)} coaches)")

# ============================================================================
# GENERATE ALL 29 FEATURES
# ============================================================================

print("\n[GENERATING] All 29 features for each game...")

X_2025_complete = []
y_2025_filtered = []
games_2025_filtered = []

for i, game in enumerate(games_2025):
    if i % 30 == 0:
        print(f"  Processing game {i+1}/{len(games_2025)}...")
    
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Handle team abbreviation variations
    team_map = {'WSH': 'WAS', 'LAR': 'LA'}  # ESPN variations
    home_team = team_map.get(home_team, home_team)
    away_team = team_map.get(away_team, away_team)
    
    if home_team not in rosters_2025['teams']:
        print(f"  ⚠ Missing roster for {home_team}")
        continue
    if away_team not in rosters_2025['teams']:
        print(f"  ⚠ Missing roster for {away_team}")
        continue
    
    home_roster = rosters_2025['teams'][home_team]
    away_roster = rosters_2025['teams'][away_team]
    
    # ===== QB FEATURES (1-6) =====
    home_qb = home_roster['qb']
    away_qb = away_roster['qb']
    
    home_qb_stats = qb_stats_historical.get(home_qb, {'games': 0, 'wins': 0})
    away_qb_stats = qb_stats_historical.get(away_qb, {'games': 0, 'wins': 0})
    
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
    
    qb_features = [
        home_qb_pres,  # 1
        away_qb_pres,  # 2
        home_qb_pres - away_qb_pres,  # 3
        max(home_qb_pres, away_qb_pres),  # 4
        min(home_qb_pres, away_qb_pres),  # 5
        home_qb_pres * away_qb_pres  # 6
    ]
    
    # ===== COACH FEATURES (7-15) =====
    home_coach = home_roster['head_coach']
    away_coach = away_roster['head_coach']
    
    home_coach_stats = coach_stats_historical.get(home_coach, {'games': 0, 'wins': 0})
    away_coach_stats = coach_stats_historical.get(away_coach, {'games': 0, 'wins': 0})
    
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
    
    coach_features = [
        home_coach_pres,  # 7
        away_coach_pres,  # 8
        home_coach_pres - away_coach_pres,  # 9
        max(home_coach_pres, away_coach_pres),  # 10
        home_coach_pres * away_coach_pres,  # 11
        1.0 if home_coach_pres > 0.70 else 0.0,  # 12
        1.0 if away_coach_pres > 0.70 else 0.0,  # 13
        home_coach_stats['games'] / 200.0,  # 14
        away_coach_stats['games'] / 200.0  # 15
    ]
    
    # ===== O-LINE FEATURES (16-19) =====
    home_oline = home_roster['oline']
    away_oline = away_roster['oline']
    
    oline_features = [
        len(home_oline),  # 16
        len(away_oline),  # 17
        len(home_oline) - len(away_oline),  # 18
        len(home_oline) * len(away_oline) / 25.0  # 19 (normalized)
    ]
    
    # ===== STAR FEATURES (20-23) =====
    home_stars = home_roster['star_players']
    away_stars = away_roster['star_players']
    
    star_features = [
        len(home_stars),  # 20
        len(away_stars),  # 21
        len(home_stars) - len(away_stars),  # 22
        len(home_stars) * len(away_stars) / 9.0  # 23 (normalized)
    ]
    
    # ===== INTERACTION FEATURES (24-29) =====
    interaction_features = [
        home_qb_pres * home_coach_pres,  # 24
        1.0 if (home_qb_pres > 0.7 and home_coach_pres > 0.7) else 0.0,  # 25
        abs(home_coach_stats['games'] - away_coach_stats['games']) / 200.0,  # 26
        home_qb_pres + home_coach_pres,  # 27
        abs(home_qb_pres - home_coach_pres),  # 28
        (home_qb_pres + home_coach_pres) / 2  # 29
    ]
    
    # Combine all 29 features
    all_features = qb_features + coach_features + oline_features + star_features + interaction_features
    X_2025_complete.append(all_features)
    y_2025_filtered.append(int(game['home_won']))
    games_2025_filtered.append(game)

X_2025 = np.array(X_2025_complete)
y_2025 = np.array(y_2025_filtered)

print(f"\n✓ Generated {X_2025.shape[1]} features for {X_2025.shape[0]} games")
print(f"  QB: 6, Coach: 9, O-line: 4, Stars: 4, Interactions: 6")

# Save features
np.savez_compressed('nfl_2025_complete_features.npz',
                   features=X_2025,
                   game_ids=[g['game_id'] for g in games_2025])

print(f"✓ Saved features")

# NOW TEST THE MODEL
print("\n" + "="*80)
print("TESTING MODEL WITH COMPLETE 2025 DATA")
print("="*80)

scaler = model_data['scaler']
model = model_data['model']

X_2025_sc = scaler.transform(X_2025)

y_pred = model.predict(X_2025_sc)
y_proba = model.predict_proba(X_2025_sc)[:, 1]

acc = (y_pred == y_2025).mean()

wins = (y_pred == y_2025).sum()
losses = len(y_2025) - wins
profit = wins * 100 - losses * 110
wagered = len(y_2025) * 110
roi = (profit / wagered) * 100

print(f"\n✓ COMPLETE DATA RESULTS:")
print(f"  Accuracy: {acc:.1%}")
print(f"  ROI: {roi:+.1f}%")
print(f"  Profit: ${profit:,.0f} on ${wagered:,.0f}")

print(f"\nCOMPARISON:")
print(f"  2024 (real data): 71.9% accuracy, +51.9% ROI")
print(f"  2025 (complete data): {acc:.1%} accuracy, {roi:+.1f}% ROI")
print(f"  Change: {(acc - 0.719)*100:+.1f}pp, {roi - 51.9:+.1f}pp")

print(f"\n{'✓ MODEL WORKS' if roi > 10 else '⚠ MODEL WEAK' if roi > 0 else '✗ MODEL FAILS'}")
print("="*80)

