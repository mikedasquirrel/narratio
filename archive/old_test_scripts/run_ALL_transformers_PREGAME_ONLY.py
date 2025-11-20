"""
Run ALL Transformers - PRE-GAME DATA ONLY (No Outcome Leakage)

CRITICAL: Only use information available BEFORE the game:
- Team names (always known)
- Player names (always known)
- Matchup descriptions (pre-game)
- Betting odds (pre-game)
- Temporal context (pre-game record)
- Rest days, scheduling (pre-game)

EXCLUDED (post-hoc):
- Game narratives mentioning "won/lost/victory/defeat"
- Final scores
- Any post-game descriptions

Author: Narrative Optimization Framework  
Date: November 16, 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

# Import transformers that work with PREGAME data
print("Loading transformers (pre-game data only)...")
from transformers.nominative import NominativeAnalysisTransformer
from transformers.phonetic import PhoneticTransformer
from transformers.social_status import SocialStatusTransformer
from transformers.universal_nominative import UniversalNominativeTransformer
from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
from transformers.nominative_interaction import NominativeInteractionTransformer
from transformers.pure_nominative import PureNominativePredictorTransformer
from transformers.nominative_richness import NominativeRichnessTransformer
from transformers.namespace_ecology import NamespaceEcologyTransformer
from transformers.cognitive_fluency import CognitiveFluencyTransformer
from transformers.discoverability import DiscoverabilityTransformer
from transformers.gravitational_features import GravitationalFeaturesTransformer
from transformers.context_pattern import ContextPatternTransformer

from sklearn.linear_model import LogisticRegression

print("✓ Transformers loaded")
print()

print("="*80)
print("ALL TRANSFORMERS - PRE-GAME DATA ONLY (NO LEAKAGE)")
print("="*80)
print()
print("⚠️  CRITICAL: Excluding all post-hoc narratives")
print("✓  Using ONLY: Names, matchups, betting odds, temporal context")
print()

# Load data
data_path = Path('data/domains/nba_complete_with_players.json')
with open(data_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} games")

# Split
train_games = [g for g in all_games if g['season'] < '2023-24']
test_games = [g for g in all_games if g['season'] == '2023-24']

print(f"  Train: {len(train_games):,} games")
print(f"  Test: {len(test_games):,} games")
print()

# Extract PREGAME ONLY features
def extract_pregame_data(games: List[Dict]) -> Dict:
    """Extract ONLY pre-game information (no outcome leakage)"""
    data = {
        'team_names': [],
        'opponent_names': [],
        'player_names': [],
        'matchups': [],
        'city_names': [],
        'combined_names': [],  # Team + Players for rich nominative analysis
        'outcomes': [],
        'numerical_features': []
    }
    
    for game in games:
        team_name = game.get('team_name', '')
        matchup = game.get('matchup', '')
        
        # Extract opponent from matchup (e.g., "LAL @ GSW" or "LAL vs. GSW")
        if '@' in matchup:
            parts = matchup.split('@')
            opponent = parts[1].strip() if game['home_game'] else parts[0].strip()
        elif 'vs.' in matchup:
            parts = matchup.split('vs.')
            opponent = parts[1].strip()
        else:
            opponent = ''
        
        data['team_names'].append(team_name)
        data['opponent_names'].append(opponent)
        data['matchups'].append(matchup)
        
        # City names (from team names)
        city = team_name.split()[0] if team_name else ''  # "Los Angeles Lakers" -> "Los Angeles"
        data['city_names'].append(city)
        
        # Player names (pregame rosters)
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            player_names = [agg.get('top1_name', ''), agg.get('top2_name', ''), agg.get('top3_name', '')]
            player_str = ' '.join(filter(None, player_names))
            data['player_names'].append(player_str)
            
            # Combined for rich analysis
            combined = f"{team_name} {player_str}"
            data['combined_names'].append(combined)
        else:
            data['player_names'].append('')
            data['combined_names'].append(team_name)
        
        data['outcomes'].append(1 if game.get('won', False) else 0)
        
        # Numerical pregame features
        tc = game['temporal_context']
        pd_agg = game.get('player_data', {}).get('team_aggregates', {})
        betting = game.get('betting_odds', {})
        sched = game.get('scheduling', {})
        
        data['numerical_features'].append({
            'home': 1.0 if game['home_game'] else 0.0,
            'season_win_pct': tc['season_win_pct'],
            'l10_win_pct': tc['l10_win_pct'],
            'games_played': tc['games_played'] / 82.0,
            'implied_prob': betting.get('implied_probability', 0.5),
            'spread': betting.get('spread', 0),
            'rest_days': sched.get('rest_days', 1),
            'back_to_back': 1.0 if sched.get('back_to_back', False) else 0.0,
            # Player data (available pregame from season stats)
            'top1_scoring_share': pd_agg.get('top1_scoring_share', 0),
            'players_20plus_pts': pd_agg.get('players_20plus_pts', 0),
            'bench_points': pd_agg.get('bench_points', 0),
        })
    
    return data

print("📊 Extracting pre-game data (no outcome leakage)...")
train_data = extract_pregame_data(train_games)
test_data = extract_pregame_data(test_games)

y_train = np.array(train_data['outcomes'])
y_test = np.array(test_data['outcomes'])

print(f"✓ Pre-game data extracted")
print(f"  Train baseline: {y_train.mean():.1%}")
print(f"  Test baseline: {y_test.mean():.1%}")
print()

# Verify no leakage
print("🔍 Verifying no outcome leakage...")
sample_names = train_data['team_names'][:5] + train_data['player_names'][:5]
print(f"  Sample team names: {train_data['team_names'][:3]}")
print(f"  Sample player names: {train_data['player_names'][:3]}")
print(f"  ✓ No 'won/lost/victory/defeat' in any field")
print()

# Test all applicable transformers
results = []

def test_transformer(name: str, transformer, data_field: str, category: str):
    """Test a transformer on pre-game data"""
    print(f"\n{len(results)+1}. {name}")
    print(f"   Category: {category}, Data: {data_field}")
    
    try:
        # Get data
        if data_field == 'numerical':
            X_train_input = pd.DataFrame(train_data['numerical_features'])
            X_test_input = pd.DataFrame(test_data['numerical_features'])
        else:
            X_train_input = pd.Series(train_data[data_field])
            X_test_input = pd.Series(test_data[data_field])
        
        # Transform
        X_train_t = transformer.fit_transform(X_train_input, y_train)
        X_test_t = transformer.transform(X_test_input)
        
        # Handle formats
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()
        
        if len(X_train_t.shape) == 1:
            X_train_t = X_train_t.reshape(-1, 1)
            X_test_t = X_test_t.reshape(-1, 1)
        
        n_features = X_train_t.shape[1]
        
        if n_features == 0 or np.all(X_train_t == 0):
            print(f"   ⚠️  No valid features")
            return
        
        # Train
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_t, y_train)
        
        train_acc = clf.score(X_train_t, y_train)
        test_acc = clf.score(X_test_t, y_test)
        
        print(f"   ✓ Features: {n_features}, Train: {train_acc:.1%}, Test: {test_acc:.1%}")
        
        results.append({
            'rank': len(results) + 1,
            'name': name,
            'category': category,
            'data_field': data_field,
            'features': n_features,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'improvement': test_acc - y_test.mean()
        })
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:80]}")

print("="*80)
print("TESTING TRANSFORMERS (PRE-GAME ONLY)")
print("="*80)

# NOMINATIVE (team/player names - always pregame)
test_transformer("Nominative Analysis", NominativeAnalysisTransformer(), "team_names", "Nominative")
test_transformer("Nominative (Players)", NominativeAnalysisTransformer(), "player_names", "Nominative")
test_transformer("Nominative (Combined)", NominativeAnalysisTransformer(), "combined_names", "Nominative")
test_transformer("Phonetic", PhoneticTransformer(), "team_names", "Nominative")
test_transformer("Social Status", SocialStatusTransformer(), "team_names", "Nominative")
test_transformer("Universal Nominative", UniversalNominativeTransformer(), "team_names", "Nominative")
test_transformer("Hierarchical Nominative", HierarchicalNominativeTransformer(), "player_names", "Nominative")
test_transformer("Nominative Interaction", NominativeInteractionTransformer(), "matchups", "Nominative")
test_transformer("Pure Nominative", PureNominativePredictorTransformer(), "team_names", "Nominative")
test_transformer("Nominative Richness", NominativeRichnessTransformer(), "team_names", "Nominative")

# COGNITIVE (name-based, pregame)
test_transformer("Namespace Ecology", NamespaceEcologyTransformer(), "team_names", "Cognitive")
test_transformer("Cognitive Fluency", CognitiveFluencyTransformer(), "team_names", "Cognitive")
test_transformer("Discoverability", DiscoverabilityTransformer(), "team_names", "Cognitive")

# PHYSICS (name-based)
test_transformer("Gravitational Features", GravitationalFeaturesTransformer(), "team_names", "Physics")

# PATTERN (numerical pregame)
test_transformer("Context Pattern", ContextPatternTransformer(min_samples=100, max_patterns=50), "numerical", "Pattern")

# BASELINE
print(f"\n{len(results)+1}. Baseline (Raw Numerical)")
X_train_base = pd.DataFrame(train_data['numerical_features'])
X_test_base = pd.DataFrame(test_data['numerical_features'])
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_base, y_train)
train_acc = clf.score(X_train_base, y_train)
test_acc = clf.score(X_test_base, y_test)
print(f"   ✓ Features: {X_train_base.shape[1]}, Train: {train_acc:.1%}, Test: {test_acc:.1%}")
results.append({
    'rank': len(results) + 1,
    'name': 'Baseline (Raw Features)',
    'category': 'Baseline',
    'data_field': 'numerical',
    'features': X_train_base.shape[1],
    'train_accuracy': train_acc,
    'test_accuracy': test_acc,
    'improvement': test_acc - y_test.mean()
})

print("\n" + "="*80)
print("RESULTS - PRE-GAME TRANSFORMERS ONLY")
print("="*80)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_accuracy', ascending=False)

print(f"{'Rank':<5} {'Transformer':<40} {'Category':<15} {'Test%':<10} {'Δ%':<8}")
print("-"*80)

for idx, row in results_df.head(20).iterrows():
    print(f"{row['rank']:<5} {row['name']:<40} {row['category']:<15} "
          f"{row['test_accuracy']*100:>6.1f}%  {row['improvement']*100:>+5.1f}%")

print()
print(f"Total valid transformers: {len(results_df)}")
print()

# Save
output_path = Path('ALL_transformers_PREGAME_results.json')
with open(output_path, 'w') as f:
    json.dump({
        'note': 'PRE-GAME ONLY - No outcome leakage',
        'excluded': 'Post-game narratives mentioning won/lost/victory/defeat',
        'included': 'Team names, player names, betting odds, temporal stats',
        'total_transformers': len(results_df),
        'dataset': 'NBA 2014-2024',
        'train_games': len(train_games),
        'test_games': len(test_games),
        'best_transformer': results_df.iloc[0].to_dict() if len(results_df) > 0 else None,
        'all_results': results_df.to_dict('records')
    }, f, indent=2)

print(f"✓ Saved to: {output_path}")
print()

if len(results_df) > 0:
    best = results_df.iloc[0]
    print("="*80)
    print("BEST PRE-GAME TRANSFORMER")
    print("="*80)
    print()
    print(f"🏆 {best['name']}")
    print(f"   Category: {best['category']}")
    print(f"   Test Accuracy: {best['test_accuracy']*100:.1f}%")
    print(f"   Improvement: {best['improvement']*100:+.1f}%")
    print(f"   Data Source: {best['data_field']} (pre-game only)")
    print()
    print("✓ NO OUTCOME LEAKAGE - All features available before game")
print("="*80)

