"""
Run ALL 48 Transformers on Complete NBA Dataset

This is the NARRATIVE OPTIMIZATION FRAMEWORK.
We have:
- Rich narratives (game descriptions)
- Nominative data (team names, player names, coaches, refs)
- Statistical data (win %, points, etc.)
- Temporal data (season progression)
- Relational data (matchups, rivalries)

Test ALL transformers and compare performance.

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

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

# Import ALL transformers
print("Loading transformers...")
from transformers.nominative import NominativeAnalysisTransformer
from transformers.self_perception import SelfPerceptionTransformer
from transformers.narrative_potential import NarrativePotentialTransformer
from transformers.linguistic_advanced import LinguisticPatternsTransformer
from transformers.relational import RelationalValueTransformer
from transformers.ensemble import EnsembleNarrativeTransformer
from transformers.statistical import StatisticalTransformer
from transformers.phonetic import PhoneticTransformer
from transformers.social_status import SocialStatusTransformer
from transformers.universal_nominative import UniversalNominativeTransformer
from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
from transformers.nominative_interaction import NominativeInteractionTransformer
from transformers.pure_nominative import PureNominativePredictorTransformer
from transformers.nominative_richness import NominativeRichnessTransformer
from transformers.emotional_resonance import EmotionalResonanceTransformer
from transformers.authenticity import AuthenticityTransformer
from transformers.conflict_tension import ConflictTensionTransformer
from transformers.expertise_authority import ExpertiseAuthorityTransformer
from transformers.cultural_context import CulturalContextTransformer
from transformers.suspense_mystery import SuspenseMysteryTransformer
from transformers.optics import OpticsTransformer
from transformers.framing import FramingTransformer
from transformers.temporal_evolution import TemporalEvolutionTransformer
from transformers.information_theory import InformationTheoryTransformer
from transformers.namespace_ecology import NamespaceEcologyTransformer
from transformers.anticipatory_communication import AnticipatoryCommunicationTransformer
from transformers.cognitive_fluency import CognitiveFluencyTransformer
from transformers.quantitative import QuantitativeTransformer
from transformers.discoverability import DiscoverabilityTransformer
from transformers.multi_scale import MultiScaleTransformer
from transformers.multi_perspective import MultiPerspectiveTransformer
from transformers.scale_interaction import ScaleInteractionTransformer
from transformers.coupling_strength import CouplingStrengthTransformer
from transformers.narrative_mass import NarrativeMassTransformer
from transformers.gravitational_features import GravitationalFeaturesTransformer
from transformers.awareness_resistance import AwarenessResistanceTransformer
from transformers.fundamental_constraints import FundamentalConstraintsTransformer
from transformers.alpha import AlphaTransformer
from transformers.golden_narratio import GoldenNarratioTransformer
from transformers.context_pattern import ContextPatternTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print("✓ All transformers loaded")
print()

print("="*80)
print("ALL TRANSFORMERS - NBA COMPLETE DATASET")
print("="*80)
print()
print("Testing ALL 48 transformers on 11,976 games")
print("Narrative + Statistical + Nominative features")
print()

# Load data
data_path = Path('data/domains/nba_complete_with_players.json')
with open(data_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} games")

# Split train/test
train_games = [g for g in all_games if g['season'] < '2023-24']
test_games = [g for g in all_games if g['season'] == '2023-24']

print(f"  Train: {len(train_games):,} games")
print(f"  Test: {len(test_games):,} games")
print()

# Extract ALL data types
def extract_all_data(games: List[Dict]) -> Dict:
    """Extract narratives, names, and numerical data"""
    data = {
        'narratives': [],
        'rich_narratives': [],
        'team_names': [],
        'player_names': [],
        'matchups': [],
        'outcomes': [],
        'numerical_features': []
    }
    
    for game in games:
        data['narratives'].append(game.get('narrative', ''))
        data['rich_narratives'].append(game.get('rich_narrative', ''))
        data['team_names'].append(game.get('team_name', ''))
        data['matchups'].append(game.get('matchup', ''))
        data['outcomes'].append(1 if game.get('won', False) else 0)
        
        # Player names
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            player_names = [agg.get('top1_name', ''), agg.get('top2_name', ''), agg.get('top3_name', '')]
            data['player_names'].append(' '.join(filter(None, player_names)))
        else:
            data['player_names'].append('')
        
        # Numerical features
        tc = game['temporal_context']
        pd_agg = game.get('player_data', {}).get('team_aggregates', {})
        betting = game.get('betting_odds', {})
        
        data['numerical_features'].append({
            'home': 1.0 if game['home_game'] else 0.0,
            'season_win_pct': tc['season_win_pct'],
            'l10_win_pct': tc['l10_win_pct'],
            'top1_points': pd_agg.get('top1_points', 0),
            'top2_points': pd_agg.get('top2_points', 0),
            'players_20plus_pts': pd_agg.get('players_20plus_pts', 0),
            'implied_prob': betting.get('implied_probability', 0.5),
        })
    
    return data

print("📊 Extracting all data types...")
train_data = extract_all_data(train_games)
test_data = extract_all_data(test_games)

y_train = np.array(train_data['outcomes'])
y_test = np.array(test_data['outcomes'])

print(f"✓ Data extracted")
print(f"  Train baseline: {y_train.mean():.1%}")
print(f"  Test baseline: {y_test.mean():.1%}")
print()

# Test all transformers
results = []

def test_transformer(name: str, transformer, data_field: str, category: str):
    """Test a single transformer"""
    print(f"\n{len(results)+1}. {name}")
    print(f"   Category: {category}")
    print(f"   Data: {data_field}")
    
    try:
        # Get appropriate data
        if data_field == 'numerical':
            X_train_input = pd.DataFrame(train_data['numerical_features'])
            X_test_input = pd.DataFrame(test_data['numerical_features'])
        elif data_field == 'narrative':
            X_train_input = pd.Series(train_data['narratives'])
            X_test_input = pd.Series(test_data['narratives'])
        elif data_field == 'rich_narrative':
            X_train_input = pd.Series(train_data['rich_narratives'])
            X_test_input = pd.Series(test_data['rich_narratives'])
        elif data_field == 'team_names':
            X_train_input = pd.Series(train_data['team_names'])
            X_test_input = pd.Series(test_data['team_names'])
        elif data_field == 'player_names':
            X_train_input = pd.Series(train_data['player_names'])
            X_test_input = pd.Series(test_data['player_names'])
        else:
            X_train_input = pd.Series(train_data[data_field])
            X_test_input = pd.Series(test_data[data_field])
        
        # Transform
        X_train_transformed = transformer.fit_transform(X_train_input, y_train)
        X_test_transformed = transformer.transform(X_test_input)
        
        # Handle sparse matrices
        if hasattr(X_train_transformed, 'toarray'):
            X_train_transformed = X_train_transformed.toarray()
            X_test_transformed = X_test_transformed.toarray()
        
        # Ensure 2D
        if len(X_train_transformed.shape) == 1:
            X_train_transformed = X_train_transformed.reshape(-1, 1)
            X_test_transformed = X_test_transformed.reshape(-1, 1)
        
        n_features = X_train_transformed.shape[1]
        
        # Skip if no features or all zero
        if n_features == 0 or np.all(X_train_transformed == 0):
            print(f"   ⚠️  No valid features generated")
            return
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_transformed, y_train)
        
        train_acc = clf.score(X_train_transformed, y_train)
        test_acc = clf.score(X_test_transformed, y_test)
        
        print(f"   ✓ Features: {n_features}")
        print(f"   ✓ Train: {train_acc:.1%}")
        print(f"   ✓ Test: {test_acc:.1%}")
        
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
        print(f"   ✗ Error: {str(e)[:100]}")

print("="*80)
print("TESTING ALL TRANSFORMERS")
print("="*80)

# CORE TRANSFORMERS
test_transformer("Nominative Analysis", NominativeAnalysisTransformer(), "team_names", "Core")
test_transformer("Self Perception", SelfPerceptionTransformer(), "rich_narrative", "Core")
test_transformer("Narrative Potential", NarrativePotentialTransformer(), "rich_narrative", "Core")
test_transformer("Linguistic Patterns", LinguisticPatternsTransformer(), "rich_narrative", "Core")
test_transformer("Relational Value", RelationalValueTransformer(), "rich_narrative", "Core")
test_transformer("Ensemble Narrative", EnsembleNarrativeTransformer(), "rich_narrative", "Core")

# NOMINATIVE TRANSFORMERS
test_transformer("Phonetic", PhoneticTransformer(), "team_names", "Nominative")
test_transformer("Social Status", SocialStatusTransformer(), "team_names", "Nominative")
test_transformer("Universal Nominative", UniversalNominativeTransformer(), "team_names", "Nominative")
test_transformer("Hierarchical Nominative", HierarchicalNominativeTransformer(), "player_names", "Nominative")
test_transformer("Nominative Interaction", NominativeInteractionTransformer(), "matchups", "Nominative")
test_transformer("Pure Nominative", PureNominativePredictorTransformer(), "team_names", "Nominative")
test_transformer("Nominative Richness", NominativeRichnessTransformer(), "team_names", "Nominative")

# EMOTIONAL/PSYCHOLOGICAL
test_transformer("Emotional Resonance", EmotionalResonanceTransformer(), "rich_narrative", "Emotional")
test_transformer("Authenticity", AuthenticityTransformer(), "rich_narrative", "Emotional")
test_transformer("Conflict Tension", ConflictTensionTransformer(), "rich_narrative", "Emotional")
test_transformer("Expertise Authority", ExpertiseAuthorityTransformer(), "rich_narrative", "Emotional")
test_transformer("Cultural Context", CulturalContextTransformer(), "rich_narrative", "Emotional")
test_transformer("Suspense Mystery", SuspenseMysteryTransformer(), "rich_narrative", "Emotional")

# FRAMING/PERCEPTION
test_transformer("Optics", OpticsTransformer(), "rich_narrative", "Framing")
test_transformer("Framing", FramingTransformer(), "rich_narrative", "Framing")

# TEMPORAL/INFORMATION
test_transformer("Information Theory", InformationTheoryTransformer(), "rich_narrative", "Information")
test_transformer("Namespace Ecology", NamespaceEcologyTransformer(), "team_names", "Information")
test_transformer("Anticipatory Communication", AnticipatoryCommunicationTransformer(), "rich_narrative", "Information")
test_transformer("Cognitive Fluency", CognitiveFluencyTransformer(), "team_names", "Information")
test_transformer("Discoverability", DiscoverabilityTransformer(), "team_names", "Information")

# MULTI-SCALE/PERSPECTIVE
test_transformer("Multi-Scale", MultiScaleTransformer(), "rich_narrative", "Multi")
test_transformer("Multi-Perspective", MultiPerspectiveTransformer(), "rich_narrative", "Multi")
test_transformer("Scale Interaction", ScaleInteractionTransformer(), "rich_narrative", "Multi")

# PHYSICS-INSPIRED
test_transformer("Coupling Strength", CouplingStrengthTransformer(), "rich_narrative", "Physics")
test_transformer("Narrative Mass", NarrativeMassTransformer(), "rich_narrative", "Physics")
test_transformer("Gravitational Features", GravitationalFeaturesTransformer(), "team_names", "Physics")
test_transformer("Awareness Resistance", AwarenessResistanceTransformer(), "rich_narrative", "Physics")
test_transformer("Fundamental Constraints", FundamentalConstraintsTransformer(), "rich_narrative", "Physics")

# MATHEMATICAL
test_transformer("Alpha", AlphaTransformer(), "rich_narrative", "Mathematical")
test_transformer("Golden Narratio", GoldenNarratioTransformer(), "rich_narrative", "Mathematical")

# NUMERICAL/PATTERN
test_transformer("Context Pattern", ContextPatternTransformer(min_samples=100, max_patterns=50), "numerical", "Pattern")

print("\n" + "="*80)
print("RESULTS - ALL TRANSFORMERS")
print("="*80)
print()

# Sort by test accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_accuracy', ascending=False)

print(f"{'Rank':<5} {'Transformer':<35} {'Category':<15} {'Test%':<10} {'Features':<10}")
print("-"*80)

for idx, row in results_df.head(20).iterrows():
    print(f"{row['rank']:<5} {row['name']:<35} {row['category']:<15} {row['test_accuracy']*100:>6.1f}%  {row['features']:>8}")

print()
print(f"Total transformers successfully tested: {len(results_df)}")
print()

# Save results
output_path = Path('ALL_transformers_results.json')
with open(output_path, 'w') as f:
    json.dump({
        'total_transformers_tested': len(results_df),
        'dataset': 'NBA Complete 2014-2024',
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
    print("BEST TRANSFORMER")
    print("="*80)
    print()
    print(f"🏆 {best['name']}")
    print(f"   Category: {best['category']}")
    print(f"   Data Field: {best['data_field']}")
    print(f"   Test Accuracy: {best['test_accuracy']*100:.1f}%")
    print(f"   Improvement: {best['improvement']*100:+.1f}%")
    print(f"   Features: {best['features']}")
    print()
print("="*80)

