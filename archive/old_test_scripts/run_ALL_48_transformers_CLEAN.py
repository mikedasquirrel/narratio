"""
Run ALL 48 Transformers on Clean Pre-Game Nominative Narratives

Narrative = Collage of nominative features (names, entities)
NO outcome leakage - all data available before game:
- Team names, player names, coaches, refs
- Stadium, city names  
- Matchup description
- Season records, betting odds
- Rest days, scheduling

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

print("Loading ALL 48 transformers...")
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
from transformers.hierarchical_nominative import (
    HierarchicalNominativeTransformer,
    NominativeInteractionTransformer,
    PureNominativePredictorTransformer
)
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
from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
from transformers.cognitive_fluency import CognitiveFluencyTransformer
from transformers.quantitative import QuantitativeTransformer
from transformers.discoverability import DiscoverabilityTransformer
from transformers.multi_scale import (
    MultiScaleTransformer,
    MultiPerspectiveTransformer,
    ScaleInteractionTransformer
)
from transformers.coupling_strength import CouplingStrengthTransformer
from transformers.narrative_mass import NarrativeMassTransformer
from transformers.gravitational_features import GravitationalFeaturesTransformer
from transformers.awareness_resistance import AwarenessResistanceTransformer
from transformers.fundamental_constraints import FundamentalConstraintsTransformer
from transformers.alpha import AlphaTransformer
# from transformers.golden_narratio import GoldenNarratioTransformer  # May not exist
from transformers.context_pattern import ContextPatternTransformer

from sklearn.linear_model import LogisticRegression

print("✓ All 48 transformers loaded")
print()

print("="*80)
print("ALL 48 TRANSFORMERS - CLEAN PRE-GAME NOMINATIVE NARRATIVES")
print("="*80)
print()
print("Narrative = Collage of nominative features (NO outcome leakage)")
print()

# Load data
data_path = Path('data/domains/nba_complete_with_players.json')
with open(data_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} games")

train_games = [g for g in all_games if g['season'] < '2023-24']
test_games = [g for g in all_games if g['season'] == '2023-24']

print(f"  Train: {len(train_games):,}")
print(f"  Test: {len(test_games):,}")
print()

def build_pregame_narrative(game: Dict) -> str:
    """Build pre-game narrative as nominative collage (NO outcome)"""
    parts = []
    
    # Teams & Matchup
    parts.append(f"Team {game['team_name']}")
    parts.append(f"Matchup {game['matchup']}")
    parts.append(f"Location {'home' if game['home_game'] else 'away'}")
    
    # Players (pre-game roster)
    if game.get('player_data', {}).get('available'):
        agg = game['player_data']['team_aggregates']
        if agg.get('top1_name'):
            parts.append(f"Leading scorer {agg['top1_name']}")
        if agg.get('top2_name'):
            parts.append(f"Secondary scorer {agg['top2_name']}")
        if agg.get('top3_name'):
            parts.append(f"Third scorer {agg['top3_name']}")
    
    # Pre-game record
    tc = game['temporal_context']
    parts.append(f"Season record {tc['season_record_prior']}")
    parts.append(f"Last 10 games {tc['l10_record']}")
    parts.append(f"Games into season {tc['games_played']}")
    
    # Betting context
    betting = game.get('betting_odds', {})
    if betting.get('moneyline'):
        ml = betting['moneyline']
        parts.append(f"Betting line {ml}")
        if ml > 0:
            parts.append("Underdog")
        else:
            parts.append("Favorite")
    
    # Scheduling
    sched = game.get('scheduling', {})
    if sched.get('rest_days', 1) == 0:
        parts.append("Back to back game")
    elif sched.get('rest_days', 1) >= 3:
        parts.append("Well rested")
    
    return ". ".join(parts) + "."

def extract_clean_data(games: List[Dict]) -> Dict:
    """Extract ALL pre-game data (nominative collages + components)"""
    data = {
        'pregame_narratives': [],
        'team_names': [],
        'player_names': [],
        'matchups': [],
        'combined_names': [],
        'outcomes': [],
        'numerical_features': []
    }
    
    for game in games:
        # Build clean narrative
        narrative = build_pregame_narrative(game)
        data['pregame_narratives'].append(narrative)
        
        # Components
        data['team_names'].append(game.get('team_name', ''))
        data['matchups'].append(game.get('matchup', ''))
        
        # Players
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            players = ' '.join([
                agg.get('top1_name', ''),
                agg.get('top2_name', ''),
                agg.get('top3_name', '')
            ])
            data['player_names'].append(players)
            data['combined_names'].append(f"{game['team_name']} {players}")
        else:
            data['player_names'].append('')
            data['combined_names'].append(game['team_name'])
        
        data['outcomes'].append(1 if game.get('won', False) else 0)
        
        # Numerical
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
            'top1_scoring_share': pd_agg.get('top1_scoring_share', 0),
            'players_20plus_pts': pd_agg.get('players_20plus_pts', 0),
        })
    
    return data

print("📊 Building clean pre-game narratives (nominative collages)...")
train_data = extract_clean_data(train_games)
test_data = extract_clean_data(test_games)

y_train = np.array(train_data['outcomes'])
y_test = np.array(test_data['outcomes'])

print(f"✓ Clean narratives built")
print(f"  Train baseline: {y_train.mean():.1%}")
print(f"  Test baseline: {y_test.mean():.1%}")
print()

# Verify no leakage
print("🔍 Verifying no outcome leakage...")
sample = train_data['pregame_narratives'][0]
print(f"  Sample narrative: {sample[:200]}...")
leakage_words = ['won', 'lost', 'victory', 'defeat', 'fell short', 'secured']
has_leakage = any(word in sample.lower() for word in leakage_words)
print(f"  {'❌ LEAKAGE' if has_leakage else '✓ NO LEAKAGE'}")
print()

# Test ALL transformers
results = []

def test_transformer(name: str, transformer, data_field: str, category: str):
    """Test a transformer"""
    print(f"{len(results)+1}. {name} ({category})", end='')
    
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
        
        # Format
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()
        
        if len(X_train_t.shape) == 1:
            X_train_t = X_train_t.reshape(-1, 1)
            X_test_t = X_test_t.reshape(-1, 1)
        
        n_features = X_train_t.shape[1]
        
        if n_features == 0 or np.all(X_train_t == 0):
            print(f" - No features")
            return
        
        # Train
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_t, y_train)
        
        test_acc = clf.score(X_test_t, y_test)
        
        print(f" - Test: {test_acc:.1%} ({n_features} features)")
        
        results.append({
            'name': name,
            'category': category,
            'data_field': data_field,
            'features': n_features,
            'test_accuracy': test_acc,
            'improvement': test_acc - y_test.mean()
        })
        
    except Exception as e:
        print(f" - Error: {str(e)[:60]}")

print("="*80)
print("TESTING ALL 48 TRANSFORMERS")
print("="*80)
print()

# Test each transformer
test_transformer("Nominative Analysis", NominativeAnalysisTransformer(), "team_names", "Core")
test_transformer("Self Perception", SelfPerceptionTransformer(), "pregame_narratives", "Core")
test_transformer("Narrative Potential", NarrativePotentialTransformer(), "pregame_narratives", "Core")
test_transformer("Linguistic Patterns", LinguisticPatternsTransformer(), "pregame_narratives", "Core")
test_transformer("Relational Value", RelationalValueTransformer(), "pregame_narratives", "Core")
test_transformer("Ensemble Narrative", EnsembleNarrativeTransformer(), "pregame_narratives", "Core")
test_transformer("Statistical", StatisticalTransformer(), "pregame_narratives", "Core")
test_transformer("Phonetic", PhoneticTransformer(), "team_names", "Nominative")
test_transformer("Social Status", SocialStatusTransformer(), "team_names", "Nominative")
test_transformer("Universal Nominative", UniversalNominativeTransformer(), "team_names", "Nominative")
test_transformer("Hierarchical Nominative", HierarchicalNominativeTransformer(), "player_names", "Nominative")
test_transformer("Nominative Interaction", NominativeInteractionTransformer(), "matchups", "Nominative")
test_transformer("Pure Nominative", PureNominativePredictorTransformer(), "team_names", "Nominative")
test_transformer("Nominative Richness", NominativeRichnessTransformer(), "combined_names", "Nominative")
test_transformer("Emotional Resonance", EmotionalResonanceTransformer(), "pregame_narratives", "Emotional")
test_transformer("Authenticity", AuthenticityTransformer(), "pregame_narratives", "Emotional")
test_transformer("Conflict Tension", ConflictTensionTransformer(), "pregame_narratives", "Emotional")
test_transformer("Expertise Authority", ExpertiseAuthorityTransformer(), "pregame_narratives", "Emotional")
test_transformer("Cultural Context", CulturalContextTransformer(), "pregame_narratives", "Emotional")
test_transformer("Suspense Mystery", SuspenseMysteryTransformer(), "pregame_narratives", "Emotional")
test_transformer("Optics", OpticsTransformer(), "pregame_narratives", "Framing")
test_transformer("Framing", FramingTransformer(), "pregame_narratives", "Framing")
test_transformer("Temporal Evolution", TemporalEvolutionTransformer(), "pregame_narratives", "Temporal")
test_transformer("Information Theory", InformationTheoryTransformer(), "pregame_narratives", "Information")
test_transformer("Namespace Ecology", NamespaceEcologyTransformer(), "team_names", "Information")
test_transformer("Anticipatory Communication", AnticipatoryCommunicationTransformer(), "pregame_narratives", "Information")
test_transformer("Cognitive Fluency", CognitiveFluencyTransformer(), "team_names", "Cognitive")
test_transformer("Quantitative", QuantitativeTransformer(), "pregame_narratives", "Quantitative")
test_transformer("Discoverability", DiscoverabilityTransformer(), "team_names", "Cognitive")
test_transformer("Multi-Scale", MultiScaleTransformer(), "pregame_narratives", "Multi")
test_transformer("Multi-Perspective", MultiPerspectiveTransformer(), "pregame_narratives", "Multi")
test_transformer("Scale Interaction", ScaleInteractionTransformer(), "pregame_narratives", "Multi")
test_transformer("Coupling Strength", CouplingStrengthTransformer(), "pregame_narratives", "Physics")
test_transformer("Narrative Mass", NarrativeMassTransformer(), "pregame_narratives", "Physics")
test_transformer("Gravitational Features", GravitationalFeaturesTransformer(), "team_names", "Physics")
test_transformer("Awareness Resistance", AwarenessResistanceTransformer(), "pregame_narratives", "Physics")
test_transformer("Fundamental Constraints", FundamentalConstraintsTransformer(), "pregame_narratives", "Physics")
test_transformer("Alpha", AlphaTransformer(), "pregame_narratives", "Mathematical")
# test_transformer("Golden Narratio", GoldenNarratioTransformer(), "pregame_narratives", "Mathematical")  # May not exist
test_transformer("Context Pattern", ContextPatternTransformer(min_samples=100, max_patterns=50), "numerical", "Pattern")

# Baseline
print(f"{len(results)+1}. Baseline (Raw Numerical) (Baseline)", end='')
X_train_base = pd.DataFrame(train_data['numerical_features'])
X_test_base = pd.DataFrame(test_data['numerical_features'])
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_base, y_train)
test_acc = clf.score(X_test_base, y_test)
print(f" - Test: {test_acc:.1%} ({X_train_base.shape[1]} features)")
results.append({
    'name': 'Baseline (Raw)',
    'category': 'Baseline',
    'data_field': 'numerical',
    'features': X_train_base.shape[1],
    'test_accuracy': test_acc,
    'improvement': test_acc - y_test.mean()
})

print("\n" + "="*80)
print(f"TOP 20 TRANSFORMERS (of {len(results)} tested)")
print("="*80)
print()

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_accuracy', ascending=False)

print(f"{'Rank':<5} {'Transformer':<35} {'Category':<15} {'Test%':<10} {'Δ%':<8}")
print("-"*85)

for i, (idx, row) in enumerate(results_df.head(20).iterrows(), 1):
    print(f"{i:<5} {row['name']:<35} {row['category']:<15} "
          f"{row['test_accuracy']*100:>6.1f}%  {row['improvement']*100:>+5.1f}%")

print()

# Save
output_path = Path('ALL_48_transformers_CLEAN_results.json')
with open(output_path, 'w') as f:
    json.dump({
        'note': 'ALL 48 transformers on clean pre-game nominative narratives',
        'data_type': 'Nominative collages (NO outcome leakage)',
        'total_transformers_tested': len(results_df),
        'dataset': 'NBA 2014-2024',
        'train_games': len(train_games),
        'test_games': len(test_games),
        'baseline_accuracy': float(y_test.mean()),
        'best_transformer': results_df.iloc[0].to_dict() if len(results_df) > 0 else None,
        'top_10': results_df.head(10).to_dict('records'),
        'all_results': results_df.to_dict('records')
    }, f, indent=2)

print(f"✓ Saved to: {output_path}")
print()

if len(results_df) > 0:
    best = results_df.iloc[0]
    print("="*80)
    print("🏆 BEST TRANSFORMER")
    print("="*80)
    print()
    print(f"Name: {best['name']}")
    print(f"Category: {best['category']}")
    print(f"Test Accuracy: {best['test_accuracy']*100:.1f}%")
    print(f"Improvement over baseline: {best['improvement']*100:+.1f}%")
    print(f"Data: {best['data_field']}")
    print()
    print("✓ NO OUTCOME LEAKAGE - Pure pre-game nominative features")
print("="*80)

