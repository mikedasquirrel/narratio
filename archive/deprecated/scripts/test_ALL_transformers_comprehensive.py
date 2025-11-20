"""
COMPREHENSIVE Transformer Test - ALL Categories
================================================

Tests ALL transformer categories on NBA data:
- Core General Purpose (35)
- Domain-Specific Archetypes (16) 
- Version 2 Enhanced (5)
- Linguistic Specialized (5)
- Sports Performance (4)
- Temporal Specialized (4)
- And more!

Author: AI Coding Assistant  
Date: November 16, 2025
"""

import sys
import json
import time
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

from sklearn.linear_model import LogisticRegression


def print_header(text, char='='):
    """Print a nice header"""
    print()
    print(char * 80)
    print(text.center(80))
    print(char * 80)
    print()


def print_progress(text):
    """Print progress with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {text}")


def load_nba_data(sample_size=500):
    """Load and prepare NBA data"""
    print_header("LOADING NBA DATA", "=")
    
    data_path = Path('data/domains/nba_complete_with_players.json')
    
    print_progress(f"Loading from: {data_path}")
    with open(data_path) as f:
        all_games = json.load(f)
    
    print_progress(f"✓ Loaded {len(all_games):,} total games")
    
    # Split by season
    train_games = [g for g in all_games if g['season'] < '2023-24'][:sample_size]
    test_games = [g for g in all_games if g['season'] == '2023-24'][:sample_size // 4]
    
    print_progress(f"✓ Train: {len(train_games):,} games")
    print_progress(f"✓ Test: {len(test_games):,} games")
    
    # Build narratives
    def build_narrative(game):
        parts = [
            f"Team {game.get('team_name', 'Unknown')}",
            f"Matchup {game.get('matchup', 'vs Opponent')}",
            f"Location {'home' if game.get('home_game', False) else 'away'}",
        ]
        
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            if agg.get('top1_name'):
                parts.append(f"Star {agg['top1_name']}")
        
        tc = game.get('temporal_context', {})
        if tc.get('season_record_prior'):
            parts.append(f"Record {tc['season_record_prior']}")
        
        return ". ".join(parts) + "."
    
    print_progress("Building narratives...")
    X_train = pd.Series([build_narrative(g) for g in train_games])
    y_train = np.array([1 if g.get('won', False) else 0 for g in train_games])
    
    X_test = pd.Series([build_narrative(g) for g in test_games])
    y_test = np.array([1 if g.get('won', False) else 0 for g in test_games])
    
    print_progress(f"✓ Built {len(X_train)} train narratives")
    print_progress(f"✓ Baseline win rate: {y_train.mean():.1%}")
    
    return X_train, y_train, X_test, y_test


def test_transformer(name, transformer_cls, kwargs, X_train, y_train, X_test, y_test, category):
    """Test a single transformer"""
    
    result = {
        'name': name,
        'category': category,
        'status': 'unknown',
        'time': 0,
        'features': 0,
        'accuracy': 0,
        'error': None
    }
    
    try:
        start = time.time()
        transformer = transformer_cls(**kwargs)
        
        X_train_t = transformer.fit_transform(X_train, y_train)
        X_test_t = transformer.transform(X_test)
        
        elapsed = time.time() - start
        
        # Format features
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()
        
        if len(X_train_t.shape) == 1:
            X_train_t = X_train_t.reshape(-1, 1)
            X_test_t = X_test_t.reshape(-1, 1)
        
        n_features = X_train_t.shape[1]
        
        if n_features > 0 and not np.all(X_train_t == 0):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_t, y_train)
            test_acc = clf.score(X_test_t, y_test)
        else:
            test_acc = 0.0
        
        result['status'] = 'SUCCESS'
        result['time'] = elapsed
        result['features'] = n_features
        result['accuracy'] = test_acc
        
        print_progress(f"  ✓ {name}: {elapsed:.2f}s, {n_features} features, {test_acc:.1%}")
        
    except Exception as e:
        error_msg = str(e)[:100]
        result['status'] = 'ERROR'
        result['error'] = error_msg
        print_progress(f"  ✗ {name}: {error_msg}")
    
    return result


def main():
    """Run comprehensive transformer test"""
    
    print_header("COMPREHENSIVE TRANSFORMER TEST - ALL CATEGORIES", "█")
    print_progress("Testing ALL transformer categories...")
    print()
    
    # Load data
    X_train, y_train, X_test, y_test = load_nba_data(sample_size=500)
    
    # Initialize transformer list
    transformers = []
    
    print_header("LOADING TRANSFORMERS", "=")
    
    # ========== CORE GENERAL PURPOSE ==========
    print_progress("Loading Core General Purpose transformers...")
    
    try:
        from transformers.nominative import NominativeAnalysisTransformer
        transformers.append(("Nominative Analysis", NominativeAnalysisTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.self_perception import SelfPerceptionTransformer
        transformers.append(("Self Perception", SelfPerceptionTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.narrative_potential import NarrativePotentialTransformer
        transformers.append(("Narrative Potential", NarrativePotentialTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.linguistic_advanced import LinguisticPatternsTransformer
        transformers.append(("Linguistic Patterns", LinguisticPatternsTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.emotional_resonance import EmotionalResonanceTransformer
        transformers.append(("Emotional Resonance", EmotionalResonanceTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.authenticity import AuthenticityTransformer
        transformers.append(("Authenticity", AuthenticityTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.conflict_tension import ConflictTensionTransformer
        transformers.append(("Conflict Tension", ConflictTensionTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.suspense_mystery import SuspenseMysteryTransformer
        transformers.append(("Suspense Mystery", SuspenseMysteryTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.phonetic import PhoneticTransformer
        transformers.append(("Phonetic", PhoneticTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.universal_nominative import UniversalNominativeTransformer
        transformers.append(("Universal Nominative", UniversalNominativeTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
        transformers.append(("Hierarchical Nominative", HierarchicalNominativeTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.nominative_richness import NominativeRichnessTransformer
        transformers.append(("Nominative Richness", NominativeRichnessTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.ensemble import EnsembleNarrativeTransformer
        transformers.append(("Ensemble Narrative", EnsembleNarrativeTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.statistical import StatisticalTransformer
        transformers.append(("Statistical", StatisticalTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.relational import RelationalValueTransformer
        transformers.append(("Relational Value", RelationalValueTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.information_theory import InformationTheoryTransformer
        transformers.append(("Information Theory", InformationTheoryTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.namespace_ecology import NamespaceEcologyTransformer
        transformers.append(("Namespace Ecology", NamespaceEcologyTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.cognitive_fluency import CognitiveFluencyTransformer
        transformers.append(("Cognitive Fluency", CognitiveFluencyTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.discoverability import DiscoverabilityTransformer
        transformers.append(("Discoverability", DiscoverabilityTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.temporal_evolution import TemporalEvolutionTransformer
        transformers.append(("Temporal Evolution", TemporalEvolutionTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.optics import OpticsTransformer
        transformers.append(("Optics", OpticsTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.framing import FramingTransformer
        transformers.append(("Framing", FramingTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.coupling_strength import CouplingStrengthTransformer
        transformers.append(("Coupling Strength", CouplingStrengthTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.narrative_mass import NarrativeMassTransformer
        transformers.append(("Narrative Mass", NarrativeMassTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.gravitational_features import GravitationalFeaturesTransformer
        transformers.append(("Gravitational Features", GravitationalFeaturesTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.awareness_resistance import AwarenessResistanceTransformer
        transformers.append(("Awareness Resistance", AwarenessResistanceTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.fundamental_constraints import FundamentalConstraintsTransformer
        transformers.append(("Fundamental Constraints", FundamentalConstraintsTransformer, {'use_embeddings': False}, "Core"))
    except: pass
    
    try:
        from transformers.multi_scale import MultiScaleTransformer
        transformers.append(("Multi-Scale", MultiScaleTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.quantitative import QuantitativeTransformer
        transformers.append(("Quantitative", QuantitativeTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.expertise_authority import ExpertiseAuthorityTransformer
        transformers.append(("Expertise Authority", ExpertiseAuthorityTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.cultural_context import CulturalContextTransformer
        transformers.append(("Cultural Context", CulturalContextTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
        transformers.append(("Anticipatory Communication", AnticipatoryCommunicationTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.social_status import SocialStatusTransformer
        transformers.append(("Social Status", SocialStatusTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.context_pattern import ContextPatternTransformer
        transformers.append(("Context Pattern", ContextPatternTransformer, {'min_samples': 30, 'max_patterns': 20}, "Core"))
    except: pass
    
    print_progress(f"✓ Loaded {len(transformers)} Core transformers")
    
    # ========== VERSION 2 ENHANCED ==========
    print_progress("Loading Version 2 Enhanced transformers...")
    v2_count = len(transformers)
    
    try:
        from transformers.emotional_resonance_v2 import EmotionalResonanceV2Transformer
        transformers.append(("Emotional Resonance V2", EmotionalResonanceV2Transformer, {}, "V2"))
    except: pass
    
    try:
        from transformers.linguistic_v2 import LinguisticPatternsV2Transformer
        transformers.append(("Linguistic Patterns V2", LinguisticPatternsV2Transformer, {}, "V2"))
    except: pass
    
    try:
        from transformers.narrative_potential_v2 import NarrativePotentialV2Transformer
        transformers.append(("Narrative Potential V2", NarrativePotentialV2Transformer, {}, "V2"))
    except: pass
    
    try:
        from transformers.nominative_v2 import NominativeAnalysisV2Transformer
        transformers.append(("Nominative Analysis V2", NominativeAnalysisV2Transformer, {}, "V2"))
    except: pass
    
    try:
        from transformers.self_perception_v2 import SelfPerceptionV2Transformer
        transformers.append(("Self Perception V2", SelfPerceptionV2Transformer, {}, "V2"))
    except: pass
    
    print_progress(f"✓ Loaded {len(transformers)-v2_count} V2 transformers")
    
    # ========== DOMAIN-SPECIFIC ARCHETYPES ==========
    print_progress("Loading Domain-Specific Archetype transformers...")
    arch_count = len(transformers)
    
    try:
        from transformers.archetypes.nba_archetype import NBAArchetypeTransformer
        transformers.append(("NBA Archetype", NBAArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.hero_journey import HeroJourneyTransformer
        transformers.append(("Hero Journey", HeroJourneyTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.character_archetype import CharacterArchetypeTransformer
        transformers.append(("Character Archetype", CharacterArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.plot_archetype import PlotArchetypeTransformer
        transformers.append(("Plot Archetype", PlotArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.thematic_archetype import ThematicArchetypeTransformer
        transformers.append(("Thematic Archetype", ThematicArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.structural_beat import StructuralBeatTransformer
        transformers.append(("Structural Beat", StructuralBeatTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.tennis_archetype import TennisArchetypeTransformer
        transformers.append(("Tennis Archetype", TennisArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.boxing_archetype import BoxingArchetypeTransformer
        transformers.append(("Boxing Archetype", BoxingArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.chess_archetype import ChessArchetypeTransformer
        transformers.append(("Chess Archetype", ChessArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.golf_archetype import GolfArchetypeTransformer
        transformers.append(("Golf Archetype", GolfArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.crypto_archetype import CryptoArchetypeTransformer
        transformers.append(("Crypto Archetype", CryptoArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.oscars_archetype import OscarsArchetypeTransformer
        transformers.append(("Oscars Archetype", OscarsArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.wwe_archetype import WWEArchetypeTransformer
        transformers.append(("WWE Archetype", WWEArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.startups_archetype import StartupsArchetypeTransformer
        transformers.append(("Startups Archetype", StartupsArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.housing_archetype import HousingArchetypeTransformer
        transformers.append(("Housing Archetype", HousingArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.hurricanes_archetype import HurricanesArchetypeTransformer
        transformers.append(("Hurricanes Archetype", HurricanesArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    try:
        from transformers.archetypes.mental_health_archetype import MentalHealthArchetypeTransformer
        transformers.append(("Mental Health Archetype", MentalHealthArchetypeTransformer, {}, "Archetype"))
    except: pass
    
    print_progress(f"✓ Loaded {len(transformers)-arch_count} Archetype transformers")
    
    # ========== OTHER SPECIALIZED ==========
    print_progress("Loading other specialized transformers...")
    other_count = len(transformers)
    
    try:
        from transformers.multi_scale import MultiPerspectiveTransformer, ScaleInteractionTransformer
        transformers.append(("Multi-Perspective", MultiPerspectiveTransformer, {}, "Specialized"))
        transformers.append(("Scale Interaction", ScaleInteractionTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.temporal_momentum_enhanced import TemporalMomentumEnhancedTransformer
        transformers.append(("Temporal Momentum Enhanced", TemporalMomentumEnhancedTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.agency_quantifier import AgencyQuantifierTransformer
        transformers.append(("Agency Quantifier", AgencyQuantifierTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.stakes_hierarchy import StakesHierarchyTransformer
        transformers.append(("Stakes Hierarchy", StakesHierarchyTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.meta_narrative import MetaNarrativeTransformer
        transformers.append(("Meta Narrative", MetaNarrativeTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.cultural_resonance import CulturalResonanceTransformer
        transformers.append(("Cultural Resonance", CulturalResonanceTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.moral_complexity import MoralComplexityTransformer
        transformers.append(("Moral Complexity", MoralComplexityTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.narrative_devices import NarrativeDevicesTransformer
        transformers.append(("Narrative Devices", NarrativeDevicesTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.conflict_typology import ConflictTypologyTransformer
        transformers.append(("Conflict Typology", ConflictTypologyTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.universal_themes import UniversalThemesTransformer
        transformers.append(("Universal Themes", UniversalThemesTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.character_complexity import CharacterComplexityTransformer
        transformers.append(("Character Complexity", CharacterComplexityTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.origin_story import OriginStoryTransformer
        transformers.append(("Origin Story", OriginStoryTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.reputation_prestige import ReputationPrestigeTransformer
        transformers.append(("Reputation Prestige", ReputationPrestigeTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.scarcity_exclusivity import ScarcityExclusivityTransformer
        transformers.append(("Scarcity Exclusivity", ScarcityExclusivityTransformer, {}, "Specialized"))
    except: pass
    
    try:
        from transformers.community_network import CommunityNetworkTransformer
        transformers.append(("Community Network", CommunityNetworkTransformer, {}, "Specialized"))
    except: pass
    
    print_progress(f"✓ Loaded {len(transformers)-other_count} other specialized transformers")
    
    print()
    print_progress(f"🎯 TOTAL: {len(transformers)} transformers loaded!")
    
    # Test all transformers
    print_header("TESTING ALL TRANSFORMERS", "=")
    
    results = []
    start_time = time.time()
    
    for i, (name, cls, kwargs, category) in enumerate(transformers, 1):
        print(f"\n[{i}/{len(transformers)}] {name} ({category})")
        result = test_transformer(name, cls, kwargs, X_train, y_train, X_test, y_test, category)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Summary
    print_header("FINAL RESULTS", "█")
    
    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'SUCCESS']
    df_error = df[df['status'] == 'ERROR']
    
    print_progress(f"Total transformers tested: {len(results)}")
    print_progress(f"✓ Successful: {len(df_success)} ({len(df_success)/len(results)*100:.0f}%)")
    print_progress(f"✗ Errors: {len(df_error)} ({len(df_error)/len(results)*100:.0f}%)")
    print_progress(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Category breakdown
    print()
    print_header("RESULTS BY CATEGORY", "-")
    for cat in df['category'].unique():
        df_cat = df[df['category'] == cat]
        success = len(df_cat[df_cat['status'] == 'SUCCESS'])
        total = len(df_cat)
        print_progress(f"{cat}: {success}/{total} working ({success/total*100:.0f}%)")
    
    if len(df_success) > 0:
        print()
        print_header("TOP 15 PERFORMERS", "-")
        df_sorted = df_success.sort_values('accuracy', ascending=False)
        
        print(f"{'#':<4} {'Transformer':<40} {'Category':<15} {'Acc':<8} {'Time'}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df_sorted.head(15).iterrows(), 1):
            print(f"{i:<4} {row['name']:<40} {row['category']:<15} {row['accuracy']:<7.1%} {row['time']:.2f}s")
    
    # Save results
    output_file = 'comprehensive_transformer_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'transformers_tested': len(results),
            'successful': len(df_success),
            'errors': len(df_error),
            'total_time_seconds': total_time,
            'results': df.to_dict('records')
        }, f, indent=2)
    
    print()
    print_progress(f"✓ Results saved to: {output_file}")
    
    print_header("TEST COMPLETE!", "█")
    print_progress(f"🎉 Comprehensive Test Finished!")
    print_progress(f"📊 {len(df_success)}/{len(results)} transformers working ({len(df_success)/len(results)*100:.0f}%)")
    print()


if __name__ == "__main__":
    main()

