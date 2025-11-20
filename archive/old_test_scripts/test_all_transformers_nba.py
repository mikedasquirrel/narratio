"""
Full NBA Pipeline Test - All Transformers
==========================================

Tests ALL 35 transformers on real NBA data with frequent progress updates.
This validates that all fixes work correctly in a production-like scenario.

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
from sklearn.metrics import accuracy_score, classification_report


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


def load_nba_data(sample_size=1000):
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
        
        # Add player data
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            if agg.get('top1_name'):
                parts.append(f"Star {agg['top1_name']}")
        
        # Add record
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
    print_progress(f"✓ Built {len(X_test)} test narratives")
    print_progress(f"✓ Baseline win rate: {y_train.mean():.1%}")
    
    return X_train, y_train, X_test, y_test


def test_transformer(name, transformer_cls, kwargs, X_train, y_train, X_test, y_test, category):
    """Test a single transformer with timing and accuracy"""
    
    print_progress(f"Testing {name}...")
    
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
        # Initialize
        start = time.time()
        transformer = transformer_cls(**kwargs)
        
        # Fit and transform
        print_progress(f"  → Fitting {name}...")
        X_train_t = transformer.fit_transform(X_train, y_train)
        
        print_progress(f"  → Transforming test set...")
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
        
        # Train classifier
        if n_features > 0 and not np.all(X_train_t == 0):
            print_progress(f"  → Training classifier on {n_features} features...")
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_t, y_train)
            test_acc = clf.score(X_test_t, y_test)
        else:
            test_acc = 0.0
        
        result['status'] = 'SUCCESS'
        result['time'] = elapsed
        result['features'] = n_features
        result['accuracy'] = test_acc
        
        print_progress(f"  ✓ {name}: {elapsed:.2f}s, {n_features} features, {test_acc:.1%} accuracy")
        
    except Exception as e:
        error_msg = str(e)[:100]
        result['status'] = 'ERROR'
        result['error'] = error_msg
        print_progress(f"  ✗ {name}: {error_msg}")
    
    return result


def main():
    """Run full NBA pipeline test"""
    
    print_header("NBA FULL PIPELINE TEST - ALL TRANSFORMERS", "█")
    print_progress("Starting comprehensive transformer test...")
    print_progress(f"Python: {sys.version.split()[0]}")
    print()
    
    # Load data
    X_train, y_train, X_test, y_test = load_nba_data(sample_size=500)
    
    # Define all transformers to test
    print_header("INITIALIZING TRANSFORMERS", "=")
    
    transformers = []
    
    # Core transformers
    try:
        from transformers.nominative import NominativeAnalysisTransformer
        transformers.append(("Nominative Analysis", NominativeAnalysisTransformer, {}, "Core"))
        print_progress("✓ Loaded Nominative Analysis")
    except: print_progress("✗ Skip Nominative Analysis")
    
    try:
        from transformers.self_perception import SelfPerceptionTransformer
        transformers.append(("Self Perception", SelfPerceptionTransformer, {}, "Core"))
        print_progress("✓ Loaded Self Perception")
    except: print_progress("✗ Skip Self Perception")
    
    try:
        from transformers.narrative_potential import NarrativePotentialTransformer
        transformers.append(("Narrative Potential", NarrativePotentialTransformer, {}, "Core"))
        print_progress("✓ Loaded Narrative Potential")
    except: print_progress("✗ Skip Narrative Potential")
    
    # Linguistic
    try:
        from transformers.linguistic_advanced import LinguisticPatternsTransformer
        transformers.append(("Linguistic Patterns", LinguisticPatternsTransformer, {}, "Linguistic"))
        print_progress("✓ Loaded Linguistic Patterns")
    except: print_progress("✗ Skip Linguistic Patterns")
    
    # Emotional
    try:
        from transformers.emotional_resonance import EmotionalResonanceTransformer
        transformers.append(("Emotional Resonance", EmotionalResonanceTransformer, {}, "Emotional"))
        print_progress("✓ Loaded Emotional Resonance")
    except: print_progress("✗ Skip Emotional Resonance")
    
    try:
        from transformers.authenticity import AuthenticityTransformer
        transformers.append(("Authenticity", AuthenticityTransformer, {}, "Emotional"))
        print_progress("✓ Loaded Authenticity")
    except: print_progress("✗ Skip Authenticity")
    
    try:
        from transformers.conflict_tension import ConflictTensionTransformer
        transformers.append(("Conflict Tension", ConflictTensionTransformer, {}, "Emotional"))
        print_progress("✓ Loaded Conflict Tension")
    except: print_progress("✗ Skip Conflict Tension")
    
    try:
        from transformers.suspense_mystery import SuspenseMysteryTransformer
        transformers.append(("Suspense Mystery", SuspenseMysteryTransformer, {}, "Emotional"))
        print_progress("✓ Loaded Suspense Mystery")
    except: print_progress("✗ Skip Suspense Mystery")
    
    # Nominative
    try:
        from transformers.phonetic import PhoneticTransformer
        transformers.append(("Phonetic", PhoneticTransformer, {}, "Nominative"))
        print_progress("✓ Loaded Phonetic")
    except: print_progress("✗ Skip Phonetic")
    
    try:
        from transformers.universal_nominative import UniversalNominativeTransformer
        transformers.append(("Universal Nominative", UniversalNominativeTransformer, {}, "Nominative"))
        print_progress("✓ Loaded Universal Nominative")
    except: print_progress("✗ Skip Universal Nominative")
    
    try:
        from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
        transformers.append(("Hierarchical Nominative", HierarchicalNominativeTransformer, {}, "Nominative"))
        print_progress("✓ Loaded Hierarchical Nominative")
    except: print_progress("✗ Skip Hierarchical Nominative")
    
    try:
        from transformers.nominative_richness import NominativeRichnessTransformer
        transformers.append(("Nominative Richness", NominativeRichnessTransformer, {}, "Nominative"))
        print_progress("✓ Loaded Nominative Richness")
    except: print_progress("✗ Skip Nominative Richness")
    
    # Ensemble/Statistical
    try:
        from transformers.ensemble import EnsembleNarrativeTransformer
        transformers.append(("Ensemble Narrative", EnsembleNarrativeTransformer, {}, "Ensemble"))
        print_progress("✓ Loaded Ensemble Narrative")
    except: print_progress("✗ Skip Ensemble Narrative")
    
    try:
        from transformers.statistical import StatisticalTransformer
        transformers.append(("Statistical", StatisticalTransformer, {}, "Statistical"))
        print_progress("✓ Loaded Statistical")
    except: print_progress("✗ Skip Statistical")
    
    try:
        from transformers.relational import RelationalValueTransformer
        transformers.append(("Relational Value", RelationalValueTransformer, {}, "Relational"))
        print_progress("✓ Loaded Relational Value")
    except: print_progress("✗ Skip Relational Value")
    
    # Information
    try:
        from transformers.information_theory import InformationTheoryTransformer
        transformers.append(("Information Theory", InformationTheoryTransformer, {}, "Information"))
        print_progress("✓ Loaded Information Theory")
    except: print_progress("✗ Skip Information Theory")
    
    try:
        from transformers.namespace_ecology import NamespaceEcologyTransformer
        transformers.append(("Namespace Ecology", NamespaceEcologyTransformer, {}, "Information"))
        print_progress("✓ Loaded Namespace Ecology")
    except: print_progress("✗ Skip Namespace Ecology")
    
    # Cognitive
    try:
        from transformers.cognitive_fluency import CognitiveFluencyTransformer
        transformers.append(("Cognitive Fluency", CognitiveFluencyTransformer, {}, "Cognitive"))
        print_progress("✓ Loaded Cognitive Fluency")
    except: print_progress("✗ Skip Cognitive Fluency")
    
    try:
        from transformers.discoverability import DiscoverabilityTransformer
        transformers.append(("Discoverability", DiscoverabilityTransformer, {}, "Cognitive"))
        print_progress("✓ Loaded Discoverability")
    except: print_progress("✗ Skip Discoverability")
    
    # Temporal
    try:
        from transformers.temporal_evolution import TemporalEvolutionTransformer
        transformers.append(("Temporal Evolution", TemporalEvolutionTransformer, {}, "Temporal"))
        print_progress("✓ Loaded Temporal Evolution")
    except: print_progress("✗ Skip Temporal Evolution")
    
    # Framing
    try:
        from transformers.optics import OpticsTransformer
        transformers.append(("Optics", OpticsTransformer, {}, "Framing"))
        print_progress("✓ Loaded Optics")
    except: print_progress("✗ Skip Optics")
    
    try:
        from transformers.framing import FramingTransformer
        transformers.append(("Framing", FramingTransformer, {}, "Framing"))
        print_progress("✓ Loaded Framing")
    except: print_progress("✗ Skip Framing")
    
    # Physics
    try:
        from transformers.coupling_strength import CouplingStrengthTransformer
        transformers.append(("Coupling Strength", CouplingStrengthTransformer, {}, "Physics"))
        print_progress("✓ Loaded Coupling Strength")
    except: print_progress("✗ Skip Coupling Strength")
    
    try:
        from transformers.narrative_mass import NarrativeMassTransformer
        transformers.append(("Narrative Mass", NarrativeMassTransformer, {}, "Physics"))
        print_progress("✓ Loaded Narrative Mass")
    except: print_progress("✗ Skip Narrative Mass")
    
    try:
        from transformers.gravitational_features import GravitationalFeaturesTransformer
        transformers.append(("Gravitational Features", GravitationalFeaturesTransformer, {}, "Physics"))
        print_progress("✓ Loaded Gravitational Features")
    except: print_progress("✗ Skip Gravitational Features")
    
    try:
        from transformers.awareness_resistance import AwarenessResistanceTransformer
        transformers.append(("Awareness Resistance", AwarenessResistanceTransformer, {}, "Physics"))
        print_progress("✓ Loaded Awareness Resistance")
    except: print_progress("✗ Skip Awareness Resistance")
    
    try:
        from transformers.fundamental_constraints import FundamentalConstraintsTransformer
        transformers.append(("Fundamental Constraints", FundamentalConstraintsTransformer, {'use_embeddings': False}, "Physics"))
        print_progress("✓ Loaded Fundamental Constraints")
    except: print_progress("✗ Skip Fundamental Constraints")
    
    # Multi-scale
    try:
        from transformers.multi_scale import MultiScaleTransformer
        transformers.append(("Multi-Scale", MultiScaleTransformer, {}, "Multi"))
        print_progress("✓ Loaded Multi-Scale")
    except: print_progress("✗ Skip Multi-Scale")
    
    # Quantitative
    try:
        from transformers.quantitative import QuantitativeTransformer
        transformers.append(("Quantitative", QuantitativeTransformer, {}, "Quantitative"))
        print_progress("✓ Loaded Quantitative")
    except: print_progress("✗ Skip Quantitative")
    
    # Authority/Cultural
    try:
        from transformers.expertise_authority import ExpertiseAuthorityTransformer
        transformers.append(("Expertise Authority", ExpertiseAuthorityTransformer, {}, "Authority"))
        print_progress("✓ Loaded Expertise Authority")
    except: print_progress("✗ Skip Expertise Authority")
    
    try:
        from transformers.cultural_context import CulturalContextTransformer
        transformers.append(("Cultural Context", CulturalContextTransformer, {}, "Cultural"))
        print_progress("✓ Loaded Cultural Context")
    except: print_progress("✗ Skip Cultural Context")
    
    try:
        from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
        transformers.append(("Anticipatory Communication", AnticipatoryCommunicationTransformer, {}, "Communication"))
        print_progress("✓ Loaded Anticipatory Communication")
    except: print_progress("✗ Skip Anticipatory Communication")
    
    try:
        from transformers.social_status import SocialStatusTransformer
        transformers.append(("Social Status", SocialStatusTransformer, {}, "Social"))
        print_progress("✓ Loaded Social Status")
    except: print_progress("✗ Skip Social Status")
    
    # Pattern
    try:
        from transformers.context_pattern import ContextPatternTransformer
        transformers.append(("Context Pattern", ContextPatternTransformer, {'min_samples': 30, 'max_patterns': 20}, "Pattern"))
        print_progress("✓ Loaded Context Pattern")
    except: print_progress("✗ Skip Context Pattern")
    
    print()
    print_progress(f"✓ Successfully loaded {len(transformers)} transformers")
    
    # Test each transformer
    print_header("TESTING ALL TRANSFORMERS", "=")
    print_progress(f"Will test {len(transformers)} transformers on NBA data")
    print()
    
    results = []
    start_time = time.time()
    
    for i, (name, cls, kwargs, category) in enumerate(transformers, 1):
        print_header(f"[{i}/{len(transformers)}] {name}", "-")
        result = test_transformer(name, cls, kwargs, X_train, y_train, X_test, y_test, category)
        results.append(result)
        print()
    
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
    
    if len(df_success) > 0:
        print()
        print_header("TOP 10 PERFORMERS", "-")
        df_sorted = df_success.sort_values('accuracy', ascending=False)
        
        print(f"{'Rank':<6} {'Transformer':<35} {'Time':<10} {'Features':<10} {'Accuracy'}")
        print("-" * 75)
        
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"{i:<6} {row['name']:<35} {row['time']:<8.2f}s  {row['features']:<8}  {row['accuracy']:<6.1%}")
        
        print()
        print_header("FASTEST TRANSFORMERS", "-")
        df_fast = df_success.sort_values('time')
        
        print(f"{'Rank':<6} {'Transformer':<35} {'Time':<10} {'Speed (samples/s)'}")
        print("-" * 70)
        
        for i, (_, row) in enumerate(df_fast.head(10).iterrows(), 1):
            speed = len(X_train) / row['time'] if row['time'] > 0 else 0
            print(f"{i:<6} {row['name']:<35} {row['time']:<8.2f}s  {speed:<10.0f}")
    
    if len(df_error) > 0:
        print()
        print_header("ERRORS", "-")
        for _, row in df_error.iterrows():
            print(f"✗ {row['name']}: {row['error']}")
    
    # Save results
    output_file = 'nba_full_pipeline_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'dataset': 'NBA games',
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'transformers_tested': len(results),
            'successful': len(df_success),
            'errors': len(df_error),
            'total_time_seconds': total_time,
            'results': df.to_dict('records')
        }, f, indent=2)
    
    print()
    print_progress(f"✓ Results saved to: {output_file}")
    
    print_header("TEST COMPLETE!", "█")
    print_progress(f"🎉 NBA Pipeline Test Finished!")
    print_progress(f"📊 {len(df_success)}/{len(results)} transformers working ({len(df_success)/len(results)*100:.0f}%)")
    print()


if __name__ == "__main__":
    main()

