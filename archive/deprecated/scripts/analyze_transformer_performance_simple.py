"""
Simple Transformer Performance Analysis
========================================

Analyzes ALL transformers for execution speed and quality.
No external dependencies beyond what's already installed.

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

from sklearn.linear_model import LogisticRegression


def load_sample_data(sample_size: int = 1000) -> Tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
    """Load sample NBA data for testing"""
    print("\n" + "="*80)
    print("LOADING SAMPLE DATA")
    print("="*80)
    
    data_path = Path('data/domains/nba_complete_with_players.json')
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    with open(data_path) as f:
        all_games = json.load(f)
    
    print(f"✓ Loaded {len(all_games):,} games")
    
    # Take sample for speed
    train_games = [g for g in all_games if g['season'] < '2023-24'][:sample_size]
    test_games = [g for g in all_games if g['season'] == '2023-24'][:sample_size // 4]
    
    print(f"✓ Using {len(train_games)} train, {len(test_games)} test samples")
    
    # Build narratives
    def build_narrative(game: Dict) -> str:
        parts = [
            f"Team {game.get('team_name', 'Unknown')}",
            f"Matchup {game.get('matchup', 'Unknown')}",
            f"Location {'home' if game.get('home_game', False) else 'away'}",
        ]
        
        # Add player data if available
        if game.get('player_data', {}).get('available'):
            agg = game['player_data']['team_aggregates']
            if agg.get('top1_name'):
                parts.append(f"Star player {agg['top1_name']}")
        
        # Add temporal context
        tc = game.get('temporal_context', {})
        if tc.get('season_record_prior'):
            parts.append(f"Record {tc['season_record_prior']}")
        
        return ". ".join(parts) + "."
    
    X_train = pd.Series([build_narrative(g) for g in train_games])
    y_train = np.array([1 if g.get('won', False) else 0 for g in train_games])
    
    X_test = pd.Series([build_narrative(g) for g in test_games])
    y_test = np.array([1 if g.get('won', False) else 0 for g in test_games])
    
    print(f"✓ Baseline: {y_train.mean():.1%} wins")
    
    return X_train, y_train, X_test, y_test


def profile_transformer(
    name: str,
    transformer_cls,
    transformer_kwargs: Dict[str, Any],
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    category: str = "Unknown"
) -> Dict[str, Any]:
    """Profile a single transformer"""
    
    print(f"\n{'='*80}")
    print(f"[{name}] - {category}")
    print(f"{'='*80}")
    
    try:
        # Initialize
        transformer = transformer_cls(**transformer_kwargs)
        
        # Fit timing
        start_fit = time.time()
        X_train_t = transformer.fit_transform(X_train, y_train)
        fit_time = time.time() - start_fit
        
        # Transform timing
        start_transform = time.time()
        X_test_t = transformer.transform(X_test)
        transform_time = time.time() - start_transform
        
        # Format features
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()
        
        if len(X_train_t.shape) == 1:
            X_train_t = X_train_t.reshape(-1, 1)
            X_test_t = X_test_t.reshape(-1, 1)
        
        n_features = X_train_t.shape[1]
        
        # Test predictive power (quick logistic regression)
        if n_features > 0 and not np.all(X_train_t == 0):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train_t, y_train)
            test_acc = clf.score(X_test_t, y_test)
        else:
            test_acc = 0.0
        
        total_time = fit_time + transform_time
        samples_per_sec = len(X_train) / total_time if total_time > 0 else 0
        
        # Classify performance
        is_slow = total_time > 10.0  # seconds
        
        # Determine reformulation priority
        if total_time > 60:
            priority = "CRITICAL"
        elif total_time > 30:
            priority = "HIGH"
        elif total_time > 10:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        print(f"✓ Total Time: {total_time:.2f}s (Fit: {fit_time:.2f}s, Transform: {transform_time:.2f}s)")
        print(f"✓ Features: {n_features}")
        print(f"✓ Speed: {samples_per_sec:.0f} samples/sec")
        print(f"✓ Test Accuracy: {test_acc:.1%}")
        print(f"✓ Priority: {priority}")
        
        return {
            'name': name,
            'category': category,
            'fit_time': fit_time,
            'transform_time': transform_time,
            'total_time': total_time,
            'features_generated': n_features,
            'samples_per_second': samples_per_sec,
            'test_accuracy': test_acc,
            'error': None,
            'is_slow': is_slow,
            'reformulation_priority': priority
        }
        
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"✗ ERROR: {error_msg}")
        if len(str(e)) > 200:
            print(f"   {traceback.format_exc()[:500]}")
        
        return {
            'name': name,
            'category': category,
            'fit_time': 0,
            'transform_time': 0,
            'total_time': 0,
            'features_generated': 0,
            'samples_per_second': 0,
            'test_accuracy': 0,
            'error': error_msg,
            'is_slow': False,
            'reformulation_priority': "ERROR"
        }


def main():
    """Run comprehensive performance analysis"""
    
    print("\n" + "="*80)
    print("TRANSFORMER PERFORMANCE ANALYSIS")
    print("="*80)
    print("\nThis will profile ALL transformers for speed and quality.")
    print("Expected runtime: 15-30 minutes\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_sample_data(sample_size=1000)
    
    print("\n" + "="*80)
    print("PROFILING TRANSFORMERS")
    print("="*80)
    
    # Define transformers to test
    transformers_to_test = []
    
    # Core transformers
    try:
        from transformers.nominative import NominativeAnalysisTransformer
        transformers_to_test.append(("Nominative Analysis", NominativeAnalysisTransformer, {}, "Core"))
    except ImportError as e:
        print(f"Skip: Nominative Analysis - {e}")
    
    try:
        from transformers.self_perception import SelfPerceptionTransformer
        transformers_to_test.append(("Self Perception", SelfPerceptionTransformer, {}, "Core"))
    except ImportError as e:
        print(f"Skip: Self Perception - {e}")
    
    try:
        from transformers.narrative_potential import NarrativePotentialTransformer
        transformers_to_test.append(("Narrative Potential", NarrativePotentialTransformer, {}, "Core"))
    except ImportError as e:
        print(f"Skip: Narrative Potential - {e}")
    
    try:
        from transformers.linguistic_advanced import LinguisticPatternsTransformer
        transformers_to_test.append(("Linguistic Patterns", LinguisticPatternsTransformer, {}, "Linguistic"))
    except ImportError as e:
        print(f"Skip: Linguistic Patterns - {e}")
    
    try:
        from transformers.emotional_resonance import EmotionalResonanceTransformer
        transformers_to_test.append(("Emotional Resonance", EmotionalResonanceTransformer, {}, "Emotional"))
    except ImportError as e:
        print(f"Skip: Emotional Resonance - {e}")
    
    try:
        from transformers.authenticity import AuthenticityTransformer
        transformers_to_test.append(("Authenticity", AuthenticityTransformer, {}, "Emotional"))
    except ImportError as e:
        print(f"Skip: Authenticity - {e}")
    
    try:
        from transformers.conflict_tension import ConflictTensionTransformer
        transformers_to_test.append(("Conflict Tension", ConflictTensionTransformer, {}, "Emotional"))
    except ImportError as e:
        print(f"Skip: Conflict Tension - {e}")
    
    try:
        from transformers.suspense_mystery import SuspenseMysteryTransformer
        transformers_to_test.append(("Suspense Mystery", SuspenseMysteryTransformer, {}, "Emotional"))
    except ImportError as e:
        print(f"Skip: Suspense Mystery - {e}")
    
    try:
        from transformers.phonetic import PhoneticTransformer
        transformers_to_test.append(("Phonetic", PhoneticTransformer, {}, "Nominative"))
    except ImportError as e:
        print(f"Skip: Phonetic - {e}")
    
    try:
        from transformers.ensemble import EnsembleNarrativeTransformer
        transformers_to_test.append(("Ensemble Narrative", EnsembleNarrativeTransformer, {}, "Ensemble"))
    except ImportError as e:
        print(f"Skip: Ensemble Narrative - {e}")
    
    try:
        from transformers.statistical import StatisticalTransformer
        transformers_to_test.append(("Statistical", StatisticalTransformer, {}, "Statistical"))
    except ImportError as e:
        print(f"Skip: Statistical - {e}")
    
    try:
        from transformers.information_theory import InformationTheoryTransformer
        transformers_to_test.append(("Information Theory", InformationTheoryTransformer, {}, "Information"))
    except ImportError as e:
        print(f"Skip: Information Theory - {e}")
    
    try:
        from transformers.cognitive_fluency import CognitiveFluencyTransformer
        transformers_to_test.append(("Cognitive Fluency", CognitiveFluencyTransformer, {}, "Cognitive"))
    except ImportError as e:
        print(f"Skip: Cognitive Fluency - {e}")
    
    try:
        from transformers.temporal_evolution import TemporalEvolutionTransformer
        transformers_to_test.append(("Temporal Evolution", TemporalEvolutionTransformer, {}, "Temporal"))
    except ImportError as e:
        print(f"Skip: Temporal Evolution - {e}")
    
    try:
        from transformers.optics import OpticsTransformer
        transformers_to_test.append(("Optics", OpticsTransformer, {}, "Framing"))
    except ImportError as e:
        print(f"Skip: Optics - {e}")
    
    try:
        from transformers.framing import FramingTransformer
        transformers_to_test.append(("Framing", FramingTransformer, {}, "Framing"))
    except ImportError as e:
        print(f"Skip: Framing - {e}")
    
    try:
        from transformers.coupling_strength import CouplingStrengthTransformer
        transformers_to_test.append(("Coupling Strength", CouplingStrengthTransformer, {}, "Physics"))
    except ImportError as e:
        print(f"Skip: Coupling Strength - {e}")
    
    try:
        from transformers.narrative_mass import NarrativeMassTransformer
        transformers_to_test.append(("Narrative Mass", NarrativeMassTransformer, {}, "Physics"))
    except ImportError as e:
        print(f"Skip: Narrative Mass - {e}")
    
    try:
        from transformers.gravitational_features import GravitationalFeaturesTransformer
        transformers_to_test.append(("Gravitational Features", GravitationalFeaturesTransformer, {}, "Physics"))
    except ImportError as e:
        print(f"Skip: Gravitational Features - {e}")
    
    try:
        from transformers.awareness_resistance import AwarenessResistanceTransformer
        transformers_to_test.append(("Awareness Resistance", AwarenessResistanceTransformer, {}, "Physics"))
    except ImportError as e:
        print(f"Skip: Awareness Resistance - {e}")
    
    try:
        from transformers.fundamental_constraints import FundamentalConstraintsTransformer
        transformers_to_test.append(("Fundamental Constraints", FundamentalConstraintsTransformer, {}, "Physics"))
    except ImportError as e:
        print(f"Skip: Fundamental Constraints - {e}")
    
    try:
        from transformers.alpha import AlphaTransformer
        transformers_to_test.append(("Alpha", AlphaTransformer, {}, "Mathematical"))
    except ImportError as e:
        print(f"Skip: Alpha - {e}")
    
    try:
        from transformers.multi_scale import MultiScaleTransformer
        transformers_to_test.append(("Multi-Scale", MultiScaleTransformer, {}, "Multi"))
    except ImportError as e:
        print(f"Skip: Multi-Scale - {e}")
    
    try:
        from transformers.quantitative import QuantitativeTransformer
        transformers_to_test.append(("Quantitative", QuantitativeTransformer, {}, "Quantitative"))
    except ImportError as e:
        print(f"Skip: Quantitative - {e}")
    
    try:
        from transformers.relational import RelationalValueTransformer
        transformers_to_test.append(("Relational Value", RelationalValueTransformer, {}, "Relational"))
    except ImportError as e:
        print(f"Skip: Relational Value - {e}")
    
    try:
        from transformers.universal_nominative import UniversalNominativeTransformer
        transformers_to_test.append(("Universal Nominative", UniversalNominativeTransformer, {}, "Nominative"))
    except ImportError as e:
        print(f"Skip: Universal Nominative - {e}")
    
    try:
        from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
        transformers_to_test.append(("Hierarchical Nominative", HierarchicalNominativeTransformer, {}, "Nominative"))
    except ImportError as e:
        print(f"Skip: Hierarchical Nominative - {e}")
    
    try:
        from transformers.nominative_richness import NominativeRichnessTransformer
        transformers_to_test.append(("Nominative Richness", NominativeRichnessTransformer, {}, "Nominative"))
    except ImportError as e:
        print(f"Skip: Nominative Richness - {e}")
    
    try:
        from transformers.expertise_authority import ExpertiseAuthorityTransformer
        transformers_to_test.append(("Expertise Authority", ExpertiseAuthorityTransformer, {}, "Authority"))
    except ImportError as e:
        print(f"Skip: Expertise Authority - {e}")
    
    try:
        from transformers.cultural_context import CulturalContextTransformer
        transformers_to_test.append(("Cultural Context", CulturalContextTransformer, {}, "Cultural"))
    except ImportError as e:
        print(f"Skip: Cultural Context - {e}")
    
    try:
        from transformers.namespace_ecology import NamespaceEcologyTransformer
        transformers_to_test.append(("Namespace Ecology", NamespaceEcologyTransformer, {}, "Information"))
    except ImportError as e:
        print(f"Skip: Namespace Ecology - {e}")
    
    try:
        from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
        transformers_to_test.append(("Anticipatory Communication", AnticipatoryCommunicationTransformer, {}, "Communication"))
    except ImportError as e:
        print(f"Skip: Anticipatory Communication - {e}")
    
    try:
        from transformers.discoverability import DiscoverabilityTransformer
        transformers_to_test.append(("Discoverability", DiscoverabilityTransformer, {}, "Cognitive"))
    except ImportError as e:
        print(f"Skip: Discoverability - {e}")
    
    try:
        from transformers.social_status import SocialStatusTransformer
        transformers_to_test.append(("Social Status", SocialStatusTransformer, {}, "Social"))
    except ImportError as e:
        print(f"Skip: Social Status - {e}")
    
    try:
        from transformers.context_pattern import ContextPatternTransformer
        transformers_to_test.append(("Context Pattern", ContextPatternTransformer, {'min_samples': 50, 'max_patterns': 30}, "Pattern"))
    except ImportError as e:
        print(f"Skip: Context Pattern - {e}")
    
    print(f"\n✓ Found {len(transformers_to_test)} transformers to profile\n")
    
    # Profile each
    results = []
    for i, (name, cls, kwargs, category) in enumerate(transformers_to_test, 1):
        print(f"\n[{i}/{len(transformers_to_test)}] ", end="")
        result = profile_transformer(
            name=name,
            transformer_cls=cls,
            transformer_kwargs=kwargs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            category=category
        )
        results.append(result)
        time.sleep(0.1)  # Small pause between transformers
    
    # Analysis
    print("\n\n" + "="*80)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df_working = df[df['error'].isna()].copy()
    
    if len(df_working) == 0:
        print("\n✗ No transformers completed successfully")
        # Still save all results
        output_path = Path('transformer_performance_analysis.json')
        with open(output_path, 'w') as f:
            json.dump({
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'sample_size': len(X_train),
                'transformers_tested': len(results),
                'transformers_successful': 0,
                'transformers_failed': len(df),
                'detailed_results': df.to_dict('records')
            }, f, indent=2)
        print(f"\n✓ Error details saved to: {output_path}")
        return
    
    print(f"\n✓ {len(df_working)} transformers completed successfully")
    print(f"✗ {len(df) - len(df_working)} transformers failed")
    
    # Sort by speed
    df_working = df_working.sort_values('total_time', ascending=False)
    
    # Print slowest transformers
    print("\n" + "="*80)
    print("SLOWEST TRANSFORMERS (Top 20)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Transformer':<40} {'Time':<12} {'Features':<10} {'Priority':<12}")
    print("-"*85)
    
    for i, (_, row) in enumerate(df_working.head(20).iterrows(), 1):
        print(f"{i:<5} {row['name']:<40} {row['total_time']:>8.2f}s   {row['features_generated']:>6}     {row['reformulation_priority']:<12}")
    
    # Print fastest transformers
    print("\n" + "="*80)
    print("FASTEST TRANSFORMERS (Top 20)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Transformer':<40} {'Time':<12} {'Features':<10} {'Accuracy':<10}")
    print("-"*85)
    
    df_fast = df_working.sort_values('total_time', ascending=True)
    for i, (_, row) in enumerate(df_fast.head(20).iterrows(), 1):
        print(f"{i:<5} {row['name']:<40} {row['total_time']:>8.2f}s   {row['features_generated']:>6}     {row['test_accuracy']:>6.1%}")
    
    # Reformulation priorities
    print("\n" + "="*80)
    print("REFORMULATION PRIORITIES")
    print("="*80)
    
    for priority in ["CRITICAL", "HIGH", "MEDIUM"]:
        df_priority = df_working[df_working['reformulation_priority'] == priority]
        if len(df_priority) > 0:
            print(f"\n{priority} Priority ({len(df_priority)} transformers):")
            for _, row in df_priority.iterrows():
                reason = f"{row['total_time']:.1f}s"
                if row['features_generated'] == 0:
                    reason += " (no features)"
                elif row['test_accuracy'] < 0.5:
                    reason += f" (low accuracy: {row['test_accuracy']:.1%})"
                print(f"  • {row['name']}: {reason}")
    
    # Statistics
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    
    print(f"\nExecution Time:")
    print(f"  Mean: {df_working['total_time'].mean():.2f}s")
    print(f"  Median: {df_working['total_time'].median():.2f}s")
    print(f"  Min: {df_working['total_time'].min():.2f}s")
    print(f"  Max: {df_working['total_time'].max():.2f}s")
    print(f"  Std: {df_working['total_time'].std():.2f}s")
    
    print(f"\nFeature Generation:")
    print(f"  Mean: {df_working['features_generated'].mean():.0f} features")
    print(f"  Median: {df_working['features_generated'].median():.0f} features")
    print(f"  Max: {df_working['features_generated'].max():.0f} features")
    
    print(f"\nProcessing Speed:")
    print(f"  Mean: {df_working['samples_per_second'].mean():.0f} samples/sec")
    print(f"  Median: {df_working['samples_per_second'].median():.0f} samples/sec")
    
    print(f"\nTest Accuracy:")
    print(f"  Mean: {df_working['test_accuracy'].mean():.1%}")
    print(f"  Median: {df_working['test_accuracy'].median():.1%}")
    print(f"  Max: {df_working['test_accuracy'].max():.1%}")
    
    # Save results
    output_path = Path('transformer_performance_analysis.json')
    output_data = {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sample_size': len(X_train),
        'transformers_tested': len(results),
        'transformers_successful': len(df_working),
        'transformers_failed': len(df) - len(df_working),
        'statistics': {
            'mean_time': float(df_working['total_time'].mean()),
            'median_time': float(df_working['total_time'].median()),
            'max_time': float(df_working['total_time'].max()),
            'mean_accuracy': float(df_working['test_accuracy'].mean()),
            'max_accuracy': float(df_working['test_accuracy'].max()),
        },
        'slow_transformers': df_working[df_working['is_slow'] == True]['name'].tolist(),
        'critical_priority': df_working[df_working['reformulation_priority'] == 'CRITICAL']['name'].tolist(),
        'high_priority': df_working[df_working['reformulation_priority'] == 'HIGH']['name'].tolist(),
        'medium_priority': df_working[df_working['reformulation_priority'] == 'MEDIUM']['name'].tolist(),
        'detailed_results': df.to_dict('records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_path}")
    
    # Save CSV for easy analysis
    csv_path = Path('transformer_performance_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV exported to: {csv_path}")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    critical = df_working[df_working['reformulation_priority'] == 'CRITICAL']
    high = df_working[df_working['reformulation_priority'] == 'HIGH']
    
    if len(critical) > 0:
        print(f"\n🔴 CRITICAL: {len(critical)} transformers taking >60s")
        print("   → Immediate reformulation required")
        print("   → Consider caching, vectorization, or complete redesign")
    
    if len(high) > 0:
        print(f"\n🟠 HIGH: {len(high)} transformers taking 30-60s")
        print("   → Reformulation recommended")
        print("   → Profile for bottlenecks and optimize")
    
    medium = df_working[df_working['reformulation_priority'] == 'MEDIUM']
    if len(medium) > 0:
        print(f"\n🟡 MEDIUM: {len(medium)} transformers taking 10-30s")
        print("   → Consider optimization if frequently used")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nReview {output_path} for detailed results.")
    print(f"Focus reformulation efforts on CRITICAL and HIGH priority transformers.\n")


if __name__ == "__main__":
    main()

