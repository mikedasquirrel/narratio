"""
Comprehensive Transformer Performance Analysis
===============================================

Analyzes ALL transformers for:
- Execution speed (fit + transform time)
- Memory efficiency
- Feature generation quality
- Scalability metrics

Identifies slow transformers that need reformulation.

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
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import psutil

warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'narrative_optimization' / 'src'))

from sklearn.linear_model import LogisticRegression


@dataclass
class TransformerPerformance:
    """Performance metrics for a transformer"""
    name: str
    category: str
    fit_time: float
    transform_time: float
    total_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    features_generated: int
    samples_per_second: float
    test_accuracy: float
    error: Optional[str] = None
    is_slow: bool = False
    reformulation_priority: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


class TransformerProfiler:
    """Profile transformer performance"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TransformerPerformance] = []
        
    def profile_transformer(
        self,
        name: str,
        transformer_cls,
        transformer_kwargs: Dict[str, Any],
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        category: str = "Unknown"
    ) -> TransformerPerformance:
        """Profile a single transformer"""
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Profiling: {name}")
            print(f"Category: {category}")
            print(f"{'='*80}")
        
        try:
            # Initialize
            transformer = transformer_cls(**transformer_kwargs)
            
            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Fit timing
            start_fit = time.time()
            mem_fit, X_train_t = memory_usage(
                (transformer.fit_transform, (X_train, y_train)),
                retval=True,
                max_usage=True
            )
            fit_time = time.time() - start_fit
            
            # Transform timing
            start_transform = time.time()
            mem_transform, X_test_t = memory_usage(
                (transformer.transform, (X_test,)),
                retval=True,
                max_usage=True
            )
            transform_time = time.time() - start_transform
            
            # Memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            mem_peak = max(mem_fit, mem_transform)
            
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
            
            result = TransformerPerformance(
                name=name,
                category=category,
                fit_time=fit_time,
                transform_time=transform_time,
                total_time=total_time,
                memory_peak_mb=mem_peak,
                memory_delta_mb=mem_delta,
                features_generated=n_features,
                samples_per_second=samples_per_sec,
                test_accuracy=test_acc,
                is_slow=is_slow,
                reformulation_priority=priority
            )
            
            if self.verbose:
                print(f"✓ Total Time: {total_time:.2f}s")
                print(f"  - Fit: {fit_time:.2f}s")
                print(f"  - Transform: {transform_time:.2f}s")
                print(f"✓ Memory: {mem_delta:.1f}MB delta, {mem_peak:.1f}MB peak")
                print(f"✓ Features: {n_features}")
                print(f"✓ Speed: {samples_per_sec:.0f} samples/sec")
                print(f"✓ Test Accuracy: {test_acc:.1%}")
                print(f"✓ Priority: {priority}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)[:200]
            if self.verbose:
                print(f"✗ ERROR: {error_msg}")
                print(traceback.format_exc())
            
            return TransformerPerformance(
                name=name,
                category=category,
                fit_time=0,
                transform_time=0,
                total_time=0,
                memory_peak_mb=0,
                memory_delta_mb=0,
                features_generated=0,
                samples_per_second=0,
                test_accuracy=0,
                error=error_msg,
                is_slow=False,
                reformulation_priority="ERROR"
            )


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


def main():
    """Run comprehensive performance analysis"""
    
    print("\n" + "="*80)
    print("TRANSFORMER PERFORMANCE ANALYSIS")
    print("="*80)
    print("\nThis will profile ALL transformers for speed, memory, and quality.")
    print("Expected runtime: 15-30 minutes\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_sample_data(sample_size=1000)
    
    # Initialize profiler
    profiler = TransformerProfiler(verbose=True)
    
    print("\n" + "="*80)
    print("PROFILING TRANSFORMERS")
    print("="*80)
    
    # Import and test transformers
    transformers_to_test = []
    
    try:
        from transformers.nominative import NominativeAnalysisTransformer
        transformers_to_test.append(("Nominative Analysis", NominativeAnalysisTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.self_perception import SelfPerceptionTransformer
        transformers_to_test.append(("Self Perception", SelfPerceptionTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.narrative_potential import NarrativePotentialTransformer
        transformers_to_test.append(("Narrative Potential", NarrativePotentialTransformer, {}, "Core"))
    except: pass
    
    try:
        from transformers.linguistic_advanced import LinguisticPatternsTransformer
        transformers_to_test.append(("Linguistic Patterns", LinguisticPatternsTransformer, {}, "Linguistic"))
    except: pass
    
    try:
        from transformers.emotional_resonance import EmotionalResonanceTransformer
        transformers_to_test.append(("Emotional Resonance", EmotionalResonanceTransformer, {}, "Emotional"))
    except: pass
    
    try:
        from transformers.authenticity import AuthenticityTransformer
        transformers_to_test.append(("Authenticity", AuthenticityTransformer, {}, "Emotional"))
    except: pass
    
    try:
        from transformers.conflict_tension import ConflictTensionTransformer
        transformers_to_test.append(("Conflict Tension", ConflictTensionTransformer, {}, "Emotional"))
    except: pass
    
    try:
        from transformers.suspense_mystery import SuspenseMysteryTransformer
        transformers_to_test.append(("Suspense Mystery", SuspenseMysteryTransformer, {}, "Emotional"))
    except: pass
    
    try:
        from transformers.phonetic import PhoneticTransformer
        transformers_to_test.append(("Phonetic", PhoneticTransformer, {}, "Nominative"))
    except: pass
    
    try:
        from transformers.ensemble import EnsembleNarrativeTransformer
        transformers_to_test.append(("Ensemble Narrative", EnsembleNarrativeTransformer, {}, "Ensemble"))
    except: pass
    
    try:
        from transformers.statistical import StatisticalTransformer
        transformers_to_test.append(("Statistical", StatisticalTransformer, {}, "Statistical"))
    except: pass
    
    try:
        from transformers.information_theory import InformationTheoryTransformer
        transformers_to_test.append(("Information Theory", InformationTheoryTransformer, {}, "Information"))
    except: pass
    
    try:
        from transformers.cognitive_fluency import CognitiveFluencyTransformer
        transformers_to_test.append(("Cognitive Fluency", CognitiveFluencyTransformer, {}, "Cognitive"))
    except: pass
    
    try:
        from transformers.temporal_evolution import TemporalEvolutionTransformer
        transformers_to_test.append(("Temporal Evolution", TemporalEvolutionTransformer, {}, "Temporal"))
    except: pass
    
    try:
        from transformers.optics import OpticsTransformer
        transformers_to_test.append(("Optics", OpticsTransformer, {}, "Framing"))
    except: pass
    
    try:
        from transformers.framing import FramingTransformer
        transformers_to_test.append(("Framing", FramingTransformer, {}, "Framing"))
    except: pass
    
    try:
        from transformers.coupling_strength import CouplingStrengthTransformer
        transformers_to_test.append(("Coupling Strength", CouplingStrengthTransformer, {}, "Physics"))
    except: pass
    
    try:
        from transformers.narrative_mass import NarrativeMassTransformer
        transformers_to_test.append(("Narrative Mass", NarrativeMassTransformer, {}, "Physics"))
    except: pass
    
    try:
        from transformers.gravitational_features import GravitationalFeaturesTransformer
        transformers_to_test.append(("Gravitational Features", GravitationalFeaturesTransformer, {}, "Physics"))
    except: pass
    
    try:
        from transformers.awareness_resistance import AwarenessResistanceTransformer
        transformers_to_test.append(("Awareness Resistance", AwarenessResistanceTransformer, {}, "Physics"))
    except: pass
    
    try:
        from transformers.fundamental_constraints import FundamentalConstraintsTransformer
        transformers_to_test.append(("Fundamental Constraints", FundamentalConstraintsTransformer, {}, "Physics"))
    except: pass
    
    try:
        from transformers.alpha import AlphaTransformer
        transformers_to_test.append(("Alpha", AlphaTransformer, {}, "Mathematical"))
    except: pass
    
    try:
        from transformers.multi_scale import MultiScaleTransformer
        transformers_to_test.append(("Multi-Scale", MultiScaleTransformer, {}, "Multi"))
    except: pass
    
    try:
        from transformers.quantitative import QuantitativeTransformer
        transformers_to_test.append(("Quantitative", QuantitativeTransformer, {}, "Quantitative"))
    except: pass
    
    try:
        from transformers.relational import RelationalValueTransformer
        transformers_to_test.append(("Relational Value", RelationalValueTransformer, {}, "Relational"))
    except: pass
    
    # Additional transformers
    try:
        from transformers.universal_nominative import UniversalNominativeTransformer
        transformers_to_test.append(("Universal Nominative", UniversalNominativeTransformer, {}, "Nominative"))
    except: pass
    
    try:
        from transformers.hierarchical_nominative import HierarchicalNominativeTransformer
        transformers_to_test.append(("Hierarchical Nominative", HierarchicalNominativeTransformer, {}, "Nominative"))
    except: pass
    
    try:
        from transformers.nominative_richness import NominativeRichnessTransformer
        transformers_to_test.append(("Nominative Richness", NominativeRichnessTransformer, {}, "Nominative"))
    except: pass
    
    try:
        from transformers.expertise_authority import ExpertiseAuthorityTransformer
        transformers_to_test.append(("Expertise Authority", ExpertiseAuthorityTransformer, {}, "Authority"))
    except: pass
    
    try:
        from transformers.cultural_context import CulturalContextTransformer
        transformers_to_test.append(("Cultural Context", CulturalContextTransformer, {}, "Cultural"))
    except: pass
    
    try:
        from transformers.namespace_ecology import NamespaceEcologyTransformer
        transformers_to_test.append(("Namespace Ecology", NamespaceEcologyTransformer, {}, "Information"))
    except: pass
    
    try:
        from transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
        transformers_to_test.append(("Anticipatory Communication", AnticipatoryCommunicationTransformer, {}, "Communication"))
    except: pass
    
    try:
        from transformers.discoverability import DiscoverabilityTransformer
        transformers_to_test.append(("Discoverability", DiscoverabilityTransformer, {}, "Cognitive"))
    except: pass
    
    try:
        from transformers.social_status import SocialStatusTransformer
        transformers_to_test.append(("Social Status", SocialStatusTransformer, {}, "Social"))
    except: pass
    
    try:
        from transformers.context_pattern import ContextPatternTransformer
        transformers_to_test.append(("Context Pattern", ContextPatternTransformer, {'min_samples': 50, 'max_patterns': 30}, "Pattern"))
    except: pass
    
    print(f"\n✓ Found {len(transformers_to_test)} transformers to profile\n")
    
    # Profile each
    results = []
    for i, (name, cls, kwargs, category) in enumerate(transformers_to_test, 1):
        print(f"\n[{i}/{len(transformers_to_test)}] ", end="")
        result = profiler.profile_transformer(
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
    
    # Analysis
    print("\n\n" + "="*80)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    df_working = df[df['error'].isna()].copy()
    
    if len(df_working) == 0:
        print("\n✗ No transformers completed successfully")
        return
    
    print(f"\n✓ {len(df_working)} transformers completed successfully")
    print(f"✗ {len(df) - len(df_working)} transformers failed")
    
    # Sort by speed
    df_working = df_working.sort_values('total_time', ascending=False)
    
    # Print slowest transformers
    print("\n" + "="*80)
    print("SLOWEST TRANSFORMERS (Top 15)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Transformer':<35} {'Time':<10} {'Mem(MB)':<10} {'Features':<10} {'Priority':<10}")
    print("-"*90)
    
    for i, (_, row) in enumerate(df_working.head(15).iterrows(), 1):
        print(f"{i:<5} {row['name']:<35} {row['total_time']:>6.1f}s   "
              f"{row['memory_delta_mb']:>6.1f}    {row['features_generated']:>6}     {row['reformulation_priority']:<10}")
    
    # Print fastest transformers
    print("\n" + "="*80)
    print("FASTEST TRANSFORMERS (Top 15)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Transformer':<35} {'Time':<10} {'Mem(MB)':<10} {'Features':<10} {'Accuracy':<10}")
    print("-"*90)
    
    df_fast = df_working.sort_values('total_time', ascending=True)
    for i, (_, row) in enumerate(df_fast.head(15).iterrows(), 1):
        print(f"{i:<5} {row['name']:<35} {row['total_time']:>6.1f}s   "
              f"{row['memory_delta_mb']:>6.1f}    {row['features_generated']:>6}     {row['test_accuracy']:>5.1%}")
    
    # Reformulation priorities
    print("\n" + "="*80)
    print("REFORMULATION PRIORITIES")
    print("="*80)
    
    for priority in ["CRITICAL", "HIGH", "MEDIUM"]:
        df_priority = df_working[df_working['reformulation_priority'] == priority]
        if len(df_priority) > 0:
            print(f"\n{priority} Priority ({len(df_priority)} transformers):")
            for _, row in df_priority.iterrows():
                print(f"  • {row['name']}: {row['total_time']:.1f}s ({row['features_generated']} features)")
    
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
    
    print(f"\nMemory Usage:")
    print(f"  Mean: {df_working['memory_delta_mb'].mean():.1f}MB")
    print(f"  Median: {df_working['memory_delta_mb'].median():.1f}MB")
    print(f"  Max: {df_working['memory_delta_mb'].max():.1f}MB")
    
    print(f"\nFeature Generation:")
    print(f"  Mean: {df_working['features_generated'].mean():.0f} features")
    print(f"  Median: {df_working['features_generated'].median():.0f} features")
    print(f"  Max: {df_working['features_generated'].max():.0f} features")
    
    print(f"\nProcessing Speed:")
    print(f"  Mean: {df_working['samples_per_second'].mean():.0f} samples/sec")
    print(f"  Median: {df_working['samples_per_second'].median():.0f} samples/sec")
    
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
            'mean_memory_mb': float(df_working['memory_delta_mb'].mean()),
            'max_memory_mb': float(df_working['memory_delta_mb'].max()),
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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nReview {output_path} for detailed results.")
    print(f"Focus reformulation efforts on CRITICAL and HIGH priority transformers.\n")


if __name__ == "__main__":
    main()

