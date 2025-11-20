"""
Transformer Interaction Analysis

Analyzes how transformers work together:
- Complementary: Capture different aspects
- Redundant: Overlapping information
- Synergistic: Amplify each other
- Antagonistic: Conflict with each other
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader

# Import transformers directly (same as analyze_novels_complete.py)
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.phonetic import PhoneticTransformer
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from narrative_optimization.src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from narrative_optimization.src.transformers.relational import RelationalValueTransformer
from narrative_optimization.src.transformers.ensemble import EnsembleNarrativeTransformer
from narrative_optimization.src.transformers.universal_nominative import UniversalNominativeTransformer
from narrative_optimization.src.transformers.hierarchical_nominative import (
    HierarchicalNominativeTransformer,
    NominativeInteractionTransformer,
    PureNominativePredictorTransformer
)
from narrative_optimization.src.transformers.social_status import SocialStatusTransformer
from narrative_optimization.src.transformers.namespace_ecology import NamespaceEcologyTransformer
from narrative_optimization.src.transformers.nominative_richness import NominativeRichnessTransformer
from narrative_optimization.src.transformers.gravitational_features import GravitationalFeaturesTransformer
from narrative_optimization.src.transformers.optics import OpticsTransformer
from narrative_optimization.src.transformers.framing import FramingTransformer
from narrative_optimization.src.transformers.temporal_evolution import TemporalEvolutionTransformer
from narrative_optimization.src.transformers.information_theory import InformationTheoryTransformer
from narrative_optimization.src.transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
from narrative_optimization.src.transformers.cognitive_fluency import CognitiveFluencyTransformer
from narrative_optimization.src.transformers.emotional_resonance import EmotionalResonanceTransformer
from narrative_optimization.src.transformers.authenticity import AuthenticityTransformer
from narrative_optimization.src.transformers.conflict_tension import ConflictTensionTransformer
from narrative_optimization.src.transformers.expertise_authority import ExpertiseAuthorityTransformer
from narrative_optimization.src.transformers.cultural_context import CulturalContextTransformer
from narrative_optimization.src.transformers.suspense_mystery import SuspenseMysteryTransformer
from narrative_optimization.src.transformers.multi_scale import MultiScaleTransformer
from narrative_optimization.src.transformers.coupling_strength import CouplingStrengthTransformer
from narrative_optimization.src.transformers.narrative_mass import NarrativeMassTransformer
from narrative_optimization.src.transformers.awareness_resistance import AwarenessResistanceTransformer
from narrative_optimization.src.transformers.fundamental_constraints import FundamentalConstraintsTransformer

def get_transformers():
    """Get list of transformers (same as analyze_novels_complete.py)."""
    return [
        ('statistical', StatisticalTransformer(max_features=100)),
        ('nominative', NominativeAnalysisTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('relational', RelationalValueTransformer()),
        ('ensemble', EnsembleNarrativeTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('universal_nominative', UniversalNominativeTransformer()),
        ('hierarchical_nominative', HierarchicalNominativeTransformer()),
        ('nominative_interaction', NominativeInteractionTransformer()),
        ('pure_nominative', PureNominativePredictorTransformer()),
        ('social_status', SocialStatusTransformer()),
        ('namespace_ecology', NamespaceEcologyTransformer()),
        ('nominative_richness', NominativeRichnessTransformer()),
        ('gravitational_features', GravitationalFeaturesTransformer()),
        ('optics', OpticsTransformer()),
        ('framing', FramingTransformer()),
        ('temporal', TemporalEvolutionTransformer()),
        ('information_theory', InformationTheoryTransformer()),
        ('anticipatory', AnticipatoryCommunicationTransformer()),
        ('cognitive_fluency', CognitiveFluencyTransformer()),
        ('emotional', EmotionalResonanceTransformer()),
        ('authenticity', AuthenticityTransformer()),
        ('conflict', ConflictTensionTransformer()),
        ('expertise', ExpertiseAuthorityTransformer()),
        ('cultural', CulturalContextTransformer()),
        ('suspense', SuspenseMysteryTransformer()),
        ('multi_scale', MultiScaleTransformer()),
        ('coupling_strength', CouplingStrengthTransformer()),
        ('narrative_mass', NarrativeMassTransformer()),
        ('awareness_resistance', AwarenessResistanceTransformer()),
        ('fundamental_constraints', FundamentalConstraintsTransformer()),
    ]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def analyze_transformer_interactions(
    X: np.ndarray,
    y: np.ndarray,
    transformer_names: List[str],
    transformer_stats: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze interactions between transformers.
    
    Parameters
    ----------
    X : np.ndarray
        Combined feature matrix
    y : np.ndarray
        Target outcomes
    transformer_names : list
        Names of transformers
    transformer_stats : dict
        Statistics for each transformer (n_features)
    
    Returns
    -------
    interactions : dict
        Interaction analysis results
    """
    print("\nAnalyzing transformer interactions...")
    
    # Extract feature ranges for each transformer
    feature_ranges = {}
    idx = 0
    for name in transformer_names:
        if name in transformer_stats:
            n_features = transformer_stats[name]['n_features']
            feature_ranges[name] = (idx, idx + n_features)
            idx += n_features
    
    # Calculate correlation matrix between transformer outputs
    print("  Calculating correlations between transformers...")
    transformer_outputs = {}
    for name, (start, end) in feature_ranges.items():
        # Average features from this transformer
        transformer_outputs[name] = np.mean(X[:, start:end], axis=1)
    
    # Correlation matrix
    n_transformers = len(transformer_outputs)
    correlation_matrix = np.zeros((n_transformers, n_transformers))
    transformer_list = list(transformer_outputs.keys())
    
    for i, name1 in enumerate(transformer_list):
        for j, name2 in enumerate(transformer_list):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                r, p = pearsonr(transformer_outputs[name1], transformer_outputs[name2])
                correlation_matrix[i, j] = r
    
    # Identify interaction types
    print("  Identifying interaction types...")
    
    # Thresholds
    high_correlation = 0.7
    low_correlation = 0.3
    negative_correlation = -0.3
    
    interactions = {
        'complementary': [],  # Low correlation, both important
        'redundant': [],      # High correlation
        'synergistic': [],    # Moderate correlation, high combined importance
        'antagonistic': []    # Negative correlation
    }
    
    # Calculate individual importances
    individual_importances = {}
    for name in transformer_list:
        start, end = feature_ranges[name]
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X[:, start:end], y)
        individual_importances[name] = r2_score(y, model.predict(X[:, start:end]))
    
    # Analyze pairs
    for i, name1 in enumerate(transformer_list):
        for j, name2 in enumerate(transformer_list[i+1:], i+1):
            corr = correlation_matrix[i, j]
            
            # Combined importance
            start1, end1 = feature_ranges[name1]
            start2, end2 = feature_ranges[name2]
            combined_X = np.hstack([X[:, start1:end1], X[:, start2:end2]])
            combined_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            combined_model.fit(combined_X, y)
            combined_importance = r2_score(y, combined_model.predict(combined_X))
            
            imp1 = individual_importances[name1]
            imp2 = individual_importances[name2]
            expected_combined = max(imp1, imp2)  # Best individual
            
            pair = {
                'transformer1': name1,
                'transformer2': name2,
                'correlation': float(corr),
                'individual1_importance': float(imp1),
                'individual2_importance': float(imp2),
                'combined_importance': float(combined_importance),
                'synergy': float(combined_importance - expected_combined)
            }
            
            # Classify interaction
            if corr < low_correlation and imp1 > 0.1 and imp2 > 0.1:
                interactions['complementary'].append(pair)
            elif corr > high_correlation:
                interactions['redundant'].append(pair)
            elif combined_importance > expected_combined * 1.1:  # 10% improvement
                interactions['synergistic'].append(pair)
            elif corr < negative_correlation:
                interactions['antagonistic'].append(pair)
    
    # Summary statistics
    summary = {
        'n_complementary': len(interactions['complementary']),
        'n_redundant': len(interactions['redundant']),
        'n_synergistic': len(interactions['synergistic']),
        'n_antagonistic': len(interactions['antagonistic']),
        'mean_correlation': float(np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
        'max_correlation': float(np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])),
        'min_correlation': float(np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
    }
    
    return {
        'correlation_matrix': correlation_matrix.tolist(),
        'transformer_names': transformer_list,
        'interactions': interactions,
        'summary': summary,
        'individual_importances': individual_importances
    }


def main():
    """Run transformer interaction analysis."""
    print("="*80)
    print("TRANSFORMER INTERACTION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    
    texts = [n['full_narrative'] for n in novels]
    outcomes = np.array([n['success_score'] for n in novels])
    
    print(f"✓ Loaded {len(novels)} novels")
    
    # Get transformers
    print("\n[2/5] Extracting features...")
    transformers = get_transformers()
    
    all_features = []
    transformer_names = []
    transformer_stats = {}
    
    for trans_name, transformer in transformers:
        try:
            if hasattr(transformer, 'fit_transform'):
                features = transformer.fit_transform(texts)
            else:
                transformer.fit(texts)
                features = transformer.transform(texts)
            
            if hasattr(features, 'toarray'):
                features = features.toarray()
            elif isinstance(features, np.ndarray):
                if features.ndim == 1:
                    features = features.reshape(-1, 1)
            
            all_features.append(features)
            transformer_names.append(trans_name)
            transformer_stats[trans_name] = {'n_features': features.shape[1]}
            
        except Exception as e:
            print(f"  ⚠️  Skipping {trans_name}: {e}")
            continue
    
    X = np.hstack(all_features)
    print(f"✓ Extracted {X.shape[1]} features from {len(transformer_names)} transformers")
    
    # Analyze interactions
    print("\n[3/5] Analyzing interactions...")
    interactions = analyze_transformer_interactions(
        X, outcomes, transformer_names, transformer_stats
    )
    
    # Print results
    print("\n[4/5] Interaction Results:")
    print(f"\nSummary:")
    print(f"  Complementary pairs: {interactions['summary']['n_complementary']}")
    print(f"  Redundant pairs: {interactions['summary']['n_redundant']}")
    print(f"  Synergistic pairs: {interactions['summary']['n_synergistic']}")
    print(f"  Antagonistic pairs: {interactions['summary']['n_antagonistic']}")
    print(f"  Mean correlation: {interactions['summary']['mean_correlation']:.3f}")
    
    print(f"\nTop 5 Synergistic Pairs:")
    sorted_synergistic = sorted(
        interactions['interactions']['synergistic'],
        key=lambda x: x['synergy'],
        reverse=True
    )[:5]
    for i, pair in enumerate(sorted_synergistic, 1):
        print(f"  {i}. {pair['transformer1']} + {pair['transformer2']}")
        print(f"     Synergy: {pair['synergy']:.4f}, Correlation: {pair['correlation']:.3f}")
    
    print(f"\nTop 5 Complementary Pairs:")
    sorted_complementary = sorted(
        interactions['interactions']['complementary'],
        key=lambda x: x['individual1_importance'] + x['individual2_importance'],
        reverse=True
    )[:5]
    for i, pair in enumerate(sorted_complementary, 1):
        print(f"  {i}. {pair['transformer1']} + {pair['transformer2']}")
        print(f"     Correlation: {pair['correlation']:.3f}")
    
    # Save results
    print("\n[5/5] Saving results...")
    output_path = Path(__file__).parent / 'transformer_interactions.json'
    
    # Convert numpy types for JSON
    results = {
        'correlation_matrix': interactions['correlation_matrix'],
        'transformer_names': interactions['transformer_names'],
        'interactions': {
            'complementary': interactions['interactions']['complementary'],
            'redundant': interactions['interactions']['redundant'],
            'synergistic': interactions['interactions']['synergistic'],
            'antagonistic': interactions['interactions']['antagonistic']
        },
        'summary': interactions['summary'],
        'individual_importances': {
            k: float(v) for k, v in interactions['individual_importances'].items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)


if __name__ == '__main__':
    main()

