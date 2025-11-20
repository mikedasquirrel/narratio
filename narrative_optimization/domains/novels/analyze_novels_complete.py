"""
Novels Complete Analysis - Full Narrative Assemblage

Comprehensive analysis with ALL nominative and phonetic transformers:
- NominativeAnalysisTransformer
- PhoneticTransformer
- UniversalNominativeTransformer
- HierarchicalNominativeTransformer
- NominativeInteractionTransformer
- PureNominativePredictorTransformer
- SocialStatusTransformer
- NamespaceEcologyTransformer
- NominativeRichnessTransformer
- GravitationalFeaturesTransformer

Plus all other transformers for complete narrative assemblage analysis.
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader

# Import ALL transformers - especially nominative and phonetic
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.phonetic import PhoneticTransformer
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from narrative_optimization.src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from narrative_optimization.src.transformers.relational import RelationalValueTransformer
from narrative_optimization.src.transformers.ensemble import EnsembleNarrativeTransformer

# Nominative transformers (CRITICAL)
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

# Other transformers
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

# Phase 7 transformers
from narrative_optimization.src.transformers.awareness_resistance import AwarenessResistanceTransformer
from narrative_optimization.src.transformers.fundamental_constraints import FundamentalConstraintsTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def main():
    """Run complete novels analysis with all transformers."""
    print("="*80)
    print("NOVELS COMPLETE ANALYSIS - FULL NARRATIVE ASSEMBLAGE")
    print("="*80)
    print("\nIncluding ALL nominative and phonetic transformers")
    print("Focus: Character ensembles, roles, plot relative to characters")
    
    # Load data
    print("\n[1/10] Loading novels dataset...")
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    
    if not novels:
        print("❌ No novels loaded!")
        return
    
    print(f"✓ Loaded {len(novels)} novels")
    
    # Extract texts and outcomes
    texts = [n['full_narrative'] for n in novels]
    outcomes = np.array([n['success_score'] for n in novels])
    
    # Extract nominatives for specialized transformers
    author_names = [n.get('author_name', '') for n in novels]
    book_titles = [n.get('book_title', '') for n in novels]
    character_names_list = [n.get('character_names', []) for n in novels]
    all_nominatives_list = [n.get('all_nominatives', []) for n in novels]
    
    print(f"\n[2/10] Nominative Statistics:")
    print(f"  Authors: {len(set(author_names))} unique")
    print(f"  Total characters: {sum(len(chars) for chars in character_names_list)}")
    print(f"  Avg characters per novel: {np.mean([len(chars) for chars in character_names_list]):.1f}")
    print(f"  Max characters: {max(len(chars) for chars in character_names_list)}")
    
    # Initialize ALL transformers
    print(f"\n[3/10] Initializing transformers...")
    print(f"  Including ALL nominative and phonetic transformers")
    
    transformers = [
        # Statistical baseline
        ('statistical', StatisticalTransformer(max_features=100)),
        
        # Core narrative transformers
        ('nominative', NominativeAnalysisTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('relational', RelationalValueTransformer()),
        ('ensemble', EnsembleNarrativeTransformer()),
        
        # PHONETIC TRANSFORMER (CRITICAL)
        ('phonetic', PhoneticTransformer()),
        
        # ALL NOMINATIVE TRANSFORMERS (CRITICAL)
        ('universal_nominative', UniversalNominativeTransformer()),
        ('hierarchical_nominative', HierarchicalNominativeTransformer()),
        ('nominative_interaction', NominativeInteractionTransformer()),
        ('pure_nominative', PureNominativePredictorTransformer()),
        ('social_status', SocialStatusTransformer()),
        ('namespace_ecology', NamespaceEcologyTransformer()),
        ('nominative_richness', NominativeRichnessTransformer()),
        ('gravitational_features', GravitationalFeaturesTransformer()),
        
        # Other transformers
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
        
        # Phase 7 transformers
        ('awareness_resistance', AwarenessResistanceTransformer()),
        ('fundamental_constraints', FundamentalConstraintsTransformer()),
    ]
    
    print(f"✓ Initialized {len(transformers)} transformers")
    print(f"  Nominative transformers: 9")
    print(f"  Phonetic transformers: 1")
    print(f"  Total transformers: {len(transformers)}")
    
    # Extract features
    print(f"\n[4/10] Extracting features from all transformers...")
    all_features = []
    feature_names_list = []
    transformer_stats = {}
    
    for trans_name, transformer in transformers:
        try:
            print(f"  Processing {trans_name}...", end=' ', flush=True)
            
            # Fit and transform
            if hasattr(transformer, 'fit_transform'):
                features = transformer.fit_transform(texts)
            else:
                transformer.fit(texts)
                features = transformer.transform(texts)
            
            # Handle different feature shapes
            if hasattr(features, 'toarray'):
                features = features.toarray()
            elif isinstance(features, np.ndarray):
                if features.ndim == 1:
                    features = features.reshape(-1, 1)
            
            # Get feature names if available
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out()
                    feature_names_list.extend([f"{trans_name}_{n}" for n in names])
                except:
                    feature_names_list.extend([f"{trans_name}_{i}" for i in range(features.shape[1])])
            else:
                feature_names_list.extend([f"{trans_name}_{i}" for i in range(features.shape[1])])
            
            all_features.append(features)
            transformer_stats[trans_name] = {
                'n_features': features.shape[1],
                'mean': np.mean(features),
                'std': np.std(features)
            }
            
            print(f"✓ ({features.shape[1]} features)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Combine all features
    print(f"\n[5/10] Combining features...")
    if all_features:
        X = np.hstack(all_features)
        print(f"✓ Combined feature matrix: {X.shape}")
        print(f"  Total features: {X.shape[1]}")
    else:
        print("❌ No features extracted!")
        return
    
    # Train model
    print(f"\n[6/10] Training predictive model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, outcomes)
    
    # Evaluate
    predictions = model.predict(X)
    r2 = r2_score(outcomes, predictions)
    rmse = np.sqrt(mean_squared_error(outcomes, predictions))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, outcomes, cv=5, scoring='r2')
    
    print(f"✓ Model performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Feature importance
    print(f"\n[7/10] Analyzing feature importance...")
    importances = model.feature_importances_
    
    # Group by transformer
    transformer_importance = {}
    idx = 0
    for trans_name, stats in transformer_stats.items():
        n_features = stats['n_features']
        transformer_importance[trans_name] = {
            'total_importance': np.sum(importances[idx:idx+n_features]),
            'mean_importance': np.mean(importances[idx:idx+n_features]),
            'max_importance': np.max(importances[idx:idx+n_features]),
            'n_features': n_features
        }
        idx += n_features
    
    # Sort by importance
    sorted_transformers = sorted(
        transformer_importance.items(),
        key=lambda x: x[1]['total_importance'],
        reverse=True
    )
    
    print(f"\nTop 10 transformers by importance:")
    for i, (name, stats) in enumerate(sorted_transformers[:10], 1):
        print(f"  {i:2d}. {name:25s} - {stats['total_importance']:.4f} (mean: {stats['mean_importance']:.6f})")
    
    # Nominative transformer analysis
    print(f"\n[8/10] Nominative transformer analysis:")
    nominative_transformers = [
        'nominative', 'phonetic', 'universal_nominative', 'hierarchical_nominative',
        'nominative_interaction', 'pure_nominative', 'social_status', 'namespace_ecology',
        'nominative_richness', 'gravitational_features'
    ]
    
    nominative_total = sum(
        transformer_importance.get(name, {}).get('total_importance', 0)
        for name in nominative_transformers
    )
    
    print(f"  Total importance from nominative/phonetic transformers: {nominative_total:.4f}")
    print(f"  Percentage of total: {nominative_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance):.1%}")
    
    # Save results
    print(f"\n[9/10] Saving results...")
    results = {
        'domain': 'novels',
        'n_samples': len(novels),
        'n_features': X.shape[1],
        'n_transformers': len(transformers),
        'performance': {
            'r2': float(r2),
            'rmse': float(rmse),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std())
        },
        'transformer_importance': {
            name: {
                'total': float(stats['total_importance']),
                'mean': float(stats['mean_importance']),
                'max': float(stats['max_importance']),
                'n_features': stats['n_features']
            }
            for name, stats in transformer_importance.items()
        },
        'nominative_analysis': {
            'total_importance': float(nominative_total),
            'percentage': float(nominative_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance))
        },
        'nominative_statistics': {
            'unique_authors': len(set(author_names)),
            'total_characters': sum(len(chars) for chars in character_names_list),
            'avg_characters_per_novel': float(np.mean([len(chars) for chars in character_names_list])),
            'max_characters': max(len(chars) for chars in character_names_list)
        }
    }
    
    output_path = Path(__file__).parent / 'novels_complete_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print(f"\n[10/10] Analysis Summary")
    print("="*80)
    print(f"✓ Processed {len(novels)} novels")
    print(f"✓ Extracted {X.shape[1]} features from {len(transformers)} transformers")
    print(f"✓ Model R²: {r2:.4f}")
    print(f"✓ Nominative/phonetic transformers contribute {nominative_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance):.1%} of importance")
    print("="*80)


if __name__ == '__main__':
    main()

