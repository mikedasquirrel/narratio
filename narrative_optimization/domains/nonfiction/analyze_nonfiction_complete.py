"""
Nonfiction Complete Analysis - Full Narrative Assemblage

Comprehensive analysis with ALL nominative and phonetic transformers.
Focus on author names, key figures, and nonfiction-specific patterns.
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.nonfiction.data_loader import NonfictionDataLoader

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

# Nonfiction-specific transformers
from narrative_optimization.src.transformers.expertise_authority import ExpertiseAuthorityTransformer
from narrative_optimization.src.transformers.authenticity import AuthenticityTransformer
from narrative_optimization.src.transformers.framing import FramingTransformer
from narrative_optimization.src.transformers.information_theory import InformationTheoryTransformer

# Other transformers
from narrative_optimization.src.transformers.optics import OpticsTransformer
from narrative_optimization.src.transformers.temporal_evolution import TemporalEvolutionTransformer
from narrative_optimization.src.transformers.anticipatory_commitment import AnticipatoryCommunicationTransformer
from narrative_optimization.src.transformers.cognitive_fluency import CognitiveFluencyTransformer
from narrative_optimization.src.transformers.emotional_resonance import EmotionalResonanceTransformer
from narrative_optimization.src.transformers.conflict_tension import ConflictTensionTransformer
from narrative_optimization.src.transformers.cultural_context import CulturalContextTransformer
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
    """Run complete nonfiction analysis with all transformers."""
    print("="*80)
    print("NONFICTION COMPLETE ANALYSIS - FULL NARRATIVE ASSEMBLAGE")
    print("="*80)
    print("\nIncluding ALL nominative and phonetic transformers")
    print("Focus: Author names, key figures, nonfiction patterns")
    
    # Load data
    print("\n[1/10] Loading nonfiction dataset...")
    loader = NonfictionDataLoader()
    books = loader.load_full_dataset()
    
    if not books:
        print("❌ No books loaded!")
        return
    
    print(f"✓ Loaded {len(books)} nonfiction books")
    
    # Extract texts and outcomes
    texts = [b['full_narrative'] for b in books]
    outcomes = np.array([b['success_score'] for b in books])
    
    # Extract nominatives
    author_names = [b.get('author_name', '') for b in books]
    book_titles = [b.get('book_title', '') for b in books]
    key_figures_list = [b.get('key_figures', []) for b in books]
    all_nominatives_list = [b.get('all_nominatives', []) for b in books]
    
    print(f"\n[2/10] Nominative Statistics:")
    print(f"  Authors: {len(set(author_names))} unique")
    print(f"  Total key figures: {sum(len(figs) for figs in key_figures_list)}")
    print(f"  Avg key figures per book: {np.mean([len(figs) for figs in key_figures_list]):.1f}")
    
    # Initialize ALL transformers
    print(f"\n[3/10] Initializing transformers...")
    
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
        
        # Nonfiction-specific transformers
        ('expertise', ExpertiseAuthorityTransformer()),
        ('authenticity', AuthenticityTransformer()),
        ('framing', FramingTransformer()),
        ('information_theory', InformationTheoryTransformer()),
        
        # Other transformers
        ('optics', OpticsTransformer()),
        ('temporal', TemporalEvolutionTransformer()),
        ('anticipatory', AnticipatoryCommunicationTransformer()),
        ('cognitive_fluency', CognitiveFluencyTransformer()),
        ('emotional', EmotionalResonanceTransformer()),
        ('conflict', ConflictTensionTransformer()),
        ('cultural', CulturalContextTransformer()),
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
    print(f"  Nonfiction-specific: 4")
    
    # Extract features
    print(f"\n[4/10] Extracting features from all transformers...")
    all_features = []
    feature_names_list = []
    transformer_stats = {}
    
    for trans_name, transformer in transformers:
        try:
            print(f"  Processing {trans_name}...", end=' ', flush=True)
            
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
    
    # Nonfiction-specific analysis
    print(f"\n[9/10] Nonfiction-specific transformer analysis:")
    nonfiction_transformers = ['expertise', 'authenticity', 'framing', 'information_theory']
    nonfiction_total = sum(
        transformer_importance.get(name, {}).get('total_importance', 0)
        for name in nonfiction_transformers
    )
    print(f"  Total importance from nonfiction-specific transformers: {nonfiction_total:.4f}")
    print(f"  Percentage of total: {nonfiction_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance):.1%}")
    
    # Save results
    print(f"\n[10/10] Saving results...")
    results = {
        'domain': 'nonfiction',
        'n_samples': len(books),
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
        'nonfiction_analysis': {
            'total_importance': float(nonfiction_total),
            'percentage': float(nonfiction_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance))
        },
        'nominative_statistics': {
            'unique_authors': len(set(author_names)),
            'total_key_figures': sum(len(figs) for figs in key_figures_list),
            'avg_key_figures_per_book': float(np.mean([len(figs) for figs in key_figures_list]))
        }
    }
    
    output_path = Path(__file__).parent / 'nonfiction_complete_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print(f"\n" + "="*80)
    print("Analysis Summary")
    print("="*80)
    print(f"✓ Processed {len(books)} nonfiction books")
    print(f"✓ Extracted {X.shape[1]} features from {len(transformers)} transformers")
    print(f"✓ Model R²: {r2:.4f}")
    print(f"✓ Nominative/phonetic transformers contribute {nominative_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance):.1%} of importance")
    print(f"✓ Nonfiction-specific transformers contribute {nonfiction_total / sum(transformer_importance[n]['total_importance'] for n in transformer_importance):.1%} of importance")
    print("="*80)


if __name__ == '__main__':
    main()

