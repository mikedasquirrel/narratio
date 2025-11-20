"""
Combined Books Analysis - Novels + Nonfiction

Cross-domain analysis comparing novels vs nonfiction with all transformers.
Tests universal vs genre-specific narrative patterns.
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader
from narrative_optimization.domains.nonfiction.data_loader import NonfictionDataLoader

# Import ALL transformers
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.phonetic import PhoneticTransformer
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from narrative_optimization.src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from narrative_optimization.src.transformers.relational import RelationalValueTransformer
from narrative_optimization.src.transformers.ensemble import EnsembleNarrativeTransformer

# All nominative transformers
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
from narrative_optimization.src.transformers.awareness_resistance import AwarenessResistanceTransformer
from narrative_optimization.src.transformers.fundamental_constraints import FundamentalConstraintsTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def main():
    """Run combined analysis comparing novels and nonfiction."""
    print("="*80)
    print("COMBINED BOOKS ANALYSIS - NOVELS + NONFICTION")
    print("="*80)
    print("\nCross-domain comparison with all transformers")
    
    # Load both datasets
    print("\n[1/10] Loading datasets...")
    novels_loader = NovelsDataLoader()
    nonfiction_loader = NonfictionDataLoader()
    
    novels = novels_loader.load_full_dataset()
    nonfiction = nonfiction_loader.load_full_dataset()
    
    print(f"✓ Loaded {len(novels)} novels")
    print(f"✓ Loaded {len(nonfiction)} nonfiction books")
    
    # Combine datasets
    all_books = []
    for novel in novels:
        book = novel.copy()
        book['book_type'] = 0  # 0 = novel
        all_books.append(book)
    
    for book in nonfiction:
        book['book_type'] = 1  # 1 = nonfiction
        all_books.append(book)
    
    print(f"✓ Combined dataset: {len(all_books)} total books")
    
    # Extract texts and outcomes
    texts = [b['full_narrative'] for b in all_books]
    outcomes = np.array([b['success_score'] for b in all_books])
    book_types = np.array([b['book_type'] for b in all_books])
    
    # Extract nominatives
    author_names = [b.get('author_name', '') for b in all_books]
    
    print(f"\n[2/10] Dataset Statistics:")
    print(f"  Novels: {np.sum(book_types == 0)}")
    print(f"  Nonfiction: {np.sum(book_types == 1)}")
    print(f"  Unique authors: {len(set(author_names))}")
    
    # Initialize transformers
    print(f"\n[3/10] Initializing transformers...")
    transformers = [
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
    
    print(f"✓ Initialized {len(transformers)} transformers")
    
    # Extract features
    print(f"\n[4/10] Extracting features...")
    all_features = []
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
            
            all_features.append(features)
            transformer_stats[trans_name] = {'n_features': features.shape[1]}
            print(f"✓ ({features.shape[1]} features)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    # Combine features
    print(f"\n[5/10] Combining features...")
    if all_features:
        X = np.hstack(all_features)
        # Add book_type as feature
        X = np.hstack([X, book_types.reshape(-1, 1)])
        print(f"✓ Combined feature matrix: {X.shape}")
    else:
        print("❌ No features extracted!")
        return
    
    # Train model
    print(f"\n[6/10] Training predictive model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, outcomes)
    
    predictions = model.predict(X)
    r2 = r2_score(outcomes, predictions)
    rmse = np.sqrt(mean_squared_error(outcomes, predictions))
    cv_scores = cross_val_score(model, X, outcomes, cv=5, scoring='r2')
    
    print(f"✓ Model performance:")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  CV R² (mean): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Compare novels vs nonfiction
    print(f"\n[7/10] Comparing novels vs nonfiction...")
    novels_mask = book_types == 0
    nonfiction_mask = book_types == 1
    
    novels_r2 = r2_score(outcomes[novels_mask], predictions[novels_mask])
    nonfiction_r2 = r2_score(outcomes[nonfiction_mask], predictions[nonfiction_mask])
    
    print(f"  Novels R²: {novels_r2:.4f}")
    print(f"  Nonfiction R²: {nonfiction_r2:.4f}")
    print(f"  Difference: {abs(novels_r2 - nonfiction_r2):.4f}")
    
    # Feature importance
    print(f"\n[8/10] Analyzing feature importance...")
    importances = model.feature_importances_
    
    # Book type importance
    book_type_importance = importances[-1]
    print(f"  Book type (novel vs nonfiction) importance: {book_type_importance:.6f}")
    
    # Transformer importance
    transformer_importance = {}
    idx = 0
    for trans_name, stats in transformer_stats.items():
        n_features = stats['n_features']
        transformer_importance[trans_name] = {
            'total_importance': np.sum(importances[idx:idx+n_features]),
            'mean_importance': np.mean(importances[idx:idx+n_features]),
            'n_features': n_features
        }
        idx += n_features
    
    sorted_transformers = sorted(
        transformer_importance.items(),
        key=lambda x: x[1]['total_importance'],
        reverse=True
    )
    
    print(f"\nTop 10 transformers by importance:")
    for i, (name, stats) in enumerate(sorted_transformers[:10], 1):
        print(f"  {i:2d}. {name:25s} - {stats['total_importance']:.4f}")
    
    # Cross-domain validation
    print(f"\n[9/10] Cross-domain validation...")
    # Train on novels, test on nonfiction
    novels_X = X[novels_mask]
    novels_y = outcomes[novels_mask]
    nonfiction_X = X[nonfiction_mask]
    nonfiction_y = outcomes[nonfiction_mask]
    
    novels_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    novels_model.fit(novels_X, novels_y)
    nonfiction_predictions = novels_model.predict(nonfiction_X)
    cross_domain_r2 = r2_score(nonfiction_y, nonfiction_predictions)
    
    print(f"  Train on novels, test on nonfiction R²: {cross_domain_r2:.4f}")
    
    # Save results
    print(f"\n[10/10] Saving results...")
    results = {
        'domain': 'books_combined',
        'n_samples': len(all_books),
        'n_novels': int(np.sum(book_types == 0)),
        'n_nonfiction': int(np.sum(book_types == 1)),
        'n_features': X.shape[1],
        'performance': {
            'overall_r2': float(r2),
            'novels_r2': float(novels_r2),
            'nonfiction_r2': float(nonfiction_r2),
            'cross_domain_r2': float(cross_domain_r2),
            'rmse': float(rmse),
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std())
        },
        'book_type_importance': float(book_type_importance),
        'transformer_importance': {
            name: {
                'total': float(stats['total_importance']),
                'mean': float(stats['mean_importance']),
                'n_features': stats['n_features']
            }
            for name, stats in transformer_importance.items()
        }
    }
    
    output_path = Path(__file__).parent / 'combined_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    
    # Summary
    print(f"\n" + "="*80)
    print("Combined Analysis Summary")
    print("="*80)
    print(f"✓ Processed {len(all_books)} books ({np.sum(book_types == 0)} novels, {np.sum(book_types == 1)} nonfiction)")
    print(f"✓ Overall R²: {r2:.4f}")
    print(f"✓ Novels R²: {novels_r2:.4f}")
    print(f"✓ Nonfiction R²: {nonfiction_r2:.4f}")
    print(f"✓ Cross-domain R²: {cross_domain_r2:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()

