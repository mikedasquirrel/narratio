"""
Feature Attribution Analysis

Uses SHAP values and ablation studies to determine which transformers
contribute most to predictions.
"""

import json
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.domains.novels.data_loader import NovelsDataLoader
from narrative_optimization.domains.novels.transformer_interactions import get_transformers

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    transformer_names: List[str],
    transformer_stats: Dict[str, Dict],
    baseline_r2: float
) -> Dict[str, Any]:
    """
    Perform ablation study: remove one transformer at a time.
    
    Parameters
    ----------
    X : np.ndarray
        Full feature matrix
    y : np.ndarray
        Target outcomes
    transformer_names : list
        Transformer names
    transformer_stats : dict
        Feature ranges for each transformer
    baseline_r2 : float
        Baseline R² with all transformers
    
    Returns
    -------
    ablation_results : dict
        Ablation study results
    """
    print("\nPerforming ablation study...")
    
    # Get feature ranges
    feature_ranges = {}
    idx = 0
    for name in transformer_names:
        if name in transformer_stats:
            n_features = transformer_stats[name]['n_features']
            feature_ranges[name] = (idx, idx + n_features)
            idx += n_features
    
    ablation_results = {}
    
    for name in transformer_names:
        if name not in feature_ranges:
            continue
        
        print(f"  Removing {name}...", end=' ', flush=True)
        
        # Create feature matrix without this transformer
        start, end = feature_ranges[name]
        X_ablated = np.hstack([
            X[:, :start],
            X[:, end:]
        ])
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_ablated, y)
        predictions = model.predict(X_ablated)
        r2_ablated = r2_score(y, predictions)
        
        # Calculate impact
        impact = baseline_r2 - r2_ablated
        
        ablation_results[name] = {
            'r2_without': float(r2_ablated),
            'impact': float(impact),
            'relative_impact': float(impact / baseline_r2) if baseline_r2 > 0 else 0.0
        }
        
        print(f"✓ Impact: {impact:.4f}")
    
    return ablation_results


def permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    transformer_names: List[str],
    transformer_stats: Dict[str, Dict],
    model: RandomForestRegressor,
    n_iterations: int = 10
) -> Dict[str, Any]:
    """
    Calculate permutation importance for each transformer.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target outcomes
    transformer_names : list
        Transformer names
    transformer_stats : dict
        Feature ranges
    model : RandomForestRegressor
        Trained model
    n_iterations : int
        Number of permutation iterations
    
    Returns
    -------
    importance_results : dict
        Permutation importance results
    """
    print("\nCalculating permutation importance...")
    
    baseline_score = r2_score(y, model.predict(X))
    
    # Get feature ranges
    feature_ranges = {}
    idx = 0
    for name in transformer_names:
        if name in transformer_stats:
            n_features = transformer_stats[name]['n_features']
            feature_ranges[name] = (idx, idx + n_features)
            idx += n_features
    
    importance_results = {}
    
    for name in transformer_names:
        if name not in feature_ranges:
            continue
        
        print(f"  Permuting {name}...", end=' ', flush=True)
        
        start, end = feature_ranges[name]
        importances = []
        
        for _ in range(n_iterations):
            X_permuted = X.copy()
            # Permute features for this transformer
            perm_indices = np.random.permutation(len(y))
            X_permuted[:, start:end] = X_permuted[perm_indices, start:end]
            
            score = r2_score(y, model.predict(X_permuted))
            importances.append(baseline_score - score)
        
        importance_results[name] = {
            'mean_importance': float(np.mean(importances)),
            'std_importance': float(np.std(importances)),
            'min_importance': float(np.min(importances)),
            'max_importance': float(np.max(importances))
        }
        
        print(f"✓ Mean: {np.mean(importances):.4f}")
    
    return importance_results


def main():
    """Run feature attribution analysis."""
    print("="*80)
    print("FEATURE ATTRIBUTION ANALYSIS")
    print("="*80)
    print("\nUsing ablation studies and permutation importance")
    
    # Load data
    print("\n[1/6] Loading data...")
    loader = NovelsDataLoader()
    novels = loader.load_full_dataset()
    
    texts = [n['full_narrative'] for n in novels]
    outcomes = np.array([n['success_score'] for n in novels])
    
    print(f"✓ Loaded {len(novels)} novels")
    
    # Extract features
    print("\n[2/6] Extracting features...")
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
    
    # Train baseline model
    print("\n[3/6] Training baseline model...")
    baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    baseline_model.fit(X, outcomes)
    baseline_predictions = baseline_model.predict(X)
    baseline_r2 = r2_score(outcomes, baseline_predictions)
    
    print(f"✓ Baseline R²: {baseline_r2:.4f}")
    
    # Ablation study
    print("\n[4/6] Ablation study...")
    ablation_results = ablation_study(X, outcomes, transformer_names, transformer_stats, baseline_r2)
    
    # Sort by impact
    sorted_ablation = sorted(
        ablation_results.items(),
        key=lambda x: x[1]['impact'],
        reverse=True
    )
    
    print("\nTop 10 transformers by ablation impact:")
    for i, (name, results) in enumerate(sorted_ablation[:10], 1):
        print(f"  {i:2d}. {name:25s} - Impact: {results['impact']:.4f} ({results['relative_impact']:.1%})")
    
    # Permutation importance
    print("\n[5/6] Permutation importance...")
    perm_results = permutation_importance(X, outcomes, transformer_names, transformer_stats, baseline_model)
    
    sorted_perm = sorted(
        perm_results.items(),
        key=lambda x: x[1]['mean_importance'],
        reverse=True
    )
    
    print("\nTop 10 transformers by permutation importance:")
    for i, (name, results) in enumerate(sorted_perm[:10], 1):
        print(f"  {i:2d}. {name:25s} - Importance: {results['mean_importance']:.4f} (±{results['std_importance']:.4f})")
    
    # Save results
    print("\n[6/6] Saving results...")
    output_path = Path(__file__).parent / 'feature_attribution.json'
    
    results = {
        'baseline_r2': float(baseline_r2),
        'ablation_study': {
            name: {
                'r2_without': results['r2_without'],
                'impact': results['impact'],
                'relative_impact': results['relative_impact']
            }
            for name, results in ablation_results.items()
        },
        'permutation_importance': {
            name: {
                'mean': results['mean_importance'],
                'std': results['std_importance'],
                'min': results['min_importance'],
                'max': results['max_importance']
            }
            for name, results in perm_results.items()
        },
        'top_transformers': {
            'ablation': [name for name, _ in sorted_ablation[:10]],
            'permutation': [name for name, _ in sorted_perm[:10]]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {output_path}")
    print("\n" + "="*80)
    print("Feature Attribution Analysis Complete")
    print("="*80)


if __name__ == '__main__':
    main()

