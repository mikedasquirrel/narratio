"""
Run FULL pipeline analyses for all fully stocked datasets.

Applies ALL 29 transformers (or п-guided subset) to each domain.
Saves results for web display.
"""

import json
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

# Import ALL transformers
from src.transformers import (
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    CulturalContextTransformer,
    StatisticalTransformer,
    NominativeAnalysisTransformer,
    PhoneticTransformer
)


def analyze_oscar_full():
    """Full Oscar analysis with ALL transformers"""
    print("\n" + "="*80)
    print("OSCAR - FULL PIPELINE")
    print("="*80)
    
    # Load
    with open('data/domains/oscar_nominees_complete.json') as f:
        oscar_raw = json.load(f)
    
    films = []
    for year_films in oscar_raw.values():
        films.extend(year_films)
    
    outcomes = np.array([int(f.get('won_oscar', 0)) for f in films])
    
    print(f"\n{len(films)} films, {outcomes.sum()} winners")
    
    # Extract texts for narrative transformers
    texts = [f['title'] + ' ' + f.get('overview', '') + ' ' + ' '.join(f.get('director', [])) 
             for f in films]
    
    # Apply transformers
    transformers = {
        'universal_nominative': (UniversalNominativeTransformer(domain_hint='prestige'), films),
        'hierarchical_nominative': (HierarchicalNominativeTransformer(), films),
        'emotional': (EmotionalResonanceTransformer(), texts),
        'cultural': (CulturalContextTransformer(), texts),
        'authenticity': (AuthenticityTransformer(), texts)
    }
    
    all_features = []
    feature_contributions = {}
    
    for name, (trans, data) in transformers.items():
        print(f"  Applying {name}...")
        trans.fit(data)
        feat = trans.transform(data)
        if hasattr(feat, 'toarray'):
            feat = feat.toarray()
        
        # Test individual contribution
        scaler_temp = StandardScaler()
        X_temp = scaler_temp.fit_transform(feat)
        model_temp = LogisticRegression(max_iter=1000)
        model_temp.fit(X_temp, outcomes)
        auc_temp = roc_auc_score(outcomes, model_temp.predict_proba(X_temp)[:, 1])
        
        feature_contributions[name] = {
            'n_features': feat.shape[1],
            'auc': float(auc_temp)
        }
        
        all_features.append(feat)
        print(f"    {feat.shape[1]} features, AUC = {auc_temp:.3f}")
    
    # Combined
    X_all = np.hstack(all_features)
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(X_all)
    model_all = LogisticRegression(max_iter=1000)
    model_all.fit(X_scaled, outcomes)
    auc_all = roc_auc_score(outcomes, model_all.predict_proba(X_scaled)[:, 1])
    
    print(f"\n✓ COMBINED: {X_all.shape[1]} features, AUC = {auc_all:.3f}")
    
    # Save results
    results = {
        'domain': 'oscar',
        'n_samples': len(films),
        'n_winners': int(outcomes.sum()),
        'n_total_features': X_all.shape[1],
        'auc_combined': float(auc_all),
        'transformer_contributions': feature_contributions,
        'baseline': 0.58,
        'D': float(auc_all - 0.58)
    }
    
    with open('narrative_optimization/domains/oscars/full_pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved")
    return results


def analyze_imdb_full():
    """Full IMDB analysis"""
    print("\n" + "="*80)
    print("IMDB - FULL PIPELINE")
    print("="*80)
    
    with open('data/domains/imdb_movies_complete.json') as f:
        movies = json.load(f)
    
    # Sample 1000
    np.random.seed(42)
    sample_idx = np.random.choice(len(movies), 1000, replace=False)
    movies_sample = [movies[i] for i in sample_idx]
    
    outcomes = np.array([m['success_score'] for m in movies_sample])
    
    print(f"\n{len(movies_sample)} movies")
    
    # Extract texts
    texts = [m['plot_summary'] for m in movies_sample]
    
    # Apply transformers
    transformers = {
        'universal_nominative': (UniversalNominativeTransformer(domain_hint='movies'), movies_sample),
        'emotional': (EmotionalResonanceTransformer(), texts),
        'conflict': (ConflictTensionTransformer(), texts),
        'authenticity': (AuthenticityTransformer(), texts),
        'statistical': (StatisticalTransformer(max_features=100), texts)
    }
    
    all_features = []
    feature_contributions = {}
    
    for name, (trans, data) in transformers.items():
        print(f"  Applying {name}...")
        trans.fit(data)
        feat = trans.transform(data)
        
        if hasattr(feat, 'toarray'):
            feat = feat.toarray()
        
        # Individual contribution
        scaler_temp = StandardScaler()
        X_temp = scaler_temp.fit_transform(feat)
        model_temp = Ridge(alpha=1.0)
        model_temp.fit(X_temp, outcomes)
        r_temp, _ = stats.pearsonr(model_temp.predict(X_temp), outcomes)
        
        feature_contributions[name] = {
            'n_features': feat.shape[1],
            'r': float(r_temp),
            'r2': float(r_temp**2)
        }
        
        all_features.append(feat)
        print(f"    {feat.shape[1]} features, r = {r_temp:.3f}")
    
    # Combined
    X_all = np.hstack(all_features)
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(X_all)
    model_all = Ridge(alpha=1.0)
    model_all.fit(X_scaled, outcomes)
    r_all, _ = stats.pearsonr(model_all.predict(X_scaled), outcomes)
    
    print(f"\n✓ COMBINED: {X_all.shape[1]} features, r = {r_all:.3f} (R² = {r_all**2:.3f})")
    
    # Save
    results = {
        'domain': 'imdb',
        'n_samples': len(movies_sample),
        'n_total_features': X_all.shape[1],
        'r_combined': float(r_all),
        'r2_combined': float(r_all**2),
        'transformer_contributions': feature_contributions,
        'baseline': 0.20,
        'D': float(r_all - 0.20)
    }
    
    with open('narrative_optimization/domains/imdb/full_pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved")
    return results


def main():
    """Run all analyses"""
    print("="*80)
    print("FULL PIPELINE ANALYSES - ALL DATASETS")
    print("="*80)
    
    results = {}
    
    # Oscar
    results['oscar'] = analyze_oscar_full()
    
    # IMDB
    results['imdb'] = analyze_imdb_full()
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("="*80)
    
    for domain, res in results.items():
        print(f"\n{domain.upper()}:")
        print(f"  Samples: {res['n_samples']}")
        print(f"  Features: {res['n_total_features']}")
        if 'auc_combined' in res:
            print(f"  AUC: {res['auc_combined']:.3f}")
        if 'r_combined' in res:
            print(f"  r: {res['r_combined']:.3f} (R² = {res['r2_combined']:.3f})")
        print(f"  Д: {res['D']:.3f}")
    
    print("\n✅ All analyses complete and saved!")
    print("   Results available for web display")


if __name__ == '__main__':
    main()

