"""
Feature Importance Rankings

Generates feature importance rankings for each domain using:
1. Univariate correlation with outcomes
2. Permutation importance
3. Random Forest feature importance

Saves top features per domain for interpretation.

Author: Narrative Integration System
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


def load_domain_features(domain_name: str, features_dir: Path) -> Tuple:
    """Load features for a domain."""
    feature_path = features_dir / f'{domain_name}_all_features.npz'
    
    if not feature_path.exists():
        return None, None, None
    
    data = np.load(feature_path, allow_pickle=True)
    features = data['features']
    feature_names = data['feature_names'].tolist() if 'feature_names' in data else [
        f"feature_{i}" for i in range(features.shape[1])
    ]
    
    # Create synthetic outcomes for now (will be replaced with real data)
    n_samples = features.shape[0]
    outcomes = np.random.randint(0, 2, n_samples)
    
    return features, outcomes, feature_names


def compute_univariate_importance(
    features: np.ndarray,
    outcomes: np.ndarray,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Compute univariate feature importance using correlation.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    outcomes : np.ndarray
        Outcome labels
    feature_names : list
        Feature names
    
    Returns
    -------
    rankings : list of (feature_name, abs_correlation)
        Sorted by importance
    """
    correlations = []
    
    for i, name in enumerate(feature_names):
        try:
            # Use Spearman for robustness
            corr, pval = spearmanr(features[:, i], outcomes)
            correlations.append((name, abs(corr), corr, pval))
        except:
            correlations.append((name, 0.0, 0.0, 1.0))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    return [(name, abs_corr, corr, pval) for name, abs_corr, corr, pval in correlations]


def compute_rf_importance(
    features: np.ndarray,
    outcomes: np.ndarray,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Compute Random Forest feature importance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    outcomes : np.ndarray
        Outcome labels
    feature_names : list
        Feature names
    
    Returns
    -------
    rankings : list of (feature_name, importance)
        Sorted by importance
    """
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(features_scaled, outcomes)
        
        importances = rf.feature_importances_
        
        rankings = list(zip(feature_names, importances))
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    except Exception as e:
        print(f"  ⚠️ RF importance error: {e}")
        return [(name, 0.0) for name in feature_names]


def compute_permutation_importance(
    features: np.ndarray,
    outcomes: np.ndarray,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    Compute permutation importance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    outcomes : np.ndarray
        Outcome labels
    feature_names : list
        Feature names
    
    Returns
    -------
    rankings : list of (feature_name, importance)
        Sorted by importance
    """
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(features_scaled, outcomes)
        
        perm_import = permutation_importance(
            rf, features_scaled, outcomes,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        importances = perm_import.importances_mean
        
        rankings = list(zip(feature_names, importances))
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    except Exception as e:
        print(f"  ⚠️ Permutation importance error: {e}")
        return [(name, 0.0) for name in feature_names]


def analyze_domain_features(
    domain_name: str,
    features: np.ndarray,
    outcomes: np.ndarray,
    feature_names: List[str],
    domain_pi: float
) -> Dict:
    """
    Comprehensive feature analysis for a domain.
    
    Parameters
    ----------
    domain_name : str
        Domain identifier
    features : np.ndarray
        Feature matrix
    outcomes : np.ndarray
        Outcome labels
    feature_names : list
        Feature names
    domain_pi : float
        Domain narrativity
    
    Returns
    -------
    analysis : dict
        Feature importance analysis
    """
    print(f"\nAnalyzing {domain_name} (π={domain_pi})...")
    print(f"  • Samples: {features.shape[0]}")
    print(f"  • Features: {features.shape[1]}")
    
    # Univariate importance
    print("  • Computing univariate correlations...")
    univariate = compute_univariate_importance(features, outcomes, feature_names)
    
    # RF importance
    print("  • Computing RF importance...")
    rf_importance = compute_rf_importance(features, outcomes, feature_names)
    
    # Permutation importance
    print("  • Computing permutation importance...")
    perm_importance = compute_permutation_importance(features, outcomes, feature_names)
    
    # Aggregate rankings (simple averaging of ranks)
    feature_ranks = {}
    
    for i, (name, _, _, _) in enumerate(univariate):
        feature_ranks[name] = feature_ranks.get(name, 0) + i
    
    for i, (name, _) in enumerate(rf_importance):
        feature_ranks[name] = feature_ranks.get(name, 0) + i
    
    for i, (name, _) in enumerate(perm_importance):
        feature_ranks[name] = feature_ranks.get(name, 0) + i
    
    # Convert to average rank
    avg_ranks = {name: rank / 3.0 for name, rank in feature_ranks.items()}
    top_features_by_avg = sorted(avg_ranks.items(), key=lambda x: x[1])[:20]
    
    print(f"\n  Top 10 Features (aggregated):")
    for i, (name, rank) in enumerate(top_features_by_avg[:10], 1):
        print(f"    {i}. {name:40s} (avg rank: {rank:.1f})")
    
    return {
        'domain': domain_name,
        'pi': domain_pi,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'top_univariate': univariate[:20],
        'top_rf': rf_importance[:20],
        'top_permutation': perm_importance[:20],
        'top_aggregated': top_features_by_avg
    }


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE RANKINGS")
    print("="*70)
    
    DOMAIN_PI_VALUES = {
        'lottery': 0.04,
        'aviation': 0.12,
        'nba': 0.49,
        'nfl': 0.57,
        'mental_health': 0.55,
        'imdb': 0.65,
        'golf': 0.70,
        'golf_enhanced': 0.70,
        'ufc': 0.722,
        'tennis': 0.75,
        'crypto': 0.76,
        'startups': 0.76,
        'oscars': 0.75,
        'housing': 0.92,
        'self_rated': 0.95,
        'wwe': 0.974
    }
    
    features_dir = project_root / 'narrative_optimization' / 'data' / 'features'
    output_dir = project_root / 'narrative_optimization' / 'results' / 'feature_rankings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_analyses = []
    
    for domain_name, domain_pi in DOMAIN_PI_VALUES.items():
        features, outcomes, feature_names = load_domain_features(domain_name, features_dir)
        
        if features is None:
            print(f"\n⚠️ Skipping {domain_name} (no features found)")
            continue
        
        analysis = analyze_domain_features(
            domain_name, features, outcomes, feature_names, domain_pi
        )
        all_analyses.append(analysis)
        
        # Save per-domain ranking
        domain_output = output_dir / f'{domain_name}_top_features.json'
        with open(domain_output, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    # Save summary
    summary = {
        'n_domains': len(all_analyses),
        'domains': [a['domain'] for a in all_analyses]
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Rankings saved to: {output_dir}")
    print("\n" + "="*70)
    print("RANKING ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

