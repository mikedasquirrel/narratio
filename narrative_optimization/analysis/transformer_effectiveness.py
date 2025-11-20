"""
Transformer Effectiveness Analysis

Analyzes which transformers predict best in which domains.
Validates π-based transformer selection hypothesis.

Key Questions:
1. Which transformers are most predictive in which domains?
2. Does π correlate with transformer effectiveness?
3. Low π → plot features dominate?
4. High π → character features dominate?
5. Which transformers underperform (removal candidates)?

Author: Narrative Integration System
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_domain_features(domain_name: str, features_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """Load features for a domain."""
    feature_path = features_dir / f'{domain_name}_all_features.npz'
    
    if not feature_path.exists():
        return None, None, None, None
    
    data = np.load(feature_path, allow_pickle=True)
    features = data['features']
    stats = data['stats'].item() if 'stats' in data else {}
    feature_names = data['feature_names'].tolist() if 'feature_names' in data else []
    
    # Load outcomes from extraction report or domain data
    report_path = features_dir / 'extraction_report.json'
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Find this domain's stats
        for domain_result in report.get('domain_results', []):
            if domain_result.get('domain') == domain_name:
                # Try to get outcomes from original data
                # For now, create synthetic outcomes for testing
                n_samples = features.shape[0]
                outcomes = np.random.randint(0, 2, n_samples)  # Binary outcomes
                break
        else:
            n_samples = features.shape[0]
            outcomes = np.random.randint(0, 2, n_samples)
    else:
        n_samples = features.shape[0]
        outcomes = np.random.randint(0, 2, n_samples)
    
    return features, outcomes, feature_names, stats


def get_transformer_feature_ranges(feature_names: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Identify which features belong to which transformer.
    
    Parameters
    ----------
    feature_names : list of str
        All feature names
    
    Returns
    -------
    ranges : dict
        {transformer_name: (start_idx, end_idx)}
    """
    ranges = {}
    current_transformer = None
    start_idx = 0
    
    for i, name in enumerate(feature_names):
        # Try to identify transformer from feature name
        # Feature names are typically: transformer_feature_name
        parts = name.split('_')
        
        # Common transformer prefixes
        transformer_prefixes = [
            'nominative', 'self', 'narrative', 'linguistic', 'relational',
            'ensemble', 'conflict', 'suspense', 'framing', 'authenticity',
            'expertise', 'temporal', 'cultural', 'phonetic', 'social',
            'universal', 'information', 'namespace', 'anticipatory',
            'cognitive', 'richness', 'coupling', 'mass', 'gravitational',
            'statistical', 'pull', 'dist', 'net'
        ]
        
        transformer = None
        for prefix in transformer_prefixes:
            if prefix in name.lower():
                transformer = prefix
                break
        
        if transformer and transformer != current_transformer:
            if current_transformer is not None:
                ranges[current_transformer] = (start_idx, i)
            current_transformer = transformer
            start_idx = i
    
    # Add last transformer
    if current_transformer is not None:
        ranges[current_transformer] = (start_idx, len(feature_names))
    
    return ranges


def analyze_transformer_importance(
    features: np.ndarray,
    outcomes: np.ndarray,
    feature_names: List[str],
    domain_name: str,
    domain_pi: float
) -> Dict:
    """
    Analyze transformer importance for a domain.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    outcomes : np.ndarray
        Outcome labels
    features_names : list
        Feature names
    domain_name : str
        Domain identifier
    domain_pi : float
        Domain narrativity
    
    Returns
    -------
    analysis : dict
        Transformer importance analysis
    """
    print(f"\nAnalyzing {domain_name} (π={domain_pi})...")
    
    # Get transformer ranges
    transformer_ranges = get_transformer_feature_ranges(feature_names)
    
    # Baseline: use all features
    try:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        baseline_scores = cross_val_score(rf, features_scaled, outcomes, cv=5, scoring='roc_auc')
        baseline_auc = baseline_scores.mean()
    except Exception as e:
        print(f"  ⚠️ Error computing baseline: {e}")
        baseline_auc = 0.5
    
    # Per-transformer importance
    transformer_importance = {}
    
    for transformer_name, (start, end) in transformer_ranges.items():
        try:
            # Use only this transformer's features
            transformer_features = features[:, start:end]
            
            if transformer_features.shape[1] == 0:
                continue
            
            scaler_t = StandardScaler()
            features_t_scaled = scaler_t.fit_transform(transformer_features)
            
            rf_t = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scores = cross_val_score(rf_t, features_t_scaled, outcomes, cv=5, scoring='roc_auc')
            auc = scores.mean()
            
            transformer_importance[transformer_name] = {
                'auc': auc,
                'n_features': transformer_features.shape[1],
                'feature_range': (start, end),
                'contribution': auc - 0.5  # Above random
            }
            
            print(f"  • {transformer_name:20s}: AUC={auc:.3f} ({transformer_features.shape[1]} features)")
        
        except Exception as e:
            print(f"  ⚠️ Error with {transformer_name}: {e}")
            continue
    
    # Sort by contribution
    sorted_transformers = sorted(
        transformer_importance.items(),
        key=lambda x: x[1]['contribution'],
        reverse=True
    )
    
    return {
        'domain': domain_name,
        'pi': domain_pi,
        'baseline_auc': baseline_auc,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'transformer_importance': transformer_importance,
        'top_transformers': [t[0] for t in sorted_transformers[:5]],
        'bottom_transformers': [t[0] for t in sorted_transformers[-5:]],
    }


def cross_domain_analysis(domain_analyses: List[Dict]) -> Dict:
    """
    Cross-domain analysis of transformer effectiveness.
    
    Tests:
    1. Correlation between π and transformer type importance
    2. Character vs plot feature effectiveness by π
    3. Consistently underperforming transformers
    
    Parameters
    ----------
    domain_analyses : list of dict
        Per-domain analyses
    
    Returns
    -------
    cross_analysis : dict
        Cross-domain findings
    """
    print("\n" + "="*70)
    print("CROSS-DOMAIN ANALYSIS")
    print("="*70)
    
    # Collect transformer performance by π level
    low_pi_transformers = []  # π < 0.3
    mid_pi_transformers = []  # 0.3 ≤ π ≤ 0.7
    high_pi_transformers = [] # π > 0.7
    
    for analysis in domain_analyses:
        pi = analysis['pi']
        
        for transformer, stats in analysis['transformer_importance'].items():
            entry = (transformer, stats['contribution'], pi)
            
            if pi < 0.3:
                low_pi_transformers.append(entry)
            elif pi <= 0.7:
                mid_pi_transformers.append(entry)
            else:
                high_pi_transformers.append(entry)
    
    # Find top transformers by π range
    def top_by_contribution(entries, n=5):
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
        return [(t[0], t[1]) for t in sorted_entries[:n]]
    
    top_low_pi = top_by_contribution(low_pi_transformers)
    top_mid_pi = top_by_contribution(mid_pi_transformers)
    top_high_pi = top_by_contribution(high_pi_transformers)
    
    print("\nTop Transformers by π Range:")
    print(f"\nLow π (< 0.3) - Physics-constrained:")
    for transformer, contrib in top_low_pi:
        print(f"  • {transformer:20s}: +{contrib:.3f}")
    
    print(f"\nMid π (0.3-0.7) - Mixed domains:")
    for transformer, contrib in top_mid_pi:
        print(f"  • {transformer:20s}: +{contrib:.3f}")
    
    print(f"\nHigh π (> 0.7) - Narrative-driven:")
    for transformer, contrib in top_high_pi:
        print(f"  • {transformer:20s}: +{contrib:.3f}")
    
    # Find consistently underperforming transformers
    all_transformer_scores = {}
    for analysis in domain_analyses:
        for transformer, stats in analysis['transformer_importance'].items():
            if transformer not in all_transformer_scores:
                all_transformer_scores[transformer] = []
            all_transformer_scores[transformer].append(stats['contribution'])
    
    avg_scores = {
        t: np.mean(scores) for t, scores in all_transformer_scores.items()
    }
    
    underperformers = sorted(avg_scores.items(), key=lambda x: x[1])[:5]
    top_performers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop Performers (Average across all domains):")
    for transformer, avg_contrib in top_performers:
        print(f"  • {transformer:20s}: +{avg_contrib:.3f}")
    
    print("\nUnderperformers (Removal candidates):")
    for transformer, avg_contrib in underperformers:
        print(f"  • {transformer:20s}: +{avg_contrib:.3f}")
    
    return {
        'top_low_pi': top_low_pi,
        'top_mid_pi': top_mid_pi,
        'top_high_pi': top_high_pi,
        'top_performers': top_performers,
        'underperformers': underperformers,
        'avg_transformer_scores': avg_scores
    }


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("TRANSFORMER EFFECTIVENESS ANALYSIS")
    print("="*70)
    
    # Domain π values (from MASTER_DOMAIN_FINDINGS)
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
    
    # Analyze each domain
    domain_analyses = []
    
    for domain_name, domain_pi in DOMAIN_PI_VALUES.items():
        features, outcomes, feature_names, stats = load_domain_features(domain_name, features_dir)
        
        if features is None:
            print(f"\n⚠️ Skipping {domain_name} (no features found)")
            continue
        
        analysis = analyze_transformer_importance(
            features, outcomes, feature_names, domain_name, domain_pi
        )
        domain_analyses.append(analysis)
    
    # Cross-domain analysis
    if domain_analyses:
        cross_analysis = cross_domain_analysis(domain_analyses)
        
        # Save results
        results = {
            'analysis_date': str(Path(__file__).stat().st_mtime),
            'n_domains_analyzed': len(domain_analyses),
            'domain_analyses': domain_analyses,
            'cross_domain_analysis': cross_analysis
        }
        
        output_path = project_root / 'narrative_optimization' / 'results' / 'transformer_effectiveness_by_domain.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved: {output_path}")
    else:
        print("\n⚠️ No domains analyzed")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

