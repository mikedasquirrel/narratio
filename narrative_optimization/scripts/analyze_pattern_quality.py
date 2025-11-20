"""
Pattern Quality Analyzer

Analyzes quality of discovered patterns across domains.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_patterns():
    """Load all discovered patterns."""
    patterns_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains'
    
    all_patterns = defaultdict(dict)
    
    for domain_dir in patterns_dir.iterdir():
        if not domain_dir.is_dir():
            continue
        
        pattern_file = domain_dir / 'discovered_archetypes.json'
        
        if pattern_file.exists():
            with open(pattern_file) as f:
                patterns = json.load(f)
                all_patterns[domain_dir.name] = patterns
    
    return all_patterns


def analyze_pattern_quality(patterns: Dict) -> Dict:
    """
    Analyze quality metrics for patterns.
    
    Parameters
    ----------
    patterns : dict
        All patterns by domain
    
    Returns
    -------
    dict
        Quality analysis
    """
    quality_metrics = {
        'by_domain': {},
        'overall': {
            'total_patterns': 0,
            'avg_correlation': [],
            'avg_coherence': [],
            'avg_frequency': []
        }
    }
    
    for domain, domain_patterns in patterns.items():
        if not domain_patterns:
            continue
        
        domain_metrics = {
            'n_patterns': len(domain_patterns),
            'correlations': [],
            'coherences': [],
            'frequencies': []
        }
        
        for pattern_name, pattern_data in domain_patterns.items():
            corr = pattern_data.get('correlation', 0)
            coh = pattern_data.get('coherence', 0)
            freq = pattern_data.get('frequency', 0)
            
            domain_metrics['correlations'].append(corr)
            domain_metrics['coherences'].append(coh)
            domain_metrics['frequencies'].append(freq)
            
            quality_metrics['overall']['avg_correlation'].append(corr)
            quality_metrics['overall']['avg_coherence'].append(coh)
            quality_metrics['overall']['avg_frequency'].append(freq)
        
        domain_metrics['avg_correlation'] = np.mean(domain_metrics['correlations'])
        domain_metrics['avg_coherence'] = np.mean(domain_metrics['coherences'])
        domain_metrics['avg_frequency'] = np.mean(domain_metrics['frequencies'])
        
        quality_metrics['by_domain'][domain] = domain_metrics
        quality_metrics['overall']['total_patterns'] += domain_metrics['n_patterns']
    
    # Calculate overall averages
    if len(quality_metrics['overall']['avg_correlation']) > 0:
        quality_metrics['overall']['avg_correlation'] = np.mean(quality_metrics['overall']['avg_correlation'])
        quality_metrics['overall']['avg_coherence'] = np.mean(quality_metrics['overall']['avg_coherence'])
        quality_metrics['overall']['avg_frequency'] = np.mean(quality_metrics['overall']['avg_frequency'])
    
    return quality_metrics


def identify_high_quality_patterns(patterns: Dict, top_n: int = 10) -> List:
    """Identify highest quality patterns across all domains."""
    all_patterns_list = []
    
    for domain, domain_patterns in patterns.items():
        for pattern_name, pattern_data in domain_patterns.items():
            all_patterns_list.append({
                'domain': domain,
                'name': pattern_name,
                'correlation': pattern_data.get('correlation', 0),
                'coherence': pattern_data.get('coherence', 0),
                'frequency': pattern_data.get('frequency', 0),
                'quality_score': pattern_data.get('correlation', 0) * pattern_data.get('coherence', 0)
            })
    
    # Sort by quality score
    all_patterns_list.sort(key=lambda x: x['quality_score'], reverse=True)
    
    return all_patterns_list[:top_n]


def run_quality_analysis():
    """Run complete pattern quality analysis."""
    print("="*80)
    print("PATTERN QUALITY ANALYSIS")
    print("="*80)
    
    print("\nLoading patterns...")
    patterns = load_all_patterns()
    
    print(f"  ✓ Loaded patterns from {len(patterns)} domains")
    
    print("\nAnalyzing quality...")
    quality = analyze_pattern_quality(patterns)
    
    # Print results
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}\n")
    
    overall = quality['overall']
    print(f"Total patterns: {overall['total_patterns']}")
    print(f"Avg correlation: {overall.get('avg_correlation', 0):.3f}")
    print(f"Avg coherence: {overall.get('avg_coherence', 0):.3f}")
    print(f"Avg frequency: {overall.get('avg_frequency', 0):.1%}")
    
    # By domain
    print(f"\n{'='*80}")
    print("BY DOMAIN")
    print(f"{'='*80}\n")
    
    by_domain = quality['by_domain']
    
    for domain in sorted(by_domain.keys()):
        metrics = by_domain[domain]
        print(f"{domain:20s} n={metrics['n_patterns']}, r={metrics['avg_correlation']:.2f}, coh={metrics['avg_coherence']:.2f}")
    
    # High quality patterns
    print(f"\n{'='*80}")
    print("TOP 10 HIGHEST QUALITY PATTERNS")
    print(f"{'='*80}\n")
    
    top_patterns = identify_high_quality_patterns(patterns, top_n=10)
    
    for i, pattern in enumerate(top_patterns, 1):
        print(f"{i}. {pattern['name']} ({pattern['domain']})")
        print(f"   Quality: {pattern['quality_score']:.3f} (r={pattern['correlation']:.2f}, coh={pattern['coherence']:.2f})")
    
    # Save analysis
    output_path = Path(__file__).parent.parent / 'pattern_quality_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(quality, f, indent=2, default=str)
    
    print(f"\n✓ Analysis saved: {output_path}")


if __name__ == '__main__':
    run_quality_analysis()

