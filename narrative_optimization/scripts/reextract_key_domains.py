"""
Re-Extract Phase 7 Features with Enriched Patterns

Re-runs extraction on key domains to validate improved variation.

Focus domains:
- Golf (should show high θ from mental game language)
- Tennis (should show variation)
- NBA, NFL (test if we get more than baseline)
- Mental Health (should show awareness)

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
import gc

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.src.transformers import (
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer
)


KEY_DOMAINS = [
    ('golf_enhanced_narratives', 'data/domains/golf_enhanced_narratives.json', 'Golf Enhanced'),
    ('tennisset', 'data/domains/tennis_complete_dataset.json', 'Tennis'),
    ('mental_health', '../mental_health_complete_200_disorders.json', 'Mental Health'),
]


def reextract_domain(domain_id, data_path, display_name):
    """Re-extract with enriched patterns"""
    print(f"\n{'='*80}")
    print(f"RE-EXTRACTING: {display_name}")
    print(f"{'='*80}")
    
    full_path = project_root / 'narrative_optimization' / data_path
    
    if not full_path.exists():
        # Try alternative path
        full_path = project_root / data_path
    
    if not full_path.exists():
        print(f"⚠️  Data not found: {full_path}")
        return None
    
    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Extract narratives
        if isinstance(data, list):
            narratives = [item.get('narrative', '') or item.get('text', '') or item.get('name', '') 
                         for item in data]
        elif isinstance(data, dict) and 'disorders' in data:
            narratives = [d.get('disorder_name', '') for d in data['disorders']]
        else:
            print("⚠️  Unknown format")
            return None
        
        # Limit to first 1000 for speed
        narratives = narratives[:1000]
        narratives = [n for n in narratives if n]
        
        print(f"✓ Loaded {len(narratives)} narratives")
        
        # Extract with ENRICHED transformers
        print(f"\nExtracting with enriched patterns...")
        
        # θ
        theta_transformer = AwarenessResistanceTransformer()
        theta_features = theta_transformer.fit_transform(narratives)
        theta_values = theta_features[:, 14]
        
        # λ
        lambda_transformer = FundamentalConstraintsTransformer()
        lambda_features = lambda_transformer.fit_transform(narratives)
        lambda_values = lambda_features[:, 11]
        
        # Statistics
        print(f"\nθ (Awareness):")
        print(f"  Mean: {theta_values.mean():.3f}")
        print(f"  Std:  {theta_values.std():.3f}")
        print(f"  Range: [{theta_values.min():.3f}, {theta_values.max():.3f}]")
        
        print(f"\nλ (Constraints):")
        print(f"  Mean: {lambda_values.mean():.3f}")
        print(f"  Std:  {lambda_values.std():.3f}")
        print(f"  Range: [{lambda_values.min():.3f}, {lambda_values.max():.3f}]")
        
        # Save enriched version
        output_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'phase7_enriched'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{domain_id}_enriched.npz'
        np.savez_compressed(
            output_file,
            theta_features=theta_features,
            lambda_features=lambda_features,
            theta_values=theta_values,
            lambda_values=lambda_values,
            n_samples=len(narratives)
        )
        
        print(f"\n✓ Saved to: {output_file}")
        
        # Clean up
        del theta_features, lambda_features, narratives
        gc.collect()
        
        return {
            'domain': domain_id,
            'display_name': display_name,
            'samples': len(theta_values),
            'theta_mean': float(theta_values.mean()),
            'theta_std': float(theta_values.std()),
            'theta_range': float(theta_values.max() - theta_values.min()),
            'lambda_mean': float(lambda_values.mean()),
            'lambda_std': float(lambda_values.std()),
            'lambda_range': float(lambda_values.max() - lambda_values.min())
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Re-extract key domains"""
    print("="*80)
    print("RE-EXTRACTION WITH ENRICHED PATTERNS")
    print("="*80)
    print("\nProcessing key domains to validate improved variation")
    
    results = []
    
    for domain_id, data_path, display_name in KEY_DOMAINS:
        result = reextract_domain(domain_id, data_path, display_name)
        if result:
            results.append(result)
    
    if not results:
        print("\n⚠️  No domains successfully re-extracted")
        return
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON: ORIGINAL VS ENRICHED")
    print(f"{'='*80}")
    
    print(f"\n{'Domain':<20} {'θ_std_old':<12} {'θ_std_new':<12} {'λ_std_old':<12} {'λ_std_new':<12}")
    print("-"*70)
    
    # Original values (from phase7_extraction_summary.json)
    original = {
        'golf_enhanced_narratives': {'theta_std': 0.028, 'lambda_std': 0.041},
        'tennisset': {'theta_std': 0.034, 'lambda_std': 0.061},
        'mental_health': {'theta_std': 0.081, 'lambda_std': 0.060}
    }
    
    for r in results:
        if r['domain'] in original:
            old = original[r['domain']]
            print(f"{r['display_name']:<20} {old['theta_std']:<12.3f} {r['theta_std']:<12.3f} {old['lambda_std']:<12.3f} {r['lambda_std']:<12.3f}")
    
    # Summary
    avg_theta_std = np.mean([r['theta_std'] for r in results])
    avg_lambda_std = np.mean([r['lambda_std'] for r in results])
    
    print(f"\n{'='*80}")
    print("IMPROVEMENT ASSESSMENT")
    print(f"{'='*80}")
    
    print(f"\nAverage θ std: {avg_theta_std:.3f}")
    print(f"Average λ std: {avg_lambda_std:.3f}")
    
    if avg_theta_std > 0.05:
        print(f"✅ θ variation improved")
    else:
        print(f"⚠️  θ variation still weak")
    
    if avg_lambda_std > 0.05:
        print(f"✅ λ variation improved")
    else:
        print(f"⚠️  λ variation still weak")
    
    # Save results
    summary = {
        'timestamp': '2025-11-12',
        'method': 'enriched_patterns',
        'domains_reextracted': len(results),
        'results': results,
        'average_theta_std': float(avg_theta_std),
        'average_lambda_std': float(avg_lambda_std)
    }
    
    output_path = project_root / 'narrative_optimization' / 'data' / 'phase7_enriched_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary to: {output_path}")
    print(f"\n{'='*80}")
    print("✓ RE-EXTRACTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

