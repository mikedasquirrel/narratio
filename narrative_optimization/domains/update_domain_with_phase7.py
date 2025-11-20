"""
Universal Domain Updater for Phase 7 Transformers

Adds θ and λ transformers to ANY existing domain analysis.
Works by importing domain data, extracting Phase 7 features, and saving them.

Usage:
    python narrative_optimization/domains/update_domain_with_phase7.py

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.src.transformers import (
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer
)


def find_domain_data_files():
    """Find all available domain data files"""
    print("Scanning for domain data files...")
    
    data_files = []
    
    # Check common locations
    locations = [
        project_root / 'narrative_optimization' / 'domains',
        project_root / 'narrative_optimization' / 'data' / 'domains',
        project_root,
    ]
    
    for location in locations:
        if location.exists():
            # Find JSON files
            json_files = list(location.rglob('*.json'))
            
            # Filter for data files (not results/config)
            for json_file in json_files:
                if any(skip in json_file.name.lower() for skip in ['result', 'config', 'summary', 'metadata']):
                    continue
                    
                # Check if it looks like domain data
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                    # Check if it has narrative-like structure
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict):
                            # Has some text field?
                            text_fields = ['narrative', 'text', 'description', 'name', 'disorder_name']
                            if any(field in data[0] for field in text_fields):
                                data_files.append((json_file, len(data)))
                    elif isinstance(data, dict) and 'disorders' in data:
                        # Mental health format
                        disorders = data['disorders']
                        data_files.append((json_file, len(disorders)))
                        
                except:
                    continue
    
    return sorted(data_files, key=lambda x: x[1], reverse=True)  # Sort by size


def extract_phase7_features(json_file, max_samples=1000):
    """Extract θ and λ features from domain data file"""
    print(f"\n{'='*80}")
    print(f"Processing: {json_file.name}")
    print(f"{'='*80}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract narratives based on structure
        narratives = []
        outcomes = []
        
        if isinstance(data, list):
            for item in data[:max_samples]:
                # Find text field
                text = (item.get('narrative') or item.get('text') or 
                       item.get('description') or item.get('name') or '')
                if text:
                    narratives.append(str(text))
                    # Try to find outcome
                    outcome = item.get('outcome', item.get('win', item.get('success', 0)))
                    outcomes.append(outcome)
                    
        elif isinstance(data, dict) and 'disorders' in data:
            # Mental health format
            for disorder in data['disorders'][:max_samples]:
                name = disorder.get('disorder_name', '')
                if name:
                    narratives.append(name)
                    stigma = disorder.get('predicted_stigma', 0)
                    outcomes.append(stigma)
        
        if not narratives:
            print("⚠️  No narratives found in this file")
            return None
        
        print(f"✓ Found {len(narratives)} narratives")
        
        # Extract Phase 7 features
        print(f"\nExtracting Phase 7 features...")
        
        # θ (Awareness Resistance)
        print(f"  [1/2] Computing θ (Awareness Resistance)...")
        theta_transformer = AwarenessResistanceTransformer()
        theta_features = theta_transformer.fit_transform(narratives)
        theta_values = theta_features[:, 14]
        
        print(f"    ✓ θ: Mean={theta_values.mean():.3f}, Std={theta_values.std():.3f}, Range=[{theta_values.min():.3f}, {theta_values.max():.3f}]")
        
        # λ (Fundamental Constraints)
        print(f"  [2/2] Computing λ (Fundamental Constraints)...")
        lambda_transformer = FundamentalConstraintsTransformer()
        lambda_features = lambda_transformer.fit_transform(narratives)
        lambda_values = lambda_features[:, 11]
        
        print(f"    ✓ λ: Mean={lambda_values.mean():.3f}, Std={lambda_values.std():.3f}, Range=[{lambda_values.min():.3f}, {lambda_values.max():.3f}]")
        
        # Determine domain name from file
        domain_name = json_file.stem.replace('_complete', '').replace('_200_disorders', '').replace('_data', '')
        
        # Save Phase 7 features
        output_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'phase7'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{domain_name}_phase7.npz'
        np.savez_compressed(
            output_file,
            theta_features=theta_features,
            lambda_features=lambda_features,
            theta_values=theta_values,
            lambda_values=lambda_values,
            narratives=narratives[:100],  # Save first 100 for reference
            n_samples=len(narratives),
            source_file=str(json_file)
        )
        
        print(f"\n✓ Saved Phase 7 features to: {output_file}")
        
        return {
            'domain': domain_name,
            'file': str(json_file),
            'samples': len(narratives),
            'theta_mean': float(theta_values.mean()),
            'theta_std': float(theta_values.std()),
            'lambda_mean': float(lambda_values.mean()),
            'lambda_std': float(lambda_values.std())
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Process all available domain data files"""
    print("="*80)
    print("PHASE 7 FEATURE EXTRACTION - REAL DOMAIN DATA")
    print("="*80)
    print("\nExtracting θ (awareness) and λ (constraints) from all available domains")
    print("Using REAL data only, no synthetic generation")
    
    # Find data files
    data_files = find_domain_data_files()
    
    print(f"\n✓ Found {len(data_files)} domain data files")
    
    if not data_files:
        print("\n⚠️  No domain data files found")
        print("Expected locations:")
        print("  - narrative_optimization/domains/*/data/*.json")
        print("  - *.json in project root")
        return
    
    # Show what was found
    print("\nAvailable data files:")
    for i, (file_path, n_samples) in enumerate(data_files[:10], 1):  # Show first 10
        rel_path = file_path.relative_to(project_root) if file_path.is_relative_to(project_root) else file_path
        print(f"  {i}. {rel_path.name} ({n_samples:,} samples)")
    
    if len(data_files) > 10:
        print(f"  ... and {len(data_files) - 10} more")
    
    # Process each file
    results = []
    
    for i, (file_path, n_samples) in enumerate(data_files, 1):
        print(f"\n[{i}/{len(data_files)}]")
        result = extract_phase7_features(file_path, max_samples=5000)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    
    if results:
        print(f"\n✓ Processed {len(results)} domains")
        print(f"\nForce Statistics:")
        print(f"{'Domain':<30} {'Samples':<10} {'θ(mean)':<10} {'λ(mean)':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['domain']:<30} {r['samples']:<10} {r['theta_mean']:<10.3f} {r['lambda_mean']:<10.3f}")
        
        # Save summary
        summary = {
            'timestamp': pd.Timestamp.now().isoformat() if 'pd' in dir() else 'now',
            'domains_processed': len(results),
            'results': results
        }
        
        summary_path = project_root / 'narrative_optimization' / 'data' / 'phase7_extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved summary to: {summary_path}")
        print(f"\nPhase 7 features saved to: data/features/phase7/")
    else:
        print("\n⚠️  No domains successfully processed")
    
    print(f"\n{'='*80}")
    print("✓ PHASE 7 FEATURES EXTRACTED FROM REAL DATA")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

