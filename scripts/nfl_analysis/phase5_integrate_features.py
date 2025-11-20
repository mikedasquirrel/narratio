#!/usr/bin/env python3
"""
Phase 5: Feature Integration
Combines all feature matrices into complete feature matrix
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("="*60)
    print(f"PHASE 5: FEATURE INTEGRATION - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "domains"
    
    # Load all feature CSVs
    print("\n📂 Loading feature matrices...")
    
    feature_files = {
        'domain': data_dir / "nfl_domain_features.csv",
        'nominative': data_dir / "nfl_nominative_features.csv",
        'narrative': data_dir / "nfl_narrative_features.csv",
    }
    
    all_features = []
    total_features = 0
    
    for name, path in feature_files.items():
        if path.exists():
            df = pd.read_csv(path)
            all_features.append(df)
            print(f"  ✓ {name}: {df.shape[1]} features ({path.stat().st_size / 1024:.1f} KB)")
            total_features += df.shape[1]
        else:
            print(f"  ✗ {name}: NOT FOUND")
    
    if not all_features:
        print("\n✗ No feature files found")
        return 1
    
    # Combine all features
    print("\n🔄 Combining feature matrices...")
    combined = pd.concat(all_features, axis=1)
    
    print(f"  ✓ Combined shape: {combined.shape}")
    print(f"  ✓ Total features: {combined.shape[1]}")
    
    # Handle missing values
    print("\n🔄 Handling missing values...")
    missing_before = combined.isnull().sum().sum()
    combined = combined.fillna(0)
    missing_after = combined.isnull().sum().sum()
    print(f"  ✓ Filled {missing_before:,} missing values")
    
    # Save combined features
    output_path = data_dir / "nfl_complete_features.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Complete feature matrix saved: {output_path.name}")
    print(f"  Shape: {combined.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Features: {list(combined.columns)[:10]}...")
    
    print(f"\n{'='*60}")
    print("PHASE 5 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

