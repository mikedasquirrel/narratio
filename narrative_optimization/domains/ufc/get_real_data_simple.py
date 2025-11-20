"""
Get REAL UFC data - Simple Approach

Try multiple sources to get actual UFC fight data with real outcomes
"""

import pandas as pd
import requests
import json
from pathlib import Path

def try_load_from_csv_urls():
    """Try loading UFC CSV data from various online sources"""
    
    print("="*80)
    print("LOADING REAL UFC DATA FROM CSV SOURCES")
    print("="*80)
    
    # Try direct CSV URLs (public datasets)
    sources = [
        {
            'name': 'UFC Master Dataset (GitHub)',
            'url': 'https://raw.githubusercontent.com/mdabbert/Ultimate-UFC-Dataset/master/data.csv'
        },
        {
            'name': 'UFC Fight Data (Alternative)',
            'url': 'https://raw.githubusercontent.com/WarrierRajeev/UFC-Stats/master/ufc_data.csv'
        },
    ]
    
    for source in sources:
        print(f"\nTrying: {source['name']}")
        try:
            df = pd.read_csv(source['url'])
            print(f"✓ SUCCESS! Loaded {len(df)} fights")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample columns: {list(df.columns)[:10]}")
            
            # Save
            output_path = Path('data/domains/ufc_real_data.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\n✓ Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    return None

def try_ufcstats_package():
    """Try using ufcstats package"""
    
    print("\n" + "="*80)
    print("TRYING UFCSTATS PACKAGE")
    print("="*80)
    
    try:
        from ufcstats import ufcstats
        
        # Try getting stats for well-known fighters
        test_fighters = ['Jon Jones', 'Conor McGregor', 'Khabib Nurmagomedov']
        
        all_data = []
        for fighter in test_fighters:
            print(f"\nFetching: {fighter}")
            try:
                stats = ufcstats.getStats(fighter)
                print(f"✓ Got data: {type(stats)}")
                print(f"  Data: {stats}")
                all_data.append(stats)
            except Exception as e:
                print(f"✗ Error: {e}")
        
        if all_data:
            print(f"\n✓ Successfully retrieved data for {len(all_data)} fighters")
            return all_data
        
    except Exception as e:
        print(f"✗ Package error: {e}")
    
    return None

def create_realistic_sample():
    """
    As fallback, create a SMALL but REALISTIC sample where:
    - Physical stats actually correlate with outcomes
    - Based on real fighter archetypes
    """
    
    print("\n" + "="*80)
    print("CREATING REALISTIC SAMPLE")
    print("="*80)
    
    print("\nThis will be a SMALL but REALISTIC dataset where:")
    print("  - Physical stats ACTUALLY predict outcomes")
    print("  - Fighter archetypes match real UFC patterns")
    print("  - Outcomes based on statistical advantages")
    
    # This would be created with more care
    # But for now, indicate we need real data
    
    print("\n✗ Better to get real data than create synthetic")
    return None

def main():
    """Main function"""
    
    # Try method 1: Load from CSV
    df = try_load_from_csv_urls()
    
    if df is not None:
        print("\n" + "="*80)
        print("SUCCESS - REAL DATA LOADED")
        print("="*80)
        print(f"\nDataset: {len(df)} fights")
        print(f"Columns: {list(df.columns)[:20]}")
        
        # Check for key columns
        key_cols = ['R_fighter', 'B_fighter', 'Winner', 'R_odds', 'B_odds']
        has_keys = [col for col in key_cols if col in df.columns or col.lower() in [c.lower() for c in df.columns]]
        print(f"\nKey columns present: {has_keys}")
        
        print(f"\nReady to run rigorous analysis!")
        return df
    
    # Try method 2: ufcstats package
    data = try_ufcstats_package()
    
    if data:
        return data
    
    # If all fails
    print("\n" + "="*80)
    print("COULD NOT GET REAL DATA")
    print("="*80)
    print("\nOptions:")
    print("1. Manually download from: https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset")
    print("2. Place CSV in: data/domains/ufc_real_data.csv")
    print("3. Rerun analysis")
    
    return None

if __name__ == "__main__":
    main()

