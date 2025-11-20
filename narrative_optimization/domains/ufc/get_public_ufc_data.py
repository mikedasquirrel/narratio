"""
Get UFC Data from Public Sources (No Authentication Required)

Try multiple public UFC datasets available without credentials
"""

import pandas as pd
import requests
from pathlib import Path
import json

def try_public_sources():
    """Try various public UFC data sources"""
    
    print("="*80)
    print("DOWNLOADING UFC DATA FROM PUBLIC SOURCES")
    print("="*80)
    
    # Public UFC datasets (no authentication)
    sources = [
        {
            'name': 'UFC Stats (rawgit/CDN)',
            'url': 'https://cdn.jsdelivr.net/gh/rajeevw/ufcstats@master/data/data.csv',
            'format': 'csv'
        },
        {
            'name': 'UFC Dataset (statso.io)',
            'url': 'https://projects.fivethirtyeight.com/complete-history-of-the-ufc/ufc-stats.csv',
            'format': 'csv'
        },
        {
            'name': 'MMA Stats (data.world)',
            'url': 'https://query.data.world/s/abcdefghijk',  # Would need actual URL
            'format': 'csv'
        },
    ]
    
    # Try each source
    for source in sources:
        print(f"\nTrying: {source['name']}")
        print(f"URL: {source['url']}")
        
        try:
            # Try to read CSV directly
            df = pd.read_csv(source['url'])
            
            print(f"✓ SUCCESS! Loaded {len(df)} rows")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample: {list(df.columns)[:10]}")
            
            # Save
            output_path = Path('data/domains/ufc_real_data.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\n✓ Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
            continue
    
    return None

def scrape_ufcstats_sample():
    """
    Scrape a small sample directly from ufcstats.com
    Just enough to demonstrate the methodology works
    """
    
    print("\n" + "="*80)
    print("ALTERNATIVE: CREATE SMALL REAL SAMPLE")
    print("="*80)
    
    print("\nWould you like to:")
    print("1. Use a small REAL sample (100 fights) to demonstrate")
    print("2. Continue waiting for full dataset")
    print("\nSmall sample would let us validate methodology immediately")
    
    return None

def main():
    """Main download function"""
    
    # Try public sources first
    df = try_public_sources()
    
    if df is not None:
        print("\n" + "="*80)
        print("SUCCESS - REAL DATA OBTAINED")
        print("="*80)
        print(f"\nReady to run rigorous analysis!")
        return df
    
    # Suggest alternatives
    print("\n" + "="*80)
    print("PUBLIC SOURCES UNAVAILABLE")
    print("="*80)
    
    print("\nOptions:")
    print("\n1. FASTEST: Set up Kaggle credentials (2 minutes)")
    print("   - Go to: https://www.kaggle.com/settings/account")
    print("   - Click 'Create New API Token'")
    print("   - Save to ~/.kaggle/kaggle.json")
    print("   - Rerun: python3 download_ufc_with_package.py")
    
    print("\n2. ALTERNATIVE: Manual download")
    print("   - Visit: https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset")
    print("   - Download ZIP")
    print("   - Extract to: data/domains/")
    
    print("\n3. USE EXISTING: Run on validated domains")
    print("   - NBA, NFL, Tennis already have REAL data")
    print("   - Focus on proven results")
    
    return None

if __name__ == "__main__":
    main()

