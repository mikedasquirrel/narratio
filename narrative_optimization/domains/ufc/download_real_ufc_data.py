"""
Download REAL UFC Data

Problem: Our synthetic data has no predictive signal
Solution: Get real historical UFC data with actual:
- Fighter statistics
- Fight outcomes  
- Betting odds
- Fighter records/streaks

Sources:
1. Try Kaggle UFC dataset (5,144 fights, 145 columns)
2. Try other public datasets
3. Create reduced but REAL sample
"""

import pandas as pd
import requests
from pathlib import Path
import json

def try_download_real_data():
    """Try to download real UFC data from various sources"""
    
    print("="*80)
    print("ATTEMPTING TO DOWNLOAD REAL UFC DATA")
    print("="*80)
    
    # Try several public UFC datasets
    sources = [
        {
            'name': 'UFC Data (Rajeev Warrier)',
            'url': 'https://raw.githubusercontent.com/rajeevw/ufcstats/master/data/data.csv',
            'format': 'csv'
        },
        {
            'name': 'UFC Stats (mtoto)',
            'url': 'https://raw.githubusercontent.com/mtoto/ufc.stats/master/data/fightdata.csv',
            'format': 'csv'
        },
    ]
    
    for source in sources:
        print(f"\nTrying: {source['name']}")
        print(f"URL: {source['url']}")
        
        try:
            response = requests.get(source['url'], timeout=30)
            
            if response.status_code == 200:
                print(f"✓ Downloaded successfully!")
                
                # Save
                output_path = Path('data/domains/ufc_real_data.csv')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                
                # Load and inspect
                df = pd.read_csv(output_path)
                
                print(f"\nDataset info:")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Columns: {list(df.columns)[:20]}")
                
                return df, str(output_path)
                
            else:
                print(f"✗ Failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print("\n✗ Could not download real data from any source")
    return None, None


def main():
    """Main function"""
    
    df, path = try_download_real_data()
    
    if df is not None:
        print("\n" + "="*80)
        print("SUCCESS! Real UFC data downloaded")
        print("="*80)
        print(f"\nSaved to: {path}")
        print(f"\nNext step: Rerun analysis with REAL data")
        print(f"  python3 analyze_ufc_rigorous.py --real-data")
    else:
        print("\n" + "="*80)
        print("MANUAL DATA NEEDED")
        print("="*80)
        print("\nTo get real UFC data:")
        print("1. Download from Kaggle: https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset")
        print("2. Or download from: https://www.kaggle.com/datasets/rajeevw/ufcdata")
        print("3. Save as: data/domains/ufc_real_data.csv")
        print("4. Rerun analysis")


if __name__ == "__main__":
    main()

