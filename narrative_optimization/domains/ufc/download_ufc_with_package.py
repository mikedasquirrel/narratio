"""
Download Real UFC Data using Python Package (opendatasets)

This downloads the Ultimate UFC Dataset from Kaggle programmatically
"""

import opendatasets as od
from pathlib import Path
import pandas as pd

def download_ufc_data():
    """Download UFC dataset using opendatasets package"""
    
    print("="*80)
    print("DOWNLOADING REAL UFC DATA VIA PYTHON PACKAGE")
    print("="*80)
    
    # Dataset URL from Kaggle
    dataset_url = 'https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset'
    
    print(f"\nDataset: {dataset_url}")
    print("Downloading...")
    print("\nNote: First time may ask for Kaggle credentials")
    print("      Just provide username/API key when prompted")
    
    try:
        # Download to data/domains/
        download_dir = 'data/domains/ufc_kaggle'
        
        print(f"\nDownloading to: {download_dir}")
        od.download(dataset_url, data_dir=download_dir)
        
        print("\n✓ Download complete!")
        
        # Find the CSV file
        data_dir = Path(download_dir) / 'ultimate-ufc-dataset'
        
        if not data_dir.exists():
            # Try alternate location
            data_dir = Path(download_dir)
        
        csv_files = list(data_dir.glob('*.csv'))
        
        if len(csv_files) > 0:
            print(f"\n✓ Found {len(csv_files)} CSV file(s):")
            for csv in csv_files:
                print(f"  - {csv.name}")
            
            # Load the main data file
            main_csv = csv_files[0]  # Usually 'data.csv' or similar
            
            print(f"\nLoading: {main_csv.name}")
            df = pd.read_csv(main_csv)
            
            print(f"\n✓ Loaded {len(df)} fights")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample columns: {list(df.columns)[:15]}")
            
            # Save to standard location
            output_path = Path('data/domains/ufc_real_data.csv')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\n✓ Saved to: {output_path}")
            
            # Show data quality
            print(f"\nData Quality Check:")
            
            # Check for key columns
            key_cols = ['R_fighter', 'B_fighter', 'Winner']
            for col in key_cols:
                if col in df.columns:
                    print(f"  ✓ {col}: {df[col].notna().sum()} non-null")
                else:
                    # Try case-insensitive
                    found = [c for c in df.columns if col.lower() in c.lower()]
                    if found:
                        print(f"  ✓ Found similar: {found[0]}")
            
            # Check for betting odds
            odds_cols = [c for c in df.columns if 'odds' in c.lower()]
            if odds_cols:
                print(f"  ✓ Betting odds columns: {odds_cols[:3]}")
            
            # Date range
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                print(f"  ✓ Date column: {date_cols[0]}")
                try:
                    dates = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    print(f"    Range: {dates.min()} to {dates.max()}")
                except:
                    pass
            
            return df
            
        else:
            print(f"\n✗ No CSV files found in {data_dir}")
            return None
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf you see 'invalid authentication credentials':")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Or provide credentials when prompted")
        return None

def main():
    """Main function"""
    
    df = download_ufc_data()
    
    if df is not None:
        print("\n" + "="*80)
        print("SUCCESS! REAL UFC DATA READY")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Run: python3 narrative_optimization/domains/ufc/analyze_ufc_rigorous.py")
        print(f"2. This will use REAL data with REAL correlations")
        print(f"3. Results will be empirically validated!")
    else:
        print("\n" + "="*80)
        print("DOWNLOAD FAILED")
        print("="*80)
        print("\nTroubleshooting:")
        print("1. Ensure you have Kaggle account")
        print("2. Set up API credentials")
        print("3. Try again")

if __name__ == "__main__":
    main()

