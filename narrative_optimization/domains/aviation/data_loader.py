"""
Data loader for aviation domain.

Loads airports and airlines data from CSV files with all extracted features.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def load_airports() -> pd.DataFrame:
    """
    Load airport data with features.
    
    Returns
    -------
    pd.DataFrame
        500 airports with codes, names, traffic, and nomenclature features
    """
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'aviation' / 'airports_with_features.csv'
    df = pd.read_csv(data_path)
    return df


def load_airlines() -> pd.DataFrame:
    """
    Load airline data with features.
    
    Returns
    -------
    pd.DataFrame
        198 airlines with names, codes, fleet, and nomenclature features
    """
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'aviation' / 'airlines_with_features.csv'
    df = pd.read_csv(data_path)
    return df


def load_aviation_data() -> Dict[str, pd.DataFrame]:
    """
    Load all aviation data.
    
    Returns
    -------
    dict
        Dictionary with 'airports' and 'airlines' DataFrames
    """
    return {
        'airports': load_airports(),
        'airlines': load_airlines()
    }


def get_aviation_summary() -> Dict[str, any]:
    """
    Get summary statistics for aviation domain.
    
    Returns
    -------
    dict
        Summary statistics
    """
    airports = load_airports()
    airlines = load_airlines()
    
    return {
        'n_airports': len(airports),
        'n_airlines': len(airlines),
        'total_entities': len(airports) + len(airlines),
        'airports': {
            'countries': airports['country'].nunique(),
            'total_passengers': airports['annual_passengers'].sum(),
            'avg_passengers': airports['annual_passengers'].mean(),
        },
        'airlines': {
            'countries': airlines['country'].nunique() if 'country' in airlines.columns else 0,
            'total_fleet': airlines['fleet_size'].sum() if 'fleet_size' in airlines.columns else 0,
            'active': airlines['is_active'].sum() if 'is_active' in airlines.columns else 0,
        }
    }


if __name__ == '__main__':
    print("="*80)
    print("AVIATION DOMAIN DATA LOADER")
    print("="*80)
    
    data = load_aviation_data()
    summary = get_aviation_summary()
    
    print(f"\nLoaded aviation data:")
    print(f"  Airports: {summary['n_airports']}")
    print(f"  Airlines: {summary['n_airlines']}")
    print(f"  Total entities: {summary['total_entities']}")
    
    print(f"\nAirports:")
    print(f"  Countries: {summary['airports']['countries']}")
    print(f"  Total annual passengers: {summary['airports']['total_passengers']:,}")
    
    print(f"\nAirlines:")
    print(f"  Countries: {summary['airlines']['countries']}")
    
    print(f"\nAirport columns: {list(data['airports'].columns)}")
    print(f"\nAirline columns: {list(data['airlines'].columns)}")
    
    print("\n✓ Data loader working correctly")

