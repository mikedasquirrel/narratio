"""
Hurricane Data Collector - COMPREHENSIVE

Downloads and processes ALL named hurricanes from NOAA HURDAT2 database:
- Atlantic Basin: 1953-2024 (systematic naming era)
- Pacific Basin: 1959-2024 
- Total: 1000+ named storms expected

Data sources:
- NOAA HURDAT2: https://www.nhc.noaa.gov/data/hurdat/
- Storm tracks, intensities, categories, landfall data

Author: Narrative Integration System
Date: November 2025
"""

import urllib.request
import re
import json
from datetime import datetime
from pathlib import Path
import csv
from collections import defaultdict

print("="*80)
print("HURRICANE DATA COLLECTION - COMPREHENSIVE")
print("="*80)
print("\nDownloading ALL named Atlantic & Pacific hurricanes (1953-2024)")
print("Expected: 1000+ storms\n")

# NOAA HURDAT2 URLs (public, free access)
ATLANTIC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
PACIFIC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-050124.txt"


def download_hurdat_data(url, basin_name):
    """Download HURDAT2 data from NOAA"""
    print(f"Downloading {basin_name} basin data from NOAA...")
    print(f"URL: {url}")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        print(f"✓ Downloaded {len(data)} characters")
        return data
    except Exception as e:
        print(f"✗ Error downloading {basin_name}: {e}")
        return None


def parse_hurdat2(data_text, basin_name):
    """
    Parse HURDAT2 format into structured storm data
    
    Format:
    AL011851,            UNNAMED,     18,
    18510625, 0000,  , HU, 28.0N,  94.8W,  80, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999,
    ...
    
    Header: ID, NAME, ENTRIES
    Data: YYYYMMDD, HHMM, RECORD, STATUS, LAT, LON, WIND, PRES, ...
    """
    
    print(f"\nParsing {basin_name} HURDAT2 data...")
    
    lines = data_text.strip().split('\n')
    storms = []
    current_storm = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if header line (storm ID, name, entries)
        if re.match(r'^[A-Z]{2}\d{6}', line):
            # Save previous storm
            if current_storm and current_storm.get('name') != 'UNNAMED':
                storms.append(current_storm)
            
            # Parse header
            parts = [p.strip() for p in line.split(',')]
            storm_id = parts[0]
            name = parts[1] if len(parts) > 1 else 'UNNAMED'
            entries = int(parts[2]) if len(parts) > 2 else 0
            
            # Extract year from ID (positions 5-8)
            year = int(storm_id[4:8])
            
            # Only process named storms from 1953+ (Atlantic) or 1959+ (Pacific)
            min_year = 1953 if basin_name == "Atlantic" else 1959
            
            if name != 'UNNAMED' and year >= min_year:
                current_storm = {
                    'id': storm_id,
                    'name': name,
                    'year': year,
                    'basin': basin_name,
                    'entries': entries,
                    'max_wind': 0,
                    'min_pressure': 9999,
                    'category': 0,
                    'tracks': [],
                    'landfall': False,
                    'landfall_locations': []
                }
        
        # Data line for current storm
        elif current_storm:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 7:
                continue
            
            try:
                date = parts[0]
                time = parts[1]
                status = parts[3] if len(parts) > 3 else ''
                lat_str = parts[4] if len(parts) > 4 else ''
                lon_str = parts[5] if len(parts) > 5 else ''
                wind = int(parts[6]) if len(parts) > 6 and parts[6].strip() and parts[6] != '-999' else 0
                pressure = int(parts[7]) if len(parts) > 7 and parts[7].strip() and parts[7] != '-999' else 9999
                
                # Update maximum values
                if wind > current_storm['max_wind']:
                    current_storm['max_wind'] = wind
                
                if pressure < current_storm['min_pressure'] and pressure > 0:
                    current_storm['min_pressure'] = pressure
                
                # Check for landfall (status includes 'L')
                if 'L' in status:
                    current_storm['landfall'] = True
                    current_storm['landfall_locations'].append({
                        'date': date,
                        'lat': lat_str,
                        'lon': lon_str
                    })
                
                # Add track point
                current_storm['tracks'].append({
                    'date': date,
                    'time': time,
                    'status': status,
                    'lat': lat_str,
                    'lon': lon_str,
                    'wind': wind,
                    'pressure': pressure
                })
                
            except (ValueError, IndexError) as e:
                continue
    
    # Add final storm
    if current_storm and current_storm.get('name') != 'UNNAMED':
        storms.append(current_storm)
    
    # Calculate category based on Saffir-Simpson scale
    for storm in storms:
        wind = storm['max_wind']
        if wind >= 137:
            storm['category'] = 5
        elif wind >= 113:
            storm['category'] = 4
        elif wind >= 96:
            storm['category'] = 3
        elif wind >= 83:
            storm['category'] = 2
        elif wind >= 64:
            storm['category'] = 1
        else:
            storm['category'] = 0  # Tropical Storm
    
    print(f"✓ Parsed {len(storms)} named storms from {basin_name}")
    
    return storms


def calculate_storm_statistics(storms):
    """Calculate statistics across all storms"""
    
    by_decade = defaultdict(int)
    by_category = defaultdict(int)
    by_basin = defaultdict(int)
    total_landfall = 0
    
    for storm in storms:
        decade = (storm['year'] // 10) * 10
        by_decade[decade] += 1
        by_category[storm['category']] += 1
        by_basin[storm['basin']] += 1
        if storm['landfall']:
            total_landfall += 1
    
    return {
        'total_storms': len(storms),
        'by_decade': dict(sorted(by_decade.items())),
        'by_category': dict(sorted(by_category.items())),
        'by_basin': dict(by_basin),
        'total_landfall': total_landfall,
        'landfall_percentage': (total_landfall / len(storms)) * 100 if storms else 0,
        'date_range': {
            'first': min(s['year'] for s in storms) if storms else None,
            'last': max(s['year'] for s in storms) if storms else None
        }
    }


def save_dataset(storms, output_dir):
    """Save complete dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    stats = calculate_storm_statistics(storms)
    
    # Create dataset
    dataset = {
        'metadata': {
            'domain': 'hurricanes',
            'description': 'All named Atlantic and Pacific hurricanes, 1953-2024',
            'data_source': 'NOAA HURDAT2',
            'collection_date': datetime.now().isoformat(),
            'atlantic_url': ATLANTIC_URL,
            'pacific_url': PACIFIC_URL
        },
        'statistics': stats,
        'storms': storms
    }
    
    # Save JSON
    output_file = output_dir / 'hurricane_complete_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return dataset, output_file


def print_summary(dataset):
    """Print comprehensive summary"""
    
    stats = dataset['statistics']
    storms = dataset['storms']
    
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal Named Storms: {stats['total_storms']:,}")
    print(f"Date Range: {stats['date_range']['first']}-{stats['date_range']['last']}")
    print(f"Landfall Storms: {stats['total_landfall']:,} ({stats['landfall_percentage']:.1f}%)")
    
    print(f"\nBy Basin:")
    for basin, count in stats['by_basin'].items():
        pct = (count / stats['total_storms']) * 100
        print(f"  {basin}: {count:,} ({pct:.1f}%)")
    
    print(f"\nBy Category (Saffir-Simpson):")
    category_names = {
        0: "Tropical Storm",
        1: "Category 1",
        2: "Category 2",
        3: "Category 3 (Major)",
        4: "Category 4 (Major)",
        5: "Category 5 (Major)"
    }
    for cat in sorted(stats['by_category'].keys()):
        count = stats['by_category'][cat]
        pct = (count / stats['total_storms']) * 100
        name = category_names.get(cat, f"Category {cat}")
        print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    print(f"\nBy Decade:")
    for decade, count in stats['by_decade'].items():
        print(f"  {decade}s: {count:,}")
    
    print(f"\nNotable Storms (Category 5):")
    cat5_storms = [s for s in storms if s['category'] == 5]
    for storm in sorted(cat5_storms, key=lambda x: x['year'])[:20]:  # Show first 20
        landfall_str = " (Landfall)" if storm['landfall'] else ""
        print(f"  {storm['year']} {storm['name']}: {storm['max_wind']} mph, {storm['min_pressure']} mb{landfall_str}")
    if len(cat5_storms) > 20:
        print(f"  ... and {len(cat5_storms) - 20} more Category 5 storms")
    
    print("\n" + "="*80)


def main():
    """Main execution"""
    
    all_storms = []
    
    # Download and parse Atlantic basin
    atlantic_data = download_hurdat_data(ATLANTIC_URL, "Atlantic")
    if atlantic_data:
        atlantic_storms = parse_hurdat2(atlantic_data, "Atlantic")
        all_storms.extend(atlantic_storms)
        print(f"  Atlantic: {len(atlantic_storms)} storms")
    
    # Download and parse Pacific basin
    pacific_data = download_hurdat_data(PACIFIC_URL, "Pacific")
    if pacific_data:
        pacific_storms = parse_hurdat2(pacific_data, "Pacific")
        all_storms.extend(pacific_storms)
        print(f"  Pacific: {len(pacific_storms)} storms")
    
    print(f"\n✓ Total storms collected: {len(all_storms)}")
    
    # Save dataset
    output_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'hurricanes'
    dataset, output_file = save_dataset(all_storms, output_dir)
    
    # Print summary
    print_summary(dataset)
    
    print("\n✓ Hurricane data collection complete!")
    print("✓ Ready for π calculation and name characterization")
    
    return dataset


if __name__ == '__main__':
    dataset = main()

