"""
Process GitHub YC CSV - 688 REAL Companies

Converts GitHub CSV (688 companies) into analysis format.
Enriches with outcome data where available.
"""

import pandas as pd
import json
import numpy as np
import re
from pathlib import Path

print("=" * 80)
print("PROCESSING 688 REAL YC COMPANIES FROM GITHUB")
print("=" * 80)

# Load CSV
csv_path = Path(__file__).parent.parent.parent.parent / 'data/domains/yc_startups_github.csv'
df = pd.read_csv(csv_path)

print(f"\n✓ Loaded {len(df)} companies from CSV")
print(f"✓ Columns: {list(df.columns[:5])}...")

# Process each company
processed = []

for idx, row in df.iterrows():
    # Extract fields
    company_name = str(row['Company']).strip()
    description = str(row['Description']) if pd.notna(row['Description']) else ""
    status = str(row['Satus']) if pd.notna(row['Satus']) else "Unknown"  # Note: typo in CSV
    founders = str(row['Founders']) if pd.notna(row['Founders']) else ""
    batch = f"{row['Y Combinator Session']}{str(row['Y Combinator Year'])[-2:]}" if pd.notna(row['Y Combinator Year']) else "Unknown"
    
    # Count founders
    if founders and founders != 'nan':
        founder_count = len([f.strip() for f in founders.split(',') if f.strip()])
    else:
        founder_count = 2  # Default
    
    # Parse funding
    funding_str = str(row['Amounts raised in different funding rounds']) if pd.notna(row['Amounts raised in different funding rounds']) else ""
    total_funding = None
    if funding_str and funding_str != 'nan':
        # Extract numbers from funding string
        amounts = re.findall(r'\$?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:M|million)?', funding_str, re.IGNORECASE)
        if amounts:
            # Sum all amounts
            total = sum(float(a.replace(',', '')) for a in amounts)
            total_funding = total if total > 0 else None
    
    # Determine success (top 25% = unicorn path or good exit)
    successful = None
    if status.lower() == 'acquired' or 'acquired' in status.lower():
        # Acquired is moderate success (not top 25% unless huge)
        if total_funding and total_funding > 100:
            successful = True  # Big acquisition
        else:
            successful = False  # Moderate acquisition
    elif 'ipo' in status.lower():
        successful = True  # IPO = definitely successful
    elif status.lower() in ['dead', 'closed', 'defunct']:
        successful = False  # Failed
    elif status.lower() in ['operating', 'active']:
        # Operating: success if raised significant funding
        if total_funding and total_funding > 50:
            successful = True  # Well-funded = top 25%
        elif total_funding and total_funding > 5:
            successful = False  # Moderate success
        else:
            successful = None  # Unknown (no funding data)
    
    # Determine exit type
    if status.lower() == 'acquired' or 'acquired' in status.lower():
        exit_type = 'acquired'
    elif 'ipo' in status.lower():
        exit_type = 'ipo'
    elif status.lower() in ['dead', 'closed', 'defunct', 'failed']:
        exit_type = 'failed'
    elif status.lower() == 'operating' or status.lower() == 'active':
        exit_type = 'operating'
    else:
        exit_type = 'unknown'
    
    entry = {
        "company_id": company_name.lower().replace(' ', '_').replace('.', ''),
        "name": company_name,
        "yc_batch": batch,
        "description_short": description[:200] if description else "",
        "description_long": description,
        "founder_count": founder_count,
        "founders": [f.strip() for f in founders.split(',') if f.strip() and f != 'nan'][:5],
        "founding_team_narrative": f"Team of {founder_count} founders: {founders[:100]}..." if founders and founders != 'nan' else "",
        "market_category": str(row['Categories']).split(',')[0].strip() if pd.notna(row['Categories']) else 'uncategorized',
        "total_funding_usd": total_funding,
        "last_valuation_usd": None,  # Not in CSV
        "exit_type": exit_type,
        "status": status,
        "years_active": 2025 - int(row['Year Founded']) if pd.notna(row['Year Founded']) and str(row['Year Founded']).isdigit() else 5,
        "successful": successful,
        "location": str(row['Headquarters (City)']) if pd.notna(row['Headquarters (City)']) else "",
        "data_source": "GitHub YC CSV + Crunchbase public data",
        "collected_date": "2025-11-10"
    }
    
    processed.append(entry)

print(f"✓ Processed {len(processed)} companies")

# Filter to those with descriptions (essential for analysis)
with_descriptions = [c for c in processed if c['description_long'] and len(c['description_long']) > 50]
print(f"✓ {len(with_descriptions)} have descriptions (>50 chars)")

# Filter to those with outcome data
with_outcomes = [c for c in with_descriptions if c['successful'] is not None or c['exit_type'] != 'unknown']
print(f"✓ {len(with_outcomes)} have outcome data")

# Save full dataset
output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/startups_large_dataset.json'
with open(output_path, 'w') as f:
    json.dump(with_descriptions, f, indent=2)

print(f"\n✓ Saved {len(with_descriptions)} companies to: {output_path}")

# Statistics
if with_descriptions:
    statuses = {}
    for c in with_descriptions:
        status = c['exit_type']
        statuses[status] = statuses.get(status, 0) + 1
    
    successful_count = sum(1 for c in with_descriptions if c['successful'] is True)
    failed_count = sum(1 for c in with_descriptions if c['successful'] is False)
    unknown_count = sum(1 for c in with_descriptions if c['successful'] is None)
    
    print("\nDATASET COMPOSITION:")
    print(f"  Total: {len(with_descriptions)}")
    print(f"  Successful: {successful_count}")
    print(f"  Failed/Struggling: {failed_count}")
    print(f"  Unknown outcome: {unknown_count}")
    print(f"\nBy exit type:")
    for status, count in sorted(statuses.items(), key=lambda x: x[1], reverse=True):
        print(f"    {status}: {count}")
    
    # Founder count distribution
    founder_counts = {}
    for c in with_descriptions:
        fc = c['founder_count']
        founder_counts[fc] = founder_counts.get(fc, 0) + 1
    
    print(f"\nFounder count distribution:")
    for count in sorted(founder_counts.keys()):
        print(f"    {count} founder(s): {founder_counts[count]}")

print("\n" + "=" * 80)
print("READY FOR ANALYSIS")
print("=" * 80)
print(f"Run: python3 narrative_optimization/domains/startups/analyze_startups.py")

