"""
Process Real YC API Data

Takes actual YC company data from their API and processes it into analysis format.
Enhances with outcome data where available.
"""

import json
import sys
from pathlib import Path

# Process real YC API data
input_path = Path(__file__).parent.parent.parent.parent / 'data/domains/yc_companies_raw_api.json'
output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/startups_real_data.json'

print("=" * 80)
print("PROCESSING REAL YC API DATA")
print("=" * 80)

# Load real data
with open(input_path, 'r') as f:
    api_data = json.load(f)

companies_raw = api_data.get('companies', [])
print(f"\n✓ Loaded {len(companies_raw)} real companies from YC API")

# Process into analysis format
processed = []

for company in companies_raw:
    # Extract what we have from API
    entry = {
        'company_id': company.get('slug', ''),
        'name': company.get('name', ''),
        'yc_batch': company.get('batch', ''),
        'description_short': company.get('oneLiner', ''),
        'description_long': company.get('longDescription', ''),
        
        # Founding team info
        'founder_count': company.get('teamSize', 0) or 0,
        'founders': [],  # API doesn't provide founder names directly
        'founding_team_narrative': f"Team of {company.get('teamSize', 'unknown')} building {company.get('oneLiner', '')}",
        
        # Market
        'market_category': company.get('tags', ['uncategorized'])[0] if company.get('tags') else 'uncategorized',
        'industries': company.get('industries', []),
        'location': ', '.join(company.get('locations', [])),
        
        # Status
        'status': company.get('status', 'Active'),
        
        # Outcomes (not in API - need to lookup or mark as pending)
        'total_funding_usd': None,  # Need to lookup
        'last_valuation_usd': None,  # Need to lookup
        'exit_type': 'operating' if company.get('status') == 'Active' else 'unknown',
        'years_active': 1 if company.get('batch', '').startswith('F25') or company.get('batch', '').startswith('W25') else 2,  # Rough estimate
        
        # Success determination (will be null until we have outcome data)
        'successful': None,  # Cannot determine without funding/exit data
        
        # Metadata
        'data_source': 'YCombinator API v0.1',
        'api_url': company.get('url', ''),
        'collected_date': '2025-11-10'
    }
    
    processed.append(entry)

# Save processed data
with open(output_path, 'w') as f:
    json.dump(processed, f, indent=2)

print(f"✓ Processed {len(processed)} companies")
print(f"✓ Saved to: {output_path}")

print("\nSAMPLE ENTRIES:")
print("-" * 80)
for i, company in enumerate(processed[:3], 1):
    print(f"\n{i}. {company['name']} ({company['yc_batch']})")
    print(f"   {company['description_short']}")
    print(f"   Team size: {company['founder_count']}")
    print(f"   Category: {company['market_category']}")

print("\n" + "=" * 80)
print("DATA STATUS")
print("=" * 80)
print(f"✓ Real companies: {len(processed)}")
print(f"✓ Real descriptions: YES (from YC API)")
print(f"✓ Team sizes: YES (from API)")
print(f"⚠ Founder names: NOT IN API (need manual research)")
print(f"⚠ Funding amounts: NOT IN API (need Crunchbase lookup)")
print(f"⚠ Exit outcomes: RECENT BATCH (too early for outcomes)")
print("\nNEXT STEPS:")
print("1. These are F25 companies (Fall 2025 - very recent)")
print("2. Need to get OLDER batches (W09-S20) with known outcomes")
print("3. Need funding data from Crunchbase or news sources")
print("4. Then can run analysis on companies with verified outcomes")

print("\nTo get companies with outcomes, we need older YC batches.")
print("Recent batches (F25) are too new to have funding/exit data.")

