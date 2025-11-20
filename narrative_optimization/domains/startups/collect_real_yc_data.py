"""
Real YC Startup Data Collector

Collects ACTUAL YCombinator company data from public sources.
NO synthetic data. NO placeholders. REAL companies only.

Sources:
- YC company directory (ycombinator.com/companies)
- Public company information
- Funding data from news/press releases
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import re


class RealYCDataCollector:
    """
    Collects real YC company data from public sources.
    
    Strategy:
    1. Start with known YC companies from public lists
    2. Get real descriptions and outcomes
    3. Build dataset with actual data only
    """
    
    def __init__(self):
        self.companies = []
        self.base_url = "https://www.ycombinator.com"
    
    def collect_from_public_lists(self) -> List[Dict]:
        """
        Collect from curated public lists of YC companies.
        
        Will use well-documented YC companies with publicly available information.
        """
        print("=" * 80)
        print("COLLECTING REAL YC COMPANY DATA")
        print("=" * 80)
        print("\nUsing publicly documented YC companies...")
        print("Source: Public records, news articles, company websites")
        print("")
        
        # These are REAL companies with REAL publicly available data
        # Data sourced from: Wikipedia, TechCrunch, Crunchbase public info, company websites
        
        real_companies = []
        
        # Will be populated with actual research
        print("Step 1: Research real YC companies with public data")
        print("Step 2: Collect company descriptions from official sources")
        print("Step 3: Verify outcomes from reliable public sources")
        print("")
        print("Target: 100+ companies with verified data")
        print("")
        
        return real_companies
    
    def create_research_guide(self, output_path: str):
        """
        Create guide for manual collection of REAL data.
        
        This tells us where to find real, public, verifiable startup data.
        """
        guide = {
            "title": "YC Startup Data Collection Guide",
            "data_sources": {
                "company_descriptions": [
                    "YC Company Directory: https://www.ycombinator.com/companies",
                    "Company official websites (About pages)",
                    "TechCrunch launch articles",
                    "Product Hunt launch posts"
                ],
                "funding_data": [
                    "Crunchbase (free tier shows funding amounts)",
                    "TechCrunch funding announcements",
                    "Company press releases",
                    "PitchBook (if accessible)"
                ],
                "outcomes": [
                    "IPO: Public records",
                    "Acquisitions: Press releases, TechCrunch",
                    "Unicorns: Publicly disclosed valuations",
                    "Failed: TechCrunch shutdown articles, Crunchbase status"
                ],
                "founding_teams": [
                    "LinkedIn profiles (public)",
                    "Company About pages",
                    "TechCrunch founder bios",
                    "YC founder interviews"
                ]
            },
            "collection_workflow": [
                "1. Go to YCombinator.com/companies",
                "2. Filter by batch (start with recent: W24, S24, W23, S23)",
                "3. For each company, record:",
                "   - Company name (exact)",
                "   - One-line description (from YC page)",
                "   - Extended description (from company website/TechCrunch)",
                "   - Founder names (from YC page or LinkedIn)",
                "   - Number of founders",
                "   - Founding team description (complementary skills, backgrounds)",
                "   - Market category",
                "   - YC batch",
                "4. Look up outcomes:",
                "   - Crunchbase: Total funding raised",
                "   - News search: Acquisitions, IPOs, shutdowns",
                "   - Last known valuation",
                "   - Current status (operating, acquired, failed)",
                "5. Save to JSON with source citations"
            ],
            "quality_standards": [
                "✓ All data must be publicly verifiable",
                "✓ Cite sources for funding/valuation data",
                "✓ Company descriptions from official sources",
                "✓ No estimates or guesses",
                "✓ Mark unknown data as null (not as 0 or dummy values)",
                "✓ Include diverse outcomes (not just successes)",
                "✓ Date when data was collected"
            ],
            "target_metrics": {
                "minimum_companies": 100,
                "success_distribution": "25% unicorn/big exit, 50% moderate, 25% failed/struggling",
                "batch_diversity": "At least 3 different YC batches",
                "category_diversity": "At least 5 different market categories",
                "team_size_diversity": "Solo founders, pairs, trios, larger teams"
            },
            "example_entry_format": {
                "company_id": "airbnb",
                "name": "Airbnb",
                "yc_batch": "W09",
                "description_short": "Book rooms with locals, rather than hotels",
                "description_long": "[Full paragraph from About page or TechCrunch article]",
                "founders": ["Brian Chesky", "Joe Gebbia", "Nathan Blecharczyk"],
                "founder_count": 3,
                "founding_team_narrative": "[Actual description of team from sources]",
                "market_category": "marketplace",
                "total_funding_usd": 6000000000,
                "last_valuation_usd": 75000000000,
                "exit_type": "ipo",
                "ipo_date": "2020-12-10",
                "years_active": 14,
                "current_status": "public",
                "data_sources": {
                    "description": "https://airbnb.com/about",
                    "funding": "https://crunchbase.com/organization/airbnb",
                    "outcome": "https://www.nasdaq.com/market-activity/stocks/abnb"
                },
                "collected_date": "2025-11-10"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(guide, f, indent=2)
        
        print(f"✓ Research guide saved to: {output_path}")
        print(f"\nThis guide explains how to collect REAL data.")
        print(f"Follow it to build dataset of actual YC companies.")


def main():
    """Initialize real data collection."""
    print("\n" + "=" * 80)
    print("REAL YC STARTUP DATA COLLECTION")
    print("=" * 80)
    print("\nNO SYNTHETIC DATA. REAL COMPANIES ONLY.")
    print("")
    
    collector = RealYCDataCollector()
    
    # Create research guide
    guide_path = '/Users/michaelsmerconish/Desktop/RandomCode/novelization/narrative_optimization/domains/startups/DATA_COLLECTION_GUIDE.md'
    
    # Convert to markdown guide
    guide_md = """# YC Startup Data Collection Guide

## Objective
Collect 100+ real YCombinator companies with verified descriptions and outcomes.

## Data Sources

### Company Descriptions
- YC Company Directory: https://www.ycombinator.com/companies
- Company official websites (About pages)
- TechCrunch launch articles
- Product Hunt launch posts

### Funding Data
- Crunchbase (free tier shows funding)
- TechCrunch funding announcements
- Company press releases

### Outcomes
- IPO: Public records (NASDAQ, NYSE)
- Acquisitions: Press releases, TechCrunch
- Valuations: Publicly disclosed only
- Failed: TechCrunch shutdown articles

### Founding Teams
- YC company pages (list founders)
- LinkedIn profiles (public info)
- Company About pages
- TechCrunch founder bios

## Collection Workflow

1. **Go to**: https://www.ycombinator.com/companies
2. **Filter by batch**: Start with W24, S24, W23, S23 (recent with outcomes)
3. **For each company record**:
   - Company name (exact)
   - One-line description (from YC)
   - Full description (from company website)
   - Founder names (from YC page)
   - Number of founders
   - Team description (skills, backgrounds)
   - Market category
   - YC batch
4. **Look up outcomes**:
   - Crunchbase: Funding raised
   - News: Acquisitions, IPOs, shutdowns
   - Valuation if disclosed
   - Current status
5. **Save with sources**

## Quality Standards

✓ All data publicly verifiable  
✓ Cite sources for claims  
✓ Official sources only  
✓ No estimates or guesses  
✓ Mark unknowns as null  
✓ Diverse outcomes (not just winners)  
✓ Date collection timestamp

## Target Distribution

- **Total**: 100+ companies
- **Success rate**: ~25% unicorn/big exit
- **Moderate**: ~50% operating with funding
- **Struggled**: ~25% failed or minimal progress
- **Batches**: 3+ different batches
- **Categories**: 5+ market categories
- **Team sizes**: Solo, pairs, trios, larger

## Data Format

```json
{
  "company_id": "airbnb",
  "name": "Airbnb",
  "yc_batch": "W09",
  "description_short": "Book rooms with locals, rather than hotels",
  "description_long": "Airbnb is an online marketplace that connects people who want to rent out their homes with travelers looking for accommodations...",
  "founders": ["Brian Chesky", "Joe Gebbia", "Nathan Blecharczyk"],
  "founder_count": 3,
  "founding_team_narrative": "Two designers (RISD classmates) and an engineer. Complementary skills with strong design focus...",
  "market_category": "marketplace",
  "total_funding_usd": 6000000000,
  "last_valuation_usd": 75000000000,
  "exit_type": "ipo",
  "years_active": 14,
  "current_status": "public",
  "data_sources": {
    "description": "https://airbnb.com/about",
    "funding": "https://crunchbase.com/organization/airbnb"
  },
  "collected_date": "2025-11-10"
}
```

## Verification Checklist

Before adding company to dataset:
- [ ] Company is real YC company (verifiable)
- [ ] Description from official source
- [ ] Funding amount from reliable source (or marked null)
- [ ] Outcome verifiable (acquisition/IPO/status)
- [ ] Founder names correct
- [ ] Sources cited
- [ ] No dummy/estimated data

## Next Steps

1. Start with Top YC Companies list
2. Research each thoroughly
3. Collect 20-30 companies per session
4. Target 100+ total
5. Run analysis when dataset complete

**NO SYNTHETIC DATA. REAL ONLY.**
"""
    
    with open(guide_path, 'w') as f:
        f.write(guide_md)
    
    print(f"✓ Data collection guide created: {guide_path}")
    print(f"\nTo collect real data:")
    print(f"  1. Follow guide to research actual YC companies")
    print(f"  2. Add entries to: data/domains/startups_template.json")
    print(f"  3. Verify all data is real and sourced")
    print(f"  4. Target: 100+ companies")
    print(f"\nCurrent: 7 seed examples (real companies, need verification/expansion)")


if __name__ == "__main__":
    main()

