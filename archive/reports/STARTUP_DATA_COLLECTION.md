# Startup Data Collection - REAL DATA ONLY

**Target**: 100+ actual YCombinator companies  
**Status**: Ready for manual collection  
**Format**: Verified public data with sources

---

## Data Collection Process

### Step 1: Visit YC Company Directory

**URL**: https://www.ycombinator.com/companies

**Action**: Browse companies, focus on those with:
- Clear descriptions
- Known outcomes (funding disclosed, exit announced, or clear failure)
- Founding team information available

### Step 2: For Each Company, Collect

**Required Fields** (all must be real/verifiable):

1. **Company Name** (exact from YC)
2. **YC Batch** (e.g., W09, S12)
3. **Description** (from YC one-liner)
4. **Extended Description** (from company website About page)
5. **Founders** (names from YC page or LinkedIn)
6. **Founder Count** (count them)
7. **Team Narrative** (from About page, TechCrunch articles, or YC interviews - how they describe themselves)
8. **Market Category** (SaaS, marketplace, fintech, etc.)

**Outcome Fields** (verify from sources):

9. **Total Funding** (from Crunchbase or press releases - USD millions)
10. **Valuation** (if publicly disclosed - USD millions, else null)
11. **Exit Type**: operating | acquired | ipo | failed | unknown
12. **Years Active** (founding year to now or exit year)

**Source Fields**:

13. **Data Sources** (URLs where you got information)
14. **Collection Date** (today's date)

### Step 3: Data Quality Rules

**MUST**:
- ✓ Be a real YC company (verifiable on YC website)
- ✓ Have actual description (not made up)
- ✓ Have verifiable outcome data (or marked as unknown)
- ✓ Cite sources

**NEVER**:
- ✗ Make up descriptions
- ✗ Estimate funding (use null if unknown)
- ✗ Guess outcomes
- ✗ Use placeholder/dummy data

---

## Target Distribution

**Aim for balanced dataset**:
- 25 unicorns/big exits (>$1B valuation or major acquisition)
- 50 moderate successes (raised $10M+, still operating)
- 25 failed or minimal progress

**Batch diversity**:
- Mix of recent (2023-2024) and older (2009-2015)
- Recent: Outcomes still evolving
- Older: Clear outcomes visible

**Category diversity**:
- SaaS, marketplace, fintech, developer tools, consumer, B2B

---

## Example Entry (REAL DATA)

```json
{
  "company_id": "stripe",
  "name": "Stripe",
  "yc_batch": "S09",
  "description_short": "Developer-first payments platform",
  "description_long": "Stripe is a technology company that builds economic infrastructure for the internet. Businesses of every size use our software to accept payments and manage their businesses online.",
  "founders": ["Patrick Collison", "John Collison"],
  "founder_count": 2,
  "founding_team_narrative": "Two Irish brothers, both technical prodigies. Patrick was youngest winner of Young Scientist award. Both dropped out of college. Strong sibling partnership with complementary strengths in product and growth.",
  "market_category": "fintech",
  "total_funding_usd": 2200,
  "last_valuation_usd": 95000,
  "exit_type": "operating",
  "years_active": 13,
  "current_status": "private_unicorn",
  "data_sources": {
    "description": "https://stripe.com/about",
    "funding": "https://www.crunchbase.com/organization/stripe",
    "team": "https://techcrunch.com/tag/stripe/"
  },
  "collected_date": "2025-11-10"
}
```

---

## Collection Tools

### Quick Search Queries

For each company, use these searches:

**Funding**:
- `[Company Name] funding crunchbase`
- `[Company Name] raises series`
- `[Company Name] valuation`

**Outcome**:
- `[Company Name] acquisition`
- `[Company Name] IPO`
- `[Company Name] shuts down`

**Team**:
- `[Company Name] founders linkedin`
- `[Company Name] team about`

### Verification

**Check multiple sources**:
- If Crunchbase says $50M funding, verify with TechCrunch article
- If claiming acquisition, find official press release
- If IPO, check NASDAQ/NYSE listings

---

## Start With These (Research Required)

### Top YC Companies (Research outcomes)

**Unicorns/Big Exits**:
- Airbnb (IPO 2020)
- Stripe (private, $95B)
- Coinbase (IPO 2021)
- Dropbox (IPO 2018)
- DoorDash (IPO 2020)
- Instacart (IPO 2023)
- Reddit (IPO 2024)
- Twitch (acquired by Amazon)

**Moderate Successes** (research current status):
- Amplitude, Brex, Checkr, Faire, Ginkgo Bioworks, etc.

**Known Failures**:
- Homejoy (shut down 2015)
- Exec (shut down 2014)
- Kivo (shut down)
- Ridejoy (shut down)

**Recent Batches** (W24, S23 - outcomes still evolving):
- Research recent batches
- May have less outcome data but current descriptions

---

## Output File

**Save collected data to**:  
`/Users/michaelsmerconish/Desktop/RandomCode/novelization/data/domains/startups_real_data.json`

**Format**: JSON array of company objects (see example above)

---

## Timeline

**Week 1**: Collect 30 companies (10/day - 30min each)
**Week 2**: Collect 40 more (70 total)
**Week 3**: Collect 30 more (100 total)
**Week 4**: Run analysis on complete dataset

**Start now. Research is required. No shortcuts.**

---

## When Dataset Is Ready

Run: `python3 narrative_optimization/domains/startups/analyze_startups.py`

This will:
1. Load real startup data
2. Extract structural properties
3. Predict formula from structure
4. Discover empirical formula
5. Validate prediction
6. Test "better stories win" for startups
7. Generate findings

**Status**: Data collection tools ready. Begin manual research and collection.
"""
    
    with open(guide_path, 'w') as f:
        f.write(guide_md)
    
    print(f"✓ Data collection guide created: {guide_path}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Open: https://www.ycombinator.com/companies")
    print(f"  2. Follow guide to research REAL companies")
    print(f"  3. Collect data for 100+ startups")
    print(f"  4. Save to: data/domains/startups_real_data.json")
    print(f"  5. Run analysis when complete")
    print(f"\n📋 See guide for detailed instructions: {guide_path}")


if __name__ == "__main__":
    main()

