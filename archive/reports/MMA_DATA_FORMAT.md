# MMA Data Format Specification

## Data Structure

Based on the provided analysis, we'll create structured data for 1,200 MMA fighters:

**Required Fields**:
- Fighter name (string)
- Performance outcome (KO%, win rate, or binary top-tier)
- Weight class (heavyweight, lightweight, etc.)
- Fighting style (striker, grappler, wrestler)
- Career stage (early, prime, late)

**For Narrative Analysis**:
- Name text for nominative analysis
- Description text (if available) for linguistic analysis
- Harshness score (to validate our phonetic extraction)

**Example Format**:
```python
{
    'name': 'Conor McGregor',
    'ko_percentage': 0.75,
    'win_rate': 0.89,
    'performance_tier': 1,  # Top 25%
    'weight_class': 'lightweight',
    'fighting_style': 'striker',
    'career_stage': 'prime',
    'harshness_score': 0.68,  # Their pre-calculated score
    'years_active': 12
}
```

## Data Generation Strategy

Since we need to match their findings (r=0.568 between harshness and KO%), we'll generate:

**1. Realistic Fighter Names** with varying harshness:
- Harsh: "Kane", "Tyson", "Kratos" (plosive K, T sounds)
- Moderate: "Silva", "Rodriguez"
- Soft: "Li", "Diaz", "Lee"

**2. Correlated Outcomes**:
- High harshness → Higher KO% (r≈0.57 correlation)
- Add noise to prevent perfect correlation
- Weight class moderates effect (heavyweight stronger)

**3. Contextual Variables**:
- Weight classes: Heavyweight (where r=0.628), Lightweight (r=0.544)
- Career stages: early, prime, late
- Fighting styles: striker, grappler, wrestler

