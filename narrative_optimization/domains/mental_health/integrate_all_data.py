"""
Master Data Integration Script

Merges all data sources into complete mental health disorder database:
1. Base disorders (names, phonetics) - 510 from NARRATIVE_EXPORT
2. Automated data (NIH, PubMed, mortality)
3. Stigma scores (from published literature)
4. Clinical outcomes (SAMHSA, meta-analyses)

Assigns quality tiers and creates expanded dataset for analysis.

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import json
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data_loader import MentalHealthDataLoader
from stigma_database import StigmaDatabase
from clinical_outcomes_database import ClinicalOutcomesDatabase


def integrate_all_data_sources():
    """
    Integrate all data sources into complete database.
    
    Returns
    -------
    dict
        Complete integrated dataset with quality tiers
    """
    print("\n" + "="*70)
    print("MENTAL HEALTH DATA INTEGRATION")
    print("="*70 + "\n")
    
    # Load base disorders
    loader = MentalHealthDataLoader()
    disorders = loader.load_disorders()
    
    print(f"Step 1: Loaded {len(disorders)} base disorders from NARRATIVE_EXPORT\n")
    
    # Load automated data (if exists)
    automated_file = Path(__file__).parent / 'data' / 'disorders_with_automated_data.json'
    if automated_file.exists():
        with open(automated_file, 'r') as f:
            automated_data = json.load(f)
        automated_disorders = {d.get('disorder_name'): d 
                             for d in automated_data.get('disorders', [])}
        print(f"Step 2: Loaded automated data for {len(automated_disorders)} disorders\n")
    else:
        automated_disorders = {}
        print("Step 2: No automated data file found (run collect_automated_data.py first)\n")
    
    # Load stigma database
    stigma_db = StigmaDatabase()
    print(f"Step 3: Loaded stigma data for {len(stigma_db.stigma_data)} disorders\n")
    
    # Load clinical outcomes
    outcomes_db = ClinicalOutcomesDatabase()
    print(f"Step 4: Loaded clinical outcomes for {len(outcomes_db.outcomes_data)} disorders\n")
    
    # Integrate all sources
    print("Step 5: Integrating all sources...\n")
    
    integrated_disorders = []
    tier_counts = {'gold': 0, 'silver': 0, 'bronze': 0, 'incomplete': 0}
    
    for disorder in disorders:
        name = disorder.get('disorder_name', '')
        
        if not name:
            continue
        
        # Start with base disorder
        integrated = disorder.copy()
        
        # Merge automated data if available
        if name in automated_disorders:
            auto_data = automated_disorders[name]
            integrated['nih_funding'] = auto_data.get('nih_funding', {})
            integrated['pubmed_data'] = auto_data.get('pubmed_data', {})
            integrated['mortality_data'] = auto_data.get('mortality_data', {})
        
        # Add stigma data
        stigma_data = stigma_db.get_stigma_with_fallback(name)
        integrated['stigma_data'] = stigma_data
        
        # Merge with existing social_impact if present
        if 'social_impact' not in integrated:
            integrated['social_impact'] = {}
        integrated['social_impact']['stigma_score'] = stigma_data.get('stigma_score')
        integrated['social_impact']['data_quality'] = stigma_data.get('data_quality', 'estimated')
        
        # Add clinical outcomes
        outcomes = outcomes_db.get_outcomes_with_fallback(name)
        integrated['clinical_outcomes_expanded'] = outcomes
        
        # Merge with existing clinical_outcomes if present
        if 'clinical_outcomes' not in integrated:
            integrated['clinical_outcomes'] = {}
        integrated['clinical_outcomes']['treatment_seeking_rate'] = outcomes.get('treatment_seeking_rate')
        integrated['clinical_outcomes']['hospitalization_rate_annual'] = outcomes.get('hospitalization_rate_annual')
        integrated['clinical_outcomes']['treatment_delay_months'] = outcomes.get('treatment_delay_months')
        
        # Assign quality tier
        tier = assess_data_quality(integrated)
        integrated['data_tier'] = tier
        tier_counts[tier] += 1
        
        integrated_disorders.append(integrated)
    
    # Summary
    print("\n" + "-"*70)
    print("INTEGRATION COMPLETE")
    print("-"*70)
    print(f"\nTotal Disorders: {len(integrated_disorders)}")
    print(f"\nQuality Tiers:")
    print(f"  🥇 Gold (all fields, high quality): {tier_counts['gold']}")
    print(f"  🥈 Silver (stigma + 2 outcomes): {tier_counts['silver']}")
    print(f"  🥉 Bronze (stigma + 1 outcome): {tier_counts['bronze']}")
    print(f"  ⚪ Incomplete (missing critical fields): {tier_counts['incomplete']}")
    
    # Calculate completeness
    complete_for_analysis = tier_counts['gold'] + tier_counts['silver'] + tier_counts['bronze']
    print(f"\n✅ Ready for analysis: {complete_for_analysis}/{len(integrated_disorders)} ({complete_for_analysis/len(integrated_disorders)*100:.1f}%)")
    
    # Save integrated dataset
    output_file = Path(__file__).parent / 'data' / 'integrated_disorders_complete.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'n_disorders': len(integrated_disorders),
                'n_complete': complete_for_analysis,
                'quality_tiers': tier_counts,
                'data_sources': [
                    'NARRATIVE_EXPORT (base disorders)',
                    'NIH RePORTER (funding)',
                    'PubMed (article counts)',
                    'Published literature (stigma)',
                    'SAMHSA/meta-analyses (clinical outcomes)'
                ],
                'integration_date': '2025-11-10'
            },
            'disorders': integrated_disorders
        }, f, indent=2)
    
    print(f"\n💾 Integrated dataset saved to: {output_file}")
    
    # Generate analysis-ready subset
    analysis_ready = [d for d in integrated_disorders 
                     if d.get('data_tier') in ['gold', 'silver', 'bronze']]
    
    analysis_file = Path(__file__).parent / 'data' / 'analysis_ready_disorders.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis_ready, f, indent=2)
    
    print(f"💾 Analysis-ready subset saved to: {analysis_file}")
    print(f"   ({len(analysis_ready)} disorders with complete data)")
    
    print("\n" + "="*70 + "\n")
    
    return {
        'integrated': integrated_disorders,
        'analysis_ready': analysis_ready,
        'tier_counts': tier_counts
    }


def assess_data_quality(disorder: Dict) -> str:
    """
    Assess data quality tier for a disorder.
    
    Tiers:
    - Gold: All critical fields from primary sources
    - Silver: Stigma + 2 clinical outcomes  
    - Bronze: Stigma + 1 clinical outcome
    - Incomplete: Missing critical data
    
    Parameters
    ----------
    disorder : dict
        Integrated disorder record
    
    Returns
    -------
    str
        Quality tier: 'gold', 'silver', 'bronze', or 'incomplete'
    """
    # Check critical fields
    has_stigma = (disorder.get('social_impact', {}).get('stigma_score') is not None or
                 disorder.get('stigma_data', {}).get('stigma_score') is not None)
    
    has_treatment_seeking = disorder.get('clinical_outcomes', {}).get('treatment_seeking_rate') is not None
    has_mortality = disorder.get('clinical_outcomes', {}).get('mortality_rate_per_100k') is not None
    has_hospitalization = disorder.get('clinical_outcomes', {}).get('hospitalization_rate_annual') is not None
    has_delay = disorder.get('clinical_outcomes', {}).get('treatment_delay_months') is not None
    
    # Count clinical outcome metrics
    clinical_count = sum([has_treatment_seeking, has_mortality, has_hospitalization, has_delay])
    
    # Check data quality flags
    stigma_quality = (disorder.get('stigma_data', {}).get('data_quality', 'estimated') == 'high' or
                     disorder.get('social_impact', {}).get('data_quality', 'estimated') == 'high')
    
    # Assign tier
    if has_stigma and clinical_count >= 3 and stigma_quality:
        return 'gold'
    elif has_stigma and clinical_count >= 2:
        return 'silver'
    elif has_stigma and clinical_count >= 1:
        return 'bronze'
    else:
        return 'incomplete'


if __name__ == '__main__':
    result = integrate_all_data_sources()
    
    print(f"🎉 SUCCESS: {len(result['analysis_ready'])} disorders ready for statistical analysis")
    print(f"\nBreakdown:")
    for tier, count in result['tier_counts'].items():
        print(f"  {tier}: {count}")
    
    print("\nNext step: Re-run experiments with expanded dataset")

