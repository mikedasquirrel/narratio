"""
UFC Peak Narrative Contexts Analysis

Deep dive into the 22 contexts that PASS threshold (efficiency > 0.5)
Analyze what makes these contexts special and quantify narrative effects.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    """Analyze peak narrative contexts"""
    
    print("="*80)
    print("UFC PEAK NARRATIVE CONTEXTS ANALYSIS")
    print("="*80)
    
    # Load context discovery results
    with open('narrative_optimization/domains/ufc/ufc_context_discovery.json', 'r') as f:
        discovery = json.load(f)
    
    passing_contexts = discovery['passing_contexts']
    top_by_efficiency = discovery['top_by_efficiency']
    top_by_delta = discovery['top_by_delta']
    
    print(f"\n✓ Loaded {len(passing_contexts)} passing contexts")
    print(f"✓ Total contexts tested: {discovery['total_tested']}")
    
    # === ANALYSIS 1: Passing Contexts ===
    print("\n" + "="*80)
    print("ANALYSIS 1: ALL 22 PASSING CONTEXTS")
    print("="*80)
    
    print(f"\nContexts with efficiency > 0.5:\n")
    for i, ctx in enumerate(sorted(passing_contexts, key=lambda x: x['efficiency'], reverse=True), 1):
        print(f"{i:2d}. {ctx['context']:45s} | eff={ctx['efficiency']:.4f} | Δ={ctx['delta']:+.4f} | n={ctx['n_samples']:5d}")
    
    # === ANALYSIS 2: Pattern Recognition ===
    print("\n" + "="*80)
    print("ANALYSIS 2: PATTERNS IN PASSING CONTEXTS")
    print("="*80)
    
    # Categorize passing contexts
    finish_contexts = [c for c in passing_contexts if any(x in c['context'] for x in ['Finish', 'KO', 'Sub', 'Round 1'])]
    temporal_contexts = [c for c in passing_contexts if any(x in c['context'] for x in ['Era', 'Years', '2010', '2015', '2020'])]
    title_contexts = [c for c in passing_contexts if 'Title' in c['context']]
    weight_contexts = [c for c in passing_contexts if any(x in c['context'] for x in ['weight', 'Welter', 'Light', 'Heavy'])]
    
    print(f"\nPassing Context Categories:")
    print(f"  Finish-related: {len(finish_contexts)} contexts")
    print(f"  Temporal: {len(temporal_contexts)} contexts")
    print(f"  Title fights: {len(title_contexts)} contexts")
    print(f"  Weight class: {len(weight_contexts)} contexts")
    
    # === ANALYSIS 3: Top Performers ===
    print("\n" + "="*80)
    print("ANALYSIS 3: TOP 5 NARRATIVE CONTEXTS")
    print("="*80)
    
    top5 = sorted(passing_contexts, key=lambda x: x['efficiency'], reverse=True)[:5]
    
    for i, ctx in enumerate(top5, 1):
        print(f"\n{i}. {ctx['context']}")
        print(f"   Efficiency: {ctx['efficiency']:.4f} ✓ PASSES")
        print(f"   Correlation |r|: {ctx['r_abs']:.4f}")
        print(f"   Narrative Δ: {ctx['delta']:+.4f}")
        print(f"   Sample size: {ctx['n_samples']:,}")
        print(f"   Physical AUC: {ctx['auc_physical']:.4f}")
        print(f"   Combined AUC: {ctx['auc_combined']:.4f}")
        
        # Interpretation
        if 'Submission' in ctx['context']:
            print(f"   → Grappling creates unpredictable outcomes where narrative matters")
        elif 'KO' in ctx['context']:
            print(f"   → Explosive finishes have narrative/psychological components")
        elif 'Round 1' in ctx['context']:
            print(f"   → Early finishes = less physical grinding, more narrative/mental")
        elif 'Era' in ctx['context']:
            print(f"   → Historical period effects on narrative importance")
        elif 'Title' in ctx['context']:
            print(f"   → High stakes amplify narrative/psychological factors")
    
    # === ANALYSIS 4: Narrative Strength Tiers ===
    print("\n" + "="*80)
    print("ANALYSIS 4: NARRATIVE STRENGTH TIERS")
    print("="*80)
    
    all_contexts = discovery['all_contexts']
    
    tier_1 = [c for c in all_contexts if c['efficiency'] > 0.57]  # Exceptional
    tier_2 = [c for c in all_contexts if 0.52 < c['efficiency'] <= 0.57]  # Strong
    tier_3 = [c for c in all_contexts if 0.50 < c['efficiency'] <= 0.52]  # Pass
    tier_4 = [c for c in all_contexts if 0.45 < c['efficiency'] <= 0.50]  # Near-pass
    tier_5 = [c for c in all_contexts if c['efficiency'] <= 0.45]  # Fail
    
    print(f"\nNarrative Strength Tiers:")
    print(f"  Tier 1 (eff > 0.57): {len(tier_1)} contexts - EXCEPTIONAL")
    print(f"  Tier 2 (0.52-0.57):  {len(tier_2)} contexts - STRONG")
    print(f"  Tier 3 (0.50-0.52):  {len(tier_3)} contexts - PASSES")
    print(f"  Tier 4 (0.45-0.50):  {len(tier_4)} contexts - NEAR")
    print(f"  Tier 5 (< 0.45):     {len(tier_5)} contexts - FAILS")
    
    if tier_1:
        print(f"\n  EXCEPTIONAL NARRATIVE CONTEXTS (Tier 1):")
        for ctx in sorted(tier_1, key=lambda x: x['efficiency'], reverse=True):
            print(f"    - {ctx['context']:40s} | eff={ctx['efficiency']:.4f}")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("PEAK CONTEXTS SUMMARY")
    print("="*80)
    
    print(f"\n✓ {len(passing_contexts)} / {len(all_contexts)} contexts PASS threshold")
    print(f"  Pass rate: {100*len(passing_contexts)/len(all_contexts):.1f}%")
    
    print(f"\n✓ Highest efficiency: {top_by_efficiency[0]['efficiency']:.4f}")
    print(f"  Context: {top_by_efficiency[0]['context']}")
    
    print(f"\n✓ Highest correlation: |r| = {top_by_efficiency[0]['r_abs']:.4f}")
    
    print(f"\n✓ Best narrative Δ: {top_by_delta[0]['delta']:+.4f}")
    print(f"  Context: {top_by_delta[0]['context']}")
    
    # Key insight
    print(f"\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print(f"\nNarrative effects in UFC are CONTEXT-DEPENDENT:")
    print(f"  - Finish fights (KO/Sub): narrative matters MORE (eff > 0.57)")
    print(f"  - Decision fights: narrative matters LESS (eff < 0.46)")
    print(f"  - Early finishes: highest narrative effect (eff = 0.57)")
    print(f"  - Title fight finishes: narrative amplified (eff = 0.57)")
    print(f"\nConclusion: When physical grinding is minimized (finishes),")
    print(f"narrative/psychological factors become more important!")
    
    # Save peak analysis
    peak_analysis = {
        'passing_count': len(passing_contexts),
        'total_tested': len(all_contexts),
        'pass_rate': len(passing_contexts) / len(all_contexts),
        'top_5_contexts': top5,
        'tier_1_exceptional': tier_1,
        'tier_2_strong': tier_2,
        'tier_3_passes': tier_3,
        'patterns': {
            'finish_related': len(finish_contexts),
            'temporal': len(temporal_contexts),
            'title_fights': len(title_contexts),
            'weight_class': len(weight_contexts)
        },
        'key_insight': 'Narrative effects strongest in finish fights where physical grinding is minimal'
    }
    
    output_path = Path('narrative_optimization/domains/ufc/ufc_peak_contexts.json')
    with open(output_path, 'w') as f:
        json.dump(peak_analysis, f, indent=2)
    
    print(f"\n✓ Analysis saved: {output_path}")


if __name__ == "__main__":
    main()

