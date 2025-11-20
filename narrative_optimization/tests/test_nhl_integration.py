"""
Test NHL integration with enhanced narrative transformers.

Shows how the new transformers integrate into the NHL pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.transformer_selector import TransformerSelector


def main():
    """Test NHL transformer selection with new features."""
    print("NHL TRANSFORMER SELECTION TEST")
    print("="*60)
    
    selector = TransformerSelector()
    
    # Select transformers for NHL (high narrativity sport)
    transformers = selector.select_transformers(
        'nhl', 
        pi_value=0.65,  # High narrativity
        domain_type='sports'
    )
    
    print(f"\nSelected {len(transformers)} transformers for NHL:")
    print("-"*60)
    
    # Group transformers by type
    core = []
    renovation = []
    sports = []
    narrative = []
    other = []
    
    narrative_transformers = {
        'DeepArchetypeTransformer',
        'LinguisticResonanceTransformer',
        'NarrativeCompletionPressureTransformer',
        'TemporalNarrativeCyclesTransformer',
        'CulturalZeitgeistTransformer',
        'RitualCeremonyTransformer',
        'MetaNarrativeAwarenessTransformer',
        'GeographicNarrativeTransformer'
    }
    
    for t in transformers:
        if t in selector.CORE_TRANSFORMERS:
            core.append(t)
        elif t in narrative_transformers:
            narrative.append(t)
        elif 'Temporal' in t or 'Cultural' in t or 'Linguistic' in t:
            renovation.append(t)
        elif t in selector.SPORTS_TRANSFORMERS:
            sports.append(t)
        else:
            other.append(t)
    
    # Print categorized list
    print(f"\nCORE TRANSFORMERS ({len(core)}):")
    for t in core:
        print(f"  - {t}")
        
    print(f"\nNARRATIVE ENHANCEMENT ({len(narrative)}):")
    for t in narrative:
        print(f"  ✨ {t}")
        
    print(f"\nSPORTS-SPECIFIC ({len(sports)}):")
    for t in sports:
        print(f"  - {t}")
        
    print(f"\nRENOVATION TRANSFORMERS ({len(renovation)}):")
    for t in renovation:
        print(f"  - {t}")
        
    if other:
        print(f"\nOTHER ({len(other)}):")
        for t in other:
            print(f"  - {t}")
    
    # Show summary
    selector.print_selection_summary('nhl')
    
    # Estimate feature count
    from src.transformers.transformer_selector import estimate_feature_count
    feature_count = estimate_feature_count(transformers)
    
    print(f"\nTOTAL ESTIMATED FEATURES: {feature_count}")
    
    # Check if our narrative transformers are included
    print("\n" + "="*60)
    print("NARRATIVE TRANSFORMER INCLUSION CHECK:")
    print("-"*60)
    
    included = sum(1 for t in narrative_transformers if t in transformers)
    print(f"✓ {included}/{len(narrative_transformers)} narrative transformers included")
    
    missing = [t for t in narrative_transformers if t not in transformers]
    if missing:
        print("\nMissing transformers:")
        for t in missing:
            print(f"  ✗ {t}")
    else:
        print("\n✓ All narrative transformers included!")


if __name__ == '__main__':
    main()
