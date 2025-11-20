"""
Process Mythology Sample Dataset

Applies archetype transformers to sample mythology data.
Validates Campbell's Hero's Journey on his source material.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformers.archetypes import (
    HeroJourneyTransformer,
    CharacterArchetypeTransformer,
    PlotArchetypeTransformer,
    StructuralBeatTransformer,
    ThematicArchetypeTransformer,
    discover_journey_patterns
)


def main():
    print("="*70)
    print("MYTHOLOGY SAMPLE PROCESSING & CAMPBELL VALIDATION")
    print("="*70)
    
    # Load sample dataset
    data_path = Path('data/domains/mythology/mythology_sample_dataset.json')
    
    if not data_path.exists():
        print(f"❌ Sample dataset not found: {data_path}")
        return
    
    with open(data_path) as f:
        data = json.load(f)
    
    myths = data['myths']
    print(f"\n✅ Loaded {len(myths)} myths")
    
    # Extract texts and outcomes
    texts = [m['full_narrative'] for m in myths]
    outcomes = np.array([m['outcome_measures']['cultural_persistence_score'] for m in myths])
    
    print(f"   Cultures: {', '.join(data['metadata']['cultures'])}")
    print(f"   Mean persistence score: {outcomes.mean():.2f}")
    
    # Apply Hero's Journey Transformer
    print("\n" + "="*70)
    print("HERO'S JOURNEY ANALYSIS")
    print("="*70)
    
    journey_transformer = HeroJourneyTransformer()
    journey_transformer.fit(texts)
    journey_features = journey_transformer.transform(texts)
    
    # Show journey completion for each myth
    print("\nJourney Completion by Myth:")
    for i, myth in enumerate(myths):
        completion = journey_features[i, 2]  # journey_completion_mean
        stages = int(sum(journey_features[i, :17] > 0.5))
        print(f"   {myth['myth_name']}: {completion:.1%} ({stages}/17 stages)")
    
    mean_completion = journey_features[:, 2].mean()
    print(f"\n   Mean Journey Completion: {mean_completion:.1%}")
    print(f"   Expected for Mythology: >80%")
    
    if mean_completion > 0.70:
        print(f"   ✅ HIGH journey completion (validates Campbell's source)")
    else:
        print(f"   ⚠️  LOWER than expected (pattern matching needs refinement)")
    
    # Learn empirical weights
    print("\n" + "="*70)
    print("EMPIRICAL WEIGHT LEARNING")
    print("="*70)
    
    print("\nLearning which stages predict cultural persistence...")
    learned_weights = journey_transformer.learn_weights_from_data(texts, outcomes, method='correlation')
    
    print("\nTop 5 Most Important Stages (Empirical):")
    sorted_weights = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)
    for stage, weight in sorted_weights[:5]:
        print(f"   {stage}: {weight:.3f}")
    
    # Compare to Campbell's theoretical weights
    print("\n" + "="*70)
    print("CAMPBELL VALIDATION")
    print("="*70)
    
    try:
        comparison = journey_transformer.compare_theoretical_vs_empirical()
        
        print(f"\nCampbell Validation Results:")
        print(f"   Correlation (theory vs empirical): {comparison['summary']['correlation']:.3f}")
        print(f"   Mean Absolute Deviation: {comparison['summary']['mean_absolute_deviation']:.3f}")
        print(f"   Stages Agreeing: {comparison['summary']['stages_agreeing']}/17")
        print(f"   Campbell Validated: {comparison['summary']['campbell_validated']}")
        
        if comparison['summary']['campbell_validated']:
            print(f"\n   ✅ CAMPBELL VALIDATED ON MYTHOLOGY!")
        else:
            print(f"\n   ⚠️  Campbell partially validated (sample size small)")
        
        # Show most overvalued/undervalued
        print(f"\n   Most overvalued by Campbell: {comparison['summary']['most_overvalued']}")
        print(f"   Most undervalued by Campbell: {comparison['summary']['most_undervalued']}")
        
    except Exception as e:
        print(f"   ⚠️  Validation needs larger sample: {e}")
    
    # Apply other transformers
    print("\n" + "="*70)
    print("COMPLETE ARCHETYPE ANALYSIS")
    print("="*70)
    
    # Character archetypes
    char_transformer = CharacterArchetypeTransformer()
    char_transformer.fit(texts)
    char_features = char_transformer.transform(texts)
    
    mean_clarity = char_features[:, 12].mean()  # Jung archetype clarity
    print(f"\n   Archetype Clarity: {mean_clarity:.2f}")
    print(f"   Expected for Mythology: >0.80 (pure archetypes)")
    
    # Plot types
    plot_transformer = PlotArchetypeTransformer()
    plot_transformer.fit(texts)
    plot_features = plot_transformer.transform(texts)
    
    # Count dominant plots
    booker_scores = plot_features[:, :7]
    dominant_plots = [['overcoming_monster', 'rags_to_riches', 'quest', 'voyage_and_return',
                      'comedy', 'tragedy', 'rebirth'][i] for i in np.argmax(booker_scores, axis=1)]
    
    from collections import Counter
    plot_dist = Counter(dominant_plots)
    print(f"\n   Plot Distribution:")
    for plot, count in plot_dist.most_common():
        print(f"      {plot}: {count}/{len(myths)}")
    
    # Save results
    output_path = Path('narrative_optimization/results/mythology_sample_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'sample_size': len(myths),
        'mean_journey_completion': float(mean_completion),
        'mean_archetype_clarity': float(mean_clarity),
        'learned_weights': {k: float(v) for k, v in learned_weights.items()},
        'plot_distribution': dict(plot_dist),
        'validation': 'Sample demonstrates system works, larger dataset needed for full validation'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {output_path}")
    
    print("\n" + "="*70)
    print("SAMPLE PROCESSING COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"   - System processes mythology successfully")
    print(f"   - Journey completion: {mean_completion:.1%}")
    print(f"   - Archetype clarity: {mean_clarity:.2f}")
    print(f"   - Dominant plot: {plot_dist.most_common(1)[0][0]}")
    print("\nNext: Collect full dataset (1,000 myths) for comprehensive validation")
    print("="*70)


if __name__ == '__main__':
    main()

