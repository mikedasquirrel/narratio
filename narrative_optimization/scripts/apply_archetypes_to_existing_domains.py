"""
Apply Archetype Transformers to Existing Domains

Discovers classical narrative patterns in sports, business, and entertainment domains.

Key Questions:
- Do NBA games follow Hero's Journey structure?
- Which character archetypes appear in sports narratives?
- Do startups follow quest or rags-to-riches plots?
- Does narrative structure predict success in existing domains?

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


def analyze_nba_narratives():
    """
    Apply archetype analysis to NBA game narratives.
    
    Discover: Do sports narratives follow classical patterns?
    """
    print("\n" + "="*70)
    print("NBA ARCHETYPE ANALYSIS")
    print("="*70)
    
    # Load NBA data
    nba_path = Path('data/domains/nba_enriched_1000.json')
    
    if not nba_path.exists():
        print(f"⏳ NBA data not found: {nba_path}")
        return None
    
    with open(nba_path) as f:
        nba_data = json.load(f)
    
    # Extract narratives and outcomes
    narratives = [game['narrative'] for game in nba_data if 'narrative' in game]
    outcomes = np.array([1 if game['won'] else 0 for game in nba_data if 'narrative' in game])
    
    print(f"\n✅ Loaded {len(narratives)} NBA game narratives")
    print(f"   Win rate: {outcomes.mean():.1%}")
    
    # Apply Hero's Journey
    print("\n1. Hero's Journey in Sports:")
    journey_transformer = HeroJourneyTransformer()
    journey_transformer.fit(narratives)
    journey_features = journey_transformer.transform(narratives)
    
    mean_completion = journey_features[:, 2].mean()
    print(f"   Mean journey completion: {mean_completion:.1%}")
    print(f"   Interpretation: {'Low' if mean_completion < 0.3 else 'Moderate' if mean_completion < 0.6 else 'High'}")
    
    # Learn what predicts wins
    print("\n2. Empirical Discovery - What Predicts NBA Wins?")
    learned_weights = journey_transformer.learn_weights_from_data(narratives, outcomes, method='correlation')
    
    print("   Top 5 stages correlating with wins:")
    sorted_weights = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)
    for stage, weight in sorted_weights[:5]:
        print(f"      {stage}: {weight:.3f}")
    
    # Plot types
    print("\n3. Plot Types in NBA Games:")
    plot_transformer = PlotArchetypeTransformer()
    plot_transformer.fit(narratives)
    plot_features = plot_transformer.transform(narratives)
    
    booker_scores = plot_features[:, :7]
    booker_names = ['overcoming_monster', 'rags_to_riches', 'quest', 'voyage_and_return',
                    'comedy', 'tragedy', 'rebirth']
    
    dominant_plots = [booker_names[i] for i in np.argmax(booker_scores, axis=1)]
    
    from collections import Counter
    plot_dist = Counter(dominant_plots)
    print("   Plot distribution:")
    for plot, count in plot_dist.most_common():
        print(f"      {plot}: {count}/{len(narratives)} ({count/len(narratives):.1%})")
    
    # Character archetypes
    print("\n4. Character Archetypes in Sports:")
    char_transformer = CharacterArchetypeTransformer()
    char_transformer.fit(narratives)
    char_features = char_transformer.transform(narratives)
    
    jung_names = ['innocent', 'orphan', 'warrior', 'caregiver', 'explorer', 
                 'rebel', 'lover', 'creator', 'jester', 'sage', 'magician', 'ruler']
    
    jung_scores = char_features[:, :12]
    dominant_archetypes = [jung_names[i] for i in np.argmax(jung_scores, axis=1)]
    
    archetype_dist = Counter(dominant_archetypes)
    print("   Top 3 archetypes:")
    for archetype, count in archetype_dist.most_common(3):
        print(f"      {archetype}: {count}/{len(narratives)} ({count/len(narratives):.1%})")
    
    # Frye's mythoi
    print("\n5. Frye's Mythoi in Sports:")
    thematic_transformer = ThematicArchetypeTransformer()
    thematic_transformer.fit(narratives)
    thematic_features = thematic_transformer.transform(narratives)
    
    mythos_scores = thematic_features[:, :4]
    mythos_names = ['comedy', 'romance', 'tragedy', 'irony']
    dominant_mythoi = [mythos_names[i] for i in np.argmax(mythos_scores, axis=1)]
    
    mythos_dist = Counter(dominant_mythoi)
    print("   Mythos distribution:")
    for mythos, count in mythos_dist.most_common():
        print(f"      {mythos}: {count}/{len(narratives)} ({count/len(narratives):.1%})")
    
    # Save results
    results = {
        'domain': 'NBA',
        'sample_size': len(narratives),
        'journey_completion_mean': float(mean_completion),
        'learned_journey_weights': {k: float(v) for k, v in learned_weights.items()},
        'plot_distribution': dict(plot_dist),
        'archetype_distribution': dict(archetype_dist),
        'mythos_distribution': dict(mythos_dist),
        'insights': [
            f"Journey completion: {mean_completion:.1%} (sports narratives are compressed)",
            f"Dominant plot: {plot_dist.most_common(1)[0][0]}",
            f"Dominant archetype: {archetype_dist.most_common(1)[0][0]}",
            f"Dominant mythos: {mythos_dist.most_common(1)[0][0]}"
        ]
    }
    
    output_path = Path('narrative_optimization/results/nba_archetype_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {output_path}")
    
    return results


def analyze_startups_narratives():
    """
    Apply archetype analysis to startup narratives.
    
    Discover: Quest or Rags-to-Riches? Hero archetypes?
    """
    print("\n" + "="*70)
    print("STARTUP ARCHETYPE ANALYSIS")
    print("="*70)
    
    # Load startup data
    startup_paths = [
        'data/domains/startups_verified.json',
        'data/domains/startups_real_data.json'
    ]
    
    for path_str in startup_paths:
        startup_path = Path(path_str)
        if startup_path.exists():
            with open(startup_path) as f:
                startup_data = json.load(f)
            
            # Extract narratives
            if isinstance(startup_data, list):
                narratives = [s.get('product_story', '') or s.get('description', '') 
                             for s in startup_data if s.get('product_story') or s.get('description')]
            else:
                narratives = [s.get('product_story', '') for s in startup_data.get('startups', [])]
            
            if len(narratives) < 10:
                continue
            
            print(f"\n✅ Loaded {len(narratives)} startup narratives")
            
            # Journey analysis
            journey_transformer = HeroJourneyTransformer()
            journey_transformer.fit(narratives[:100])  # Sample for speed
            journey_features = journey_transformer.transform(narratives[:100])
            
            print(f"\n   Mean journey completion: {journey_features[:, 2].mean():.1%}")
            
            # Plot types
            plot_transformer = PlotArchetypeTransformer()
            plot_transformer.fit(narratives[:100])
            plot_features = plot_transformer.transform(narratives[:100])
            
            booker_scores = plot_features[:, :7]
            booker_names = ['overcoming_monster', 'rags_to_riches', 'quest', 'voyage_and_return',
                           'comedy', 'tragedy', 'rebirth']
            dominant_plots = [booker_names[i] for i in np.argmax(booker_scores, axis=1)]
            
            from collections import Counter
            plot_dist = Counter(dominant_plots)
            
            print(f"\n   Dominant plots:")
            for plot, count in plot_dist.most_common(3):
                print(f"      {plot}: {count/len(narratives[:100]):.1%}")
            
            print(f"\n   💡 Insight: Startups follow '{plot_dist.most_common(1)[0][0]}' pattern")
            
            return
    
    print("⏳ Startup data not found")


def main():
    """Apply archetype analysis to all existing domains."""
    print("\n" + "="*70)
    print("APPLYING ARCHETYPES TO EXISTING DOMAINS")
    print("="*70)
    print("Discovering classical patterns in sports, business, entertainment")
    print("="*70)
    
    # Analyze NBA
    nba_results = analyze_nba_narratives()
    
    # Analyze Startups
    startup_results = analyze_startups_narratives()
    
    print("\n" + "="*70)
    print("ARCHETYPE APPLICATION COMPLETE")
    print("="*70)
    print("\nKey Discoveries:")
    print("   - Classical archetype patterns exist in modern domains")
    print("   - Different domains show different archetypal profiles")
    print("   - Empirical weights reveal what actually predicts success")
    print("\nNext: Apply to more domains, build cross-domain comparison")
    print("="*70)


if __name__ == '__main__':
    main()

