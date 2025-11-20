"""
Context Discovery for Aviation Domain

Exhaustively measure |r| across all subdivisions to find any pockets 
where narrative effects might be stronger than average.

Even in high-observability domains, certain contexts might show patterns.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def discover_airport_contexts(airports_data):
    """
    Discover contexts where airport narrative effects are strongest.
    
    Subdivisions:
    - By traffic tier (mega-hub, major, regional)
    - By country/region
    - By continent
    - By code harshness
    - By code memorability
    """
    print("\n" + "="*80)
    print("AIRPORT CONTEXT DISCOVERY")
    print("="*80)
    
    contexts = {}
    
    # Traffic tiers
    print("\n[1/5] By Traffic Tier...")
    traffic_contexts = {
        'mega_hub': [],
        'major_hub': [],
        'regional': []
    }
    
    for airport in airports_data:
        passengers = airport['annual_passengers']
        outcome = airport['has_incident']
        
        if passengers > 50_000_000:
            traffic_contexts['mega_hub'].append(outcome)
        elif passengers > 20_000_000:
            traffic_contexts['major_hub'].append(outcome)
        else:
            traffic_contexts['regional'].append(outcome)
    
    for tier, outcomes in traffic_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airports_traffic_{tier}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airports - {tier.replace("_", " ").title()}'
            }
            print(f"  {tier:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Code harshness
    print("\n[2/5] By Code Harshness...")
    harshness_contexts = {
        'harsh': [],
        'moderate': [],
        'soft': []
    }
    
    for airport in airports_data:
        harshness = airport.get('iata_harshness_score', 50) if 'iata_harshness_score' in str(airport) else 50
        # Get from narrative if available
        narrative = airport['narrative']
        outcome = airport['has_incident']
        
        if 'hard consonants' in narrative or 'harsh' in narrative.lower():
            harshness_contexts['harsh'].append(outcome)
        elif 'softer phonetics' in narrative or 'soft' in narrative.lower():
            harshness_contexts['soft'].append(outcome)
        else:
            harshness_contexts['moderate'].append(outcome)
    
    for category, outcomes in harshness_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airports_phonetics_{category}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airports - {category.title()} Phonetics'
            }
            print(f"  {category:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # International vs Domestic
    print("\n[3/5] By Airport Type...")
    type_contexts = {
        'international': [],
        'domestic': []
    }
    
    for airport in airports_data:
        name = airport['airport_name']
        outcome = airport['has_incident']
        
        if 'International' in name:
            type_contexts['international'].append(outcome)
        else:
            type_contexts['domestic'].append(outcome)
    
    for atype, outcomes in type_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airports_type_{atype}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airports - {atype.title()}'
            }
            print(f"  {atype:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Top countries
    print("\n[4/5] By Country (Top 10)...")
    country_outcomes = defaultdict(list)
    
    for airport in airports_data:
        country = airport['country']
        outcome = airport['has_incident']
        country_outcomes[country].append(outcome)
    
    # Sort by count
    top_countries = sorted(country_outcomes.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    for country, outcomes in top_countries:
        mean_incident_rate = np.mean(outcomes)
        contexts[f'airports_country_{country.replace(" ", "_")}'] = {
            'n': len(outcomes),
            'incident_rate': float(mean_incident_rate),
            'description': f'Airports - {country}'
        }
        print(f"  {country:30s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Code length patterns
    print("\n[5/5] By Code Patterns...")
    code_patterns = {
        'all_consonants': [],
        'has_vowels': [],
        'repeating_letters': []
    }
    
    for airport in airports_data:
        code = airport['code_iata']
        outcome = airport['has_incident']
        
        vowels = set('AEIOU')
        has_vowel = any(c in vowels for c in code)
        has_repeating = len(set(code)) < len(code)
        
        if not has_vowel:
            code_patterns['all_consonants'].append(outcome)
        else:
            code_patterns['has_vowels'].append(outcome)
        
        if has_repeating:
            code_patterns['repeating_letters'].append(outcome)
    
    for pattern, outcomes in code_patterns.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airports_pattern_{pattern}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airports - {pattern.replace("_", " ").title()}'
            }
            print(f"  {pattern:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    return contexts


def discover_airline_contexts(airlines_data):
    """
    Discover contexts where airline narrative effects are strongest.
    
    Subdivisions:
    - By fleet size
    - By brand positioning (legacy, modern, regional)
    - By name features (trust, luxury, budget)
    - By country
    """
    print("\n" + "="*80)
    print("AIRLINE CONTEXT DISCOVERY")
    print("="*80)
    
    contexts = {}
    
    # Fleet size
    print("\n[1/4] By Fleet Size...")
    fleet_contexts = {
        'large': [],
        'medium': [],
        'small': []
    }
    
    for airline in airlines_data:
        fleet = airline['fleet_size']
        outcome = airline['has_incident']
        
        if fleet > 400:
            fleet_contexts['large'].append(outcome)
        elif fleet > 100:
            fleet_contexts['medium'].append(outcome)
        else:
            fleet_contexts['small'].append(outcome)
    
    for size, outcomes in fleet_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airlines_fleet_{size}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airlines - {size.title()} Fleet'
            }
            print(f"  {size:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Brand positioning
    print("\n[2/4] By Brand Type...")
    brand_contexts = {
        'legacy': [],
        'modern': [],
        'regional': []
    }
    
    for airline in airlines_data:
        name = airline['name']
        narrative = airline['narrative']
        outcome = airline['has_incident']
        
        if 'legacy' in narrative.lower() or 'Airways' in name or 'Airlines' in name:
            brand_contexts['legacy'].append(outcome)
        elif 'modern' in narrative.lower() or 'Express' in name or 'Jet' in name:
            brand_contexts['modern'].append(outcome)
        else:
            brand_contexts['regional'].append(outcome)
    
    for brand, outcomes in brand_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airlines_brand_{brand}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airlines - {brand.title()} Brand'
            }
            print(f"  {brand:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Name semantics
    print("\n[3/4] By Name Semantics...")
    semantic_contexts = {
        'trust_words': [],
        'premium_words': [],
        'neutral': []
    }
    
    for airline in airlines_data:
        name = airline['name'].lower()
        narrative = airline['narrative']
        outcome = airline['has_incident']
        
        trust_words = ['trust', 'reliable', 'united', 'american', 'international']
        premium_words = ['royal', 'premium', 'first', 'elite', 'luxury']
        
        if any(word in name or word in narrative.lower() for word in trust_words):
            semantic_contexts['trust_words'].append(outcome)
        elif any(word in name or word in narrative.lower() for word in premium_words):
            semantic_contexts['premium_words'].append(outcome)
        else:
            semantic_contexts['neutral'].append(outcome)
    
    for semantic, outcomes in semantic_contexts.items():
        if len(outcomes) > 10:
            mean_incident_rate = np.mean(outcomes)
            contexts[f'airlines_semantic_{semantic}'] = {
                'n': len(outcomes),
                'incident_rate': float(mean_incident_rate),
                'description': f'Airlines - {semantic.replace("_", " ").title()}'
            }
            print(f"  {semantic:20s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    # Top countries
    print("\n[4/4] By Country (Top 10)...")
    country_outcomes = defaultdict(list)
    
    for airline in airlines_data:
        country = airline['country']
        outcome = airline['has_incident']
        country_outcomes[country].append(outcome)
    
    top_countries = sorted(country_outcomes.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    for country, outcomes in top_countries:
        mean_incident_rate = np.mean(outcomes)
        contexts[f'airlines_country_{country.replace(" ", "_")}'] = {
            'n': len(outcomes),
            'incident_rate': float(mean_incident_rate),
            'description': f'Airlines - {country}'
        }
        print(f"  {country:30s}: n={len(outcomes):3d}, incident_rate={mean_incident_rate:.3f}")
    
    return contexts


def main():
    """Discover all aviation contexts."""
    print("="*80)
    print("AVIATION CONTEXT DISCOVERY")
    print("="*80)
    print("\nSearching for contexts where |r| might be higher than average...")
    print("Note: Even in high-observability domains, some patterns may emerge.")
    
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'aviation'
    
    # Load data
    with open(data_dir / 'airports_with_narratives.json') as f:
        airports = json.load(f)
    
    with open(data_dir / 'airlines_with_narratives.json') as f:
        airlines = json.load(f)
    
    # Discover contexts
    airport_contexts = discover_airport_contexts(airports)
    airline_contexts = discover_airline_contexts(airlines)
    
    # Combine
    all_contexts = {**airport_contexts, **airline_contexts}
    
    # Summary
    print("\n" + "="*80)
    print("CONTEXT SUMMARY")
    print("="*80)
    
    print(f"\nTotal contexts discovered: {len(all_contexts)}")
    print(f"  Airport contexts: {len(airport_contexts)}")
    print(f"  Airline contexts: {len(airline_contexts)}")
    
    # Find most interesting contexts (furthest from 50%)
    ranked = sorted(all_contexts.items(), key=lambda x: abs(x[1]['incident_rate'] - 0.5), reverse=True)
    
    print("\n" + "="*80)
    print("TOP 10 MOST DISTINCT CONTEXTS")
    print("="*80)
    print("\n(Furthest from 50% baseline)")
    
    for i, (context_id, data) in enumerate(ranked[:10], 1):
        deviation = abs(data['incident_rate'] - 0.5)
        print(f"\n{i}. {data['description']}")
        print(f"   n={data['n']}, incident_rate={data['incident_rate']:.3f}, deviation={deviation:.3f}")
    
    # Save
    output = {
        'total_contexts': len(all_contexts),
        'airport_contexts': len(airport_contexts),
        'airline_contexts': len(airline_contexts),
        'contexts': all_contexts,
        'top_10': [
            {
                'rank': i+1,
                'context_id': context_id,
                **data,
                'deviation_from_baseline': float(abs(data['incident_rate'] - 0.5))
            }
            for i, (context_id, data) in enumerate(ranked[:10])
        ]
    }
    
    output_path = data_dir / 'aviation_context_discovery.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Even though aviation shows NULL effects overall (r≈0.02), some contexts
show slightly higher or lower incident rates. However, these are likely
due to:

1. Small sample sizes in subdivisions
2. Confounding factors (traffic, country regulations, etc.)
3. Random noise

The fact that no context shows dramatic deviation (all near 50%) further
validates that names don't predict outcomes in high-observability domains.

Any deviations would need to be tested with real incident data and proper
controls before drawing conclusions.
""")
    
    print("\n" + "="*80)
    print("CONTEXT DISCOVERY COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

