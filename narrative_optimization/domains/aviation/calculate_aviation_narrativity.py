"""
Calculate narrativity (π) for aviation domain.

Aviation is a highly circumscribed, high-observability domain.
Expected π ≈ 0.23 (very low, similar to dice rolls).

π components:
- π_structural: How many narrative paths are possible?
- π_temporal: Can the story unfold over time?
- π_agency: Do actors have agency/choice?
- π_interpretation: Is judgment subjective?
- π_format: Is the medium flexible?

Formula: π = 0.30×π_structural + 0.20×π_temporal + 0.25×π_agency + 
             0.15×π_interpretation + 0.10×π_format
"""

import json
from pathlib import Path


def calculate_aviation_narrativity():
    """
    Calculate π for aviation domain.
    
    Aviation characteristics:
    - Codes assigned by IATA/ICAO (not strategic choice)
    - Safety outcomes determined by engineering/regulation
    - Objective metrics (incidents are facts)
    - Highly regulated and observable
    
    This makes aviation a CONTROL domain (high observability → low narrative effects)
    """
    
    # π_structural: How constrained is the narrative space?
    # 0.0 = completely fixed (one path), 1.0 = unlimited paths
    π_structural = 0.20
    # Reasoning: Airport codes are assigned by authorities, not chosen strategically.
    # Airline names have some choice, but follow strict conventions.
    # Safety outcomes are determined by engineering, not narrative positioning.
    # Very limited narrative paths.
    
    # π_temporal: Can story unfold over time?
    # 0.0 = snapshot, 1.0 = evolves continuously
    π_temporal = 0.40
    # Reasoning: Airlines and airports do evolve over time (reputations, incidents, 
    # rebranding). But the core identity (codes, names) is relatively fixed.
    # Moderate temporal evolution.
    
    # π_agency: Do actors have agency?
    # 0.0 = no choice, 1.0 = complete autonomy
    π_agency = 0.30
    # Reasoning: Airlines have limited agency over safety outcomes.
    # Engineering quality, maintenance, pilot training matter far more than names.
    # Airports have even less agency (infrastructure determined by geography, traffic).
    # Low agency.
    
    # π_interpretation: How subjective is judgment?
    # 0.0 = objective measurement, 1.0 = pure interpretation
    π_interpretation = 0.10
    # Reasoning: Safety is OBJECTIVE. Incidents are facts, not interpretations.
    # This is why aviation is high-observability control domain.
    # Very low subjectivity.
    
    # π_format: How flexible is the format?
    # 0.0 = rigid format, 1.0 = unlimited expression
    π_format = 0.15
    # Reasoning: IATA codes must be 3 letters, ICAO 4 letters.
    # Airline names follow strict industry conventions.
    # Safety reporting follows standardized formats.
    # Very rigid format.
    
    # Calculate weighted average
    π = (0.30 * π_structural + 
         0.20 * π_temporal + 
         0.25 * π_agency + 
         0.15 * π_interpretation + 
         0.10 * π_format)
    
    return {
        'π': round(π, 3),
        'components': {
            'π_structural': π_structural,
            'π_temporal': π_temporal,
            'π_agency': π_agency,
            'π_interpretation': π_interpretation,
            'π_format': π_format,
        },
        'weights': {
            'w_structural': 0.30,
            'w_temporal': 0.20,
            'w_agency': 0.25,
            'w_interpretation': 0.15,
            'w_format': 0.10,
        },
        'interpretation': {
            'category': 'circumscribed' if π < 0.3 else 'moderate' if π < 0.6 else 'open',
            'comparison': {
                'dice_rolls': 0.12,
                'coin_flips': 0.15,
                'aviation': round(π, 3),
                'nba_games': 0.50,
                'startups': 0.76,
                'character_arcs': 0.88,
            },
            'expected_narrative_effects': 'minimal to none (high observability)',
            'hypothesis': 'Aviation should FAIL narrative law (Д/π < 0.5)',
            'scientific_value': 'Validates observability moderation theory'
        },
        'reasoning': {
            'structural': 'Codes assigned by authorities, not strategic choice. Very constrained.',
            'temporal': 'Some evolution over time (reputation, incidents), but core identity fixed.',
            'agency': 'Limited agency. Engineering/regulation determine outcomes, not names.',
            'interpretation': 'Objective metrics. Incidents are facts, not interpretations. High observability.',
            'format': 'Rigid format (3-letter IATA, 4-letter ICAO codes). Industry conventions.'
        }
    }


def main():
    """Calculate and save aviation narrativity."""
    print("="*80)
    print("CALCULATING AVIATION NARRATIVITY (π)")
    print("="*80)
    
    result = calculate_aviation_narrativity()
    
    print("\n" + "="*80)
    print("COMPONENT VALUES")
    print("="*80)
    
    for component, value in result['components'].items():
        weight_key = f"w_{component.split('_')[1]}"
        weight = result['weights'][weight_key]
        contribution = value * weight
        print(f"\n{component}:")
        print(f"  Value: {value:.2f}")
        print(f"  Weight: {weight:.2f}")
        print(f"  Contribution: {contribution:.3f}")
        print(f"  Reasoning: {result['reasoning'][component.split('_')[1]]}")
    
    print("\n" + "="*80)
    print("FINAL NARRATIVITY")
    print("="*80)
    
    π = result['π']
    print(f"\nπ (aviation) = {π:.3f}")
    print(f"\nCategory: {result['interpretation']['category'].upper()}")
    
    print("\n" + "="*80)
    print("COMPARISON TO OTHER DOMAINS")
    print("="*80)
    
    comparisons = result['interpretation']['comparison']
    sorted_domains = sorted(comparisons.items(), key=lambda x: x[1])
    
    for domain, value in sorted_domains:
        marker = " ← Aviation" if domain == 'aviation' else ""
        print(f"  {domain:20s}: π = {value:.3f}{marker}")
    
    print("\n" + "="*80)
    print("THEORETICAL IMPLICATIONS")
    print("="*80)
    
    print(f"\nExpected effects: {result['interpretation']['expected_narrative_effects']}")
    print(f"Hypothesis: {result['interpretation']['hypothesis']}")
    print(f"Scientific value: {result['interpretation']['scientific_value']}")
    
    print("\n" + "="*80)
    print("PREDICTION")
    print("="*80)
    
    print(f"""
Aviation π = {π:.3f} (very low - circumscribed domain)

Expected results:
- |r| (narrative quality × outcomes) ≈ 0.03-0.07 (very weak)
- κ (performance judgment) ≈ 0.1 (objective)
- Д = π × |r| × κ ≈ {π:.3f} × 0.05 × 0.1 = {π * 0.05 * 0.1:.4f}
- Efficiency = Д/π ≈ {(π * 0.05 * 0.1) / π:.4f} = 0.005

Threshold: Д/π > 0.5 for narrative law to hold
Expected: Д/π ≈ 0.005 → FAILS (as expected!)

Interpretation: This NULL result VALIDATES the theory.
Aviation has high observability (safety records are public).
Names should NOT predict outcomes when performance is observable.

This creates the observability gradient:
  Low observability (Crypto): r=0.65 → Names MATTER
  Medium (Hurricanes): r=0.47 → Names matter
  High (Aviation): r≈0.05 → Names DON'T matter ✓
""")
    
    # Save
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'aviation'
    output_path = data_dir / 'aviation_narrativity.json'
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*80)
    print("NARRATIVITY CALCULATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

