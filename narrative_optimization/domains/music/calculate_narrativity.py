"""
Calculate Music/Spotify Narrativity (π)

Music domain analysis for π calculation.
"""

import json
from pathlib import Path

def calculate_music_narrativity():
    """
    Calculate π for music domain
    
    Components:
    1. Structural (0-1): How constrained by physical rules?
    2. Temporal (0-1): How much temporal evolution?
    3. Agency (0-1): How much creator control?
    4. Interpretation (0-1): How subjective is judgment?
    5. Format (0-1): How free is the format?
    """
    
    print("\n" + "="*80)
    print("MUSIC NARRATIVITY CALCULATION")
    print("="*80)
    
    # 1. Structural openness
    structural = 0.70
    print(f"\n[1] Structural: {structural:.2f}")
    print("  - Chord progressions: Some physics/math constraints")
    print("  - Song structure: Mostly creative freedom")
    print("  - Production choices: High freedom")
    print("  → Moderate-high openness")
    
    # 2. Temporal evolution
    temporal = 0.75
    print(f"\n[2] Temporal: {temporal:.2f}")
    print("  - Genre evolution: Rapid and significant")
    print("  - Cultural context: Strong influence")
    print("  - Trends: Fast-moving")
    print("  → High temporal dynamics")
    
    # 3. Agency
    agency = 0.65
    print(f"\n[3] Agency: {agency:.2f}")
    print("  - Artists control: Song writing, production")
    print("  - External constraints: Label influence, market")
    print("  - Distribution: Algorithm influence (Spotify)")
    print("  → Moderate-high agency")
    
    # 4. Interpretation
    interpretation = 0.80
    print(f"\n[4] Interpretation: {interpretation:.2f}")
    print("  - Taste: Highly subjective")
    print("  - Genre preferences: Personal")
    print("  - Cultural context: Strong variation")
    print("  → High interpretive freedom")
    
    # 5. Format freedom
    format_freedom = 0.60
    print(f"\n[5] Format: {format_freedom:.2f}")
    print("  - Song length: Some conventions (3-4 min)")
    print("  - Genre constraints: Moderate")
    print("  - Medium: Relatively fixed (audio)")
    print("  → Moderate format freedom")
    
    # Weighted average (standard weights)
    π = (0.30 * structural +
         0.20 * temporal +
         0.25 * agency +
         0.15 * interpretation +
         0.10 * format_freedom)
    
    print(f"\n" + "="*80)
    print(f"MUSIC π = {π:.3f}")
    print("="*80)
    print(f"\nInterpretation: Mid-High Narrativity")
    print(f"  Similar to: Movies (π=0.65), Golf (π=0.70)")
    print(f"  Lower than: Tennis (π=0.75), Crypto (π=0.76)")
    print(f"  Higher than: Mental Health (π=0.55), NBA (π=0.49)")
    print()
    
    # Additional forces
    λ = 0.30  # Some physical constraints (audio production, harmony)
    ψ = 0.65  # Moderate awareness (people know hits are marketed)
    ν = 0.75  # Strong narrative (branding, artist persona)
    κ = 0.50  # Mixed (Spotify algorithm + user choice)
    
    result = {
        'domain': 'Music/Spotify',
        'π': round(π, 3),
        'components': {
            'structural': structural,
            'temporal': temporal,
            'agency': agency,
            'interpretation': interpretation,
            'format': format_freedom
        },
        'forces': {
            'λ': λ,
            'ψ': ψ,
            'ν': ν,
            'κ': κ
        },
        'interpretation': 'Mid-High narrativity entertainment domain',
        'comparisons': {
            'movies': 0.65,
            'golf': 0.70,
            'tennis': 0.75
        }
    }
    
    # Save
    output_path = Path(__file__).parent / 'music_narrativity.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved to: {output_path}\n")
    
    return result


if __name__ == '__main__':
    calculate_music_narrativity()









