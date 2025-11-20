"""
Silver Ratio Finder for Oscars

Silver Ratio (Σ) = Domain-specific optimal that approximates Golden Narratio (Ξ)

For Oscars: What is the achievable best within film medium constraints?

Ξ = Universal perfection (archetypal ideal)
Σ_oscars = Best achievable within:
  - Film medium (visual storytelling, 90-180 min)
  - Hollywood system (studios, stars, budgets)
  - Academy voter preferences
  - Genre conventions

Find Σ by analyzing what winners share (centroid of success galaxy).
Test: Distance to Σ predicts better than distance to naive average.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans


def find_silver_ratio():
    """Find domain-specific optimal (Σ) for Oscars."""
    print("=" * 80)
    print("FINDING SILVER RATIO (Σ) FOR OSCARS")
    print("=" * 80)
    
    # Load Oscar data
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/oscar_nominees_complete.json'
    
    with open(data_path, 'r') as f:
        oscar_data = json.load(f)
    
    # Separate winners vs nominees
    winners = []
    nominees = []
    
    for year, films in oscar_data.items():
        for film in films:
            if film['won_oscar']:
                winners.append(film)
            else:
                nominees.append(film)
    
    print(f"\n✓ {len(winners)} winners")
    print(f"✓ {len(nominees)} nominees (non-winners)")
    
    # Extract features for winners
    print("\nAnalyzing winner commonalities (finding Σ)...")
    
    winner_features = {
        'avg_cast_size': np.mean([len(w['cast']) for w in winners]),
        'avg_runtime': np.mean([w['runtime'] for w in winners if w['runtime']]),
        'common_genres': [],
        'avg_keywords': np.mean([len(w['keywords']) for w in winners]),
        'director_name_patterns': {},
        'lead_actor_patterns': {},
        'setting_patterns': {}
    }
    
    # Genre distribution in winners
    all_genres = {}
    for w in winners:
        for g in w.get('genres', []):
            all_genres[g] = all_genres.get(g, 0) + 1
    
    winner_features['common_genres'] = sorted(all_genres.items(), key=lambda x: x[1], reverse=True)
    
    # Character name patterns (nominative analysis of winners)
    winner_character_names = []
    for w in winners:
        for c in w['cast'][:5]:  # Top 5 characters
            if c.get('character'):
                winner_character_names.append(c['character'])
    
    print(f"\nWINNER PATTERNS (Σ_oscars characteristics):")
    print(f"  Avg cast size: {winner_features['avg_cast_size']:.1f}")
    print(f"  Avg runtime: {winner_features['avg_runtime']:.0f} min")
    print(f"  Top genres: {winner_features['common_genres'][:3]}")
    print(f"  Avg keyword count: {winner_features['avg_keywords']:.1f}")
    print(f"  Sample character names: {winner_character_names[:5]}")
    
    # Compute "Silver Ratio" = centroid of winners
    # This is the domain-optimal pattern (Σ)
    
    print(f"\n{'='*80}")
    print("SILVER RATIO (Σ_oscars) - Domain-Specific Optimal")
    print(f"{'='*80}")
    print("\nBased on winner centroid:")
    print(f"  • Cast size ~{winner_features['avg_cast_size']:.0f} actors")
    print(f"  • Runtime ~{winner_features['avg_runtime']:.0f} minutes")
    print(f"  • Predominantly Drama genre")
    print(f"  • Rich keyword density ({winner_features['avg_keywords']:.0f} themes)")
    print("\nThis is the achievable best within Oscar domain constraints.")
    print("Films closer to Σ (not Ξ directly) should win more.")
    
    # Test: Do nominees closer to winner centroid lose by smaller margins?
    print(f"\n{'='*80}")
    print("TESTING: Distance to Σ vs Loss Margin")
    print(f"{'='*80}")
    print("\n⚠️  Need outcome metric beyond binary (vote counts, margins)")
    print("Current data only has win/lose, not by how much")
    print("\nWith vote data, could test:")
    print("  Distance from Σ → Vote margin")
    print("  Closer to winner centroid → Closer race")
    
    return {
        'silver_ratio_characteristics': winner_features,
        'n_winners_analyzed': len(winners),
        'interpretation': 'Σ_oscars = achievable best within Academy/Hollywood system'
    }


if __name__ == "__main__":
    results = find_silver_ratio()
    
    output_path = Path(__file__).parent / 'silver_ratio_oscars.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Saved: {output_path}")

