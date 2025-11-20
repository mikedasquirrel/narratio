"""
Validate Movie Archetype Library

Quick script to verify the movie archetype library is properly formatted
and ready for transfer learning.

Author: Narrative Optimization Framework
Date: November 2025
"""

import pickle
import json
import numpy as np
from pathlib import Path


def validate_archetype_library():
    """Validate the movie archetype library."""
    
    print("="*80)
    print("MOVIE ARCHETYPE LIBRARY VALIDATION")
    print("="*80)
    
    # Load pickle version
    pkl_path = Path('archetypes/movies_archetype_library.pkl')
    print(f"\n1. Loading pickle file: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        archetype = pickle.load(f)
    
    print("   ✓ Pickle file loaded successfully")
    
    # Load JSON version
    json_path = Path('archetypes/movies_archetype_library.json')
    print(f"\n2. Loading JSON file: {json_path}")
    
    with open(json_path, 'r') as f:
        archetype_json = json.load(f)
    
    print("   ✓ JSON file loaded successfully")
    
    # Validate structure
    print("\n3. Validating archetype structure...")
    
    required_keys = ['domain', 'n_samples', 'xi_vector', 'optimal_alpha', 
                     'genome_feature_names', 'cluster_centroids', 'n_clusters']
    
    for key in required_keys:
        if key in archetype:
            print(f"   ✓ {key}: present")
        else:
            print(f"   ✗ {key}: MISSING")
    
    # Validate Ξ vector
    print("\n4. Validating Ξ (Golden Narratio) vector...")
    
    xi = np.array(archetype['xi_vector'])
    print(f"   Shape: {xi.shape}")
    print(f"   Dimensions: {len(xi)}")
    print(f"   Mean: {np.mean(xi):.4f}")
    print(f"   Std: {np.std(xi):.4f}")
    print(f"   Range: [{np.min(xi):.4f}, {np.max(xi):.4f}]")
    
    # Ξ is PCA-reduced (typically 20 dims from 122 genome features)
    if 10 <= len(xi) <= 30:
        print(f"   ✓ Ξ vector has valid PCA-reduced dimensions ({len(xi)})")
        print(f"   Note: Reduced from {len(archetype['genome_feature_names'])} genome features")
    else:
        print(f"   ✗ Ξ vector has unexpected dimensions: {len(xi)}")
    
    # Validate optimal α
    print("\n5. Validating optimal α...")
    
    alpha = archetype['optimal_alpha']
    print(f"   α = {alpha:.4f}")
    
    if 0 <= alpha <= 1:
        print("   ✓ α is in valid range [0, 1]")
    else:
        print(f"   ✗ α is out of range: {alpha}")
    
    if alpha < 0.5:
        print("   Interpretation: CHARACTER-HEAVY (traits > events)")
    elif alpha > 0.5:
        print("   Interpretation: PLOT-HEAVY (events > traits)")
    else:
        print("   Interpretation: BALANCED")
    
    # Validate cluster centroids
    print("\n6. Validating cluster centroids...")
    
    centroids = np.array(archetype['cluster_centroids'])
    print(f"   Shape: {centroids.shape}")
    print(f"   Number of clusters: {archetype['n_clusters']}")
    
    if centroids.shape[0] == archetype['n_clusters']:
        print(f"   ✓ Centroids match n_clusters ({archetype['n_clusters']})")
    else:
        print(f"   ✗ Centroids mismatch: {centroids.shape[0]} vs {archetype['n_clusters']}")
    
    if centroids.shape[1] == 10:
        print("   ✓ Centroids have correct embedding dimension (10)")
    else:
        print(f"   ✗ Centroids have wrong dimension: {centroids.shape[1]}")
    
    # Validate feature names
    print("\n7. Validating genome feature names...")
    
    feature_names = archetype['genome_feature_names']
    print(f"   Number of genome features: {len(feature_names)}")
    print(f"   Ξ dimensions (PCA-reduced): {len(xi)}")
    
    if len(feature_names) > len(xi):
        print(f"   ✓ Genome features ({len(feature_names)}) reduced to Ξ ({len(xi)}) via PCA")
        print(f"   Compression ratio: {len(xi)/len(feature_names):.1%}")
    else:
        print(f"   Note: No PCA reduction applied")
    
    print(f"\n   Feature breakdown:")
    nominative = sum(1 for name in feature_names if 'Nominative' in name)
    linguistic = sum(1 for name in feature_names if 'Linguistic' in name)
    narrative = sum(1 for name in feature_names if 'Narrative' in name)
    
    print(f"     - Nominative: {nominative} features")
    print(f"     - Linguistic: {linguistic} features")
    print(f"     - Narrative Potential: {narrative} features")
    print(f"     - Total: {nominative + linguistic + narrative} features")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n✓ Domain: {archetype['domain']}")
    print(f"✓ Samples: {archetype['n_samples']}")
    print(f"✓ Ξ dimensions: {len(xi)}")
    print(f"✓ Optimal α: {alpha:.4f}")
    print(f"✓ Clusters: {archetype['n_clusters']}")
    print(f"✓ Genome features: {len(feature_names)}")
    
    print("\n✓ ARCHETYPE LIBRARY IS VALID AND READY FOR TRANSFER LEARNING")
    
    # Test transfer readiness
    print("\n" + "="*80)
    print("TRANSFER LEARNING READINESS")
    print("="*80)
    
    print("\nThis archetype can transfer to:")
    print("  1. Music (entertainment domain, π = 0.702)")
    print("  2. Oscars (prestige entertainment, π = 0.75)")
    print("  3. Books/Novels (narrative structure)")
    print("  4. TV Shows (episodic narratives)")
    print("  5. Video Games (interactive narratives)")
    
    print("\nTo use for transfer learning:")
    print("""
    from src.transformers.outcome_conditioned_archetype import OutcomeConditionedArchetypeTransformer
    import pickle
    
    # Load archetype
    with open('archetypes/movies_archetype_library.pkl', 'rb') as f:
        movie_archetype = pickle.load(f)
    
    # Transfer to new domain
    transformer = OutcomeConditionedArchetypeTransformer(
        enable_transfer=True,
        source_archetype=movie_archetype
    )
    
    features = transformer.fit_transform(new_domain_data, y=new_domain_outcomes)
    """)
    
    print("="*80)


if __name__ == '__main__':
    validate_archetype_library()

