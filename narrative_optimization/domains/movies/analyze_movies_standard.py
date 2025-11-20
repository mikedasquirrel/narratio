"""
Movies Analysis with Standard Framework

Analyzes MovieLens dataset:
- 9,742 movies with titles and genres
- 100,836 ratings
- Applies standard transformers to extract ж
- Calculates Д = п × r × κ
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer  
from src.transformers.narrative_potential import NarrativePotentialTransformer
from scipy import stats


def analyze_movies():
    """Complete movies analysis."""
    print("=" * 80)
    print("MOVIES ANALYSIS - Complete Framework")
    print("=" * 80)
    
    # Load MovieLens data
    movies_path = Path(__file__).parent.parent.parent.parent / 'data/ml-latest-small/movies.csv'
    ratings_path = Path(__file__).parent.parent.parent.parent / 'data/ml-latest-small/ratings.csv'
    
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    print(f"\n✓ Loaded {len(movies)} movies")
    print(f"✓ Loaded {len(ratings)} ratings")
    
    # Aggregate ratings per movie
    avg_ratings = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    
    # Merge with movies
    movies_with_ratings = movies.merge(avg_ratings, on='movieId')
    
    # Filter to movies with 20+ ratings (more reliable)
    movies_with_ratings = movies_with_ratings[movies_with_ratings['count'] >= 20]
    
    print(f"✓ {len(movies_with_ratings)} movies with 20+ ratings")
    
    # Create narrative from title + genres
    narratives = (movies_with_ratings['title'] + ' ' + movies_with_ratings['genres']).values
    outcomes = movies_with_ratings['mean'].values
    
    # Sample for speed
    sample_size = min(1000, len(narratives))
    X = narratives[:sample_size]
    y = outcomes[:sample_size]
    
    print(f"✓ Analyzing {len(X)} movies")
    
    # Apply transformers
    print("\nApplying standard transformers...")
    
    transformers = {
        'nominative': NominativeAnalysisTransformer(),
        'self_perception': SelfPerceptionTransformer(),
        'narrative_potential': NarrativePotentialTransformer()
    }
    
    all_features = []
    
    for name, transformer in transformers.items():
        try:
            transformer.fit(X)
            features = transformer.transform(X)
            all_features.append(features)
            print(f"  ✓ {name}: {features.shape[1]} features")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # Combine
    ж = np.hstack(all_features)
    ю = np.mean(ж, axis=1)
    
    # Measure r
    r, p = stats.pearsonr(ю, y)
    
    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"r (predictive correlation): {r:.4f}, p={p:.4f}")
    print(f"R²: {r**2:.4f}")
    
    # Calculate Д
    п = 0.65  # Movies narrativity (storytelling medium, genre conventions)
    κ = 0.5   # Creators make, audiences judge (moderate coupling)
    Д = п * r * κ
    
    print(f"\nNarrative Agency Calculation:")
    print(f"  п (narrativity): {п}")
    print(f"  r (correlation): {r:.4f}")
    print(f"  κ (coupling): {κ}")
    print(f"  Д = п × r × κ = {Д:.4f}")
    print(f"\n  Efficiency: Д/п = {Д/п:.4f}")
    print(f"  Threshold: {'✓ PASS' if Д/п > 0.5 else '✗ FAIL'} (need > 0.5)")
    
    results = {
        'domain': 'movies',
        'n_movies': len(X),
        'narrativity': п,
        'coupling': κ,
        'r_measured': float(r),
        'p_value': float(p),
        'D_agency': float(Д),
        'efficiency': float(Д/п),
        'passes_threshold': bool(Д/п > 0.5)
    }
    
    # Save
    output_path = Path(__file__).parent / 'movies_D_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    return results


if __name__ == "__main__":
    analyze_movies()

