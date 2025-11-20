"""
Pure Data-Driven Discovery - Let Data Show Where Narrative Matters

NO PREDICTIONS. Only measurements.

Process:
1. Load real IMDB data (6,047 movies)
2. Extract ю (story quality) for ALL
3. For EVERY subdivision in data (genre, decade, cast diversity, etc.)
4. MEASURE actual r between ю and outcomes
5. RANK by measured r
6. TOP contexts = where narrative is empirically strongest

Theory comes AFTER to explain, not before to predict.
"""

import json
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.imdb.data_loader import IMDBDataLoader
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from scipy import stats


def main():
    """Pure data-driven discovery"""
    
    print("="*80)
    print("PURE DATA-DRIVEN DISCOVERY")
    print("="*80)
    print("\nPrinciple: Let data show us where narrative matters most")
    print("NO predictions - only measurements")
    
    # === LOAD ALL DATA ===
    
    print("\n" + "="*80)
    print("Load Complete Dataset")
    print("="*80)
    
    loader = IMDBDataLoader()
    movies = loader.load_full_dataset(use_cache=True, filter_data=True)
    
    print(f"✓ Loaded {len(movies)} movies")
    
    # Check what dimensions we have
    sample = movies[0]
    print(f"\nAvailable dimensions: {list(sample.keys())[:15]}")
    
    # Sample for computational speed
    np.random.seed(42)
    sample_size = 2000
    indices = np.random.choice(len(movies), min(sample_size, len(movies)), replace=False)
    movies_sample = [movies[i] for i in indices]
    
    print(f"✓ Analyzing {len(movies_sample)} movies")
    
    # === COMPUTE ю FOR ALL ===
    
    print("\n" + "="*80)
    print("Compute Story Quality (ю) for All Movies")
    print("="*80)
    
    texts = [m['full_narrative'] for m in movies_sample]
    outcomes = np.array([m['success_score'] for m in movies_sample])
    
    print("\nExtracting narrative features...")
    
    # Fast feature extraction
    nom = NominativeAnalysisTransformer()
    nom.fit(texts)
    nom_feat = nom.transform(texts)
    if hasattr(nom_feat, 'toarray'):
        nom_feat = nom_feat.toarray()
    
    ling = LinguisticPatternsTransformer()
    ling.fit(texts)
    ling_feat = ling.transform(texts)
    if hasattr(ling_feat, 'toarray'):
        ling_feat = ling_feat.toarray()
    
    # Story quality
    ю = np.mean(np.hstack([nom_feat, ling_feat]), axis=1)
    
    print(f"✓ Computed ю for all {len(ю)} movies")
    
    # === MEASURE BASELINE ===
    
    r_overall, p_overall = stats.pearsonr(ю, outcomes)
    
    print(f"\nOverall correlation (baseline):")
    print(f"  r = {r_overall:.3f} (p={p_overall:.6f})")
    
    # === DISCOVER FROM DATA ===
    
    print("\n" + "="*80)
    print("Measure r in EVERY Subdivision (Pure Discovery)")
    print("="*80)
    print("\nSegmenting by all available dimensions...")
    print("Measuring actual correlations...")
    
    discoveries = []
    
    # === 1. BY PRIMARY GENRE ===
    
    print("\n--- PRIMARY GENRE (Data Measured) ---")
    
    for idx, movie in enumerate(movies_sample):
        genre = movie.get('primary_genre', 'Unknown')
        movie['genre_for_analysis'] = genre
    
    genres = [m.get('genre_for_analysis', 'Unknown') for m in movies_sample]
    genre_counter = Counter(genres)
    
    print(f"Found {len(genre_counter)} genres")
    
    for genre, count in genre_counter.most_common():
        if count < 30:  # Min sample
            continue
        
        # Get indices for this genre
        genre_mask = np.array([g == genre for g in genres])
        genre_yu = ю[genre_mask]
        genre_outcomes = outcomes[genre_mask]
        
        if len(np.unique(genre_yu)) < 2:
            continue
        
        r, p = stats.pearsonr(genre_yu, genre_outcomes)
        
        discoveries.append({
            'context': f"Genre: {genre}",
            'dimension': 'genre',
            'value': genre,
            'n': count,
            'r': r,
            'p': p
        })
        
        print(f"  {genre:30s}: r={r:+.3f} (n={count:4d}, p={p:.4f})")
    
    # === 2. BY DECADE ===
    
    print("\n--- DECADE (Data Measured) ---")
    
    decades = [m.get('decade', 2000) for m in movies_sample]
    decade_counter = Counter(decades)
    
    for decade, count in sorted(decade_counter.items()):
        if count < 30:
            continue
        
        decade_mask = np.array([d == decade for d in decades])
        decade_yu = ю[decade_mask]
        decade_outcomes = outcomes[decade_mask]
        
        if len(np.unique(decade_yu)) < 2:
            continue
        
        r, p = stats.pearsonr(decade_yu, decade_outcomes)
        
        discoveries.append({
            'context': f"Decade: {decade}s",
            'dimension': 'decade',
            'value': decade,
            'n': count,
            'r': r,
            'p': p
        })
        
        print(f"  {decade}s: r={r:+.3f} (n={count:4d}, p={p:.4f})")
    
    # === 3. BY CAST DIVERSITY ===
    
    print("\n--- CAST DIVERSITY (Data Measured) ---")
    
    diversity_levels = []
    for m in movies_sample:
        div = m.get('cast_diversity', 0.5)
        if div < 0.3:
            level = 'Low Diversity'
        elif div < 0.7:
            level = 'Medium Diversity'
        else:
            level = 'High Diversity'
        diversity_levels.append(level)
    
    div_counter = Counter(diversity_levels)
    
    for div_level, count in div_counter.items():
        if count < 30:
            continue
        
        div_mask = np.array([d == div_level for d in diversity_levels])
        div_yu = ю[div_mask]
        div_outcomes = outcomes[div_mask]
        
        if len(np.unique(div_yu)) < 2:
            continue
        
        r, p = stats.pearsonr(div_yu, div_outcomes)
        
        discoveries.append({
            'context': f"Cast: {div_level}",
            'dimension': 'diversity',
            'value': div_level,
            'n': count,
            'r': r,
            'p': p
        })
        
        print(f"  {div_level:30s}: r={r:+.3f} (n={count:4d}, p={p:.4f})")
    
    # === RANK ALL DISCOVERIES ===
    
    print("\n" + "="*80)
    print("EMPIRICAL RANKING (Pure Data, No Theory)")
    print("="*80)
    
    # Sort by absolute r (strongest effects first)
    discoveries.sort(key=lambda x: abs(x['r']), reverse=True)
    
    print(f"\nTOP 15 CONTEXTS WHERE NARRATIVE IS STRONGEST (Measured):")
    print(f"{'Rank':<6} {'Context':<45} {'r':<10} {'n':<8} {'p-value'}")
    print("-"*85)
    
    for i, d in enumerate(discoveries[:15], 1):
        sig = "***" if d['p'] < 0.001 else "**" if d['p'] < 0.01 else "*" if d['p'] < 0.05 else ""
        print(f"{i:<6} {d['context']:<45} {d['r']:>+.3f}    {d['n']:>6}  {d['p']:.4f} {sig}")
    
    # === IDENTIFY PASSING CONTEXTS ===
    
    print("\n" + "="*80)
    print("Where Can We Optimize? (Data Shows Us)")
    print("="*80)
    
    strong_contexts = [d for d in discoveries if abs(d['r']) > 0.3]
    
    if strong_contexts:
        print(f"\n✓ Found {len(strong_contexts)} contexts with r > 0.3:")
        for d in strong_contexts:
            print(f"  • {d['context']}: r={d['r']:+.3f} (n={d['n']})")
            print(f"    → This is where narrative is EMPIRICALLY strong")
            print(f"    → Optimize formula specifically for this context")
    else:
        print("\n⚠️  No single dimension shows r > 0.3")
        print("   May need interaction effects (genre × decade, etc.)")
    
    # === SAVE DISCOVERIES ===
    
    output_path = Path(__file__).parent / 'empirical_discoveries.json'
    
    results_json = {
        'approach': 'pure_data_driven',
        'principle': 'Measure everything, rank by r, optimize top contexts',
        'overall_r': float(r_overall),
        'n_movies': len(movies_sample),
        'discoveries': [
            {
                'rank': i+1,
                'context': d['context'],
                'dimension': d['dimension'],
                'r_measured': float(d['r']),
                'n_samples': d['n'],
                'p_value': float(d['p']),
                'abs_r': float(abs(d['r']))
            }
            for i, d in enumerate(discoveries)
        ],
        'strong_contexts': [d['context'] for d in strong_contexts],
        'insight': 'Data reveals where narrative is empirically strongest - optimize those contexts'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Saved discoveries: {output_path}")
    
    # === THE PRINCIPLE ===
    
    print("\n" + "="*80)
    print("DATA-DRIVEN PRINCIPLE")
    print("="*80)
    print("\n1. ✓ MEASURED r in every available subdivision")
    print("2. ✓ RANKED by empirical strength (no theory)")
    print("3. → OPTIMIZE formula for top measured contexts")
    print("4. → EXPLAIN with theory afterwards")
    print("\nNext: Focus optimization on empirically-discovered strong contexts")
    print(f"      (Top {len(strong_contexts) if strong_contexts else 'few'} contexts from data)")
    
    return discoveries


if __name__ == '__main__':
    main()

