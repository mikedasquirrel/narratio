"""
NBA Analysis - PROPER THEORETICAL FRAMEWORK

1,000 enriched games with comprehensive narratives.
Proper п-guided analysis following complete variable system.
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.transformer_library import TransformerLibrary
from src.analysis import UniversalDomainAnalyzer

# Transformers
from src.transformers.statistical import StatisticalTransformer
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.conflict_tension import ConflictTensionTransformer
from src.transformers.temporal_evolution import TemporalEvolutionTransformer


def main():
    print("="*80)
    print("NBA PROPER ANALYSIS - Enriched Dataset")
    print("="*80)
    
    # Load enriched data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_enriched_1000.json'
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"\n✓ Loaded {len(games)} enriched games")
    
    # Extract data
    texts = [g['narrative'] for g in games]
    outcomes = np.array([int(g['won']) for g in games])
    names = [g['matchup'] for g in games]
    
    print(f"  Wins: {outcomes.sum()}")
    print(f"  Losses: {len(outcomes) - outcomes.sum()}")
    
    # === STEP 1: п CALCULATION ===
    
    print("\n" + "="*80)
    print("STEP 1: Calculate Narrativity (п)")
    print("="*80)
    
    domain_characteristics = {
        'п_structural': 0.05,  # Physics heavily constrain outcomes
        'п_temporal': 0.25,  # Season arcs exist
        'п_agency': 0.15,  # Player choice limited by physics
        'п_interpretation': 0.15,  # Some subjectivity in "quality"
        'п_format': 0.10   # Game format rigid
    }
    
    analyzer = UniversalDomainAnalyzer('nba', narrativity=0.15)
    п = analyzer.calculate_narrativity(domain_characteristics)
    
    print(f"\nCalculated п: {п:.2f} (HIGHLY CONSTRAINED)")
    print("NBA is physics-dominated - stats should predict strongly")
    
    # === STEP 2: SELECT TRANSFORMERS ===
    
    print("\n" + "="*80)
    print("STEP 2: Select Transformers (п-guided)")
    print("="*80)
    
    library = TransformerLibrary()
    selected, expected_features = library.get_for_narrativity(
        п=п,
        target_feature_count=280
    )
    
    # === STEP 3: EXTRACT ж ===
    
    print("\n" + "="*80)
    print("STEP 3: Extract Genome (ж)")
    print("="*80)
    
    transformer_map = {
        'statistical': StatisticalTransformer(max_features=200),
        'linguistic': LinguisticPatternsTransformer(),
        'nominative': NominativeAnalysisTransformer(),
        'conflict': ConflictTensionTransformer(),
        'temporal_evolution': TemporalEvolutionTransformer()
    }
    
    all_features = []
    all_feature_names = []
    
    for trans_name in selected:
        if trans_name in transformer_map:
            print(f"\nExtracting {trans_name}...")
            transformer = transformer_map[trans_name]
            
            try:
                transformer.fit(texts)
                features = transformer.transform(texts)
                
                if hasattr(features, 'toarray'):
                    features = features.toarray()
                
                all_features.append(features)
                
                if hasattr(transformer, 'get_feature_names_out'):
                    names_out = transformer.get_feature_names_out()
                    all_feature_names.extend(names_out)
                else:
                    all_feature_names.extend([f"{trans_name}_{i}" for i in range(features.shape[1])])
                
                print(f"  ✓ {features.shape[1]} features")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    ж = np.hstack(all_features)
    print(f"\n✓ Genome (ж): {ж.shape}")
    
    # === STEP 4-9: COMPLETE ANALYSIS ===
    
    print("\n" + "="*80)
    print("STEP 4-9: Complete Analysis")
    print("="*80)
    
    # Run full framework
    results = analyzer.analyze_complete(
        texts=texts,
        outcomes=outcomes,
        names=names,
        genome=ж,
        feature_names=all_feature_names,
        masses=None
    )
    
    # === GENRE ANALYSIS (NBA-specific) ===
    
    print("\n" + "="*80)
    print("GENRE ANALYSIS (NBA Context Types)")
    print("="*80)
    
    # Identify game types
    rivalry_games = []
    playoff_games = []
    regular_games = []
    
    for i, game in enumerate(games):
        team = game.get('team_abbreviation', '')
        opponent = enricher._extract_opponent(game.get('matchup', ''))
        rivalry_key = tuple(sorted([team, opponent]))
        
        importance = game.get('importance_score', 0)
        
        if importance >= 6:  # Rivalry
            rivalry_games.append(i)
        elif '2023' in game.get('season', '') or '2024' in game.get('season', ''):
            playoff_games.append(i)
        else:
            regular_games.append(i)
    
    # Analyze by genre
    genres = {
        'Rivalry Games': rivalry_games,
        'Recent High-Stakes': playoff_games[:100],
        'Regular Season': regular_games[:300]
    }
    
    from scipy import stats as scipy_stats
    
    print("\nNarrative Effect by Game Type:")
    for genre_name, indices in genres.items():
        if len(indices) > 20:
            genre_ю = results['ю'][indices]
            genre_outcomes = outcomes[indices]
            
            r, p = scipy_stats.pearsonr(genre_ю, genre_outcomes)
            
            print(f"\n  {genre_name}:")
            print(f"    n = {len(indices)}")
            print(f"    r = {r:.3f} (p={p:.4f})")
            print(f"    Д = {r - 0.40:.3f}")  # NBA baseline is high (stats predict)
    
    # === SAVE RESULTS ===
    
    output_path = Path(__file__).parent / 'nba_proper_results.json'
    
    results_save = {
        'domain': 'nba',
        'п': results['п'],
        'n_games': len(games),
        'n_features': results['n_features'],
        'Д': results['Д'],
        'r_narrative': results['r_narrative'],
        'r_baseline': results['r_baseline'],
        'genre_analysis': {
            genre: {
                'n': len(indices),
                'indices': indices[:10]  # Sample
            }
            for genre, indices in genres.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    return results


if __name__ == '__main__':
    from enrich_narratives import NBANarrativeEnricher
    
    # Need enricher for genre analysis
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_all_seasons_real.json'
    with open(data_path) as f:
        all_games = json.load(f)
    enricher = NBANarrativeEnricher(all_games)
    
    main()

