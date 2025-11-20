"""
NFL Context Discovery - Data-First Empirical Analysis

Measure |r| across ALL subdivisions to discover where narrative is strongest.
Following data-first methodology: don't predict, measure everything and let data lead.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_data():
    """Load game data, genome, and story quality."""
    # Load games
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    with open(dataset_path) as f:
        games = json.load(f)
    
    # Load genome data
    genome_path = Path(__file__).parent / 'nfl_genome_data.npz'
    genome_data = np.load(genome_path, allow_pickle=True)
    
    ю = genome_data['story_quality']
    outcomes = genome_data['outcomes']
    
    return games, ю, outcomes


def measure_correlation_by_context(
    games: List[Dict],
    ю: np.ndarray,
    outcomes: np.ndarray,
    context_key: str,
    context_extractor
) -> Dict[str, Dict]:
    """
    Measure |r| for each value of a context variable.
    
    Parameters
    ----------
    games : list
        Game data
    ю : ndarray
        Story quality scores
    outcomes : ndarray
        Outcomes
    context_key : str
        Name of context dimension
    context_extractor : callable
        Function to extract context value from game
        
    Returns
    -------
    results : dict
        {context_value: {'r': r, 'abs_r': abs_r, 'n': n_games}}
    """
    # Group by context
    context_groups = defaultdict(list)
    
    for i, game in enumerate(games):
        context_value = context_extractor(game)
        if context_value is not None:
            context_groups[context_value].append(i)
    
    # Measure |r| for each context
    results = {}
    
    for context_value, indices in context_groups.items():
        if len(indices) < 10:  # Need minimum sample size
            continue
        
        ю_subset = ю[indices]
        outcomes_subset = outcomes[indices]
        
        if len(np.unique(outcomes_subset)) < 2:  # Need variation in outcomes
            continue
        
        r = np.corrcoef(ю_subset, outcomes_subset)[0, 1]
        abs_r = abs(r)
        
        results[context_value] = {
            'r': float(r),
            'abs_r': float(abs_r),
            'n': len(indices),
            'mean_ю': float(ю_subset.mean()),
            'win_rate': float(outcomes_subset.mean())
        }
    
    return results


def main():
    """Discover optimal narrative contexts empirically."""
    print("="*80)
    print("NFL CONTEXT DISCOVERY - DATA-FIRST EMPIRICAL ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    games, ю, outcomes = load_data()
    print(f"✓ Loaded {len(games)} games with story quality and outcomes")
    
    # Overall baseline
    r_overall = np.corrcoef(ю, outcomes)[0, 1]
    abs_r_overall = abs(r_overall)
    print(f"\nBaseline |r| (all games): {abs_r_overall:.4f}")
    
    all_discoveries = {}
    
    # ========================================================================
    # TEAM-LEVEL CONTEXTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEAM-LEVEL CONTEXTS")
    print("="*80)
    
    # By home team
    print("\n[1] By Home Team...")
    team_results = measure_correlation_by_context(
        games, ю, outcomes, 'home_team',
        lambda g: g['home_team']
    )
    print(f"  Measured {len(team_results)} teams")
    all_discoveries['by_home_team'] = team_results
    
    # By season
    print("\n[2] By Season...")
    season_results = measure_correlation_by_context(
        games, ю, outcomes, 'season',
        lambda g: g['season']
    )
    print(f"  Measured {len(season_results)} seasons")
    all_discoveries['by_season'] = season_results
    
    # By week
    print("\n[3] By Week...")
    week_results = measure_correlation_by_context(
        games, ю, outcomes, 'week',
        lambda g: g['week'] if g['week'] is not None else None
    )
    print(f"  Measured {len(week_results)} weeks")
    all_discoveries['by_week'] = week_results
    
    # Playoff vs Regular
    print("\n[4] Playoff vs Regular Season...")
    playoff_results = measure_correlation_by_context(
        games, ю, outcomes, 'playoff',
        lambda g: 'playoff' if g['context'].get('playoff_game') else 'regular'
    )
    print(f"  Measured {len(playoff_results)} categories")
    all_discoveries['by_playoff_status'] = playoff_results
    
    # Division games
    print("\n[5] Division Games...")
    division_results = measure_correlation_by_context(
        games, ю, outcomes, 'division',
        lambda g: 'division' if g['context'].get('division_game') else 'non_division'
    )
    print(f"  Measured {len(division_results)} categories")
    all_discoveries['by_division_game'] = division_results
    
    # Primetime
    print("\n[6] Primetime Games...")
    primetime_results = measure_correlation_by_context(
        games, ю, outcomes, 'primetime',
        lambda g: 'primetime' if g['context'].get('primetime') else 'daytime'
    )
    print(f"  Measured {len(primetime_results)} categories")
    all_discoveries['by_primetime'] = primetime_results
    
    # Rivalry games
    print("\n[7] Rivalry Games...")
    rivalry_results = measure_correlation_by_context(
        games, ю, outcomes, 'rivalry',
        lambda g: 'rivalry' if g['context'].get('rivalry') else 'non_rivalry'
    )
    print(f"  Measured {len(rivalry_results)} categories")
    all_discoveries['by_rivalry'] = rivalry_results
    
    # ========================================================================
    # SCORE-BASED CONTEXTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SCORE-BASED CONTEXTS")
    print("="*80)
    
    # Close games vs blowouts
    print("\n[8] Close Games vs Blowouts...")
    closeness_results = measure_correlation_by_context(
        games, ю, outcomes, 'closeness',
        lambda g: 'close' if abs(g['home_score'] - g['away_score']) <= 7 else 'blowout'
    )
    print(f"  Measured {len(closeness_results)} categories")
    all_discoveries['by_closeness'] = closeness_results
    
    # High scoring vs low scoring
    print("\n[9] High Scoring vs Low Scoring...")
    scoring_results = measure_correlation_by_context(
        games, ю, outcomes, 'scoring',
        lambda g: 'high' if (g['home_score'] + g['away_score']) > 45 else 'low'
    )
    print(f"  Measured {len(scoring_results)} categories")
    all_discoveries['by_scoring'] = scoring_results
    
    # ========================================================================
    # NOMINATIVE CONTEXTS (CRITICAL FOR NFL)
    # ========================================================================
    
    print("\n" + "="*80)
    print("NOMINATIVE CONTEXTS (Critical for NFL)")
    print("="*80)
    
    # By QB name (home team)
    print("\n[10] By Home QB Name...")
    qb_results = measure_correlation_by_context(
        games, ю, outcomes, 'home_qb',
        lambda g: g['home_roster']['starting_qb']['name'] 
            if 'starting_qb' in g['home_roster'] else None
    )
    print(f"  Measured {len(qb_results)} QBs")
    all_discoveries['by_home_qb'] = qb_results
    
    # By coach name
    print("\n[11] By Home Coach...")
    coach_results = measure_correlation_by_context(
        games, ю, outcomes, 'home_coach',
        lambda g: g['home_coaches']['head_coach']
    )
    print(f"  Measured {len(coach_results)} coaches")
    all_discoveries['by_home_coach'] = coach_results
    
    # ========================================================================
    # BETTING CONTEXTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("BETTING CONTEXTS")
    print("="*80)
    
    # By spread size
    print("\n[12] By Spread Size...")
    spread_results = measure_correlation_by_context(
        games, ю, outcomes, 'spread_size',
        lambda g: 'heavy_favorite' if g['betting_odds']['spread'] < -7 
            else 'favorite' if g['betting_odds']['spread'] < -3
            else 'pick_em' if abs(g['betting_odds']['spread']) <= 3
            else 'underdog'
    )
    print(f"  Measured {len(spread_results)} categories")
    all_discoveries['by_spread_size'] = spread_results
    
    # By spread coverage
    print("\n[13] By Spread Coverage...")
    coverage_results = measure_correlation_by_context(
        games, ю, outcomes, 'spread_coverage',
        lambda g: 'covered' if g['betting_odds']['home_covered_spread'] else 'not_covered'
    )
    print(f"  Measured {len(coverage_results)} categories")
    all_discoveries['by_spread_coverage'] = coverage_results
    
    # ========================================================================
    # RANK ALL CONTEXTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("RANKING ALL CONTEXTS BY |r|")
    print("="*80)
    
    # Flatten all results
    all_contexts = []
    
    for context_name, context_results in all_discoveries.items():
        for context_value, stats in context_results.items():
            all_contexts.append({
                'context_dimension': context_name,
                'context_value': context_value,
                'abs_r': stats['abs_r'],
                'r': stats['r'],
                'n': stats['n'],
                'mean_ю': stats['mean_ю'],
                'win_rate': stats['win_rate']
            })
    
    # Sort by |r|
    all_contexts.sort(key=lambda x: x['abs_r'], reverse=True)
    
    # Display top 50
    print("\nTOP 50 CONTEXTS BY |r|:")
    print(f"{'Rank':<6} {'Context':<30} {'Value':<25} {'|r|':<8} {'r':<8} {'n':<6}")
    print("-" * 85)
    
    for i, ctx in enumerate(all_contexts[:50], 1):
        context_str = ctx['context_dimension'].replace('by_', '')[:29]
        value_str = str(ctx['context_value'])[:24]
        print(f"{i:<6} {context_str:<30} {value_str:<25} {ctx['abs_r']:<8.4f} {ctx['r']:<8.4f} {ctx['n']:<6}")
    
    # ========================================================================
    # KEY DISCOVERIES
    # ========================================================================
    
    print("\n" + "="*80)
    print("KEY EMPIRICAL DISCOVERIES")
    print("="*80)
    
    # Find strongest contexts
    top_10 = all_contexts[:10]
    
    print("\nStrongest Narrative Contexts:")
    for i, ctx in enumerate(top_10, 1):
        print(f"\n{i}. {ctx['context_dimension'].replace('by_', '')} = {ctx['context_value']}")
        print(f"   |r| = {ctx['abs_r']:.4f}, n = {ctx['n']}")
        print(f"   Pattern: {'Inverse (underdog advantage)' if ctx['r'] < 0 else 'Positive (favorite advantage)'}")
    
    # Compare to baseline
    print(f"\nBaseline |r| (all games): {abs_r_overall:.4f}")
    print(f"Strongest context |r|: {top_10[0]['abs_r']:.4f}")
    print(f"Improvement: {(top_10[0]['abs_r'] / abs_r_overall - 1) * 100:.1f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING DISCOVERIES")
    print("="*80)
    
    output = {
        'baseline': {
            'r': float(r_overall),
            'abs_r': float(abs_r_overall),
            'n': len(games)
        },
        'discoveries': all_discoveries,
        'ranked_contexts': all_contexts[:100],  # Top 100
        'summary': {
            'total_contexts_measured': len(all_contexts),
            'strongest_context': top_10[0] if top_10 else None,
            'top_10': top_10
        }
    }
    
    output_path = Path(__file__).parent / 'nfl_context_discoveries.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"  Total contexts measured: {len(all_contexts)}")
    print(f"  Top 100 ranked contexts saved")
    
    print("\n" + "="*80)
    print("CONTEXT DISCOVERY COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

