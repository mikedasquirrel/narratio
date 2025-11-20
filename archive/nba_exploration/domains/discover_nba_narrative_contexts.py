"""
NBA Empirical Discovery - DATA-FIRST

Measure narrative effects across ALL dimensions in 11,979 games.

Available dimensions:
- Season (2014-2024)
- Team (30 teams)
- Month (Oct-June)
- Home/Away
- Opponent strength
- Score differential
- Playoff vs Regular season
- Team × Season interactions
- And more...

Let DATA show us where narrative matters most.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.taxonomy.empirical_discovery import EmpiricalDiscoveryEngine
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from scipy import stats


def main():
    """Comprehensive data-driven NBA discovery"""
    
    print("="*80)
    print("NBA EMPIRICAL DISCOVERY - EXHAUSTIVE DATA-FIRST ANALYSIS")
    print("="*80)
    print("\n11,979 games across 10 seasons")
    print("Principle: MEASURE everything, let data reveal patterns")
    print("NO predictions - only discoveries")
    
    # === LOAD ALL DATA ===
    
    print("\n" + "="*80)
    print("STEP 1: Load Complete NBA Dataset")
    print("="*80)
    
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/nba_all_seasons_real.json'
    
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} games")
    print(f"  Seasons: {min(g['season'] for g in games)} - {max(g['season'] for g in games)}")
    
    # Sample for computational speed (can process all later)
    np.random.seed(42)
    sample_size = 3000
    games_sample = np.random.choice(games, size=min(sample_size, len(games)), replace=False).tolist()
    
    print(f"✓ Analyzing {len(games_sample)} games for discovery")
    
    # === EXTRACT NARRATIVE QUALITY ===
    
    print("\n" + "="*80)
    print("STEP 2: Compute Narrative Quality (ю) for All Games")
    print("="*80)
    
    texts = [g['narrative'] for g in games_sample]
    outcomes = np.array([int(g['won']) for g in games_sample])
    
    print("\nExtracting narrative features...")
    
    # Nominative (team names)
    nom = NominativeAnalysisTransformer()
    nom.fit(texts)
    nom_feat = nom.transform(texts)
    if hasattr(nom_feat, 'toarray'):
        nom_feat = nom_feat.toarray()
    
    # Linguistic (narrative structure)
    ling = LinguisticPatternsTransformer()
    ling.fit(texts)
    ling_feat = ling.transform(texts)
    if hasattr(ling_feat, 'toarray'):
        ling_feat = ling_feat.toarray()
    
    # Ensemble (team dynamics)
    ens = EnsembleNarrativeTransformer(n_top_terms=20)
    ens.fit(texts)
    ens_feat = ens.transform(texts)
    if hasattr(ens_feat, 'toarray'):
        ens_feat = ens_feat.toarray()
    
    # Combine
    all_features = np.hstack([nom_feat, ling_feat, ens_feat])
    ю = np.mean(all_features, axis=1)
    
    print(f"✓ Computed ю for {len(ю)} games")
    print(f"  Features: {all_features.shape[1]}")
    
    # === MEASURE BASELINE ===
    
    r_overall, p_overall = stats.pearsonr(ю, outcomes)
    
    print(f"\nOverall correlation (single game baseline):")
    print(f"  r = {r_overall:.3f} (p={p_overall:.6f})")
    print(f"  This is our baseline - discover where r is HIGHER")
    
    # === CREATE RICH METADATA ===
    
    print("\n" + "="*80)
    print("STEP 3: Extract All Available Dimensions")
    print("="*80)
    
    metadata_rows = []
    
    for game in games_sample:
        # Parse date for temporal dimensions
        try:
            date_obj = datetime.strptime(game['date'], '%Y-%m-%d')
            month = date_obj.month
            month_name = date_obj.strftime('%B')
        except:
            month = 1
            month_name = 'Unknown'
        
        # Determine season phase
        if month >= 10 or month <= 12:
            phase = 'Early Season'
        elif month in [1, 2]:
            phase = 'Mid Season'
        elif month in [3, 4]:
            phase = 'Late Season'
        else:
            phase = 'Playoffs'
        
        # Score differential category
        plus_minus = game.get('plus_minus', 0)
        if abs(plus_minus) <= 5:
            game_closeness = 'Nail-biter'
        elif abs(plus_minus) <= 15:
            game_closeness = 'Close'
        else:
            game_closeness = 'Blowout'
        
        metadata_rows.append({
            'team': game['team_abbreviation'],
            'season': game['season'],
            'home_game': game['home_game'],
            'month': month_name,
            'season_phase': phase,
            'game_closeness': game_closeness,
            'points_category': 'Low' if game['points'] < 100 else 'Medium' if game['points'] < 110 else 'High'
        })
    
    metadata = pd.DataFrame(metadata_rows)
    
    print("✓ Extracted metadata dimensions:")
    for col in metadata.columns:
        unique = len(metadata[col].unique())
        print(f"  • {col:20s}: {unique} unique values")
    
    # === EMPIRICAL DISCOVERY ===
    
    print("\n" + "="*80)
    print("STEP 4: MEASURE r in Every Subdivision (Pure Discovery)")
    print("="*80)
    print("\nExploring all dimensions...")
    print("(Data will show us where narrative matters most)")
    
    engine = EmpiricalDiscoveryEngine(verbose=True)
    
    contexts = engine.discover_all_contexts(
        story_quality=ю,
        outcomes=outcomes,
        metadata=metadata,
        dimensions_to_explore=['team', 'season', 'month', 'season_phase', 
                               'home_game', 'game_closeness', 'points_category'],
        min_samples=30,
        max_subdivisions=300
    )
    
    # === REPORT DISCOVERIES ===
    
    print("\n" + "="*80)
    print("NBA EMPIRICAL DISCOVERIES")
    print("="*80)
    
    engine.print_discovery_report(top_n=30)
    
    # === ANALYZE PATTERNS ===
    
    print("\n" + "="*80)
    print("PATTERN ANALYSIS (What Data Revealed)")
    print("="*80)
    
    # Group by dimension
    by_dimension = {}
    for ctx in contexts:
        dim = ctx.dimension
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(ctx)
    
    # Best per dimension
    print("\nStrongest context per dimension:")
    for dimension in sorted(by_dimension.keys()):
        best = max(by_dimension[dimension], key=lambda x: abs(x.r_measured))
        print(f"  {dimension:25s}: {best.name[:40]:40s} r={best.r_measured:+.3f} (n={best.n_samples})")
    
    # === TEMPORAL ANALYSIS ===
    
    print("\n" + "="*80)
    print("TEMPORAL PATTERN DISCOVERY")
    print("="*80)
    
    # By season phase
    phase_contexts = [c for c in contexts if 'season_phase' in c.name]
    if phase_contexts:
        print("\nNarrative strength by season phase (measured):")
        for ctx in sorted(phase_contexts, key=lambda x: x.r_measured, reverse=True):
            print(f"  {ctx.name.split('=')[1]:20s}: r={ctx.r_measured:+.3f} (n={ctx.n_samples})")
    
    # By month
    month_contexts = [c for c in contexts if 'month=' in c.name and '×' not in c.name]
    if month_contexts:
        print("\nNarrative strength by month (measured):")
        for ctx in sorted(month_contexts, key=lambda x: x.r_measured, reverse=True)[:6]:
            print(f"  {ctx.name.split('=')[1]:20s}: r={ctx.r_measured:+.3f} (n={ctx.n_samples})")
    
    # === TEAM ANALYSIS ===
    
    print("\n" + "="*80)
    print("TEAM PATTERN DISCOVERY")
    print("="*80)
    
    team_contexts = [c for c in contexts if 'team=' in c.name and '×' not in c.name]
    if team_contexts:
        print("\nTop 10 teams by narrative strength (measured):")
        for ctx in sorted(team_contexts, key=lambda x: abs(x.r_measured), reverse=True)[:10]:
            team = ctx.name.split('=')[1]
            print(f"  {team:5s}: r={ctx.r_measured:+.3f} (n={ctx.n_samples:4d})")
    
    # === GAME CONTEXT ANALYSIS ===
    
    print("\n" + "="*80)
    print("GAME CONTEXT DISCOVERY")
    print("="*80)
    
    # Home vs Away
    home_contexts = [c for c in contexts if 'home_game' in c.name and '×' not in c.name]
    if home_contexts:
        print("\nHome vs Away (measured):")
        for ctx in sorted(home_contexts, key=lambda x: x.name):
            location = "Home" if 'True' in ctx.name else "Away"
            print(f"  {location:10s}: r={ctx.r_measured:+.3f} (n={ctx.n_samples})")
    
    # Game closeness
    close_contexts = [c for c in contexts if 'game_closeness' in c.name and '×' not in c.name]
    if close_contexts:
        print("\nBy game closeness (measured):")
        for ctx in sorted(close_contexts, key=lambda x: x.r_measured, reverse=True):
            closeness = ctx.name.split('=')[1]
            print(f"  {closeness:15s}: r={ctx.r_measured:+.3f} (n={ctx.n_samples})")
    
    # === KEY INSIGHTS ===
    
    print("\n" + "="*80)
    print("KEY EMPIRICAL INSIGHTS")
    print("="*80)
    
    top_10 = contexts[:10]
    
    print("\nTop 10 contexts where narrative is EMPIRICALLY strongest:")
    for i, ctx in enumerate(top_10, 1):
        print(f"\n{i:2d}. {ctx.name} (r={ctx.r_measured:+.3f}, n={ctx.n_samples})")
        
        # Infer what this tells us
        if abs(ctx.r_measured) > 0.2:
            if 'season_phase=Playoffs' in ctx.name or 'Late Season' in ctx.name:
                print(f"    → High stakes increase narrative effects")
            elif 'Nail-biter' in ctx.name or 'Close' in ctx.name:
                print(f"    → Close games amplify narrative")
            elif 'team=' in ctx.name:
                team = ctx.name.split('=')[1].split(' ')[0]
                print(f"    → {team} has strong narrative effects")
            elif 'month=' in ctx.name:
                month = ctx.name.split('=')[1].split(' ')[0]
                print(f"    → {month} shows narrative strength")
    
    # === SAVE DISCOVERIES ===
    
    output_path = Path(__file__).parent / 'nba_empirical_discoveries_complete.json'
    engine.export_discoveries(str(output_path))
    
    print(f"\n✓ Discoveries saved: {output_path}")
    
    # === OPTIMIZATION POTENTIAL ===
    
    print("\n" + "="*80)
    print("OPTIMIZATION POTENTIAL")
    print("="*80)
    
    passing = engine.get_passing_contexts()
    
    if passing:
        print(f"\n✓ Found {len(passing)} contexts that might PASS with optimization:")
        for p in passing:
            print(f"  • {p['context']}")
            print(f"    Measured r: {p['r_measured']:.3f}")
            print(f"    Estimated efficiency: {p['efficiency']:.3f}")
    else:
        print("\n⚠️  No single contexts pass threshold")
        print(f"    Best r: {contexts[0].r_measured:.3f}")
        print("    May need:")
        print("      • Temporal aggregation (season-level)")
        print("      • Multiple-game narratives")
        print("      • Interaction effects")
    
    # === AGGREGATION RECOMMENDATION ===
    
    print("\n" + "="*80)
    print("AGGREGATION HYPOTHESIS (Data-Driven)")
    print("="*80)
    
    # Group by team-season and measure
    print("\nTesting: Does aggregation increase r?")
    print("(Measure at different scales)")
    
    df = pd.DataFrame(metadata_rows)
    df['ю'] = ю
    df['outcome'] = outcomes
    df['team_season'] = df['team'] + '_' + df['season']
    
    # Aggregate by team-season
    team_season_agg = df.groupby('team_season').agg({
        'ю': 'mean',
        'outcome': 'mean'  # Win rate
    }).reset_index()
    
    if len(team_season_agg) >= 30:
        r_season, p_season = stats.pearsonr(team_season_agg['ю'], team_season_agg['outcome'])
        
        print(f"\nSingle game level:  r={r_overall:.3f}")
        print(f"Team-season level:  r={r_season:.3f}")
        print(f"Improvement:        {abs(r_season/r_overall):.1f}x")
        
        if abs(r_season) > abs(r_overall) * 2:
            print("\n✓ DATA CONFIRMS: Aggregation increases narrative effects!")
            print("  → Season-level narratives are stronger")
            print("  → Optimize at season scale, not game scale")
    
    # === FINAL SUMMARY ===
    
    print("\n" + "="*80)
    print("NBA DATA-DRIVEN DISCOVERIES COMPLETE")
    print("="*80)
    
    print(f"\nTotal contexts measured: {len(contexts)}")
    print(f"Dimensions explored: {len(set(c.dimension for c in contexts))}")
    print(f"Strongest measured r: {contexts[0].r_measured:.3f}")
    
    print("\nNext steps (data-guided):")
    print("  1. Optimize formula for top measured contexts")
    print("  2. Test season-level aggregation (if data shows benefit)")
    print("  3. Analyze interaction effects further")
    print("  4. Explain patterns with theory (after discovery)")
    
    print("\n✓ NBA empirical discovery complete!")
    print("✓ Data has spoken - ready to optimize where it actually matters")
    
    return contexts


if __name__ == '__main__':
    main()

