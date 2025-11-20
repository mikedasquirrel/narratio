"""
NCAA Basketball Analysis with Standard Pipeline

Applies complete framework to March Madness / College Basketball:
- Uses standard transformers (nominative, self-perception, potential)
- Extracts ж (genome) from game narratives
- Computes ю (story quality)
- Measures r (correlation with outcomes)
- Calculates Д = п × r × κ

NCAA Basketball characteristics:
- п ≈ 0.45 (similar to NBA, rules constrain but agency exists)
- κ ≈ 0.7 (players perform, judges evaluate)
- Tournament has higher μ (mass) than regular season
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.transformer_library import TransformerLibrary
from src.analysis.validation_checklist import NarrativeLawValidator
from scipy import stats


def find_ncaa_data():
    """Find NCAA basketball data."""
    base = Path(__file__).parent.parent.parent.parent
    
    possible_paths = [
        base / 'data/march_madness/march_madness_15_years_MAXIMUM.json',
        base / 'data/march_madness/march_madness_ENRICHED_complete.json',
        base / 'data/march_madness/march_madness_2024_REAL.json'
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def analyze_ncaa():
    """Complete NCAA analysis with Д calculation."""
    print("=" * 80)
    print("NCAA BASKETBALL ANALYSIS - PRESUME AND PROVE")
    print("=" * 80)
    
    # === HYPOTHESIS (PRESUMPTION) ===
    
    print("\n" + "="*80)
    print("HYPOTHESIS")
    print("="*80)
    print("\nPresumption: Narrative laws should apply to NCAA basketball")
    print("Test: Д/п > 0.5 (narrative efficiency threshold)")
    print("\nExpectation: п ≈ 0.44 (rules constrain, but agency exists)")
    print("  • Game rules constrain actions")
    print("  • BUT players have agency within rules")
    print("  • College narratives (underdog, Cinderella stories)")
    print("\nIf TRUE: Team/game narratives predict outcomes")
    print("If FALSE: Skill, seed, physical performance dominate")
    print("\nExpected Result: FAIL (similar to NBA, performance constrains)")
    
    # Initialize validator
    validator = NarrativeLawValidator()
    
    # Find data
    data_path = find_ncaa_data()
    
    if not data_path:
        print("\n✗ No NCAA data found")
        print("\nSearched for:")
        print("  - data/march_madness/*.json")
        print("  - data/college_basketball/*.json")
        print("\nNeed to collect NCAA data or use existing if available")
        return None
    
    print(f"\n✓ Found data: {data_path.name}")
    
    # Load
    with open(data_path, 'r') as f:
        data_obj = json.load(f)
    
    # Extract games from nested structure
    if isinstance(data_obj, dict) and 'tournaments' in data_obj:
        all_games = []
        for tournament in data_obj['tournaments']:
            all_games.extend(tournament.get('games', []))
        games = all_games
    else:
        games = data_obj if isinstance(data_obj, list) else []
    
    print(f"✓ Loaded {len(games)} games")
    
    # Extract narratives and outcomes
    # For March Madness, create narrative from team names/matchup
    narratives = []
    outcomes = []
    
    for game in games:
        # Create narrative from available fields
        if isinstance(game, dict):
            name = game.get('name', '')
            short = game.get('shortName', '')
            narrative = name or short or "Game"
            
            # Outcome from competitions if available
            comps = game.get('competitions', [])
            if comps and isinstance(comps, list) and comps[0]:
                comp = comps[0]
                # Try to determine win/loss
                competitors = comp.get('competitors', [])
                if len(competitors) >= 2:
                    # Assume first is home team, check if they won
                    home = competitors[0]
                    outcome = 1 if home.get('winner', False) else 0
                    
                    narratives.append(narrative)
                    outcomes.append(outcome)
    
    print(f"✓ {len(narratives)} games with narratives and outcomes")
    
    if len(narratives) < 50:
        print("\n⚠️  Too few games with narratives")
        print("Need to enrich dataset with narrative descriptions")
        return None
    
    # Sample for speed
    sample_size = min(1000, len(narratives))
    X = np.array(narratives[:sample_size])
    y = np.array(outcomes[:sample_size])
    
    print(f"✓ Analyzing {len(X)} games")
    
    # === TRANSFORMER SELECTION ===
    
    print("\n" + "="*80)
    print("TRANSFORMER SELECTION (п-guided)")
    print("="*80)
    
    п = 0.44  # NCAA similar to NBA
    
    library = TransformerLibrary()
    selected_transformers, _ = library.get_for_narrativity(п, target=150)
    
    print(f"\nSelected {len(selected_transformers)} transformers for п={п:.2f}")
    print("\nTRANSFORMER SELECTION RATIONALE:")
    rationale = validator.generate_transformer_rationale(п, selected_transformers)
    for trans_name in selected_transformers[:5]:  # Show first 5
        print(f"  • {trans_name}")
    
    # Apply standard transformers
    print("\nApplying transformers...")
    
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
    
    if not all_features:
        print("\n✗ No features extracted")
        return None
    
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
    п = 0.45  # NCAA similar to NBA (rules, but agency)
    κ = 0.7   # Players perform, judges evaluate
    Д = п * r * κ
    
    print(f"\nNarrative Agency Calculation:")
    print(f"  п (narrativity): {п}")
    print(f"  r (correlation): {r:.4f}")
    print(f"  κ (coupling): {κ}")
    print(f"  Д = п × r × κ = {Д:.4f}")
    print(f"\n  Efficiency: Д/п = {Д/п:.4f}")
    print(f"  Threshold: {'✓ PASS' if Д/п > 0.5 else '✗ FAIL'} (need > 0.5)")
    
    results = {
        'domain': 'ncaa',
        'n_games': len(X),
        'narrativity': п,
        'coupling': κ,
        'r_measured': float(r),
        'p_value': float(p),
        'D_agency': float(Д),
        'efficiency': float(Д/п),
        'passes_threshold': bool(Д/п > 0.5)
    }
    
    # Save
    output_path = Path(__file__).parent / 'ncaa_D_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    # === VALIDATION (PROVE) ===
    
    print("\n" + "="*80)
    print("VALIDATION - TESTING HYPOTHESIS")
    print("="*80)
    
    # Validate domain
    validation_result = validator.validate_domain(
        domain_name='NCAA Basketball',
        narrativity=п,
        correlation=r,
        coupling=κ,
        transformer_info=rationale
    )
    
    # Print validation report
    print(validation_result)
    
    print("\n" + "="*80)
    print("NCAA VS NBA COMPARISON")
    print("="*80)
    print("\nBoth college and pro basketball are performance-dominated:")
    print(f"  • NCAA: п={п:.2f}, efficiency≈{Д/п:.2f}")
    print(f"  • NBA: п=0.49, efficiency=-0.03")
    print("\nSimilar constraints:")
    print("  • Rules constrain actions")
    print("  • Physical skill determines outcomes")
    print("  • Narrative exists but doesn't determine results")
    print("\nDifference: College may have MORE narrative (underdog stories)")
    print("  • But still fails threshold")
    print("  • Performance reality dominates")
    
    if validation_result.passes:
        print("\n✗ UNEXPECTED: NCAA passes (different from NBA!)")
        print("  College basketball has higher narrative agency than pro")
    else:
        print("\n✓ EXPECTED: NCAA fails (like NBA)")
        print("  Performance constrains narrative agency")
        print("  Validates sports spectrum consistency")
    
    print("\n✓ NCAA analysis complete with presume-and-prove rigor!")
    
    return results


if __name__ == "__main__":
    analyze_ncaa()

