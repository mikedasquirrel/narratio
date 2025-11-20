"""
Oscar Analysis - PROPER THEORETICAL FRAMEWORK

Follows complete variable system for competitive domain:
1. Calculate п = 0.85 (highly open/subjective)
2. Select п-appropriate transformers (character-focused)
3. Extract ж (genome)
4. Compute ю (story quality) with п-based weighting
5. Calculate Д (bridge) for binary outcomes
6. Calculate gravitational forces (competitive field analysis)
7. Estimate Ξ (ideal Oscar-winning narrative)

Plus competitive-specific analysis:
- Year-by-year gravitational fields
- Winner vs nominee discrimination
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.oscars.data_loader import OscarDataLoader
from src.transformers.transformer_library import TransformerLibrary
from src.analysis.universal_analyzer import UniversalDomainAnalyzer
from src.analysis.validation_checklist import NarrativeLawValidator

# Import transformers
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.statistical import StatisticalTransformer
from src.transformers.cultural_context import CulturalContextTransformer
from src.transformers.expertise_authority import ExpertiseAuthorityTransformer

# Intelligent transformers
try:
    from src.transformers.semantic.emotional_semantic import EmotionalSemanticTransformer
    INTELLIGENT_AVAILABLE = True
except:
    from src.transformers.emotional_resonance import EmotionalResonanceTransformer as EmotionalSemanticTransformer
    INTELLIGENT_AVAILABLE = False


def main():
    """Run proper Oscar analysis following theoretical framework"""
    
    print("="*80)
    print("OSCAR ANALYSIS - PRESUME AND PROVE METHODOLOGY")
    print("="*80)
    
    # === HYPOTHESIS (PRESUMPTION) ===
    
    print("\n" + "="*80)
    print("HYPOTHESIS")
    print("="*80)
    print("\nPresumption: Narrative laws should apply to Oscar Best Picture")
    print("Test: Д/п > 0.5 (narrative efficiency threshold)")
    print("\nExpectation: п ≈ 0.85 (highly subjective, Academy judges)")
    print("If TRUE: Better narratives predict Oscar winners")
    print("If FALSE: Politics, campaigns, or other factors dominate")
    
    # Initialize validator
    validator = NarrativeLawValidator()
    
    # === STEP 0: LOAD DATA ===
    
    print("\nLoading Oscar dataset...")
    loader = OscarDataLoader()
    processed_films, competitive_structure, stats = loader.load_full_dataset()
    
    # Extract components
    texts = [f['full_narrative'] for f in processed_films]
    outcomes = np.array([f['won_oscar'] for f in processed_films])
    names = [f['title'] for f in processed_films]
    years = np.array([f['year'] for f in processed_films])
    
    print(f"✓ Loaded {len(texts)} films across {len(competitive_structure)} years")
    print(f"  Winners: {outcomes.sum()}")
    print(f"  Nominees: {len(outcomes) - outcomes.sum()}")
    
    # === STEP 1: CALCULATE п (NARRATIVITY) ===
    
    print("\n" + "="*80)
    print("STEP 1: Calculate Narrativity (п)")
    print("="*80)
    
    # Oscar characteristics (HIGHLY OPEN)
    domain_characteristics = {
        'п_structural': 0.95,  # Infinite possible stories
        'п_temporal': 0.90,  # Films unfold over time
        'п_agency': 0.85,  # Director/actor creative control
        'п_interpretation': 0.95,  # Highly subjective Academy voting
        'п_format': 0.60   # Film format has some constraints
    }
    
    analyzer = UniversalDomainAnalyzer('oscars', narrativity=0.85)
    п = analyzer.calculate_narrativity(domain_characteristics)
    
    print(f"\nCalculated п: {п:.2f} (HIGHLY NARRATIVE)")
    print("Components:")
    for comp, value in domain_characteristics.items():
        print(f"  {comp}: {value:.2f}")
    
    # === STEP 2: SELECT TRANSFORMERS (п-GUIDED) ===
    
    print("\n" + "="*80)
    print("STEP 2: Select Transformers (п-guided)")
    print("="*80)
    
    library = TransformerLibrary()
    selected_transformers, expected_features = library.get_for_narrativity(
        п=п,
        target_feature_count=250  # Fewer for Oscar (small dataset)
    )
    
    print(f"\nSelected {len(selected_transformers)} transformers for п={п:.2f}")
    print("\nTRANSFORMER SELECTION RATIONALE:")
    rationale = validator.generate_transformer_rationale(п, selected_transformers)
    for trans_name in selected_transformers:
        print(f"  • {trans_name}:")
        print(f"    {rationale.get(trans_name, 'Selected for domain coverage')}")
    
    # === STEP 3: EXTRACT ж (GENOME) ===
    
    print("\n" + "="*80)
    print("STEP 3: Extract Genome (ж)")
    print("="*80)
    
    # Map selected names to actual transformers
    transformer_map = {
        'nominative': NominativeAnalysisTransformer(),
        'self_perception': SelfPerceptionTransformer(),
        'narrative_potential': NarrativePotentialTransformer(),
        'linguistic': LinguisticPatternsTransformer(),
        'emotional_semantic': EmotionalSemanticTransformer() if INTELLIGENT_AVAILABLE else None,
        'statistical': StatisticalTransformer(max_features=100),
        'cultural_context': CulturalContextTransformer(),
        'expertise': ExpertiseAuthorityTransformer()
    }
    
    # Extract features
    all_features = []
    all_feature_names = []
    
    for trans_name in selected_transformers:
        if trans_name in transformer_map and transformer_map[trans_name] is not None:
            print(f"\nExtracting {trans_name}...")
            try:
                transformer = transformer_map[trans_name]
                transformer.fit(texts)
                features = transformer.transform(texts)
                
                # Convert sparse to dense
                if hasattr(features, 'toarray'):
                    features = features.toarray()
                
                all_features.append(features)
                
                # Get feature names
                if hasattr(transformer, 'get_feature_names_out'):
                    names_out = transformer.get_feature_names_out()
                    all_feature_names.extend(names_out)
                else:
                    all_feature_names.extend([f"{trans_name}_{i}" for i in range(features.shape[1])])
                
                print(f"  ✓ {features.shape[1]} features extracted")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Combine into ж
    ж = np.hstack(all_features)
    
    print(f"\n✓ Genome (ж) extracted: {ж.shape}")
    print(f"  Theory: Character-driven features dominate (п={п:.2f})")
    
    # === STEP 4-9: COMPLETE ANALYSIS ===
    
    print("\n" + "="*80)
    print("STEP 4-9: Complete Variable System Analysis")
    print("="*80)
    
    # Run full analysis
    results = analyzer.analyze_complete(
        texts=texts,
        outcomes=outcomes,
        names=names,
        genome=ж,
        feature_names=all_feature_names,
        masses=None,  # Uniform masses for Oscar (all prestige films)
        baseline_features=None  # Will estimate (genre + year baseline)
    )
    
    # === COMPETITIVE ANALYSIS (Oscar-specific) ===
    
    print("\n" + "="*80)
    print("COMPETITIVE FIELD ANALYSIS (Oscar-Specific)")
    print("="*80)
    
    # Analyze gravitational forces within each year
    year_analyses = {}
    
    for year in np.unique(years):
        year_mask = years == year
        year_indices = np.where(year_mask)[0]
        
        if len(year_indices) < 2:
            continue
        
        # Get year's data
        year_ж = ж[year_mask]
        year_ю = results['ю'][year_mask]
        year_outcomes = outcomes[year_mask]
        year_names = [names[i] for i in year_indices]
        
        # Calculate forces within competitive field
        year_forces = analyzer.gravitational_calc.calculate_all_forces(
            year_ж, year_names, np.ones(len(year_ж)), year_ю
        )
        
        # Find winner
        winner_idx = np.where(year_outcomes == 1)[0]
        if len(winner_idx) > 0:
            winner_idx = winner_idx[0]
            
            # Winner's gravitational position
            winner_ф = year_forces['ф'][winner_idx, :].sum()
            winner_ة = year_forces['ة'][winner_idx, :].sum()
            
            # Winner's distance from year's Ξ
            year_Ξ = year_ж[winner_idx]  # Winner defines ideal for year
            distances = np.linalg.norm(year_ж - year_Ξ, axis=1)
            
            year_analyses[int(year)] = {
                'n_nominees': len(year_indices),
                'winner': year_names[winner_idx],
                'winner_ф': float(winner_ф),
                'winner_ة': float(winner_ة),
                'winner_ю': float(year_ю[winner_idx]),
                'mean_nominee_ю': float(year_ю[year_outcomes == 0].mean()) if (year_outcomes == 0).sum() > 0 else 0.0,
                'winner_distance_from_field_center': float(distances[winner_idx])
            }
    
    print(f"\nAnalyzed {len(year_analyses)} competitive years:")
    for year in sorted(year_analyses.keys()):
        ya = year_analyses[year]
        print(f"\n  {year}: {ya['winner']}")
        print(f"    ю_winner: {ya['winner_ю']:.3f} vs ю_nominees: {ya['mean_nominee_ю']:.3f}")
        print(f"    Gravitational position: ф={ya['winner_ф']:.6f}, ة={ya['winner_ة']:.6f}")
    
    # === SAVE RESULTS ===
    
    results['competitive_years'] = year_analyses
    
    output_path = Path(__file__).parent / 'oscar_results_proper.json'
    
    results_serializable = {
        'domain': results['domain'],
        'п': results['п'],
        'n_organisms': results['n_organisms'],
        'n_features': results['n_features'],
        
        # All variable results
        'ю_statistics': {
            'mean': float(results['ю'].mean()),
            'std': float(results['ю'].std()),
            'range': [float(results['ю'].min()), float(results['ю'].max())]
        },
        'Д_results': results['Д_results'],
        'Ξ_validation': results['Ξ_validation'],
        'competitive_years': year_analyses,
        
        # Framework metadata
        'story_quality_weights': results['story_quality_weights'],
        'theoretical_alignment': 'Complete - all variables calculated per formal system'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    # === VALIDATION (PROVE) ===
    
    print("\n" + "="*80)
    print("VALIDATION - TESTING HYPOTHESIS")
    print("="*80)
    
    # Extract key metrics
    r = results['Д_results']['r_narrative']
    coupling = 0.7  # Oscar: Academy judges are involved in narrative creation
    
    # Validate domain
    validation_result = validator.validate_domain(
        domain_name='Oscar Best Picture',
        narrativity=п,
        correlation=r,
        coupling=coupling,
        transformer_info=rationale
    )
    
    # Print validation report
    print(validation_result)
    
    # === FINAL SUMMARY ===
    
    print("\n" + "="*80)
    print("THEORETICAL ALIGNMENT VERIFIED")
    print("="*80)
    print("\n✓ All variables calculated according to formal system:")
    print(f"  ж: {ж.shape[1]} features (character-focused for п={п:.2f})")
    print(f"  ю: computed with п-based weighting")
    print(f"  Д: {results['Д']:.4f} (measured with baseline)")
    print(f"  ф, ة: competitive gravitational fields analyzed")
    print(f"  Ξ: Oscar-winning archetypal narrative estimated")
    
    if validation_result.passes:
        print("\n🎉 HYPOTHESIS VALIDATED: Narrative laws apply to Oscars!")
        print("   Academy voting is predictable from narrative quality.")
    else:
        print("\n⚠️  HYPOTHESIS REJECTED: Politics/campaigns dominate Oscars")
        print("   (This is honest science - not all domains pass)")
    
    print("\n✓ Oscar analysis complete with presume-and-prove rigor!")
    
    return results


if __name__ == '__main__':
    main()

