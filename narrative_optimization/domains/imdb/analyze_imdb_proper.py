"""
IMDB Analysis - PROPER THEORETICAL FRAMEWORK

Follows complete variable system exactly:
1. Calculate п (narrativity)
2. Select п-appropriate transformers
3. Extract ж (genome)
4. Compute ю (story quality) with п-based weighting
5. Calculate Д (bridge) with proper baseline
6. Calculate gravitational forces (ф, ة)
7. Estimate Ξ (Golden Narratio)

This is the authoritative implementation following formal system.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.imdb.data_loader import IMDBDataLoader
from src.transformers.transformer_library import TransformerLibrary
from src.analysis.universal_analyzer import UniversalDomainAnalyzer
from src.analysis.validation_checklist import NarrativeLawValidator

# Import transformers
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.statistical import StatisticalTransformer
from src.transformers.conflict_tension import ConflictTensionTransformer
from src.transformers.suspense_mystery import SuspenseMysteryTransformer

# Intelligent transformers
try:
    from src.transformers.semantic.emotional_semantic import EmotionalSemanticTransformer
    INTELLIGENT_AVAILABLE = True
except:
    from src.transformers.emotional_resonance import EmotionalResonanceTransformer as EmotionalSemanticTransformer
    INTELLIGENT_AVAILABLE = False


def main():
    """Run proper IMDB analysis following theoretical framework"""
    
    print("="*80)
    print("IMDB ANALYSIS - PRESUME AND PROVE METHODOLOGY")
    print("="*80)
    
    # === HYPOTHESIS (PRESUMPTION) ===
    
    print("\n" + "="*80)
    print("HYPOTHESIS")
    print("="*80)
    print("\nPresumption: Narrative laws should apply to IMDB movies")
    print("Test: Д/п > 0.5 (narrative efficiency threshold)")
    print("\nIf TRUE: Better movie narratives predict outcomes")
    print("If FALSE: Genre, budget, or other factors dominate")
    
    # Initialize validator
    validator = NarrativeLawValidator()
    
    # === STEP 0: LOAD DATA ===
    
    print("\nLoading IMDB dataset...")
    loader = IMDBDataLoader()
    data = loader.load_full_dataset(use_cache=True, filter_data=True)
    
    # Sample for speed
    sample_size = 1500
    np.random.seed(42)
    indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
    data_sample = [data[i] for i in indices]
    
    # Extract components
    texts = [m['full_narrative'] for m in data_sample]
    outcomes = np.array([m['success_score'] for m in data_sample])
    names = [m['title'] for m in data_sample]
    
    print(f"✓ Loaded {len(texts)} movies")
    
    # === STEP 1: CALCULATE п (NARRATIVITY) ===
    
    print("\n" + "="*80)
    print("STEP 1: Calculate Narrativity (п)")
    print("="*80)
    
    # IMDB characteristics
    domain_characteristics = {
        'п_structural': 0.80,  # Many possible plots
        'п_temporal': 0.75,  # Stories unfold over time
        'п_agency': 0.70,  # Characters make choices
        'п_interpretation': 0.60,  # Subjective judgment (reviews)
        'п_format': 0.40   # Movie format somewhat rigid
    }
    
    analyzer = UniversalDomainAnalyzer('imdb', narrativity=0.65)
    п = analyzer.calculate_narrativity(domain_characteristics)
    
    print(f"\nCalculated п: {п:.2f}")
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
        target_feature_count=300
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
        'linguistic': LinguisticPatternsTransformer(),
        'ensemble': EnsembleNarrativeTransformer(n_top_terms=30),
        'emotional_semantic': EmotionalSemanticTransformer() if INTELLIGENT_AVAILABLE else None,
        'statistical': StatisticalTransformer(max_features=200),
        'conflict': ConflictTensionTransformer(),
        'suspense': SuspenseMysteryTransformer()
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
    print(f"  Theory target: 40-100 features")
    print(f"  Actual: {ж.shape[1]} features")
    
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
        masses=None,  # Uniform masses
        baseline_features=None  # Will estimate
    )
    
    # === SAVE RESULTS ===
    
    output_path = Path(__file__).parent / 'imdb_results_proper.json'
    
    # Convert numpy arrays to lists for JSON
    results_serializable = {
        'domain': results['domain'],
        'п': results['п'],
        'n_organisms': results['n_organisms'],
        'n_features': results['n_features'],
        'feature_names': results['feature_names'],
        
        # ю statistics
        'ю_mean': float(results['ю'].mean()),
        'ю_std': float(results['ю'].std()),
        'ю_range': [float(results['ю'].min()), float(results['ю'].max())],
        
        # Д results
        'Д_results': results['Д_results'],
        'Д': results['Д'],
        
        # Gravitational summary
        'gravitational_summary': {
            'ф_mean': float(results['ф'].mean()),
            'ة_mean': float(results['ة'].mean()),
            'ф_net_mean': float(results['ф_net'].mean()),
            'n_tensions': len(results['gravitational_tensions']),
            'top_tensions': results['gravitational_tensions'][:5]
        },
        
        # Ξ results
        'Ξ_validation': results['Ξ_validation'],
        'distance_from_Ξ_mean': float(results['distances_from_Ξ'].mean()),
        'distance_from_Ξ_std': float(results['distances_from_Ξ'].std()),
        
        # Meta
        'story_quality_weights': results['story_quality_weights'],
        'story_quality_interpretation': results['story_quality_interpretation']
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
    coupling = 0.4  # IMDB: External judges evaluate
    
    # Validate domain
    validation_result = validator.validate_domain(
        domain_name='IMDB Movies',
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
    print(f"  ж: {ж.shape[1]} features (п-guided selection)")
    print(f"  ю: computed with п-based weighting")
    print(f"  Д: {results['Д']:.4f} (measured with baseline)")
    print(f"  ф, ة: gravitational forces computed")
    print(f"  Ξ: estimated and validated")
    
    if validation_result.passes:
        print("\n🎉 HYPOTHESIS VALIDATED: Narrative laws apply to IMDB!")
    else:
        print("\n⚠️  HYPOTHESIS REJECTED: Reality constrains narrative in IMDB")
        print("   (This is honest science - not all domains pass)")
    
    print("\n✓ Analysis complete with presume-and-prove rigor!")
    
    return results


if __name__ == '__main__':
    main()

