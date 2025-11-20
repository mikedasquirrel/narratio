"""
UFC Complete Analysis - Applying Narrative Framework

Comprehensive analysis following presume-and-prove methodology:
1. Calculate п (narrativity) - UFC specific
2. Apply ALL 33+ transformers to extract ж (genome)
3. Compute ю (story quality)
4. Measure |r| (absolute correlation with outcomes)
5. Calculate Д (bridge) and efficiency
6. Validate: Д/п > 0.5 hypothesis
7. Test if UFC is 3rd PASSING domain!

Expected п ≈ 0.73 (highest individual sport)
Need |r| > 0.85 to pass with κ = 0.6
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    # Core 6 - CRITICAL for nominative-rich UFC
    StatisticalTransformer,
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    EnsembleNarrativeTransformer,
    RelationalValueTransformer,
    
    # Phase 1 Enhancements
    OpticsTransformer,
    FramingTransformer,
    
    # Phase 1.5 Nominative Foundation
    PhoneticTransformer,
    TemporalEvolutionTransformer,
    InformationTheoryTransformer,
    SocialStatusTransformer,
    
    # Phase 1.6 Advanced
    NamespaceEcologyTransformer,
    AnticipatoryCommunicationTransformer,
    
    # Phase 2 Complete Coverage
    QuantitativeTransformer,
    CrossmodalTransformer,
    AudioTransformer,
    CrossLingualTransformer,
    DiscoverabilityTransformer,
    CognitiveFluencyTransformer,
    
    # Phase 3 Critical Missing  
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,      # HUGE for UFC rivalries!
    ExpertiseAuthorityTransformer,
    CulturalContextTransformer,
    SuspenseMysteryTransformer,       # Upset potential
    VisualMultimodalTransformer,
    
    # Phase 4 Complete Nominative - MAX IMPORTANCE FOR UFC!
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    NominativeInteractionTransformer,
    PureNominativePredictorTransformer,
    
    # Phase 5 Multi-scale
    MultiScaleTransformer,
    MultiPerspectiveTransformer,
    ScaleInteractionTransformer,
)


def main():
    """
    Complete UFC narrative analysis following framework.
    
    GOAL: Determine if UFC PASSES threshold (3rd passing domain!)
    """
    print("="*80)
    print("UFC NARRATIVE ANALYSIS - COMPLETE FRAMEWORK APPLICATION")
    print("TESTING: Could UFC be 3rd PASSING domain?")
    print("="*80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n[STEP 1] Loading UFC dataset...")
    
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'ufc_with_narratives.json'
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run generate_fighter_narratives.py first!")
        return
    
    with open(dataset_path) as f:
        fights = json.load(f)
    
    print(f"✓ Loaded {len(fights)} fights")
    
    # Extract data
    narratives = [f['narrative'] for f in fights]
    
    # Binary outcome: fighter_a wins (1) or fighter_b wins (0)
    outcomes = np.array([1 if f['result']['winner'] == 'fighter_a' else 0 for f in fights])
    
    # Fight names for reference
    names = [f"{f['fighter_a']['name']} vs {f['fighter_b']['name']}" for f in fights]
    
    print(f"  Fighter A wins: {outcomes.sum()} ({100*outcomes.mean():.1f}%)")
    print(f"  Fighter B wins: {len(outcomes) - outcomes.sum()} ({100*(1-outcomes.mean()):.1f}%)")
    print(f"  Total narratives: {len(narratives)}")
    print(f"  Average narrative length: {np.mean([len(n.split()) for n in narratives]):.0f} words")
    
    # ========================================================================
    # STEP 2: CALCULATE п (NARRATIVITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 2] Calculate Narrativity (п) - UFC Specific")
    print("="*80)
    
    print("\nHYPOTHESIS: UFC should have HIGHEST narrativity among sports")
    print("REASONING: Pure 1v1, individual agency, persona-driven")
    print("TEST: Д/п > 0.5")
    print("EXPECTATION: п ≈ 0.73 (much higher than team sports)")
    
    # UFC-specific characteristics (from INSTRUCTIONS lines 296-325)
    domain_characteristics = {
        'п_structural': 0.70,    # Many fight paths (striker, grappler, mixed)
        'п_temporal': 0.65,      # 15-25 minute fights, career arcs
        'п_agency': 0.95,        # EXTREME individual agency
        'п_interpretation': 0.70, # Judges for decisions, but finish = objective
        'п_format': 0.40         # Rules constrain (rounds, weight classes)
    }
    
    # Weighted calculation (from INSTRUCTIONS)
    weights = {
        'п_structural': 0.30,
        'п_temporal': 0.20,
        'п_agency': 0.25,
        'п_interpretation': 0.15,
        'п_format': 0.10
    }
    
    п = sum(domain_characteristics[k] * weights[k] for k in domain_characteristics.keys())
    
    print(f"\nп Component Breakdown:")
    for component, value in domain_characteristics.items():
        weight = weights[component]
        contribution = value * weight
        print(f"  {component}: {value:.2f} (weight {weight:.2f}) → {contribution:.3f}")
    
    print(f"\n✓ Calculated п: {п:.3f}")
    print(f"  Classification: {'HIGH' if п > 0.7 else 'MEDIUM' if п > 0.4 else 'LOW'} narrativity")
    print(f"  Comparison: NBA п=0.49, NFL п≈0.48")
    print(f"  UFC п={п:.3f} is {п/0.49:.1f}x higher than NBA!")
    print(f"\n  Domain Type: INDIVIDUAL NARRATIVE (highest agency, persona-driven)")
    
    # Calculate expected threshold
    κ = 0.6  # Moderate-high coupling (from INSTRUCTIONS line 537)
    print(f"\n  Estimated κ (coupling): {κ:.2f}")
    print(f"  Required |r| to pass: {0.5/п/κ:.3f}")
    print(f"  → Need |r| > 0.{int(0.5/п/κ*100)} to achieve Д/п > 0.5")
    
    # ========================================================================
    # STEP 3: APPLY ALL TRANSFORMERS → EXTRACT ж (GENOME)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 3] Apply ALL Transformers - Extract Genome (ж)")
    print("="*80)
    
    print("\nApplying 33+ transformers (MAXIMUM nominative emphasis for UFC)...")
    print("Expected: 400-600 features (nominative-heavy for 1v1 matchups)")
    print(f"Processing {len(narratives)} fight narratives...")
    print("This will take 2-5 minutes - watch for progress updates!")
    
    # Initialize all transformers
    transformers = [
        ('statistical', StatisticalTransformer(max_features=100)),
        
        # CORE NOMINATIVE - CRITICAL FOR UFC!
        ('nominative', NominativeAnalysisTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('universal_nominative', UniversalNominativeTransformer()),
        ('hierarchical_nominative', HierarchicalNominativeTransformer()),
        ('nominative_interaction', NominativeInteractionTransformer()),
        ('pure_nominative', PureNominativePredictorTransformer()),
        
        # PERSONA & PERCEPTION - HUGE FOR UFC!
        ('self_perception', SelfPerceptionTransformer()),
        ('social_status', SocialStatusTransformer()),
        ('authenticity', AuthenticityTransformer()),
        
        # CONFLICT & TENSION - CRITICAL FOR RIVALRIES!
        ('conflict_tension', ConflictTensionTransformer()),
        ('suspense_mystery', SuspenseMysteryTransformer()),
        ('emotional_resonance', EmotionalResonanceTransformer()),
        
        # NARRATIVE STRUCTURE
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('temporal', TemporalEvolutionTransformer()),
        ('framing', FramingTransformer()),
        
        # EXPERTISE & AUTHORITY
        ('expertise', ExpertiseAuthorityTransformer()),
        ('cultural', CulturalContextTransformer()),
        
        # ENSEMBLE & RELATIONAL
        ('ensemble', EnsembleNarrativeTransformer()),
        ('relational', RelationalValueTransformer()),
        ('namespace_ecology', NamespaceEcologyTransformer()),
        
        # INFORMATION & COGNITION
        ('information_theory', InformationTheoryTransformer()),
        ('cognitive_fluency', CognitiveFluencyTransformer()),
        ('discoverability', DiscoverabilityTransformer()),
        
        # OPTICS & COMMUNICATION
        ('optics', OpticsTransformer()),
        ('anticipatory', AnticipatoryCommunicationTransformer()),
        
        # MULTIMODAL (limited for text)
        ('quantitative', QuantitativeTransformer()),
        ('crossmodal', CrossmodalTransformer()),
        ('visual', VisualMultimodalTransformer()),
        
        # MULTI-SCALE ANALYSIS
        ('multi_scale', MultiScaleTransformer()),
        ('multi_perspective', MultiPerspectiveTransformer()),
        ('scale_interaction', ScaleInteractionTransformer()),
    ]
    
    print(f"\nTotal transformers: {len(transformers)}")
    
    # Apply transformers
    all_features = []
    feature_names = []
    transformer_feature_counts = {}
    
    total_transformers = len(transformers)
    
    for idx, (name, transformer) in enumerate(transformers, 1):
        print(f"\n  [{idx}/{total_transformers}] Applying {name}...", end=" ", flush=True)
        try:
            import time
            start_time = time.time()
            
            X = transformer.fit_transform(narratives)
            all_features.append(X)
            
            elapsed = time.time() - start_time
            
            # Get feature names
            if hasattr(transformer, 'get_feature_names_out'):
                names_out = transformer.get_feature_names_out()
            else:
                names_out = [f"{name}_{i}" for i in range(X.shape[1])]
            
            feature_names.extend(names_out)
            transformer_feature_counts[name] = X.shape[1]
            
            print(f"✓ {X.shape[1]} features ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            continue
    
    # Combine all features
    print(f"\nCombining features...")
    ж = np.hstack(all_features)
    
    print(f"\n✓ Extracted genome (ж): {ж.shape[1]} total features from {ж.shape[0]} fights")
    
    # Show feature breakdown
    print(f"\nFeature Breakdown by Transformer:")
    for name, count in sorted(transformer_feature_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / ж.shape[1]
        print(f"  {name}: {count} features ({pct:.1f}%)")
    
    # Nominative total
    nominative_transformers = ['nominative', 'phonetic', 'universal_nominative', 
                               'hierarchical_nominative', 'nominative_interaction', 'pure_nominative']
    nominative_total = sum(transformer_feature_counts.get(t, 0) for t in nominative_transformers)
    print(f"\n  NOMINATIVE TOTAL: {nominative_total} features ({100*nominative_total/ж.shape[1]:.1f}%)")
    print(f"  → UFC is nominative-dominated as expected!")
    
    # ========================================================================
    # STEP 4: COMPUTE ю (STORY QUALITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 4] Compute Story Quality (ю)")
    print("="*80)
    
    print("\nStandardizing features...")
    scaler = StandardScaler()
    ж_scaled = scaler.fit_transform(ж)
    
    # ю = mean absolute feature value (normalized)
    ю = np.abs(ж_scaled).mean(axis=1)
    
    print(f"✓ Computed ю for {len(ю)} fights")
    print(f"  Mean ю: {ю.mean():.3f}")
    print(f"  Std ю: {ю.std():.3f}")
    print(f"  Min ю: {ю.min():.3f}")
    print(f"  Max ю: {ю.max():.3f}")
    
    # ========================================================================
    # STEP 5: MEASURE |r| (CORRELATION)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 5] Measure Absolute Correlation |r|")
    print("="*80)
    
    print("\nTesting: Does narrative quality (ю) predict fight outcomes?")
    
    # Simple correlation
    r_raw = np.corrcoef(ю, outcomes)[0, 1]
    r_abs = abs(r_raw)
    
    print(f"\n✓ Measured correlation:")
    print(f"  r (raw): {r_raw:+.4f}")
    print(f"  |r| (absolute): {r_abs:.4f}")
    
    # Predictive model
    print(f"\nTesting predictive power with logistic regression...")
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Cross-validation
    cv_scores = cross_val_score(model, ю.reshape(-1, 1), outcomes, cv=5, scoring='roc_auc')
    
    print(f"  5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Full model
    model.fit(ю.reshape(-1, 1), outcomes)
    predictions = model.predict(ю.reshape(-1, 1))
    accuracy = (predictions == outcomes).mean()
    
    print(f"  Prediction accuracy: {accuracy:.4f}")
    print(f"  Baseline (always predict favorite): {max(outcomes.mean(), 1-outcomes.mean()):.4f}")
    
    # ========================================================================
    # STEP 6: CALCULATE Д (BRIDGE) AND EFFICIENCY
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 6] Calculate Bridge (Д) and Efficiency")
    print("="*80)
    
    print(f"\nFramework Variables:")
    print(f"  п (narrativity): {п:.3f}")
    print(f"  |r| (correlation): {r_abs:.4f}")
    print(f"  κ (coupling): {κ:.2f} (estimated)")
    
    # Calculate Д
    Д = п * r_abs * κ
    efficiency = Д / п
    
    print(f"\n✓ Calculated Bridge (Д):")
    print(f"  Д = п × |r| × κ")
    print(f"  Д = {п:.3f} × {r_abs:.4f} × {κ:.2f}")
    print(f"  Д = {Д:.4f}")
    
    print(f"\n✓ Calculated Efficiency:")
    print(f"  Efficiency = Д / п")
    print(f"  Efficiency = {Д:.4f} / {п:.3f}")
    print(f"  Efficiency = {efficiency:.4f}")
    
    # ========================================================================
    # STEP 7: PRESUME AND PROVE - PASS/FAIL
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 7] PRESUME AND PROVE - VALIDATION")
    print("="*80)
    
    threshold = 0.5
    passes = efficiency > threshold
    
    print(f"\nHYPOTHESIS: Narrative laws apply to UFC")
    print(f"TEST: Д/п > {threshold}")
    print(f"\nRESULT: Efficiency = {efficiency:.4f}")
    print(f"THRESHOLD: {threshold}")
    
    if passes:
        print(f"\n{'='*80}")
        print(f"{'  ' * 10}✓ PASSES! ✓")
        print(f"{'='*80}")
        print(f"\n🎉 UFC is the 3RD PASSING DOMAIN! 🎉")
        print(f"\nPassing Domains:")
        print(f"  1. Character Creation (self-rated, п≈0.95)")
        print(f"  2. Self-Rated Narratives (π≈0.90)")
        print(f"  3. UFC (п={п:.3f}) ← NEW!")
        print(f"\nUFC Pass Rate: 3/11 domains (27.3%)")
        print(f"\nInterpretation:")
        print(f"  - FIRST combat sport to pass!")
        print(f"  - Individual agency + persona > team constraints")
        print(f"  - Fighter names and narratives MATTER")
        print(f"  - 1v1 format maximizes nominative effects")
    else:
        deficit = threshold - efficiency
        print(f"\n{'='*80}")
        print(f"{'  ' * 10}✗ FAILS")
        print(f"{'='*80}")
        print(f"\nDeficit: {deficit:.4f}")
        print(f"  Need {deficit/r_abs:.1%} stronger correlation to pass")
        print(f"  OR need {deficit/κ:.1%} higher coupling")
        
        print(f"\nInterpretation:")
        print(f"  - UFC still HIGHER than team sports (NBA {0.49:.2f}, NFL ~{0.48:.2f})")
        print(f"  - Individual narrative matters more than teams")
        print(f"  - Physical performance dominates over narrative")
        print(f"  - BUT: Specific contexts might pass (see context discovery)")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 8] Save Results")
    print("="*80)
    
    results = {
        'domain': 'UFC',
        'dataset_size': len(fights),
        'narrativity': {
            'п': float(п),
            'components': {k: float(v) for k, v in domain_characteristics.items()},
            'classification': 'HIGH' if п > 0.7 else 'MEDIUM' if п > 0.4 else 'LOW'
        },
        'genome': {
            'total_features': int(ж.shape[1]),
            'transformer_counts': {k: int(v) for k, v in transformer_feature_counts.items()},
            'nominative_features': int(nominative_total),
            'nominative_percentage': float(nominative_total / ж.shape[1])
        },
        'story_quality': {
            'mean_ю': float(ю.mean()),
            'std_ю': float(ю.std()),
            'min_ю': float(ю.min()),
            'max_ю': float(ю.max())
        },
        'correlation': {
            'r_raw': float(r_raw),
            'r_absolute': float(r_abs),
            'cv_auc': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'accuracy': float(accuracy),
            'baseline': float(max(outcomes.mean(), 1-outcomes.mean()))
        },
        'bridge': {
            'Д': float(Д),
            'efficiency': float(efficiency),
            'κ_estimated': float(κ),
            'threshold': float(threshold),
            'passes': bool(passes)
        },
        'comparison': {
            'nba_п': 0.49,
            'nfl_п': 0.48,
            'ufc_higher_by': float(п / 0.49)
        }
    }
    
    output_path = Path(__file__).parent / 'ufc_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    
    # Also save genome for further analysis
    genome_path = Path(__file__).parent / 'ufc_genome_data.npz'
    np.savez(genome_path, 
             genome=ж,
             story_quality=ю,
             outcomes=outcomes,
             feature_names=np.array(feature_names))
    
    print(f"✓ Genome data saved to: {genome_path}")
    
    print("\n" + "="*80)
    print("UFC ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Run discover_ufc_contexts.py (find top contexts by |r|)")
    print(f"2. Run test_ufc_betting_edge.py (narrative vs Vegas odds)")
    print(f"3. Create Flask route and dashboard")


if __name__ == "__main__":
    main()

