"""
Tennis Complete Analysis - Applying Narrative Framework

HIGH ROI POTENTIAL: Individual sport with mental game dominance

Comprehensive analysis following presume-and-prove methodology:
1. Calculate π (narrativity) - EXPECTED: 0.70-0.80 (HIGH)
2. Apply ALL 33 transformers to extract ж (genome)
3. Compute ю (story quality)
4. Measure |r| (absolute correlation with outcomes)
5. Calculate Δ (bridge) and efficiency
6. Validate: Δ/π > 0.5 hypothesis (may PASS unlike NFL)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    StatisticalTransformer,
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    EnsembleNarrativeTransformer,
    RelationalValueTransformer,
    OpticsTransformer,
    FramingTransformer,
    PhoneticTransformer,
    TemporalEvolutionTransformer,
    InformationTheoryTransformer,
    SocialStatusTransformer,
    NamespaceEcologyTransformer,
    AnticipatoryCommunicationTransformer,
    QuantitativeTransformer,
    CrossmodalTransformer,
    AudioTransformer,
    CrossLingualTransformer,
    DiscoverabilityTransformer,
    CognitiveFluencyTransformer,
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    ExpertiseAuthorityTransformer,
    CulturalContextTransformer,
    SuspenseMysteryTransformer,
    VisualMultimodalTransformer,
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    NominativeInteractionTransformer,
    PureNominativePredictorTransformer,
    MultiScaleTransformer,
    MultiPerspectiveTransformer,
    ScaleInteractionTransformer,
)


def main():
    """Complete tennis narrative analysis."""
    print("="*80)
    print("TENNIS NARRATIVE ANALYSIS - COMPLETE FRAMEWORK APPLICATION")
    print("="*80)
    print("\nHIGH ROI POTENTIAL: Individual sport, mental game, betting opportunities")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n[STEP 1] Loading tennis dataset...")
    
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
    
    with open(dataset_path) as f:
        all_matches = json.load(f)
    
    print(f"✓ Loaded {len(all_matches)} matches")
    
    # For transformer analysis, use strategic sample to avoid memory issues
    # Sample: Recent years (2015-2024) + Grand Slams + Top rankings
    print("\nCreating strategic sample for analysis...")
    
    sampled_matches = []
    for m in all_matches:
        # Priority: Grand Slams, Recent years, Top players
        year = m['year']
        level = m['level']
        p1_rank = m['player1'].get('ranking')
        p2_rank = m['player2'].get('ranking')
        
        include = False
        
        if level == 'grand_slam':  # All Grand Slams
            include = True
        elif year >= 2015:  # Recent matches
            if p1_rank and p2_rank:
                if p1_rank <= 50 or p2_rank <= 50:  # Top 50 players
                    include = True
        
        if include:
            sampled_matches.append(m)
        
        if len(sampled_matches) >= 5000:  # Cap at 5K for speed
            break
    
    matches = sampled_matches
    print(f"✓ Using {len(matches)} matches for analysis (strategic sample)")
    print(f"  Grand Slams: {sum(1 for m in matches if m['level'] == 'grand_slam')}")
    print(f"  Years 2015+: {sum(1 for m in matches if m['year'] >= 2015)}")
    print(f"  Full dataset remains available for context discovery")
    
    # Extract data with balanced outcomes
    print("\nExtracting match narratives and outcomes...")
    
    narratives = []
    outcomes = []
    
    for m in matches:
        # Create narrative if not present
        if 'narrative' not in m or not m['narrative']:
            p1_name = m['player1']['name']
            p2_name = m['player2']['name']
            p1_rank = m['player1'].get('ranking', 999)
            p2_rank = m['player2'].get('ranking', 999)
            surface = m['surface']
            tournament = m['tournament']
            level = m['level']
            
            narrative = f"{p1_name} (ranked #{p1_rank}) faces {p2_name} (ranked #{p2_rank}) "
            narrative += f"on {surface} court at {tournament} ({level})."
            
            if p1_rank and p2_rank and p1_rank < p2_rank:
                narrative += f" {p1_name} is the higher-ranked favorite."
            elif p1_rank and p2_rank and p2_rank < p1_rank:
                narrative += f" {p2_name} is the higher-ranked favorite."
            
            m['narrative'] = narrative
        
        narratives.append(m['narrative'])
        
        # Outcome: 1 if player1 won, 0 if player2 won
        outcomes.append(1 if m['player1_won'] else 0)
    
    outcomes = np.array(outcomes)
    
    print(f"  Player1 wins: {outcomes.sum()} ({100*outcomes.sum()/len(outcomes):.1f}%)")
    print(f"  Player2 wins: {len(outcomes) - outcomes.sum()} ({100*(1-outcomes.sum()/len(outcomes)):.1f}%)")
    
    # ========================================================================
    # STEP 2: CALCULATE π (NARRATIVITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 2] Calculate Narrativity (π)")
    print("="*80)
    
    print("\nHYPOTHESIS: Narrative laws should apply to tennis (individual sport)")
    print("TEST: Δ/π > 0.5")
    print("EXPECTATION: π ≈ 0.75 (HIGH - individual sport, mental game)")
    
    domain_characteristics = {
        'π_structural': 0.60,  # Rules exist but more open than team sports
        'π_temporal': 0.70,    # Match unfolds with momentum shifts
        'π_agency': 1.00,      # Individual player agency complete
        'π_interpretation': 0.80,  # Mental game heavily interpreted
        'π_format': 0.65       # Tennis format allows variation
    }
    
    π = np.mean(list(domain_characteristics.values()))
    
    print(f"\nπ Component Breakdown:")
    for component, value in domain_characteristics.items():
        print(f"  {component}: {value:.2f}")
    
    print(f"\n✓ Calculated π: {π:.3f}")
    print(f"  Classification: {'HIGH' if π > 0.7 else 'MEDIUM' if π > 0.4 else 'LOW'} narrativity")
    print(f"  Domain Type: INDIVIDUAL PERFORMANCE (mental game critical)")
    
    # ========================================================================
    # STEP 3: APPLY ALL TRANSFORMERS → EXTRACT ж (GENOME)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 3] Apply ALL Transformers - Extract Genome (ж)")
    print("="*80)
    
    print("\nApplying 33 transformers (nominative + mental game heavy)...")
    print("Expected: 600-800 features due to rich nominative content")
    
    # Initialize all transformers
    transformers = [
        ('statistical', StatisticalTransformer(max_features=100)),
        ('nominative', NominativeAnalysisTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('ensemble', EnsembleNarrativeTransformer()),
        ('relational', RelationalValueTransformer()),
        ('optics', OpticsTransformer()),
        ('framing', FramingTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('temporal', TemporalEvolutionTransformer()),
        ('information_theory', InformationTheoryTransformer()),
        ('social_status', SocialStatusTransformer()),
        ('namespace', NamespaceEcologyTransformer()),
        ('anticipatory', AnticipatoryCommunicationTransformer()),
        ('quantitative', QuantitativeTransformer()),
        ('crossmodal', CrossmodalTransformer()),
        ('audio', AudioTransformer()),
        ('crosslingual', CrossLingualTransformer()),
        ('discoverability', DiscoverabilityTransformer()),
        ('cognitive_fluency', CognitiveFluencyTransformer()),
        ('emotional', EmotionalResonanceTransformer()),
        ('authenticity', AuthenticityTransformer()),
        ('conflict', ConflictTensionTransformer()),
        ('expertise', ExpertiseAuthorityTransformer()),
        ('cultural', CulturalContextTransformer()),
        ('suspense', SuspenseMysteryTransformer()),
        ('visual', VisualMultimodalTransformer()),
        ('universal_nominative', UniversalNominativeTransformer()),
        ('hierarchical_nominative', HierarchicalNominativeTransformer()),
        ('nominative_interaction', NominativeInteractionTransformer()),
        ('pure_nominative', PureNominativePredictorTransformer()),
        ('multi_scale', MultiScaleTransformer()),
        ('multi_perspective', MultiPerspectiveTransformer()),
        ('scale_interaction', ScaleInteractionTransformer()),
    ]
    
    # Extract features from each transformer
    all_features = []
    all_feature_names = []
    
    for idx, (name, transformer) in enumerate(transformers, 1):
        try:
            print(f"\n  [{idx}/{len(transformers)}] Applying {name}...", end=" ", flush=True)
            transformer.fit(narratives)
            print("fitted...", end=" ", flush=True)
            features = transformer.transform(narratives)
            print("transformed...", end=" ", flush=True)
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            # Get feature names
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    feature_names = transformer.get_feature_names_out()
                except:
                    feature_names = [f'{name}_{i}' for i in range(features.shape[1])]
            else:
                feature_names = [f'{name}_{i}' for i in range(features.shape[1])]
            
            all_features.append(features)
            all_feature_names.extend(feature_names)
            
            print(f"✓ {features.shape[1]} features")
        
        except Exception as e:
            print(f"⚠ Error: {str(e)[:50]}")
            continue
    
    # Combine all features
    processed_features = []
    n_samples = len(narratives)
    
    print(f"\nProcessing {len(all_features)} feature arrays...")
    for i, feat in enumerate(all_features):
        try:
            if not isinstance(feat, np.ndarray):
                feat = np.array(feat)
            
            if feat.ndim == 0:
                continue
            
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            
            if feat.shape[0] != n_samples or feat.ndim != 2:
                continue
                
            processed_features.append(feat)
        except:
            continue
    
    print(f"✓ {len(processed_features)} valid feature arrays ready")
    
    ж = np.hstack(processed_features)
    
    print(f"\n{'='*80}")
    print(f"✓ GENOME EXTRACTED (ж)")
    print(f"  Total features: {ж.shape[1]}")
    print(f"  Matches: {ж.shape[0]}")
    print(f"  Feature space: {ж.nbytes / 1024 / 1024:.1f} MB")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 4: COMPUTE ю (STORY QUALITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 4] Compute Story Quality (ю)")
    print("="*80)
    
    print("\nWeighting features based on π...")
    print(f"π = {π:.3f} → HIGH weighting (narrative-rich domain)")
    
    # Normalize features
    scaler = StandardScaler()
    ж_normalized = scaler.fit_transform(ж)
    
    # Compute story quality with nominative/mental game emphasis
    feature_weights = np.ones(ж.shape[1])
    
    # Boost nominative and mental game features
    nominative_kw = ['nominative', 'phonetic', 'name', 'hierarchical', 'interaction', 'pure_nominative', 'universal_nominative']
    mental_kw = ['emotional', 'conflict', 'tension', 'suspense', 'authenticity', 'self_perception']
    
    for i, fname in enumerate(all_feature_names[:ж.shape[1]]):
        fname_lower = fname.lower()
        if any(kw in fname_lower for kw in nominative_kw):
            feature_weights[i] *= 2.0  # Heavy nominative boost
        elif any(kw in fname_lower for kw in mental_kw):
            feature_weights[i] *= 1.5  # Mental game boost
    
    # Normalize weights
    feature_weights = feature_weights / feature_weights.sum()
    
    # Compute ю
    ю = (ж_normalized * feature_weights).sum(axis=1)
    ю = (ю - ю.min()) / (ю.max() - ю.min())  # Normalize to [0, 1]
    
    print(f"\n✓ Story Quality (ю) computed")
    print(f"  Mean: {ю.mean():.3f}")
    print(f"  Std: {ю.std():.3f}")
    print(f"  Range: [{ю.min():.3f}, {ю.max():.3f}]")
    
    # ========================================================================
    # STEP 5: MEASURE |r| (ABSOLUTE CORRELATION)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 5] Measure Correlation |r|")
    print("="*80)
    
    print("\nCorrelating ю (story quality) with ❊ (outcomes)...")
    print("Using ABSOLUTE VALUE per framework correction")
    
    r = np.corrcoef(ю, outcomes)[0, 1]
    abs_r = abs(r)
    
    print(f"\n✓ Correlation measured:")
    print(f"  r = {r:.4f}")
    print(f"  |r| = {abs_r:.4f}")
    
    if r < 0:
        print(f"\n  ⚠ INVERSE PATTERN (unexpected for tennis)")
        print(f"     Better narratives → UNDERDOG wins")
    else:
        print(f"\n  ✓ POSITIVE PATTERN (expected)")
        print(f"     Better narratives → Favorite wins")
    
    # ========================================================================
    # STEP 6: CALCULATE Δ (BRIDGE) AND EFFICIENCY
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 6] Calculate Δ (Bridge) and Efficiency")
    print("="*80)
    
    # κ for individual performance domain with mental game
    κ = 0.40  # Mental game is judged + interpreted
    
    Δ = π * abs_r * κ
    efficiency = Δ / π
    
    print(f"\nΔ Calculation:")
    print(f"  π (narrativity) = {π:.3f}")
    print(f"  |r| (correlation) = {abs_r:.4f}")
    print(f"  κ (judgment) = {κ:.2f}")
    print(f"  Δ = π × |r| × κ = {Δ:.4f}")
    print(f"  Efficiency = Δ/π = {efficiency:.4f}")
    
    # ========================================================================
    # STEP 7: PRESUME-AND-PROVE VALIDATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 7] PRESUME-AND-PROVE VALIDATION")
    print("="*80)
    
    threshold = 0.5
    passed = efficiency > threshold
    
    print(f"\nHYPOTHESIS: Narrative laws should apply to tennis")
    print(f"TEST: Δ/π > {threshold}")
    print(f"\nRESULT: Δ/π = {efficiency:.4f}")
    
    if passed:
        print(f"\n✓ VALIDATION PASSED")
        print(f"  Narrative influence is SIGNIFICANT")
        print(f"  Tennis is narrative-driven enough to show strong effects")
        print(f"  EXPECTED BETTING EDGE LIKELY")
    else:
        print(f"\n✗ VALIDATION FAILED")
        print(f"  Narrative influence weaker than expected")
        print(f"  Physical performance still dominates")
        print(f"  Similar to other sports (NFL, NBA)")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 8] Save Results")
    print("="*80)
    
    results = {
        'domain': 'Tennis',
        'hypothesis': 'Narrative laws should apply to tennis (individual sport, mental game)',
        'test': 'Δ/π > 0.5',
        'data': {
            'matches': len(matches),
            'years': '2000-2024',
            'player1_win_rate': float(outcomes.sum() / len(outcomes)),
            'surfaces': {
                'hard': sum(1 for m in matches if m['surface'] == 'hard'),
                'clay': sum(1 for m in matches if m['surface'] == 'clay'),
                'grass': sum(1 for m in matches if m['surface'] == 'grass')
            }
        },
        'narrativity': {
            'π': float(π),
            'components': {k: float(v) for k, v in domain_characteristics.items()},
            'classification': 'HIGH' if π > 0.7 else 'MEDIUM',
            'domain_type': 'INDIVIDUAL PERFORMANCE + MENTAL GAME'
        },
        'genome': {
            'transformers_applied': len(transformers),
            'total_features': int(ж.shape[1]),
            'feature_names_sample': all_feature_names[:100]
        },
        'story_quality': {
            'ю_mean': float(ю.mean()),
            'ю_std': float(ю.std()),
            'ю_min': float(ю.min()),
            'ю_max': float(ю.max())
        },
        'correlation': {
            'r': float(r),
            'abs_r': float(abs_r),
            'pattern': 'inverse' if r < 0 else 'positive',
            'interpretation': 'Better narratives favor underdogs' if r < 0 else 'Better narratives favor favorites'
        },
        'bridge': {
            'κ': float(κ),
            'Δ': float(Δ),
            'efficiency': float(efficiency),
            'threshold': threshold
        },
        'validation': {
            'passed': bool(passed),
            'result': 'PASS' if passed else 'FAIL',
            'expected': 'PASS or borderline (individual sport)',
            'explanation': 'Individual sport with mental game should show stronger narrative effects than team sports'
        }
    }
    
    # Save results
    output_path = Path(__file__).parent / 'tennis_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save genome and story quality for further analysis
    genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
    np.savez_compressed(
        genome_path,
        genome=ж,
        story_quality=ю,
        outcomes=outcomes,
        feature_names=all_feature_names
    )
    
    print(f"✓ Genome data saved to: {genome_path}")
    
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED")
    print("="*80)
    print(f"\nValidation: {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"Next steps:")
    print(f"  1. Surface-specific optimization (clay/grass/hard)")
    print(f"  2. Context discovery (Grand Slams, rivalries, players)")
    print(f"  3. Betting edge testing (TARGET: 3-8% ROI)")


if __name__ == '__main__':
    main()

