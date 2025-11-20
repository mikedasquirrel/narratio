"""
MLB Complete Analysis - Applying Narrative Framework

MEDIUM-HIGH NARRATIVITY: Team sport with individual performance elements

Comprehensive analysis following presume-and-prove methodology:
1. Calculate π (narrativity) - EXPECTED: 0.25-0.30 (MEDIUM-HIGH)
2. Apply ALL 33 transformers to extract ж (genome)
3. Compute ю (story quality)
4. Measure |r| (absolute correlation with outcomes)
5. Calculate Δ (bridge) and efficiency
6. Validate: Δ/π > 0.5 hypothesis
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
    """Complete MLB narrative analysis."""
    print("="*80)
    print("MLB NARRATIVE ANALYSIS - COMPLETE FRAMEWORK APPLICATION")
    print("="*80)
    print("\nMEDIUM-HIGH NARRATIVITY: Team sport with individual performance elements")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n[STEP 1] Loading MLB dataset...")
    
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'mlb_complete_dataset.json'
    
    if not dataset_path.exists():
        print(f"⚠ Dataset not found at {dataset_path}")
        print("  Running data collection first...")
        from collect_mlb_data import MLBDataCollector
        collector = MLBDataCollector(years=list(range(2015, 2025)))
        collector.collect_all_games(output_path=str(dataset_path))
    
    with open(dataset_path) as f:
        all_games = json.load(f)
    
    print(f"✓ Loaded {len(all_games)} games")
    
    # For transformer analysis, use strategic sample to avoid memory issues
    # Sample: Recent years (2018-2024) + Rivalry games + Playoff race games
    print("\nCreating strategic sample for analysis...")
    
    sampled_games = []
    for g in all_games:
        year = g['season']
        is_rivalry = g['context'].get('rivalry', False)
        is_playoff_race = g['context'].get('playoff_race', False)
        
        include = False
        
        if year >= 2018:  # Recent games
            if is_rivalry or is_playoff_race:  # High narrative games
                include = True
            elif year >= 2020:  # Very recent
                include = True
        
        if include:
            sampled_games.append(g)
        
        if len(sampled_games) >= 5000:  # Cap at 5K for speed
            break
    
    games = sampled_games
    print(f"✓ Using {len(games)} games for analysis (strategic sample)")
    print(f"  Rivalry games: {sum(1 for g in games if g['context'].get('rivalry', False))}")
    print(f"  Playoff race: {sum(1 for g in games if g['context'].get('playoff_race', False))}")
    print(f"  Years 2018+: {sum(1 for g in games if g['season'] >= 2018)}")
    print(f"  Full dataset remains available for context discovery")
    
    # Extract data - Build narratives and outcomes
    print("\nExtracting narratives and outcomes...")
    
    narratives = []
    outcomes = []
    
    for g in games:
        # Use narrative from game data
        narrative = g.get('narrative', '')
        if not narrative:
            # Generate narrative if missing
            home = g['home_team']
            away = g['away_team']
            narrative = f"The {away['nickname']} visit {home['city']} to face the {home['nickname']} at {g['venue']['name']}."
        
        narratives.append(narrative)
        
        # Outcome: 1 if home team won, 0 if away team won
        winner = g['outcome'].get('winner', '')
        if winner == 'home':
            outcomes.append(1)
        elif winner == 'away':
            outcomes.append(0)
        else:
            # Skip ties or incomplete games
            continue
    
    # Filter out any games without valid outcomes
    valid_indices = [i for i, o in enumerate(outcomes) if o in [0, 1]]
    narratives = [narratives[i] for i in valid_indices]
    outcomes = [outcomes[i] for i in valid_indices]
    games = [games[i] for i in valid_indices]
    
    outcomes = np.array(outcomes)
    
    print(f"  Home wins: {outcomes.sum()} ({100*outcomes.sum()/len(outcomes):.1f}%)")
    print(f"  Away wins: {len(outcomes) - outcomes.sum()} ({100*(1-outcomes.sum()/len(outcomes)):.1f}%)")
    
    # ========================================================================
    # STEP 2: CALCULATE π (NARRATIVITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 2] Calculate Narrativity (π)")
    print("="*80)
    
    print("\nHYPOTHESIS: Narrative laws should apply to MLB (team sport with individual elements)")
    print("TEST: Δ/π > 0.5")
    print("EXPECTATION: π ≈ 0.25-0.30 (MEDIUM-HIGH - more narrative than NBA/NFL)")
    
    domain_characteristics = {
        'π_structural': 0.20,  # Rules constrain but less than NBA/NFL
        'π_temporal': 0.30,    # Season arcs, playoff races matter
        'π_agency': 0.25,      # Individual performance matters (pitchers, hitters)
        'π_interpretation': 0.35,  # Narrative matters - "clutch", "momentum", "rivalries"
        'π_format': 0.15       # Game format rigid (9 innings)
    }
    
    π = np.mean(list(domain_characteristics.values()))
    
    print(f"\nπ Component Breakdown:")
    for component, value in domain_characteristics.items():
        print(f"  {component}: {value:.2f}")
    
    print(f"\n✓ Calculated π: {π:.3f}")
    print(f"  Classification: {'HIGH' if π > 0.7 else 'MEDIUM-HIGH' if π > 0.4 else 'MEDIUM' if π > 0.2 else 'LOW'} narrativity")
    print(f"  Domain Type: TEAM SPORT + INDIVIDUAL PERFORMANCE (rivalries, narratives matter)")
    
    # ========================================================================
    # STEP 3: APPLY ALL TRANSFORMERS → EXTRACT ж (GENOME)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 3] Apply ALL Transformers - Extract Genome (ж)")
    print("="*80)
    
    print("\nApplying 33 transformers (nominative + rivalry + narrative heavy)...")
    print("Expected: 600-800 features due to rich nominative content (team names, player names)")
    
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
    print(f"  Games: {ж.shape[0]}")
    print(f"  Feature space: {ж.nbytes / 1024 / 1024:.1f} MB")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 4: COMPUTE ю (STORY QUALITY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 4] Compute Story Quality (ю)")
    print("="*80)
    
    print("\nWeighting features based on π...")
    print(f"π = {π:.3f} → MEDIUM-HIGH weighting (narrative-rich domain)")
    
    # Normalize features
    scaler = StandardScaler()
    ж_normalized = scaler.fit_transform(ж)
    
    # Compute story quality with nominative/rivalry emphasis
    feature_weights = np.ones(ж.shape[1])
    
    # Boost nominative and rivalry features
    nominative_kw = ['nominative', 'phonetic', 'name', 'hierarchical', 'interaction', 'pure_nominative', 'universal_nominative']
    rivalry_kw = ['conflict', 'tension', 'emotional', 'cultural', 'authenticity']
    
    for i, fname in enumerate(all_feature_names[:ж.shape[1]]):
        fname_lower = fname.lower()
        if any(kw in fname_lower for kw in nominative_kw):
            feature_weights[i] *= 2.0  # Heavy nominative boost
        elif any(kw in fname_lower for kw in rivalry_kw):
            feature_weights[i] *= 1.5  # Rivalry boost
    
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
        print(f"\n  ⚠ INVERSE PATTERN (unexpected for MLB)")
        print(f"     Better narratives → Away team wins")
    else:
        print(f"\n  ✓ POSITIVE PATTERN (expected)")
        print(f"     Better narratives → Home team wins")
    
    # ========================================================================
    # STEP 6: CALCULATE Δ (BRIDGE) AND EFFICIENCY
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 6] Calculate Δ (Bridge) and Efficiency")
    print("="*80)
    
    # κ for team sport with narrative elements
    κ = 0.35  # Narrative is judged + interpreted (rivalries, momentum)
    
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
    
    print(f"\nHYPOTHESIS: Narrative laws should apply to MLB")
    print(f"TEST: Δ/π > {threshold}")
    print(f"\nRESULT: Δ/π = {efficiency:.4f}")
    
    if passed:
        print(f"\n✓ VALIDATION PASSED")
        print(f"  Narrative influence is SIGNIFICANT")
        print(f"  MLB is narrative-driven enough to show strong effects")
        print(f"  EXPECTED BETTING EDGE POSSIBLE")
    else:
        print(f"\n✗ VALIDATION FAILED")
        print(f"  Narrative influence weaker than expected")
        print(f"  Physical performance still dominates")
        print(f"  Similar to other team sports (NFL, NBA)")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 8] Save Results")
    print("="*80)
    
    results = {
        'domain': 'MLB',
        'hypothesis': 'Narrative laws should apply to MLB (team sport with individual elements, rivalries)',
        'test': 'Δ/π > 0.5',
        'data': {
            'games': len(games),
            'years': f"{min(g['season'] for g in games)}-{max(g['season'] for g in games)}",
            'home_win_rate': float(outcomes.sum() / len(outcomes)),
            'rivalry_games': sum(1 for g in games if g['context'].get('rivalry', False)),
            'playoff_race_games': sum(1 for g in games if g['context'].get('playoff_race', False))
        },
        'narrativity': {
            'π': float(π),
            'components': {k: float(v) for k, v in domain_characteristics.items()},
            'classification': 'MEDIUM-HIGH' if 0.2 < π < 0.4 else 'HIGH' if π > 0.7 else 'MEDIUM',
            'domain_type': 'TEAM SPORT + INDIVIDUAL PERFORMANCE + RIVALRIES'
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
            'interpretation': 'Better narratives favor away team' if r < 0 else 'Better narratives favor home team'
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
            'expected': 'FAIL or borderline (team sport, but higher π than NBA/NFL)',
            'explanation': 'Team sport with rivalries and narratives may show moderate narrative effects'
        }
    }
    
    # Save results
    output_path = Path(__file__).parent / 'mlb_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Save genome and story quality for further analysis
    genome_path = Path(__file__).parent / 'mlb_genome_data.npz'
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
    print(f"  1. Rivalry-specific optimization (Yankees-Red Sox, etc.)")
    print(f"  2. Context discovery (playoff race, stadium effects)")
    print(f"  3. Betting edge testing (if validation passed)")


if __name__ == '__main__':
    main()

