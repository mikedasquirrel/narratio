"""
WWE Three-Force Model Analysis

Re-processes WWE with complete three-force model:
- θ (Awareness Resistance) transformer
- λ (Fundamental Constraints) transformer  
- ة (Nominative Gravity) from GravitationalFeaturesTransformer

Tests prestige formula: Д = ة + θ - λ (awareness amplifies)

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

from src.transformers import (
    # Phase 7 - Three-force transformers
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    GravitationalFeaturesTransformer,
    # Core for genome
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    StatisticalTransformer
)
from src.analysis.bridge_calculator import BridgeCalculator


def main():
    """WWE three-force analysis"""
    print("="*80)
    print("WWE THREE-FORCE MODEL ANALYSIS")
    print("="*80)
    print("\nPrestige Domain: Д = ة + θ - λ (awareness amplifies)")
    print("Expected: High ة, High θ, Low λ → High Д")
    
    # Load WWE data
    data_path = project_root / 'narrative_optimization' / 'domains' / 'wwe' / 'data' / 'wwe_storylines.csv'
    
    if not data_path.exists():
        # Try alternative location
        alt_path = project_root / 'narrative_optimization' / 'domains' / 'wwe' / 'data' / 'wwe_storylines_with_features.csv'
        if alt_path.exists():
            data_path = alt_path
        else:
            print(f"\n❌ Data file not found")
            print(f"Tried: {data_path}")
            print(f"Tried: {alt_path}")
            print("WWE data exists but in different format")
            return
    
    df = pd.read_csv(data_path)
    print(f"\n✓ Loaded {len(df)} WWE storylines")
    
    # Extract narratives - build from available columns
    if 'narrative' in df.columns:
        narratives = df['narrative'].fillna('').tolist()
    elif 'participants' in df.columns and 'storyline_type' in df.columns:
        # Build narratives from participant names and storyline types
        narratives = []
        for _, row in df.iterrows():
            narrative = f"{row.get('storyline_type', '').replace('_', ' ')} storyline featuring {row.get('participants', '')}. "
            narrative += f"Character quality {row.get('character_quality', 0.5):.2f}, plot quality {row.get('plot_quality', 0.5):.2f}."
            narratives.append(narrative)
    else:
        print("❌ Cannot extract narratives from available columns")
        return
    
    # Extract outcomes
    if 'success' in df.columns:
        outcomes = df['success'].values
    elif 'engagement' in df.columns:
        # Use engagement as proxy (above median = success)
        median_engagement = df['engagement'].median()
        outcomes = (df['engagement'] > median_engagement).astype(int).values
    else:
        print("❌ Cannot extract outcomes")
        return
    
    # Apply transformers
    print("\n" + "-"*80)
    print("Extracting Three Forces")
    print("-"*80)
    
    # θ (Awareness Resistance)
    print("\n[1/3] Extracting θ (Awareness Resistance)...")
    theta_transformer = AwarenessResistanceTransformer()
    theta_features = theta_transformer.fit_transform(narratives)
    theta_values = theta_features[:, 14]  # θ score
    
    print(f"✓ θ extracted - Mean: {theta_values.mean():.3f}, Std: {theta_values.std():.3f}")
    print(f"  Expected: High (WWE fans are meta-aware)")
    
    # λ (Fundamental Constraints)
    print("\n[2/3] Extracting λ (Fundamental Constraints)...")
    lambda_transformer = FundamentalConstraintsTransformer()
    lambda_features = lambda_transformer.fit_transform(narratives)
    lambda_values = lambda_features[:, 11]  # λ score
    
    print(f"✓ λ extracted - Mean: {lambda_values.mean():.3f}, Std: {lambda_values.std():.3f}")
    print(f"  Expected: Low (outcomes are scripted, no physical determination)")
    
    # ة (Nominative Gravity) - need genome first
    print("\n[3/3] Extracting ة (Nominative Gravity)...")
    print("  Computing genome (ж) for gravitational calculations...")
    
    # Extract basic genome
    nom_transformer = NominativeAnalysisTransformer()
    self_transformer = SelfPerceptionTransformer()
    stat_transformer = StatisticalTransformer(max_features=50)
    
    nom_features = nom_transformer.fit_transform(narratives)
    self_features = self_transformer.fit_transform(narratives)
    
    try:
        stat_features = stat_transformer.fit_transform(narratives)
        # Ensure all are 2D
        if stat_features.ndim == 1:
            stat_features = stat_features.reshape(-1, 1)
        if nom_features.ndim == 1:
            nom_features = nom_features.reshape(-1, 1)
        if self_features.ndim == 1:
            self_features = self_features.reshape(-1, 1)
        
        genome = np.hstack([nom_features, self_features, stat_features])
    except:
        # Skip statistical if it causes issues, use nom + self only
        print("  ⚠️  Statistical features skipped")
        if nom_features.ndim == 1:
            nom_features = nom_features.reshape(-1, 1)
        if self_features.ndim == 1:
            self_features = self_features.reshape(-1, 1)
        genome = np.hstack([nom_features, self_features])
    
    print(f"  ✓ Genome: {genome.shape}")
    
    # Extract gravitational features
    grav_transformer = GravitationalFeaturesTransformer()
    grav_features = grav_transformer.fit_transform(genome, y=outcomes)
    ta_values = grav_features[:, 12]  # ة (nominative gravity)
    
    print(f"✓ ة extracted - Mean: {ta_values.mean():.3f}, Std: {ta_values.std():.3f}")
    print(f"  Expected: Very High (names and personas are everything in WWE)")
    
    # Calculate Д using three-force model
    print("\n" + "="*80)
    print("CALCULATING Д (THREE-FORCE MODEL)")
    print("="*80)
    
    calc = BridgeCalculator()
    
    # Compute story quality (simple mean of genome for now)
    scaler = StandardScaler()
    genome_scaled = scaler.fit_transform(genome)
    story_quality = np.mean(genome_scaled, axis=1)
    story_quality = (story_quality - story_quality.min()) / (story_quality.max() - story_quality.min())
    
    # Use three-force formula
    results = calc.calculate_D(
        story_quality=story_quality,
        outcomes=outcomes,
        nominative_gravity=ta_values,
        awareness_resistance=theta_values,
        fundamental_constraints=lambda_values,
        is_prestige=True  # WWE is prestige domain
    )
    
    print(f"\nFormula: {results['equation']}")
    print(f"\nForce Values:")
    print(f"  ة (Nominative Gravity):     {results['nominative_gravity_mean']:.3f}")
    print(f"  θ (Awareness Resistance):    {results['awareness_resistance_mean']:.3f}")
    print(f"  λ (Fundamental Constraints): {results['fundamental_constraints_mean']:.3f}")
    print(f"\nDominant Force: {results['dominant_force']}")
    print(f"\nД = {results['Д']:.3f}")
    print(f"Passes Threshold (Д > 0.10): {results['passes_threshold']}")
    print(f"\nInterpretation: {results['interpretation']}")
    
    # Compare with expected
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Expected from framework: WWE π=0.974, Д=1.80
    expected_D = 1.80
    observed_D = results['Д']
    error = abs(expected_D - observed_D)
    
    print(f"\nExpected Д: {expected_D:.3f}")
    print(f"Observed Д: {observed_D:.3f}")
    print(f"Error: {error:.3f}")
    
    if error < 0.5:
        print("\n✓ Three-force model validates WWE as prestige domain")
    else:
        print("\n⚠ Large discrepancy - may need calibration")
    
    # Save results
    output = {
        'domain': 'wwe',
        'narrativity': 0.974,
        'three_force_model': {
            'nominative_gravity_mean': float(results['nominative_gravity_mean']),
            'awareness_resistance_mean': float(results['awareness_resistance_mean']),
            'fundamental_constraints_mean': float(results['fundamental_constraints_mean']),
            'dominant_force': results['dominant_force'],
            'is_prestige': True,
            'equation': results['equation']
        },
        'bridge': float(results['Д']),
        'expected_bridge': expected_D,
        'model_error': float(error),
        'passes_threshold': results['passes_threshold'],
        'interpretation': results['interpretation']
    }
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'wwe' / 'three_force_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_path}")
    print("\n" + "="*80)
    print("✓ WWE THREE-FORCE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

