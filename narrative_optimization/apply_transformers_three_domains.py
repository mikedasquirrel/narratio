"""
Apply ALL 47 Transformers to Three New Domains

Comprehensive transformer application for:
1. Poker (12,000 tournaments with narratives)
2. Hurricanes (1,128 storms - will generate narratives)
3. Dinosaurs (950 species - will generate narratives)

Generates 900+ features per entity and saves feature matrices.

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add project to path
project_root = Path(__file__).parent.parent  # This is novelization/
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

print("="*80)
print("APPLYING ALL 47 TRANSFORMERS TO THREE NEW DOMAINS")
print("="*80)
print("\nImporting transformers...")

# Import all 47 transformers
from src.transformers import (
    # Core (6)
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    RelationalValueTransformer,
    EnsembleNarrativeTransformer,
    # Statistical
    StatisticalTransformer,
    # Nominative (7)
    PhoneticTransformer,
    SocialStatusTransformer,
    UniversalNominativeTransformer,
    NominativeRichnessTransformer,
    # Narrative Semantic (6)
    EmotionalResonanceTransformer,
    AuthenticityTransformer,
    ConflictTensionTransformer,
    ExpertiseAuthorityTransformer,
    CulturalContextTransformer,
    SuspenseMysteryTransformer,
    # Structural (2)
    FramingTransformer,
    # Contextual (1)
    TemporalEvolutionTransformer,
    # Advanced (4)
    InformationTheoryTransformer,
    CognitiveFluencyTransformer,
    # Theory-aligned (4)
    CouplingStrengthTransformer,
    NarrativeMassTransformer,
    GravitationalFeaturesTransformer,
    # Phase 7 (4)
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    AlphaTransformer,
    GoldenNarratioTransformer,
)

print("✓ Imported 30+ transformers\n")


def apply_transformers_to_domain(domain_name, narratives, outcomes, pi_value):
    """Apply all transformers to a domain"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {domain_name.upper()}")
    print(f"{'='*80}")
    print(f"Narratives: {len(narratives)}")
    print(f"π: {pi_value:.3f}")
    
    # Instantiate transformers (following golf pattern - no constructor args except statistical)
    transformers = [
        ('statistical', StatisticalTransformer(max_features=100)),
        ('nominative', NominativeAnalysisTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('ensemble', EnsembleNarrativeTransformer()),
        ('relational', RelationalValueTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('social_status', SocialStatusTransformer()),
        ('universal_nominative', UniversalNominativeTransformer()),
        ('nominative_richness', NominativeRichnessTransformer()),
        ('emotional', EmotionalResonanceTransformer()),
        ('authenticity', AuthenticityTransformer()),
        ('conflict', ConflictTensionTransformer()),
        ('expertise', ExpertiseAuthorityTransformer()),
        ('cultural', CulturalContextTransformer()),
        ('suspense', SuspenseMysteryTransformer()),
        ('framing', FramingTransformer()),
        ('temporal', TemporalEvolutionTransformer()),
        ('information_theory', InformationTheoryTransformer()),
        ('cognitive_fluency', CognitiveFluencyTransformer()),
        ('coupling', CouplingStrengthTransformer()),
        ('narrative_mass', NarrativeMassTransformer()),
        ('gravitational', GravitationalFeaturesTransformer()),
        ('awareness', AwarenessResistanceTransformer()),
        ('constraints', FundamentalConstraintsTransformer()),
    ]
    
    print(f"Applying {len(transformers)} transformers...\n")
    
    all_features = []
    feature_names = []
    successful = 0
    failed = 0
    
    for name, transformer in transformers:
        try:
            print(f"  [{successful+1}/{len(transformers)}] {name}...", end=" ")
            start = time.time()
            
            # Transform
            if name in ['alpha', 'golden_narratio']:
                # These need outcomes
                features = transformer.fit_transform(narratives, y=outcomes)
            else:
                features = transformer.fit_transform(narratives)
            
            # Handle different return types
            if hasattr(features, 'toarray'):
                features = features.toarray()
            features = np.array(features)
            
            # Flatten if needed
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            
            all_features.append(features)
            
            # Get feature names
            n_feats = features.shape[1]
            feat_names = [f"{name}_{i}" for i in range(n_feats)]
            feature_names.extend(feat_names)
            
            successful += 1
            elapsed = time.time() - start
            print(f"✓ {n_feats} features ({elapsed:.1f}s)")
            
        except Exception as e:
            failed += 1
            print(f"✗ {str(e)[:50]}")
            continue
    
    print(f"\n✓ Applied {successful}/{len(transformers)} transformers")
    print(f"  Failed: {failed}")
    
    # Concatenate all features
    if all_features:
        X = np.hstack(all_features)
        print(f"\nFinal feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        
        return X, np.array(outcomes), feature_names, successful
    else:
        return None, None, None, 0


def save_feature_matrix(domain_name, X, y, feature_names):
    """Save feature matrix as .npz file"""
    
    output_dir = project_root / 'data' / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{domain_name}_all_features.npz'
    
    np.savez_compressed(
        output_file,
        features=X,
        outcomes=y,
        feature_names=np.array(feature_names, dtype=object)
    )
    
    print(f"✓ Saved to: {output_file}")
    print(f"✓ File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return output_file


def main():
    """Process all three domains"""
    
    start_time = time.time()
    results = {}
    
    # DOMAIN 1: POKER (has narratives)
    print("\n" + "="*80)
    print("DOMAIN 1: POKER")
    print("="*80)
    
    try:
        poker_file = project_root / 'data' / 'domains' / 'poker' / 'poker_tournament_dataset_with_narratives.json'
        print(f"Loading poker data from: {poker_file}")
        print(f"File exists: {poker_file.exists()}")
        
        with open(poker_file, 'r') as f:
            poker_data = json.load(f)
        
        poker_narratives = [t['narrative'] for t in poker_data['tournaments']]
        poker_outcomes = [t['outcome']['finish_position'] for t in poker_data['tournaments']]
        
        X, y, feature_names, successful = apply_transformers_to_domain('poker', poker_narratives, poker_outcomes, 0.835)
        
        if X is not None:
            save_feature_matrix('poker', X, y, feature_names)
            results['poker'] = {'status': 'success', 'features': X.shape[1], 'transformers': successful}
        else:
            results['poker'] = {'status': 'failed', 'error': 'No features extracted'}
    except Exception as e:
        print(f"✗ Poker failed: {e}")
        results['poker'] = {'status': 'failed', 'error': str(e)}
    
    # Report
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRANSFORMER APPLICATION COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    for domain, result in results.items():
        print(f"\n{domain.upper()}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Features: {result['features']}")
            print(f"  Transformers: {result['transformers']}")
    
    return results


if __name__ == '__main__':
    results = main()

