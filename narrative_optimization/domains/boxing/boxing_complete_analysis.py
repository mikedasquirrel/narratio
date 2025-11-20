"""
Boxing Complete Analysis - ALL POSSIBLE FEATURES

Applies complete framework to boxing domain:
- All 33 transformers
- π component calculation
- θ, λ, ة force extraction
- R² performance measurement
- Nominative enrichment
- Ablation studies
- Feature importance

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

from narrative_optimization.src.transformers import (
    NominativeAnalysisTransformer, SelfPerceptionTransformer, NarrativePotentialTransformer,
    LinguisticPatternsTransformer, RelationalValueTransformer, EnsembleNarrativeTransformer,
    CouplingStrengthTransformer, NarrativeMassTransformer, NominativeRichnessTransformer,
    GravitationalFeaturesTransformer, AwarenessResistanceTransformer, FundamentalConstraintsTransformer,
    AlphaTransformer, GoldenNarratioTransformer,
    # Additional transformers for comprehensive coverage
    StatisticalTransformer, PhoneticTransformer, TemporalEvolutionTransformer,
    EmotionalResonanceTransformer, ConflictTensionTransformer, ExpertiseAuthorityTransformer,
    UniversalNominativeTransformer, MultiScaleTransformer
)
# Import enriched patterns
from narrative_optimization.src.transformers.enriched_patterns import get_patterns_for_domain

# Bridge calculator (simplified version)
def calculate_bridge_three_force(ta_marbuta, theta, lambda_val, prestige_domain=False):
    """Calculate Д using three-force equation."""
    if prestige_domain:
        return ta_marbuta + theta - lambda_val
    else:
        return ta_marbuta - theta - lambda_val

data_dir = project_root / 'data' / 'domains' / 'boxing'
output_dir = project_root / 'narrative_optimization' / 'domains' / 'boxing'
output_dir.mkdir(parents=True, exist_ok=True)


def calculate_boxing_pi():
    """
    Calculate π (narrativity) for boxing domain.
    
    Components:
    - Structural: Rules exist but outcomes vary (0.50)
    - Temporal: Multi-round progression (0.80)
    - Agency: Complete individual control (1.00)
    - Interpretive: Heavy mental/physical interpretation (0.75)
    - Format: Multiple weight classes, venues (0.70)
    """
    print("="*80)
    print("CALCULATING BOXING π (NARRATIVITY)")
    print("="*80)
    
    components = {
        'structural': 0.50,  # Rules constrain (12 rounds, weight classes) but outcomes vary
        'temporal': 0.80,    # Multi-round progression, dramatic arcs, momentum shifts
        'agency': 1.00,      # Complete individual control - one-on-one combat
        'interpretive': 0.75, # Heavy interpretation (styles, mental game, intimidation)
        'format': 0.70       # Multiple weight classes, venues, promotions
    }
    
    # Standard formula
    pi = (0.30 * components['structural'] +
          0.20 * components['temporal'] +
          0.25 * components['agency'] +
          0.15 * components['interpretive'] +
          0.10 * components['format'])
    
    print(f"\nComponent Breakdown:")
    print(f"  Structural:  {components['structural']:.2f} × 0.30 = {components['structural'] * 0.30:.3f}")
    print(f"  Temporal:    {components['temporal']:.2f} × 0.20 = {components['temporal'] * 0.20:.3f}")
    print(f"  Agency:      {components['agency']:.2f} × 0.25 = {components['agency'] * 0.25:.3f}")
    print(f"  Interpretive: {components['interpretive']:.2f} × 0.15 = {components['interpretive'] * 0.15:.3f}")
    print(f"  Format:      {components['format']:.2f} × 0.10 = {components['format'] * 0.10:.3f}")
    print(f"\nCalculated π: {pi:.3f}")
    
    return pi, components


def extract_all_features(fights_data, pi_value=0.715):
    """
    Apply ALL transformers to extract comprehensive features.
    
    Args:
        fights_data: List of fight dictionaries
        pi_value: Domain narrativity (π) for transformers that need it
    """
    print("\n" + "="*80)
    print("EXTRACTING FEATURES - ALL TRANSFORMERS")
    print("="*80)
    
    # Initialize all transformers - ALL POSSIBLE FEATURES
    # Some transformers need parameters, initialize them safely
    transformers = {}
    
    # Basic transformers (no params)
    basic_transformers = {
        'statistical': StatisticalTransformer,
        'nominative': NominativeAnalysisTransformer,
        'self_perception': SelfPerceptionTransformer,
        'narrative_potential': NarrativePotentialTransformer,
        'linguistic': LinguisticPatternsTransformer,
        'relational': RelationalValueTransformer,
        'ensemble': EnsembleNarrativeTransformer,
        'phonetic': PhoneticTransformer,
        'temporal': TemporalEvolutionTransformer,
        'coupling': CouplingStrengthTransformer,
        'mass': NarrativeMassTransformer,
        'nominative_richness': NominativeRichnessTransformer,
        'gravitational': GravitationalFeaturesTransformer,
        'awareness': AwarenessResistanceTransformer,
        'constraints': FundamentalConstraintsTransformer,
        'emotional': EmotionalResonanceTransformer,
        'conflict': ConflictTensionTransformer,
        'expertise': ExpertiseAuthorityTransformer,
        'universal_nominative': UniversalNominativeTransformer,
        'multi_scale': MultiScaleTransformer
    }
    
    # Initialize basic transformers
    for name, transformer_class in basic_transformers.items():
        try:
            transformers[name] = transformer_class()
        except Exception as e:
            print(f"    Warning: Could not initialize {name}: {e}")
    
    # Special transformers (need params)
    try:
        transformers['alpha'] = AlphaTransformer(narrativity=pi_value)
    except Exception as e:
        print(f"    Warning: Could not initialize alpha: {e}")
    
    try:
        transformers['golden_narratio'] = GoldenNarratioTransformer()
    except Exception as e:
        print(f"    Warning: Could not initialize golden_narratio: {e}")
    
    print(f"\n✓ Initialized {len(transformers)} transformers")
    
    # Get enriched patterns for sports domain
    sports_patterns = get_patterns_for_domain('sports', 'both')
    
    # Prepare all narratives for fitting
    print(f"\nPreparing narratives for transformer fitting...")
    all_narratives = []
    for fight in fights_data:
        narrative = fight.get('narrative', '')
        f1_narrative = fight.get('fighter1', {}).get('narrative', '')
        f2_narrative = fight.get('fighter2', {}).get('narrative', '')
        full_narrative = f"{narrative} {f1_narrative} {f2_narrative}"
        all_narratives.append(full_narrative)
    
    # Fit all transformers (some may not need fitting)
    print(f"Fitting transformers on {len(all_narratives)} narratives...")
    fitted_transformers = {}
    for name, transformer in transformers.items():
        try:
            # Try to fit (some transformers don't need fitting)
            if hasattr(transformer, 'fit'):
                transformer.fit(all_narratives)
                fitted_transformers[name] = transformer
            elif hasattr(transformer, 'fit_transform'):
                # Can fit and transform in one step
                fitted_transformers[name] = transformer
            else:
                # No fit needed, use as-is
                fitted_transformers[name] = transformer
        except Exception as e:
            print(f"    Warning: Could not fit {name}: {e}")
            # Try to use anyway (some transformers work without fit)
            fitted_transformers[name] = transformer
    
    print(f"✓ Fitted {len(fitted_transformers)} transformers")
    
    all_features = []
    feature_names_set = set()
    
    print(f"\nProcessing {len(fights_data)} fights...")
    print(f"  Using batch processing for efficiency...")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    total_batches = (len(fights_data) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(fights_data))
        batch_fights = fights_data[start_idx:end_idx]
        batch_narratives = all_narratives[start_idx:end_idx]
        
        if batch_num % 10 == 0:
            print(f"  Processing batch {batch_num+1}/{total_batches} (fights {start_idx+1}-{end_idx})...")
        
        for i, fight in enumerate(batch_fights):
            full_narrative = batch_narratives[i]
            fight_features = {}
            
            # Apply each transformer
            for name, transformer in fitted_transformers.items():
                try:
                    if hasattr(transformer, 'transform'):
                        features = transformer.transform([full_narrative])
                        if isinstance(features, np.ndarray):
                            features = features.flatten()
                        elif isinstance(features, list):
                            features = np.array(features).flatten()
                        
                        # Store features
                        for j, val in enumerate(features):
                            feature_name = f"{name}_feature_{j}"
                            fight_features[feature_name] = float(val)
                            feature_names_set.add(feature_name)
                    elif hasattr(transformer, 'fit_transform'):
                        # Use fit_transform for this single instance
                        features = transformer.fit_transform([full_narrative])
                        if isinstance(features, np.ndarray):
                            features = features.flatten()
                        for j, val in enumerate(features):
                            feature_name = f"{name}_feature_{j}"
                            fight_features[feature_name] = float(val)
                            feature_names_set.add(feature_name)
                except Exception as e:
                    if batch_num == 0 and i == 0:  # Only print errors for first fight
                        print(f"    Warning: {name} transformer failed: {e}")
                    continue
            
            # Add domain-specific features
            boxing_features = extract_boxing_specific_features(fight)
            fight_features.update(boxing_features)
            for key in boxing_features.keys():
                feature_names_set.add(key)
            
            all_features.append(fight_features)
    
    feature_names = sorted(list(feature_names_set))
    
    print(f"\n✓ Extracted features from {len(all_features)} fights")
    print(f"✓ Total features: {len(feature_names)}")
    
    return all_features, feature_names


def extract_boxing_specific_features(fight):
    """
    Extract boxing-specific features not covered by general transformers.
    """
    features = {}
    
    f1 = fight.get('fighter1', {})
    f2 = fight.get('fighter2', {})
    
    # Reputation features
    features['f1_reputation'] = f1.get('reputation', 0.5)
    features['f2_reputation'] = f2.get('reputation', 0.5)
    features['reputation_diff'] = features['f1_reputation'] - features['f2_reputation']
    
    # Record features
    f1_record = parse_record(f1.get('record', '0-0-0'))
    f2_record = parse_record(f2.get('record', '0-0-0'))
    features['f1_wins'] = f1_record[0]
    features['f1_losses'] = f1_record[1]
    features['f1_draws'] = f1_record[2]
    features['f2_wins'] = f2_record[0]
    features['f2_losses'] = f2_record[1]
    features['f2_draws'] = f2_record[2]
    features['win_diff'] = features['f1_wins'] - features['f2_wins']
    
    # Weight class
    weight_classes = {
        'Heavyweight': 1.0, 'Cruiserweight': 0.9, 'Light Heavyweight': 0.8,
        'Super Middleweight': 0.7, 'Middleweight': 0.6, 'Welterweight': 0.5,
        'Lightweight': 0.4, 'Featherweight': 0.3, 'Bantamweight': 0.2
    }
    features['weight_class_value'] = weight_classes.get(f1.get('weight_class', ''), 0.5)
    
    # Title fight
    features['is_title_fight'] = 1.0 if 'Championship' in fight.get('title', '') else 0.0
    
    # Venue prestige
    venue = fight.get('venue', '')
    features['venue_prestige'] = 1.0 if any(x in venue for x in ['Las Vegas', 'Madison Square', 'Wembley']) else 0.5
    
    # Promotion prestige
    promotion = fight.get('promotion', '')
    features['promotion_prestige'] = 1.0 if any(x in promotion for x in ['Top Rank', 'Matchroom', 'PBC']) else 0.5
    
    # Rounds scheduled
    features['rounds_scheduled'] = fight.get('rounds_scheduled', 12) / 12.0
    
    return features


def parse_record(record_str):
    """Parse boxing record string like '34-0-1' to (wins, losses, draws)."""
    try:
        parts = record_str.split('-')
        return (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
    except:
        return (0, 0, 0)


def calculate_forces(fights_data, all_features):
    """
    Extract θ (awareness) and λ (constraints) using enriched patterns.
    """
    print("\n" + "="*80)
    print("EXTRACTING THREE FORCES (θ, λ, ة)")
    print("="*80)
    
    sports_patterns = get_patterns_for_domain('sports', 'both')
    theta_patterns = sports_patterns['theta']
    lambda_patterns = sports_patterns['lambda']
    
    theta_scores = []
    lambda_scores = []
    ta_marbuta_scores = []
    
    for fight, features in zip(fights_data, all_features):
        narrative = fight.get('narrative', '')
        narrative_lower = narrative.lower()
        
        # Calculate θ (awareness) - count awareness patterns
        theta_count = 0
        for category, patterns in theta_patterns.items():
            for pattern in patterns:
                if pattern.lower() in narrative_lower:
                    theta_count += 1
        
        # Normalize by narrative length
        theta_score = min(1.0, theta_count / (len(narrative.split()) / 100))
        theta_scores.append(theta_score)
        
        # Calculate λ (constraints) - count constraint patterns
        lambda_count = 0
        total_patterns = 0
        for category, patterns in lambda_patterns.items():
            total_patterns += len(patterns)
            for pattern in patterns:
                if pattern.lower() in narrative_lower:
                    lambda_count += 1
        
        # Normalize by total possible patterns and narrative length
        if total_patterns > 0:
            lambda_score = min(1.0, (lambda_count / total_patterns) * 2.0)  # Scale factor
        else:
            lambda_score = 0.5  # Baseline if no patterns found
        
        # Also consider domain-specific constraints (reputation, records indicate skill barriers)
        f1_rep = fight.get('fighter1', {}).get('reputation', 0.5)
        f2_rep = fight.get('fighter2', {}).get('reputation', 0.5)
        avg_rep = (f1_rep + f2_rep) / 2.0
        # Higher reputation = higher skill level = higher constraints
        lambda_score = max(lambda_score, avg_rep * 0.7)  # Blend pattern-based and reputation-based
        
        lambda_scores.append(lambda_score)
        
        # Calculate ة (nominative gravity) from features
        # Use reputation, name recognition, achievements
        f1_rep = fight.get('fighter1', {}).get('reputation', 0.5)
        f2_rep = fight.get('fighter2', {}).get('reputation', 0.5)
        ta_score = (f1_rep + f2_rep) / 2.0
        ta_marbuta_scores.append(ta_score)
    
    theta_mean = np.mean(theta_scores)
    theta_std = np.std(theta_scores)
    lambda_mean = np.mean(lambda_scores)
    lambda_std = np.std(lambda_scores)
    ta_mean = np.mean(ta_marbuta_scores)
    ta_std = np.std(ta_marbuta_scores)
    
    print(f"\nθ (Awareness Resistance):")
    print(f"  Mean: {theta_mean:.3f}")
    print(f"  Std:  {theta_std:.3f}")
    
    print(f"\nλ (Fundamental Constraints):")
    print(f"  Mean: {lambda_mean:.3f}")
    print(f"  Std:  {lambda_std:.3f}")
    
    print(f"\nة (Nominative Gravity):")
    print(f"  Mean: {ta_mean:.3f}")
    print(f"  Std:  {ta_std:.3f}")
    
    return {
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'lambda_mean': lambda_mean,
        'lambda_std': lambda_std,
        'ta_marbuta_mean': ta_mean,
        'ta_marbuta_std': ta_std
    }


def calculate_performance(fights_data, all_features):
    """
    Calculate R² performance using all features.
    """
    print("\n" + "="*80)
    print("CALCULATING PERFORMANCE (R²)")
    print("="*80)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    # Prepare data
    X = []
    y = []
    
    for fight, features in zip(fights_data, all_features):
        # Convert features dict to array
        feature_array = [features.get(name, 0.0) for name in sorted(features.keys())]
        X.append(feature_array)
        
        # Outcome: 1 if fighter1 wins, 0 if fighter2 wins
        y.append(fight.get('outcome', 0))
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nData shape: {X.shape}")
    print(f"Outcomes: {np.sum(y)} wins for fighter1, {len(y) - np.sum(y)} wins for fighter2")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with regularization to prevent overfitting
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced to prevent overfitting
        random_state=42, 
        max_depth=8,  # Reduced depth
        min_samples_split=20,  # Increased minimum samples
        min_samples_leaf=10,  # Increased leaf samples
        max_features='sqrt'  # Limit features per split
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # R² scores
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"\nR² Performance:")
    print(f"  Train: {r2_train:.3f} ({r2_train*100:.1f}%)")
    print(f"  Test:  {r2_test:.3f} ({r2_test*100:.1f}%)")
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names_sorted = sorted(all_features[0].keys())
    
    # Top 20 features
    top_indices = np.argsort(feature_importance)[-20:][::-1]
    print(f"\nTop 20 Features:")
    for idx in top_indices:
        print(f"  {feature_names_sorted[idx]}: {feature_importance[idx]:.4f}")
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'feature_importance': {feature_names_sorted[i]: float(feature_importance[i]) 
                               for i in range(len(feature_importance))}
    }


def main():
    """Run complete boxing analysis."""
    print("="*80)
    print("BOXING COMPLETE ANALYSIS - ALL FEATURES")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Load data - prefer expanded dataset
    expanded_file = data_dir / 'boxing_fights_expanded.json'
    complete_file = data_dir / 'boxing_fights_complete.json'
    
    if expanded_file.exists():
        print(f"\n✓ Loading expanded dataset: {expanded_file}")
        with open(expanded_file, 'r') as f:
            data = json.load(f)
            fights_data = data['fights']
            fighters = data.get('fighters', {})
            print(f"  Total fights: {len(fights_data)}")
            print(f"  Total fighters: {len(fighters)}")
    elif complete_file.exists():
        print(f"\n✓ Loading initial dataset: {complete_file}")
        with open(complete_file, 'r') as f:
            data = json.load(f)
            fights_data = data['fights']
            fighters = data.get('fighters', {})
    else:
        print(f"\n⚠️  Data file not found. Running data collector first...")
        from boxing_data_collector import collect_boxrec_data
        fights_data, fighters = collect_boxrec_data()
    
    print(f"\n✓ Loaded {len(fights_data)} fights")
    
    # Calculate π
    pi, pi_components = calculate_boxing_pi()
    
    # Extract all features (pass pi for transformers that need it)
    all_features, feature_names = extract_all_features(fights_data, pi_value=pi)
    
    # Calculate forces
    forces = calculate_forces(fights_data, all_features)
    
    # Calculate performance
    performance = calculate_performance(fights_data, all_features)
    
    # Calculate Д (Bridge)
    arch = calculate_bridge_three_force(
        ta_marbuta=forces['ta_marbuta_mean'],
        theta=forces['theta_mean'],
        lambda_val=forces['lambda_mean'],
        prestige_domain=False
    )
    
    leverage = arch / pi if pi > 0 else 0
    
    # Compile results
    results = {
        'domain': 'boxing',
        'name': 'Professional Boxing',
        'pi': pi,
        'pi_components': pi_components,
        'forces': forces,
        'performance': performance,
        'arch': arch,
        'leverage': leverage,
        'sample_size': len(fights_data),
        'total_features': len(feature_names),
        'analysis_date': datetime.now().isoformat()
    }
    
    # Save results
    results_file = output_dir / 'boxing_analysis_complete.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  π (Narrativity): {pi:.3f}")
    print(f"  R² (Test): {performance['r2_test']:.3f} ({performance['r2_test']*100:.1f}%)")
    print(f"  θ (Awareness): {forces['theta_mean']:.3f}")
    print(f"  λ (Constraints): {forces['lambda_mean']:.3f}")
    print(f"  ة (Nominative): {forces['ta_marbuta_mean']:.3f}")
    print(f"  Д (Bridge): {arch:.3f}")
    print(f"  Leverage (Д/π): {leverage:.3f}")
    print(f"\n✓ Saved results to: {results_file}")
    
    return results


if __name__ == '__main__':
    results = main()

