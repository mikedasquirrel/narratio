"""
UFC Complete Reanalysis - ALL POSSIBLE FEATURES

Deep investigation of UFC performance gap:
- Expected: 80-90% R² (based on π=0.722, Agency=1.00, θ=0.535)
- Previous: 2.5% R² (AUC delta-based measure)
- Goal: Calculate proper R² and understand the gap

Enhancements over previous analysis:
1. UFC-specific θ/λ patterns (combat sports awareness & constraints)
2. Deep phonetic analysis of fighter names and nicknames
3. UFC-specific features (finish types, styles, camps)
4. Proper R² calculation (not AUC delta)
5. Variance/randomness metrics
6. Context analysis (finish types, title fights)

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

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
    StatisticalTransformer, PhoneticTransformer, TemporalEvolutionTransformer,
    EmotionalResonanceTransformer, ConflictTensionTransformer, ExpertiseAuthorityTransformer,
    UniversalNominativeTransformer, MultiScaleTransformer
)

# Import enriched patterns (now includes UFC-specific combat sports patterns)
from narrative_optimization.src.transformers.enriched_patterns import get_patterns_for_domain

# Setup directories
data_dir = project_root / 'data' / 'domains'
output_dir = project_root / 'narrative_optimization' / 'domains' / 'ufc'
output_dir.mkdir(parents=True, exist_ok=True)


def calculate_bridge_three_force(ta_marbuta, theta, lambda_val, prestige_domain=False):
    """Calculate Д using three-force equation."""
    if prestige_domain:
        return ta_marbuta + theta - lambda_val
    else:
        return ta_marbuta - theta - lambda_val


def calculate_ufc_pi():
    """
    Calculate π (narrativity) for UFC domain.
    
    Components derived from prior analysis:
    - Structural: Multiple outcomes (KO, submission, decision) - 0.50
    - Temporal: Multi-round progression, finish timing - 0.85
    - Agency: Complete individual control - 1.00
    - Interpretive: Heavy style/matchup interpretation - 0.80
    - Format: Weight classes, venues, title/non-title - 0.75
    
    Previous π = 0.722
    """
    print("="*80)
    print("CALCULATING UFC π (NARRATIVITY)")
    print("="*80)
    
    components = {
        'structural': 0.50,  # Multiple paths to victory, but rules constrain
        'temporal': 0.85,    # Multi-round dramatic arcs, finish timing crucial
        'agency': 1.00,      # Perfect individual control - one-on-one combat
        'interpretive': 0.80, # Heavy interpretation (styles, matchups, fight IQ)
        'format': 0.75       # Multiple weight classes, venues, contexts
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
    print(f"(Previous analysis: 0.722)")
    
    return pi, components


def extract_ufc_specific_features(fight):
    """
    Extract UFC-specific features not covered by general transformers.
    
    Based on previous context discovery showing:
    - Finish types matter (submissions, KO/TKO)
    - Title fights have different dynamics
    - Early finishes show narrative effects
    """
    features = {}
    
    # Finish type features (critical for UFC)
    method = fight.get('method', '').lower()
    features['is_ko_tko'] = 1.0 if 'ko' in method or 'tko' in method else 0.0
    features['is_submission'] = 1.0 if 'submission' in method or 'sub' in method else 0.0
    features['is_decision'] = 1.0 if 'decision' in method or 'dec' in method else 0.0
    
    # Round features (early finishes matter)
    last_round = fight.get('last_round', 3)
    total_rounds = fight.get('total_rounds', 3)
    features['round_finished'] = last_round
    features['is_early_finish'] = 1.0 if last_round <= 1 else 0.0
    features['finish_percentage'] = last_round / total_rounds if total_rounds > 0 else 0.5
    
    # Title fight features (shown to have higher efficiency)
    features['is_title_fight'] = 1.0 if fight.get('title_bout', False) else 0.0
    features['is_main_event'] = 1.0 if fight.get('main_event', False) else 0.0
    
    # Fighter record/reputation features
    r_fighter = fight.get('R_fighter', {})
    b_fighter = fight.get('B_fighter', {})
    
    # Win streaks (momentum)
    features['r_win_streak'] = r_fighter.get('win_streak', 0)
    features['b_win_streak'] = b_fighter.get('win_streak', 0)
    features['win_streak_diff'] = features['r_win_streak'] - features['b_win_streak']
    
    # Finish rates (from previous analysis)
    features['r_finish_rate'] = r_fighter.get('finish_rate', 0.5)
    features['b_finish_rate'] = b_fighter.get('finish_rate', 0.5)
    features['finish_rate_diff'] = features['r_finish_rate'] - features['b_finish_rate']
    
    # Reach differential (physical advantage)
    r_reach = r_fighter.get('Reach', 0) or 0
    b_reach = b_fighter.get('Reach', 0) or 0
    if r_reach > 0 and b_reach > 0:
        features['reach_diff'] = r_reach - b_reach
    else:
        features['reach_diff'] = 0.0
    
    # Age differential
    r_age = r_fighter.get('age', 30) or 30
    b_age = b_fighter.get('age', 30) or 30
    features['age_diff'] = r_age - b_age
    
    # Ranking features (if available)
    features['r_ranked'] = 1.0 if r_fighter.get('rank', 0) > 0 else 0.0
    features['b_ranked'] = 1.0 if b_fighter.get('rank', 0) > 0 else 0.0
    
    # Weight class prestige (heavier = more prestige?)
    weight_class_prestige = {
        'Heavyweight': 1.0, 'Light Heavyweight': 0.9, 'Middleweight': 0.8,
        'Welterweight': 0.7, 'Lightweight': 0.6, 'Featherweight': 0.5,
        'Bantamweight': 0.4, 'Flyweight': 0.3, "Women's Bantamweight": 0.8,
        "Women's Featherweight": 0.7, "Women's Flyweight": 0.6, "Women's Strawweight": 0.5
    }
    weight_class = fight.get('weight_class', '')
    features['weight_class_prestige'] = weight_class_prestige.get(weight_class, 0.5)
    
    # Variance/randomness features
    # Higher finish rates = more variance
    avg_finish_rate = (features['r_finish_rate'] + features['b_finish_rate']) / 2
    features['matchup_variance'] = avg_finish_rate  # High finish rate = high variance
    
    # Puncher's chance (both have knockout power)
    features['double_knockout_threat'] = features['r_finish_rate'] * features['b_finish_rate']
    
    # Style matchup (if we have style data)
    r_style = r_fighter.get('stance', '').lower()
    b_style = b_fighter.get('stance', '').lower()
    features['style_mismatch'] = 1.0 if (r_style == 'southpaw' and b_style == 'orthodox') or \
                                        (r_style == 'orthodox' and b_style == 'southpaw') else 0.0
    
    return features


def extract_phonetic_features(fighter_name, fighter_nickname=''):
    """
    Deep phonetic analysis of fighter names.
    
    Focuses on:
    - Name memorability (syllable count, stress patterns)
    - Nickname power ("The Notorious", "Rowdy", "The Eagle")
    - Vowel dominance, consonant clusters
    - Cultural/ethnic distinctiveness
    """
    features = {}
    
    # Handle NaN or None values
    if not isinstance(fighter_name, str):
        fighter_name = ''
    if not isinstance(fighter_nickname, str):
        fighter_nickname = ''
    
    # Basic name metrics
    name_length = len(fighter_name)
    features['name_length'] = name_length
    features['name_word_count'] = len(fighter_name.split())
    
    # Vowel/consonant ratio
    vowels = 'aeiouAEIOU'
    vowel_count = sum(1 for c in fighter_name if c in vowels)
    consonant_count = sum(1 for c in fighter_name if c.isalpha() and c not in vowels)
    features['vowel_ratio'] = vowel_count / name_length if name_length > 0 else 0.0
    features['consonant_ratio'] = consonant_count / name_length if name_length > 0 else 0.0
    
    # Syllable approximation (vowel clusters = syllables)
    syllables = vowel_count  # Rough approximation
    features['syllable_count'] = syllables
    
    # Nickname features
    has_nickname = len(fighter_nickname) > 0
    features['has_nickname'] = 1.0 if has_nickname else 0.0
    features['nickname_length'] = len(fighter_nickname) if has_nickname else 0
    
    # Nickname power words (from UFC-specific patterns)
    power_nicknames = ['notorious', 'beast', 'eagle', 'spider', 'dragon', 'king', 'champion',
                       'killer', 'warrior', 'destroyer', 'pitbull', 'ice', 'bone', 'last']
    features['nickname_power'] = 1.0 if has_nickname and any(word in fighter_nickname.lower() 
                                                             for word in power_nicknames) else 0.0
    
    # Cultural distinctiveness (non-English names often more memorable)
    # Check for characters/patterns indicating non-English origin
    non_english_indicators = ['ó', 'á', 'ã', 'č', 'ć', 'š', 'ž', 'ñ', 'ü', 'ö']
    features['cultural_distinctive'] = 1.0 if any(char in fighter_name.lower() 
                                                   for char in non_english_indicators) else 0.0
    
    # Name memorability score (combination of factors)
    features['name_memorability'] = (
        0.3 * (1.0 if syllables <= 3 else 0.5) +  # Short names more memorable
        0.2 * features['has_nickname'] +
        0.2 * features['nickname_power'] +
        0.2 * features['cultural_distinctive'] +
        0.1 * (1.0 if features['vowel_ratio'] > 0.4 else 0.5)  # Vowel-rich names
    )
    
    return features


def load_ufc_data():
    """Load UFC fight data from available sources."""
    print("="*80)
    print("LOADING UFC DATA")
    print("="*80)
    
    # Load ufc_with_narratives.json (preferred format)
    ufc_file = data_dir / 'ufc_with_narratives.json'
    
    if not ufc_file.exists():
        print("  ✗ UFC data file not found!")
        return []
    
    print(f"\n✓ Found data file: {ufc_file.name}")
    
    with open(ufc_file, 'r') as f:
        fights_data = json.load(f)
    
    print(f"\n✓ Loaded {len(fights_data)} UFC fights")
    
    # Normalize fight structure to standard format
    for fight in fights_data:
        # Map fighter_a/fighter_b to R_fighter/B_fighter for consistency
        if 'fighter_a' in fight and 'R_fighter' not in fight:
            fight['R_fighter'] = fight['fighter_a']
        if 'fighter_b' in fight and 'B_fighter' not in fight:
            fight['B_fighter'] = fight['fighter_b']
        
        # Normalize winner field
        if 'result' in fight and 'winner' not in fight:
            result = fight['result']
            if isinstance(result, dict):
                fight['winner'] = result.get('winner', '')
                fight['method'] = result.get('method', '')
                fight['last_round'] = result.get('round', 3)
            elif isinstance(result, str):
                # Assume format like "Fighter A via KO"
                if 'fighter_a' in result.lower() or 'fighter a' in result.lower():
                    fight['winner'] = fight.get('fighter_a', {}).get('name', '')
                elif 'fighter_b' in result.lower() or 'fighter b' in result.lower():
                    fight['winner'] = fight.get('fighter_b', {}).get('name', '')
                
                if 'ko' in result.lower() or 'tko' in result.lower():
                    fight['method'] = 'KO/TKO'
                elif 'submission' in result.lower() or 'sub' in result.lower():
                    fight['method'] = 'Submission'
                elif 'decision' in result.lower():
                    fight['method'] = 'Decision'
        
        # Normalize title_fight field
        if 'title_fight' in fight and 'title_bout' not in fight:
            fight['title_bout'] = fight['title_fight']
        
        # Build narratives if missing
        if 'narrative' not in fight:
            r_fighter = fight.get('R_fighter', {})
            b_fighter = fight.get('B_fighter', {})
            r_name = r_fighter.get('name', 'Fighter A')
            b_name = b_fighter.get('name', 'Fighter B')
            r_nickname = r_fighter.get('nickname', '')
            b_nickname = b_fighter.get('nickname', '')
            
            # Build rich narrative
            narrative = f"{r_name}"
            if r_nickname:
                narrative += f" '{r_nickname}'"
            narrative += f" faces {b_name}"
            if b_nickname:
                narrative += f" '{b_nickname}'"
            
            # Add context
            if fight.get('title_bout') or fight.get('title_fight'):
                narrative += " in a championship bout"
            if fight.get('weight_class'):
                narrative += f" in the {fight['weight_class']} division"
            if fight.get('method'):
                narrative += f". Fight ended via {fight['method']}"
            if fight.get('location'):
                narrative += f" in {fight['location']}"
            
            fight['narrative'] = narrative
            
            # Add fighter narratives if missing
            if 'narrative' not in r_fighter:
                r_narrative = f"{r_name}"
                if r_nickname:
                    r_narrative += f" '{r_nickname}'"
                r_narrative += f", professional mixed martial artist"
                if r_fighter.get('record'):
                    r_narrative += f" with record {r_fighter['record']}"
                if r_fighter.get('stance'):
                    r_narrative += f", fights {r_fighter['stance']}"
                r_fighter['narrative'] = r_narrative
            
            if 'narrative' not in b_fighter:
                b_narrative = f"{b_name}"
                if b_nickname:
                    b_narrative += f" '{b_nickname}'"
                b_narrative += f", professional mixed martial artist"
                if b_fighter.get('record'):
                    b_narrative += f" with record {b_fighter['record']}"
                if b_fighter.get('stance'):
                    b_narrative += f", fights {b_fighter['stance']}"
                b_fighter['narrative'] = b_narrative
    
    return fights_data


def extract_all_features(fights_data, pi_value=0.722):
    """
    Apply ALL transformers + UFC-specific features.
    """
    print("\n" + "="*80)
    print("EXTRACTING FEATURES - ALL TRANSFORMERS + UFC-SPECIFIC")
    print("="*80)
    
    # Initialize all transformers
    transformers = {}
    
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
    
    for name, transformer_class in basic_transformers.items():
        try:
            transformers[name] = transformer_class()
        except Exception as e:
            print(f"    Warning: Could not initialize {name}: {e}")
    
    # Special transformers
    try:
        transformers['alpha'] = AlphaTransformer(narrativity=pi_value)
    except Exception as e:
        print(f"    Warning: Could not initialize alpha: {e}")
    
    try:
        transformers['golden_narratio'] = GoldenNarratioTransformer()
    except Exception as e:
        print(f"    Warning: Could not initialize golden_narratio: {e}")
    
    print(f"\n✓ Initialized {len(transformers)} transformers")
    
    # Prepare narratives for fitting
    print(f"\nPreparing narratives for transformer fitting...")
    all_narratives = []
    for fight in fights_data:
        narrative = fight.get('narrative', '')
        r_narrative = fight.get('R_fighter', {}).get('narrative', '')
        b_narrative = fight.get('B_fighter', {}).get('narrative', '')
        full_narrative = f"{narrative} {r_narrative} {b_narrative}"
        all_narratives.append(full_narrative)
    
    # Fit transformers
    print(f"Fitting transformers on {len(all_narratives)} narratives...")
    fitted_transformers = {}
    for name, transformer in transformers.items():
        try:
            if hasattr(transformer, 'fit'):
                transformer.fit(all_narratives)
            fitted_transformers[name] = transformer
        except Exception as e:
            print(f"    Warning: Could not fit {name}: {e}")
            fitted_transformers[name] = transformer
    
    print(f"✓ Fitted {len(fitted_transformers)} transformers")
    
    # Extract features
    all_features = []
    feature_names_set = set()
    
    print(f"\nProcessing {len(fights_data)} fights...")
    
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
                        
                        for j, val in enumerate(features):
                            feature_name = f"{name}_feature_{j}"
                            fight_features[feature_name] = float(val)
                            feature_names_set.add(feature_name)
                except Exception as e:
                    if batch_num == 0 and i == 0:
                        print(f"    Warning: {name} transformer failed: {e}")
                    continue
            
            # UFC-specific features
            ufc_features = extract_ufc_specific_features(fight)
            fight_features.update(ufc_features)
            for key in ufc_features.keys():
                feature_names_set.add(key)
            
            # Phonetic features for both fighters
            r_fighter = fight.get('R_fighter', {})
            b_fighter = fight.get('B_fighter', {})
            r_name = r_fighter.get('name', 'Fighter A')
            b_name = b_fighter.get('name', 'Fighter B')
            r_nickname = r_fighter.get('nickname', '')
            b_nickname = b_fighter.get('nickname', '')
            
            r_phonetic = extract_phonetic_features(r_name, r_nickname)
            b_phonetic = extract_phonetic_features(b_name, b_nickname)
            
            for key, val in r_phonetic.items():
                fight_features[f'r_{key}'] = val
                feature_names_set.add(f'r_{key}')
            for key, val in b_phonetic.items():
                fight_features[f'b_{key}'] = val
                feature_names_set.add(f'b_{key}')
            
            # Phonetic differentials
            fight_features['name_memorability_diff'] = r_phonetic.get('name_memorability', 0) - b_phonetic.get('name_memorability', 0)
            feature_names_set.add('name_memorability_diff')
            
            all_features.append(fight_features)
    
    feature_names = sorted(list(feature_names_set))
    
    print(f"\n✓ Extracted features from {len(all_features)} fights")
    print(f"✓ Total features: {len(feature_names)}")
    
    return all_features, feature_names


def extract_forces(fights_data):
    """
    Extract θ, λ, ة using enriched patterns (now including UFC-specific combat patterns).
    """
    print("\n" + "="*80)
    print("EXTRACTING THREE FORCES (θ, λ, ة) - WITH UFC PATTERNS")
    print("="*80)
    
    # Get enriched patterns for sports (includes combat sports patterns now)
    sports_patterns = get_patterns_for_domain('sports', 'both')
    theta_patterns = sports_patterns['theta']
    lambda_patterns = sports_patterns['lambda']
    
    theta_scores = []
    lambda_scores = []
    ta_marbuta_scores = []
    
    for fight in fights_data:
        narrative = fight.get('narrative', '')
        r_narrative = fight.get('R_fighter', {}).get('narrative', '')
        b_narrative = fight.get('B_fighter', {}).get('narrative', '')
        full_narrative = f"{narrative} {r_narrative} {b_narrative}"
        narrative_lower = full_narrative.lower()
        
        # Calculate θ (awareness) - count awareness patterns
        theta_count = 0
        for category, patterns in theta_patterns.items():
            for pattern in patterns:
                if pattern.lower() in narrative_lower:
                    theta_count += 1
        
        # Normalize by narrative length (patterns per 100 words)
        theta_score = min(1.0, theta_count / (len(full_narrative.split()) / 100))
        theta_scores.append(theta_score)
        
        # Calculate λ (constraints) - count constraint patterns
        lambda_count = 0
        total_lambda_patterns = 0
        for category, patterns in lambda_patterns.items():
            total_lambda_patterns += len(patterns)
            for pattern in patterns:
                if pattern.lower() in narrative_lower:
                    lambda_count += 1
        
        # Normalize by total possible patterns
        if total_lambda_patterns > 0:
            lambda_score = min(1.0, (lambda_count / total_lambda_patterns) * 2.0)
        else:
            lambda_score = 0.5
        
        # Also consider fighter-level constraints (finish rates, records)
        r_finish_rate = fight.get('R_fighter', {}).get('finish_rate', 0.5)
        b_finish_rate = fight.get('B_fighter', {}).get('finish_rate', 0.5)
        avg_finish_rate = (r_finish_rate + b_finish_rate) / 2.0
        # High finish rate = high skill level = higher constraints
        lambda_score = max(lambda_score, avg_finish_rate * 0.6)
        
        lambda_scores.append(lambda_score)
        
        # Calculate ة (nominative gravity)
        # Use fighter name features, rankings, finish rates
        r_name_len = len(fight.get('R_fighter', {}).get('name', ''))
        b_name_len = len(fight.get('B_fighter', {}).get('name', ''))
        avg_name_len = (r_name_len + b_name_len) / 2.0
        name_score = min(1.0, avg_name_len / 20.0)  # Names ~10-20 chars
        
        # Combine with finish rates and title fight status
        title_bonus = 0.2 if fight.get('title_bout', False) else 0.0
        ta_marbuta_score = 0.4 * name_score + 0.4 * avg_finish_rate + title_bonus
        ta_marbuta_score = min(1.0, ta_marbuta_score)
        
        ta_marbuta_scores.append(ta_marbuta_score)
    
    theta_mean = np.mean(theta_scores)
    theta_std = np.std(theta_scores)
    lambda_mean = np.mean(lambda_scores)
    lambda_std = np.std(lambda_scores)
    ta_marbuta_mean = np.mean(ta_marbuta_scores)
    ta_marbuta_std = np.std(ta_marbuta_scores)
    
    print(f"\nθ (Awareness Resistance):")
    print(f"  Mean: {theta_mean:.3f}")
    print(f"  Std:  {theta_std:.3f}")
    print(f"  Previous measurement: 0.535")
    print(f"  Change: {theta_mean - 0.535:+.3f} ({((theta_mean - 0.535) / 0.535 * 100):+.1f}%)")
    
    print(f"\nλ (Fundamental Constraints):")
    print(f"  Mean: {lambda_mean:.3f}")
    print(f"  Std:  {lambda_std:.3f}")
    
    print(f"\nة (Nominative Gravity):")
    print(f"  Mean: {ta_marbuta_mean:.3f}")
    print(f"  Std:  {ta_marbuta_std:.3f}")
    
    return {
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'lambda_mean': lambda_mean,
        'lambda_std': lambda_std,
        'ta_marbuta_mean': ta_marbuta_mean,
        'ta_marbuta_std': ta_marbuta_std
    }


def calculate_r_squared(features_data, feature_names, fights_data):
    """
    Calculate proper R² with train/test split.
    """
    print("\n" + "="*80)
    print("CALCULATING PERFORMANCE (R²)")
    print("="*80)
    
    # Convert features to matrix
    X_list = []
    y_list = []
    
    for i, fight_features in enumerate(features_data):
        # Create feature vector (fill missing with 0)
        feature_vec = [fight_features.get(fname, 0.0) for fname in feature_names]
        X_list.append(feature_vec)
        
        # Outcome: 1 if Red/A fighter wins, 0 if Blue/B fighter wins
        winner = fights_data[i].get('winner', '').lower()
        r_name = fights_data[i].get('R_fighter', {}).get('name', '').lower()
        b_name = fights_data[i].get('B_fighter', {}).get('name', '').lower()
        
        # Determine winner
        if r_name and r_name in winner:
            y = 1.0
        elif b_name and b_name in winner:
            y = 0.0
        else:
            # Check result field for fighter_a/fighter_b
            result = fights_data[i].get('result', '')
            if isinstance(result, str):
                result_lower = result.lower()
                if 'fighter_a' in result_lower or 'fighter a' in result_lower:
                    y = 1.0
                elif 'fighter_b' in result_lower or 'fighter b' in result_lower:
                    y = 0.0
                else:
                    y = 0.5  # Unknown - exclude?
            else:
                y = 0.5
        
        y_list.append(y)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nData shape: {X.shape}")
    print(f"Outcomes: {int(y.sum())} red wins, {len(y) - int(y.sum())} blue wins")
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with regularization (like Boxing)
    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
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
    print(f"\nPrevious AUC-based measure: 2.5% (delta)")
    print(f"New R² measure: {r2_test*100:.1f}%")
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print(f"\nTop 20 Features:")
    for fname, importance in top_features:
        print(f"  {fname}: {importance:.4f}")
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'model': model,
        'feature_importance': feature_importance
    }


def analyze_by_context(features_data, feature_names, fights_data):
    """
    Analyze R² by context (finish type, title fights).
    
    Previous analysis showed:
    - Submissions: 0.584 efficiency
    - KO/TKO: 0.575 efficiency
    - Round 1 finishes: 0.571 efficiency
    - Title fight finishes: 0.565 efficiency
    """
    print("\n" + "="*80)
    print("CONTEXT ANALYSIS")
    print("="*80)
    
    # Group fights by context
    contexts = {
        'All Fights': list(range(len(fights_data))),
        'Finishes': [],
        'Decisions': [],
        'KO/TKO': [],
        'Submissions': [],
        'Early Finishes (R1)': [],
        'Title Fights': [],
        'Non-Title': []
    }
    
    for i, fight in enumerate(fights_data):
        method = fight.get('method', '').lower()
        is_finish = 'ko' in method or 'tko' in method or 'submission' in method or 'sub' in method
        
        if is_finish:
            contexts['Finishes'].append(i)
        else:
            contexts['Decisions'].append(i)
        
        if 'ko' in method or 'tko' in method:
            contexts['KO/TKO'].append(i)
        if 'submission' in method or 'sub' in method:
            contexts['Submissions'].append(i)
        
        if fight.get('last_round', 3) <= 1:
            contexts['Early Finishes (R1)'].append(i)
        
        if fight.get('title_bout', False):
            contexts['Title Fights'].append(i)
        else:
            contexts['Non-Title'].append(i)
    
    print(f"\nContext sizes:")
    for context_name, indices in contexts.items():
        print(f"  {context_name}: {len(indices)} fights")
    
    # Calculate R² for each context
    context_results = {}
    for context_name, indices in contexts.items():
        if len(indices) < 50:  # Skip if too few samples
            continue
        
        # Subset data
        X_subset = []
        y_subset = []
        for i in indices:
            feature_vec = [features_data[i].get(fname, 0.0) for fname in feature_names]
            X_subset.append(feature_vec)
            
            winner = fights_data[i].get('winner', '').lower()
            r_name = fights_data[i].get('R_fighter', {}).get('name', '').lower()
            b_name = fights_data[i].get('B_fighter', {}).get('name', '').lower()
            
            if r_name and r_name in winner:
                y = 1.0
            elif b_name and b_name in winner:
                y = 0.0
            else:
                result = fights_data[i].get('result', '')
                if isinstance(result, str):
                    result_lower = result.lower()
                    if 'fighter_a' in result_lower:
                        y = 1.0
                    elif 'fighter_b' in result_lower:
                        y = 0.0
                    else:
                        y = 0.5
                else:
                    y = 0.5
            
            y_subset.append(y)
        
        X_subset = np.array(X_subset)
        y_subset = np.array(y_subset)
        X_subset = np.nan_to_num(X_subset, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Train-test split
        if len(X_subset) < 100:
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=50, random_state=42, max_depth=8,
            min_samples_split=20, min_samples_leaf=10, max_features='sqrt'
        )
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        context_results[context_name] = r2_test
    
    print(f"\nContext R² Results:")
    for context_name, r2 in sorted(context_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {context_name}: {r2:.3f} ({r2*100:.1f}%)")
    
    return context_results


def main():
    """Run complete UFC reanalysis."""
    print("="*80)
    print("UFC COMPLETE REANALYSIS")
    print("Investigating the performance gap")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    # 1. Load data
    fights_data = load_ufc_data()
    if not fights_data:
        print("ERROR: No UFC data available")
        return
    
    # 2. Calculate π
    pi, pi_components = calculate_ufc_pi()
    
    # 3. Extract features
    features_data, feature_names = extract_all_features(fights_data, pi)
    
    # 4. Extract forces
    forces = extract_forces(fights_data)
    
    # 5. Calculate R²
    performance = calculate_r_squared(features_data, feature_names, fights_data)
    
    # 6. Context analysis
    context_results = analyze_by_context(features_data, feature_names, fights_data)
    
    # 7. Calculate bridge
    theta = forces['theta_mean']
    lambda_val = forces['lambda_mean']
    ta_marbuta = forces['ta_marbuta_mean']
    bridge = calculate_bridge_three_force(ta_marbuta, theta, lambda_val, prestige_domain=False)
    leverage = bridge / pi if pi > 0 else 0
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nResults:")
    print(f"  π (Narrativity): {pi:.3f}")
    print(f"  R² (Test): {performance['r2_test']:.3f} ({performance['r2_test']*100:.1f}%)")
    print(f"  θ (Awareness): {theta:.3f}")
    print(f"  λ (Constraints): {lambda_val:.3f}")
    print(f"  ة (Nominative): {ta_marbuta:.3f}")
    print(f"  Д (Bridge): {bridge:.3f}")
    print(f"  Leverage (Д/π): {leverage:.3f}")
    
    # Save results
    output_file = output_dir / 'ufc_reanalysis_complete.json'
    results = {
        'domain': 'UFC',
        'name': 'Ultimate Fighting Championship',
        'date': datetime.now().isoformat(),
        'pi': float(pi),
        'pi_components': {k: float(v) for k, v in pi_components.items()},
        'forces': {k: float(v) for k, v in forces.items()},
        'performance': {
            'r2_train': float(performance['r2_train']),
            'r2_test': float(performance['r2_test']),
            'feature_importance': {k: float(v) for k, v in performance['feature_importance'].items()}
        },
        'context_results': {k: float(v) for k, v in context_results.items()},
        'bridge': float(bridge),
        'leverage': float(leverage),
        'sample_size': len(fights_data),
        'total_features': len(feature_names)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_file}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

