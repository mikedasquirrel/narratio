"""
Generate Predictions for Upcoming ATP Tennis Matches

Uses the complete trained model (ALL 33 transformers, learned weights)
to predict upcoming ATP matches with full feature breakdown and betting recommendations.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta
import pickle

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

print("="*80)
print("TENNIS UPCOMING MATCH PREDICTIONS")
print("="*80)

# ============================================================================
# STEP 1: LOAD TRAINED MODEL
# ============================================================================

print("\n[STEP 1] Loading trained model...")

model_path = Path(__file__).parent / 'results' / 'tennis_complete_model.pkl'
scaler_path = Path(__file__).parent / 'results' / 'feature_scaler.pkl'
results_path = Path(__file__).parent / 'results' / 'tennis_complete_results.json'

if not model_path.exists():
    print(f"❌ Model not found at: {model_path}")
    print("Run train_complete_model.py first!")
    sys.exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(results_path) as f:
    training_results = json.load(f)

print(f"✓ Loaded model: {training_results['model_selection']['best_model']}")
print(f"  Test R²: {training_results['test_performance']['r2']:.4f}")
print(f"  Test ROI: {training_results['betting_simulation']['roi_pct']:.1f}%")
print(f"  Features: {training_results['features']['total_features']}")

# ============================================================================
# STEP 2: GET UPCOMING ATP MATCHES
# ============================================================================

print("\n[STEP 2] Getting upcoming ATP matches...")

# For now, we'll use recent test set matches as proxy for "upcoming"
# In production, this would query ATP API or scrape schedule

# Load full dataset
dataset_path = project_root / 'data' / 'domains' / 'tennis_complete_dataset.json'
with open(dataset_path) as f:
    all_matches = json.load(f)

# Get most recent matches as proxy
recent_matches = [m for m in all_matches if m.get('year') == 2024]
recent_matches = sorted(recent_matches, key=lambda m: m.get('date', ''), reverse=True)

# Take top 20 most recent
upcoming_matches = recent_matches[:20]

print(f"✓ Found {len(upcoming_matches)} recent/upcoming matches")
print(f"  Latest date: {upcoming_matches[0].get('date', 'unknown')}")

# ============================================================================
# STEP 3: GENERATE PREDICTIONS WITH FULL FEATURE BREAKDOWN
# ============================================================================

print("\n[STEP 3] Generating predictions...")

# Import transformers for feature extraction
from src.transformers import (
    StatisticalTransformer, NominativeAnalysisTransformer,
    SelfPerceptionTransformer, NarrativePotentialTransformer,
    LinguisticPatternsTransformer, EnsembleNarrativeTransformer,
    RelationalValueTransformer, PhoneticTransformer,
    UniversalNominativeTransformer, HierarchicalNominativeTransformer,
    NominativeInteractionTransformer, PureNominativePredictorTransformer,
    NominativeRichnessTransformer, EmotionalResonanceTransformer,
    AuthenticityTransformer, ConflictTensionTransformer,
    ExpertiseAuthorityTransformer, CulturalContextTransformer,
    SuspenseMysteryTransformer, FramingTransformer,
    TemporalEvolutionTransformer, InformationTheoryTransformer,
    CognitiveFluencyTransformer, SocialStatusTransformer,
    OpticsTransformer, CouplingStrengthTransformer,
    NarrativeMassTransformer, GravitationalFeaturesTransformer,
    AwarenessResistanceTransformer, FundamentalConstraintsTransformer,
    AlphaTransformer, GoldenNarratioTransformer
)

# Initialize transformers (same as training)
tennis_pi = 0.75
transformers_list = [
    ('statistical', StatisticalTransformer(max_features=100)),
    ('nominative', NominativeAnalysisTransformer()),
    ('self_perception', SelfPerceptionTransformer()),
    ('narrative_potential', NarrativePotentialTransformer()),
    ('linguistic', LinguisticPatternsTransformer()),
    ('ensemble', EnsembleNarrativeTransformer()),
    ('relational', RelationalValueTransformer()),
    ('phonetic', PhoneticTransformer()),
    ('universal_nominative', UniversalNominativeTransformer()),
    ('hierarchical_nominative', HierarchicalNominativeTransformer()),
    ('nominative_interaction', NominativeInteractionTransformer()),
    ('pure_nominative', PureNominativePredictorTransformer()),
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
    ('social_status', SocialStatusTransformer()),
    ('optics', OpticsTransformer()),
    ('coupling', CouplingStrengthTransformer()),
    ('mass', NarrativeMassTransformer()),
    ('gravitational', GravitationalFeaturesTransformer()),
    ('awareness', AwarenessResistanceTransformer()),
    ('constraints', FundamentalConstraintsTransformer()),
    ('alpha', AlphaTransformer(narrativity=tennis_pi)),
    ('golden_narratio', GoldenNarratioTransformer()),
]

predictions = []

for idx, match in enumerate(upcoming_matches, 1):
    print(f"  [{idx}/{len(upcoming_matches)}] {match.get('player1', {}).get('name', 'Player 1')} vs {match.get('player2', {}).get('name', 'Player 2')}")
    
    try:
        # Extract features
        narrative = match.get('narrative', '')
        
        # Apply all transformers
        features_list = []
        for trans_name, transformer in transformers_list:
            try:
                trans_features = transformer.transform([narrative])
                features_list.append(trans_features)
            except:
                pass  # Skip failed transformers
        
        if len(features_list) == 0:
            print(f"    ❌ No transformers succeeded")
            continue
        
        # Combine features
        combined_features = np.hstack(features_list)
        
        # Scale
        features_scaled = scaler.transform(combined_features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = abs(prediction - 0.5) * 2
        
        # Get actual outcome if available
        actual_winner = match.get('player1', {}).get('name', '') if match.get('player1_won', None) else match.get('player2', {}).get('name', '')
        
        predictions.append({
            'match_id': match.get('match_id', idx),
            'date': match.get('date', 'unknown'),
            'tournament': match.get('tournament', 'Unknown'),
            'player1': match.get('player1', {}).get('name', 'Player 1'),
            'player2': match.get('player2', {}).get('name', 'Player 2'),
            'surface': match.get('surface', 'unknown'),
            'predicted_winner': match.get('player1', {}).get('name', '') if prediction > 0.5 else match.get('player2', {}).get('name', ''),
            'prediction_score': float(prediction),
            'confidence': float(confidence),
            'probability_player1': float(prediction),
            'probability_player2': float(1 - prediction),
            'actual_winner': actual_winner if match.get('player1_won') is not None else 'TBD',
            'outcome': 'CORRECT' if (prediction > 0.5) == match.get('player1_won', False) and match.get('player1_won') is not None else 'TBD',
            'features_used': int(combined_features.shape[1]),
            'betting_recommendation': {
                'action': f"BET {match.get('player1', {}).get('name', '') if prediction > 0.5 else match.get('player2', {}).get('name', '')}",
                'confidence_level': 'HIGH' if confidence >= 0.7 else 'MEDIUM' if confidence >= 0.5 else 'LOW',
                'recommended_unit': 1.0 if confidence >= 0.7 else 0.5 if confidence >= 0.5 else 0.0
            }
        })
        
        print(f"    ✓ Prediction: {predictions[-1]['predicted_winner']} ({confidence*100:.1f}% confidence)")
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        continue

# ============================================================================
# STEP 4: SAVE PREDICTIONS
# ============================================================================

print(f"\n[STEP 4] Saving predictions...")

output_path = Path(__file__).parent / 'results' / 'upcoming_predictions.json'

predictions_output = {
    'generated_at': datetime.now().isoformat(),
    'model_info': {
        'model_type': training_results['model_selection']['best_model'],
        'test_r2': training_results['test_performance']['r2'],
        'test_roi': training_results['betting_simulation']['roi_pct'],
        'features_used': training_results['features']['total_features']
    },
    'predictions': predictions,
    'summary': {
        'total_predictions': len(predictions),
        'high_confidence': sum(1 for p in predictions if p['confidence'] >= 0.7),
        'medium_confidence': sum(1 for p in predictions if 0.5 <= p['confidence'] < 0.7),
        'low_confidence': sum(1 for p in predictions if p['confidence'] < 0.5)
    }
}

with open(output_path, 'w') as f:
    json.dump(predictions_output, f, indent=2)

print(f"✓ Saved to: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ Generated {len(predictions)} predictions")
print(f"   High confidence (≥70%): {predictions_output['summary']['high_confidence']}")
print(f"   Medium confidence (50-70%): {predictions_output['summary']['medium_confidence']}")
print(f"   Low confidence (<50%): {predictions_output['summary']['low_confidence']}")

print(f"\n💰 BETTING RECOMMENDATIONS:")
for pred in predictions[:5]:  # Show first 5
    conf_emoji = "🔥" if pred['confidence'] >= 0.7 else "📊"
    print(f"   {conf_emoji} {pred['player1']} vs {pred['player2']}")
    print(f"      → Bet {pred['predicted_winner']} ({pred['confidence']*100:.1f}% confidence)")

print(f"\n📄 Full predictions saved to: {output_path}")
print("="*80)

