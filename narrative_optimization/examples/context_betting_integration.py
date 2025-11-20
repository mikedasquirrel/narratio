"""
Context Pattern Betting Integration Example

Shows how to integrate ContextPatternTransformer into a betting pipeline.

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from transformers.context_pattern import ContextPatternTransformer

print("="*80)
print("CONTEXT PATTERN BETTING INTEGRATION")
print("="*80)


def create_synthetic_sports_data(n_games=2000):
    """Create realistic synthetic sports data"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_games, freq='D'),
        'is_home': np.random.choice([0, 1], n_games),
        'team_quality': np.random.uniform(0.3, 0.7, n_games),
        'opponent_quality': np.random.uniform(0.3, 0.7, n_games),
        'recent_form': np.random.uniform(0, 1, n_games),
        'season_progress': np.linspace(0, 1, n_games),
        'rest_days': np.random.choice([1, 2, 3, 4, 5, 7], n_games),
    })
    
    # Create outcome with patterns
    win_prob = (
        0.50 +  # Base
        0.10 * data['is_home'] +  # Home advantage
        0.20 * (data['team_quality'] - data['opponent_quality']) +  # Quality diff
        0.15 * (data['recent_form'] - 0.5) +  # Form
        0.05 * (data['season_progress'] > 0.7)  # Late season boost
    )
    
    # Add noise
    win_prob = np.clip(win_prob, 0.1, 0.9)
    data['won'] = (np.random.uniform(0, 1, n_games) < win_prob).astype(int)
    
    return data


def baseline_betting_system(data):
    """Traditional betting system without context patterns"""
    print("\n[BASELINE] Traditional Betting System")
    print("-"*80)
    
    # Features
    X = data[['is_home', 'team_quality', 'opponent_quality', 'recent_form', 'season_progress', 'rest_days']].values
    y = data['won'].values
    
    # Temporal split (70/30)
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Betting strategy: bet on high confidence (> 0.6)
    high_conf_mask = (y_pred_proba > 0.6) | (y_pred_proba < 0.4)
    
    accuracy_overall = (y_pred == y_test).mean()
    accuracy_high_conf = (y_pred[high_conf_mask] == y_test[high_conf_mask]).mean()
    
    # ROI calculation (assuming -110 odds)
    n_bets_conf = high_conf_mask.sum()
    wins_conf = (y_pred[high_conf_mask] == y_test[high_conf_mask]).sum()
    roi_conf = ((wins_conf * 91 - (n_bets_conf - wins_conf) * 100) / (n_bets_conf * 100)) if n_bets_conf > 0 else 0
    
    print(f"  Overall accuracy: {accuracy_overall:.1%}")
    print(f"  High confidence bets: {n_bets_conf} ({high_conf_mask.mean():.1%} of games)")
    print(f"  High confidence accuracy: {accuracy_high_conf:.1%}")
    print(f"  High confidence ROI: {roi_conf:+.1%}")
    
    return {
        'accuracy': accuracy_overall,
        'conf_accuracy': accuracy_high_conf,
        'conf_bets': n_bets_conf,
        'roi': roi_conf
    }


def context_enhanced_system(data):
    """Context-enhanced betting system"""
    print("\n[ENHANCED] Context-Enhanced Betting System")
    print("-"*80)
    
    # Prepare data
    feature_cols = ['is_home', 'team_quality', 'opponent_quality', 'recent_form', 'season_progress', 'rest_days']
    X = data[feature_cols]
    y = data['won'].values
    
    # Temporal split
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n  Step 1: Discovering context patterns...")
    
    # Discover patterns on training data
    context_transformer = ContextPatternTransformer(
        min_accuracy=0.60,
        min_samples=30,
        max_patterns=20
    )
    
    context_transformer.fit(X_train, y_train)
    
    print(f"    Patterns discovered: {len(context_transformer.patterns_)}")
    
    if context_transformer.patterns_:
        best = context_transformer.patterns_[0]
        print(f"    Best pattern: {best.accuracy:.1%} accuracy (n={best.sample_size})")
    
    print(f"\n  Step 2: Generating betting recommendations...")
    
    # Get recommendations for test set
    recommendations = context_transformer.get_betting_recommendations(X_test)
    
    # Filter to BET recommendations
    bet_recommendations = [r for r in recommendations if r['recommendation'] == 'BET']
    
    print(f"    Total test games: {len(X_test)}")
    print(f"    BET recommendations: {len(bet_recommendations)}")
    
    if bet_recommendations:
        # Evaluate betting performance
        bet_indices = [r['sample_idx'] for r in bet_recommendations]
        bet_outcomes = y_test[bet_indices]
        
        # Assume we bet on the best pattern's prediction
        wins = bet_outcomes.sum()
        losses = len(bet_outcomes) - wins
        
        accuracy = wins / len(bet_outcomes)
        roi = ((wins * 91 - losses * 100) / (len(bet_outcomes) * 100))
        
        avg_confidence = np.mean([r['confidence'] for r in bet_recommendations])
        avg_edge = np.mean([r['expected_edge'] for r in bet_recommendations])
        
        print(f"\n  Step 3: Performance metrics...")
        print(f"    Accuracy: {accuracy:.1%}")
        print(f"    ROI: {roi:+.1%}")
        print(f"    Average confidence: {avg_confidence:.1%}")
        print(f"    Average expected edge: {avg_edge:+.1%}")
        
        # Show sample recommendations
        print(f"\n  Step 4: Sample recommendations...")
        for i, rec in enumerate(bet_recommendations[:3], 1):
            print(f"\n    Recommendation {i}:")
            print(f"      Game index: {rec['sample_idx']}")
            print(f"      Confidence: {rec['confidence']:.1%}")
            print(f"      Expected edge: {rec['expected_edge']:+.1%}")
            print(f"      Pattern sample size: {rec['sample_size']}")
        
        return {
            'accuracy': accuracy,
            'roi': roi,
            'n_bets': len(bet_recommendations),
            'avg_confidence': avg_confidence,
            'avg_edge': avg_edge
        }
    else:
        print("    ⚠ No BET recommendations generated")
        return None


def compare_systems():
    """Compare baseline vs context-enhanced"""
    print("\n" + "="*80)
    print("SYSTEM COMPARISON")
    print("="*80)
    
    # Generate data
    print("\nGenerating synthetic sports data...")
    data = create_synthetic_sports_data(n_games=2000)
    print(f"  Games: {len(data):,}")
    print(f"  Win rate: {data['won'].mean():.1%}")
    
    # Run baseline
    baseline_results = baseline_betting_system(data)
    
    # Run enhanced
    enhanced_results = context_enhanced_system(data)
    
    # Compare
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Baseline':<15} {'Context-Enhanced':<15} {'Improvement'}")
    print("-"*80)
    
    if enhanced_results:
        # Accuracy
        acc_improvement = enhanced_results['accuracy'] - baseline_results['conf_accuracy']
        print(f"{'Accuracy':<30} {baseline_results['conf_accuracy']:.1%}            {enhanced_results['accuracy']:.1%}            {acc_improvement:+.1%}")
        
        # ROI
        roi_improvement = enhanced_results['roi'] - baseline_results['roi']
        print(f"{'ROI':<30} {baseline_results['roi']:+.1%}           {enhanced_results['roi']:+.1%}           {roi_improvement:+.1%}")
        
        # Bets
        print(f"{'Number of bets':<30} {baseline_results['conf_bets']:<15} {enhanced_results['n_bets']:<15}")
        
        # Summary
        print("\n" + "-"*80)
        if enhanced_results['accuracy'] > baseline_results['conf_accuracy']:
            print("✓ Context-enhanced system OUTPERFORMS baseline")
        else:
            print("⚠ Context-enhanced system comparable to baseline")
        
        print(f"\nKey advantage: Context patterns provide {enhanced_results['avg_edge']:.1%} expected edge")
        print(f"High confidence recommendations: {enhanced_results['avg_confidence']:.1%} average")
    else:
        print("\n⚠ Context-enhanced system generated no recommendations")
        print("  Try: Lower thresholds, more data, or better features")


if __name__ == '__main__':
    compare_systems()
    
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Replace synthetic data with real domain data")
    print("  2. Tune min_accuracy and min_samples for your domain")
    print("  3. Combine with domain-specific transformers")
    print("  4. Monitor pattern performance over time")
    print("  5. Retrain periodically to capture new patterns")
    print("\nSee docs/CONTEXT_PATTERN_TRANSFORMER_GUIDE.md for details.")
    print("="*80)

