"""
NFL Betting Edge Analysis

Test if narrative provides edge over betting markets:
1. Narrative-only predictions
2. Odds-only predictions (Vegas baseline)
3. Combined model (narrative + odds)
4. Spread coverage prediction
5. Underdog identification (inverse pattern)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load data for betting analysis."""
    # Load games with betting odds
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    with open(dataset_path) as f:
        games = json.load(f)
    
    # Load story quality
    genome_path = Path(__file__).parent / 'nfl_genome_data.npz'
    genome_data = np.load(genome_path, allow_pickle=True)
    ю = genome_data['story_quality']
    outcomes = genome_data['outcomes']
    
    return games, ю, outcomes


def extract_betting_features(games):
    """Extract betting odds features."""
    spreads = np.array([g['betting_odds']['spread'] for g in games])
    moneyline_home = np.array([g['betting_odds']['moneyline_home'] for g in games])
    moneyline_away = np.array([g['betting_odds']['moneyline_away'] for g in games])
    over_under = np.array([g['betting_odds']['over_under'] for g in games])
    
    return np.column_stack([spreads, moneyline_home, moneyline_away, over_under])


def main():
    """Test narrative vs betting market edge."""
    print("="*80)
    print("NFL BETTING EDGE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    games, ю, outcomes = load_data()
    print(f"✓ Loaded {len(games)} games")
    
    betting_features = extract_betting_features(games)
    
    # ========================================================================
    # TEST 1: NARRATIVE-ONLY MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 1: NARRATIVE-ONLY MODEL")
    print("="*80)
    
    X_narrative = ю.reshape(-1, 1)
    y = outcomes
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_narrative, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model_narrative = LogisticRegression(random_state=42)
    model_narrative.fit(X_train, y_train)
    
    # Predictions
    y_pred_narrative = model_narrative.predict(X_test)
    accuracy_narrative = accuracy_score(y_test, y_pred_narrative)
    
    print(f"\nNarrative-only accuracy: {accuracy_narrative:.4f} ({accuracy_narrative*100:.2f}%)")
    print(f"Baseline (always predict home win): {y_test.mean():.4f} ({y_test.mean()*100:.2f}%)")
    print(f"Edge over baseline: {(accuracy_narrative - y_test.mean())*100:.2f} percentage points")
    
    # Cross-validation
    cv_scores = cross_val_score(model_narrative, X_narrative, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ========================================================================
    # TEST 2: ODDS-ONLY MODEL (Vegas Baseline)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 2: ODDS-ONLY MODEL (Vegas Baseline)")
    print("="*80)
    
    X_odds = betting_features
    
    # Train/test split (same split as before)
    X_train_odds, X_test_odds, _, _ = train_test_split(
        X_odds, y, test_size=0.3, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_odds = scaler.fit_transform(X_train_odds)
    X_test_odds = scaler.transform(X_test_odds)
    
    # Train model
    model_odds = LogisticRegression(random_state=42, max_iter=1000)
    model_odds.fit(X_train_odds, y_train)
    
    # Predictions
    y_pred_odds = model_odds.predict(X_test_odds)
    accuracy_odds = accuracy_score(y_test, y_pred_odds)
    
    print(f"\nOdds-only accuracy: {accuracy_odds:.4f} ({accuracy_odds*100:.2f}%)")
    print(f"Edge over narrative: {(accuracy_odds - accuracy_narrative)*100:.2f} percentage points")
    
    # Simple spread-based prediction
    spreads_test = X_test_odds[:, 0] * scaler.scale_[0] + scaler.mean_[0]  # Inverse transform
    y_pred_spread = (spreads_test < 0).astype(int)  # Home favored if spread negative
    accuracy_spread = accuracy_score(y_test, y_pred_spread)
    
    print(f"\nSimple spread prediction: {accuracy_spread:.4f} ({accuracy_spread*100:.2f}%)")
    print(f"Note: Vegas is typically ~65-70% accurate ATS")
    
    # ========================================================================
    # TEST 3: COMBINED MODEL (Narrative + Odds)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 3: COMBINED MODEL (Narrative + Odds)")
    print("="*80)
    
    X_combined = np.column_stack([X_narrative, X_odds])
    
    # Train/test split
    X_train_combined, X_test_combined, _, _ = train_test_split(
        X_combined, y, test_size=0.3, random_state=42
    )
    
    # Standardize
    scaler_combined = StandardScaler()
    X_train_combined = scaler_combined.fit_transform(X_train_combined)
    X_test_combined = scaler_combined.transform(X_test_combined)
    
    # Train model
    model_combined = LogisticRegression(random_state=42, max_iter=1000)
    model_combined.fit(X_train_combined, y_train)
    
    # Predictions
    y_pred_combined = model_combined.predict(X_test_combined)
    accuracy_combined = accuracy_score(y_test, y_pred_combined)
    
    print(f"\nCombined accuracy: {accuracy_combined:.4f} ({accuracy_combined*100:.2f}%)")
    print(f"Improvement over odds-only: {(accuracy_combined - accuracy_odds)*100:.2f} percentage points")
    print(f"Improvement over narrative-only: {(accuracy_combined - accuracy_narrative)*100:.2f} percentage points")
    
    # Feature importance
    feature_importance = abs(model_combined.coef_[0])
    narrative_importance = feature_importance[0]
    odds_importance = feature_importance[1:].mean()
    
    print(f"\nRelative feature importance:")
    print(f"  Narrative: {narrative_importance:.4f}")
    print(f"  Odds (avg): {odds_importance:.4f}")
    print(f"  Narrative/Odds ratio: {narrative_importance/odds_importance:.4f}")
    
    # ========================================================================
    # TEST 4: SPREAD COVERAGE PREDICTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 4: SPREAD COVERAGE PREDICTION")
    print("="*80)
    
    # Extract spread coverage outcomes
    spread_covered = np.array([g['betting_odds']['home_covered_spread'] for g in games])
    
    # Use narrative to predict spread coverage
    X_train_cov, X_test_cov, y_train_cov, y_test_cov = train_test_split(
        X_narrative, spread_covered, test_size=0.3, random_state=42
    )
    
    model_coverage = LogisticRegression(random_state=42)
    model_coverage.fit(X_train_cov, y_train_cov)
    
    y_pred_coverage = model_coverage.predict(X_test_cov)
    accuracy_coverage = accuracy_score(y_test_cov, y_pred_coverage)
    
    print(f"\nSpread coverage prediction accuracy: {accuracy_coverage:.4f} ({accuracy_coverage*100:.2f}%)")
    print(f"Baseline (random): 50.0%")
    print(f"Edge: {(accuracy_coverage - 0.5)*100:.2f} percentage points")
    
    # ========================================================================
    # TEST 5: UNDERDOG IDENTIFICATION (Inverse Pattern)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TEST 5: UNDERDOG IDENTIFICATION")
    print("="*80)
    
    # Check correlation sign
    r = np.corrcoef(ю, outcomes)[0, 1]
    
    print(f"\nCorrelation (r): {r:.4f}")
    
    if r < 0:
        print("✓ INVERSE PATTERN CONFIRMED (like NBA)")
        print("  Better narrative → Underdog advantage")
        
        # Test inverse betting strategy
        # Inverse strategy: Bet against the narrative (low ю → bet home, high ю → bet away)
        inverse_predictions = (ю < np.median(ю)).astype(int)
        
        X_train_inv, X_test_inv, y_train_inv, y_test_inv = train_test_split(
            np.arange(len(ю)), outcomes, test_size=0.3, random_state=42
        )
        
        inverse_pred_test = inverse_predictions[X_test_inv]
        accuracy_inverse = accuracy_score(y_test_inv, inverse_pred_test)
        
        print(f"\nInverse strategy accuracy: {accuracy_inverse:.4f} ({accuracy_inverse*100:.2f}%)")
        print(f"Interpretation: Bet on team with LOWER narrative quality")
    else:
        print("✓ POSITIVE PATTERN")
        print("  Better narrative → Favorite advantage")
        print("  Direct strategy: Bet on higher narrative quality team")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY: BETTING EDGE ANALYSIS")
    print("="*80)
    
    results = {
        'narrative_only': {
            'accuracy': float(accuracy_narrative),
            'description': 'Predict from narrative quality alone'
        },
        'odds_only': {
            'accuracy': float(accuracy_odds),
            'spread_accuracy': float(accuracy_spread),
            'description': 'Vegas baseline (betting odds)'
        },
        'combined': {
            'accuracy': float(accuracy_combined),
            'improvement_over_odds': float(accuracy_combined - accuracy_odds),
            'improvement_over_narrative': float(accuracy_combined - accuracy_narrative),
            'description': 'Narrative + Odds combined model'
        },
        'spread_coverage': {
            'accuracy': float(accuracy_coverage),
            'edge': float(accuracy_coverage - 0.5),
            'description': 'Predict if home covers spread'
        },
        'pattern': {
            'correlation': float(r),
            'pattern_type': 'inverse' if r < 0 else 'positive',
            'interpretation': 'Better narrative favors underdog' if r < 0 else 'Better narrative favors favorite'
        },
        'baseline': {
            'home_win_rate': float(y.mean()),
            'description': 'Always predict home win'
        }
    }
    
    print("\nModel Performance:")
    print(f"  Narrative-only:     {accuracy_narrative:.4f} ({accuracy_narrative*100:.2f}%)")
    print(f"  Odds-only:          {accuracy_odds:.4f} ({accuracy_odds*100:.2f}%)")
    print(f"  Combined:           {accuracy_combined:.4f} ({accuracy_combined*100:.2f}%)")
    print(f"  Spread coverage:    {accuracy_coverage:.4f} ({accuracy_coverage*100:.2f}%)")
    print(f"  Baseline:           {y.mean():.4f} ({y.mean()*100:.2f}%)")
    
    print("\nKey Findings:")
    if accuracy_combined > accuracy_odds:
        print(f"  ✓ Narrative ADDS value: +{(accuracy_combined - accuracy_odds)*100:.2f}pp")
    else:
        print(f"  ✗ Narrative does NOT add value: {(accuracy_combined - accuracy_odds)*100:.2f}pp")
    
    if accuracy_narrative > y.mean():
        print(f"  ✓ Narrative beats baseline: +{(accuracy_narrative - y.mean())*100:.2f}pp")
    else:
        print(f"  ✗ Narrative doesn't beat baseline: {(accuracy_narrative - y.mean())*100:.2f}pp")
    
    print(f"\n  Pattern: {results['pattern']['pattern_type'].upper()}")
    print(f"  Interpretation: {results['pattern']['interpretation']}")
    
    # Save results
    output_path = Path(__file__).parent / 'nfl_betting_edge_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*80)
    print("BETTING EDGE ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

