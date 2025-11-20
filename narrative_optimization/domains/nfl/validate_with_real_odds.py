"""
NFL Validation with REAL Moneylines and Spreads
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from narrative_optimization.utils.real_odds_validator import RealOddsValidator, calculate_real_roi

def main():
    print("\nNFL VALIDATION WITH REAL BETTING ODDS")
    print("="*60)
    
    # Load features and games
    features_path = Path(__file__).parent.parent.parent / 'data' / 'features' / 'nfl_all_features.npz'
    games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    
    data = np.load(features_path)
    X = data['features']
    
    with open(games_path) as f:
        games = json.load(f)
    
    # Extract outcomes and REAL moneylines
    y = np.array([1 if g['home_won'] else 0 for g in games])
    
    # Get the appropriate odds for our prediction (home win)
    odds_home = np.array([g['betting_odds']['moneyline_home'] for g in games])
    
    print(f"\nGames: {len(games)}")
    print(f"Features: {X.shape[1]}")
    print(f"Home win rate: {y.mean():.1%}")
    print(f"Sample home moneylines: {odds_home[:10]}")
    
    # Create model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # Run validation with real odds
    validator = RealOddsValidator('NFL')
    results = validator.validate_with_real_odds(X, y, odds_home, model, confidence_threshold=0.70)
    
    if results:
        # Save results
        results_path = Path(__file__).parent / 'nfl_betting_validated_results.json'
        validator.save_results(results_path)
        
        # Test different confidence thresholds
        print(f"\n{'='*60}")
        print("CONFIDENCE THRESHOLD OPTIMIZATION")
        print(f"{'='*60}")
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        odds_test = odds_home[split:]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        confidence = np.max(y_proba, axis=1)
        y_pred = model.predict(X_test)
        
        for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = confidence >= threshold
            if mask.sum() > 0:
                roi_stats = calculate_real_roi(y_pred[mask], y_test[mask], odds_test[mask])
                print(f"{threshold:.0%}: {roi_stats['win_rate']:.1%} win rate, {roi_stats['roi_pct']:+.1f}% ROI ({roi_stats['total_bets']} bets)")
        
        print("\n✓ NFL VALIDATION COMPLETE (REAL ODDS)\n")
        return results
    
    return None

if __name__ == '__main__':
    main()

