"""
Tennis Validation with REAL Decimal Odds
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from narrative_optimization.utils.real_odds_validator import RealOddsValidator, calculate_real_roi

def main():
    print("\nTENNIS VALIDATION WITH REAL BETTING ODDS")
    print("="*60)
    
    # Load tennis features (use archetype features)
    base_path = Path(__file__).parent.parent.parent / 'data' / 'archetype_features' / 'tennis'
    
    feature_types = ['hero_journey', 'character', 'plot', 'thematic', 'structural']
    all_features = []
    
    for feat_type in feature_types:
        data = np.load(base_path / f'{feat_type}_features.npz')
        all_features.append(data['features'])
    
    X = np.hstack(all_features)
    
    # Load games for outcomes and odds
    games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_with_temporal_context.json'
    with open(games_path) as f:
        games = json.load(f)
    
    # Extract outcomes and REAL decimal odds for player1
    y = np.array([1 if g['player1_won'] else 0 for g in games])
    odds_p1 = np.array([g['betting_odds']['player1_odds'] for g in games])
    
    print(f"\nGames: {len(games)}")
    print(f"Features: {X.shape[1]} (archetype only)")
    print(f"Player 1 win rate: {y.mean():.1%}")
    print(f"Sample Player1 decimal odds: {odds_p1[:10]}")
    
    # Create model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # Run validation with real odds
    validator = RealOddsValidator('Tennis')
    results = validator.validate_with_real_odds(X, y, odds_p1, model, confidence_threshold=0.70)
    
    if results:
        # Save results
        results_path = Path(__file__).parent / 'tennis_betting_validated_results_real.json'
        validator.save_results(results_path)
        
        # Test different confidence thresholds
        print(f"\n{'='*60}")
        print("CONFIDENCE THRESHOLD OPTIMIZATION")
        print(f"{'='*60}")
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        odds_test = odds_p1[split:]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        confidence = np.max(y_proba, axis=1)
        y_pred = model.predict(X_test)
        
        for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = confidence >= threshold
            if mask.sum() > 0:
                roi_stats = calculate_real_roi(y_pred[mask], y_test[mask], odds_test[mask])
                print(f"{threshold:.0%}: {roi_stats['win_rate']:.1%} win rate, {roi_stats['roi_pct']:+.1f}% ROI ({roi_stats['total_bets']} bets, ${roi_stats['net_profit']:,.0f} profit)")
        
        print("\n✓ TENNIS VALIDATION COMPLETE (REAL ODDS)\n")
        return results
    
    return None

if __name__ == '__main__':
    main()

