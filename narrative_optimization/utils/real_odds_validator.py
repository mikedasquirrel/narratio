"""
Validation Pipeline with REAL Betting Odds

Uses actual moneylines, spreads, and decimal odds from sportsbooks.
NO ASSUMED ODDS.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
import json
from pathlib import Path


def calculate_real_roi(predictions, actuals, odds, bet_amount=100):
    """
    Calculate ROI using REAL odds.
    
    Parameters
    ----------
    predictions : array
        Model predictions (0 or 1)
    actuals : array
        Actual outcomes (0 or 1)
    odds : array
        Real odds for the prediction side
        - American format: negative for favorites (e.g., -150), positive for dogs (e.g., +200)
        - Decimal format: > 1.0 (e.g., 1.67, 3.50)
    bet_amount : float
        Amount bet per game
        
    Returns
    -------
    roi_stats : dict
        ROI, total wagered, net profit, win rate
    """
    correct = (predictions == actuals)
    n_bets = len(predictions)
    n_wins = correct.sum()
    
    total_wagered = n_bets * bet_amount
    total_profit = 0
    
    for i in range(n_bets):
        if correct[i]:
            odd = odds[i]
            
            # Handle American odds
            if isinstance(odd, (int, float)) and abs(odd) > 10:
                if odd > 0:  # Underdog
                    profit = bet_amount * (odd / 100)
                else:  # Favorite
                    profit = bet_amount * (100 / abs(odd))
            # Handle decimal odds
            elif isinstance(odd, float) and 1.0 <= odd <= 20.0:
                profit = bet_amount * (odd - 1)
            else:
                # Unknown format, skip
                continue
                
            total_profit += profit
        else:
            total_profit -= bet_amount
    
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    
    return {
        'total_bets': n_bets,
        'wins': int(n_wins),
        'losses': n_bets - int(n_wins),
        'win_rate': float(n_wins / n_bets),
        'total_wagered': float(total_wagered),
        'net_profit': float(total_profit),
        'roi_pct': float(roi),
        'bet_amount': bet_amount
    }


class RealOddsValidator:
    """Validator that uses actual betting odds"""
    
    def __init__(self, sport_name):
        self.sport_name = sport_name
        self.results = {}
        
    def validate_with_real_odds(self, X, y, odds, model, confidence_threshold=0.7):
        """
        Run validation using real odds.
        
        Parameters
        ----------
        X : array
            Features
        y : array
            Actual outcomes (0 or 1)
        odds : array
            Real odds corresponding to predictions
        model : sklearn model
            Model to validate
        confidence_threshold : float
            Only bet when confidence >= threshold
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION WITH REAL ODDS: {self.sport_name.upper()}")
        print(f"{'='*60}\n")
        
        # Train/test split (temporal)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        odds_test = odds[split:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Get predictions with confidence
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            confidence = np.max(y_proba, axis=1)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            confidence = np.ones(len(y_pred))
        
        # Overall accuracy
        test_acc = (y_pred == y_test).mean()
        print(f"Test Accuracy: {test_acc:.1%}")
        
        # Filter to confident predictions
        confident_mask = confidence >= confidence_threshold
        n_confident = confident_mask.sum()
        
        if n_confident == 0:
            print(f"No predictions meet {confidence_threshold:.0%} confidence threshold")
            return None
        
        print(f"Confident predictions ({confidence_threshold:.0%}): {n_confident}/{len(y_test)}")
        
        # Calculate ROI on confident bets with REAL odds
        roi_results = calculate_real_roi(
            y_pred[confident_mask],
            y_test[confident_mask],
            odds_test[confident_mask]
        )
        
        print(f"\nRESULTS WITH REAL ODDS:")
        print(f"  Total Bets: {roi_results['total_bets']}")
        print(f"  Win Rate: {roi_results['win_rate']:.1%}")
        print(f"  Total Wagered: ${roi_results['total_wagered']:,.0f}")
        print(f"  Net Profit: ${roi_results['net_profit']:,.0f}")
        print(f"  ROI: {roi_results['roi_pct']:.1f}%")
        
        # Statistical significance
        try:
            p_value = stats.binomtest(
                roi_results['wins'], 
                roi_results['total_bets'], 
                0.5, 
                alternative='greater'
            ).pvalue
        except:
            from scipy.stats import binom_test
            p_value = binom_test(
                roi_results['wins'],
                roi_results['total_bets'],
                0.5,
                alternative='greater'
            )
        
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
        
        self.results = {
            'test_accuracy': float(test_acc),
            'confident_predictions': int(n_confident),
            'confidence_threshold': confidence_threshold,
            'roi_results': roi_results,
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'uses_real_odds': True
        }
        
        return self.results
    
    def save_results(self, filepath):
        """Save validation results"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")

