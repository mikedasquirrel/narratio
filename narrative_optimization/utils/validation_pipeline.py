"""
Universal Validation Pipeline

Runs complete statistical validation for any domain:
1. Train/test split (temporal if possible)
2. Model training
3. Performance metrics (accuracy, R², AUC)
4. Statistical tests (p-values, confidence intervals)
5. Backtesting simulation
6. Risk metrics (Sharpe, drawdown)
7. Context optimization
8. Confidence threshold testing
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from scipy import stats
import json
from pathlib import Path


class ValidationPipeline:
    """Complete validation pipeline for betting models"""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.results = {}
        
    def run_full_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        temporal_split: bool = True,
        odds: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run complete validation pipeline.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Outcomes (0/1 for binary classification)
        model : sklearn model
            Model to train/validate
        temporal_split : bool
            Use temporal validation if True
        odds : np.ndarray, optional
            Betting odds for each game (American format)
            
        Returns
        -------
        results : dict
            Complete validation results
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"VALIDATION PIPELINE: {self.domain_name.upper()}")
        print(f"{'='*60}\n")
        
        # 1. Train/Test Split
        print("Step 1: Train/Test Split")
        if temporal_split:
            # Use first 80% for train, last 20% for test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            if odds is not None:
                odds_train, odds_test = odds[:split_idx], odds[split_idx:]
            print(f"  ✓ Temporal split: {len(X_train)} train, {len(X_test)} test")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            if odds is not None:
                odds_train, odds_test = train_test_split(
                    odds, test_size=0.2, random_state=42
                )
            print(f"  ✓ Random split: {len(X_train)} train, {len(X_test)} test")
        
        # 2. Train model
        print("\nStep 2: Training Model")
        model.fit(X_train, y_train)
        print(f"  ✓ Model trained")
        
        # 3. Performance metrics
        print("\nStep 3: Performance Metrics")
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        results['train_accuracy'] = float(train_acc)
        results['test_accuracy'] = float(test_acc)
        
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy:  {test_acc:.1%}")
        
        # 4. Statistical significance
        print("\nStep 4: Statistical Significance")
        y_pred = model.predict(X_test)
        correct = (y_pred == y_test).sum()
        total = len(y_test)
        
        # Binomial test vs 50%
        try:
            # Try newer scipy API
            p_value = stats.binomtest(correct, total, 0.5, alternative='greater').pvalue
        except AttributeError:
            # Fallback to older API
            p_value = stats.binom_test(correct, total, 0.5, alternative='greater')
        results['p_value'] = float(p_value)
        results['significant'] = bool(p_value < 0.05)
        
        print(f"  Correct: {correct}/{total}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'} (α=0.05)")
        
        # 5. Confidence intervals
        print("\nStep 5: Confidence Intervals")
        from statsmodels.stats.proportion import proportion_confint
        ci_low, ci_high = proportion_confint(correct, total, alpha=0.05, method='wilson')
        results['confidence_interval_95'] = {
            'lower': float(ci_low),
            'upper': float(ci_high)
        }
        
        print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
        
        # 6. Cross-validation
        print("\nStep 6: Cross-Validation")
        cv_scores = cross_val_score(model, X, y, cv=5)
        results['cross_validation'] = {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': cv_scores.tolist()
        }
        
        print(f"  CV Mean: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
        print(f"  CV Scores: {[f'{s:.1%}' for s in cv_scores]}")
        
        # 7. Backtesting simulation
        print("\nStep 7: Backtesting Simulation")
        if odds is not None:
            backtest = self._run_backtest_with_odds(X_test, y_test, model, odds_test)
        else:
            backtest = self._run_backtest_simulation(X_test, y_test, model)
        results['backtesting'] = backtest
        
        if 'error' not in backtest:
            print(f"  Total Bets: {backtest['total_bets']}")
            print(f"  Accuracy: {backtest['accuracy']:.1%}")
            print(f"  ROI: {backtest['roi_pct']:.1%}")
            print(f"  Net Return: ${backtest['net_return']:,.2f}")
        else:
            print(f"  Error: {backtest['error']}")
        
        # 8. Confidence threshold testing
        print("\nStep 8: Confidence Threshold Testing")
        threshold_results = self._test_confidence_thresholds(X_test, y_test, model, odds_test if odds is not None else None)
        results['confidence_thresholds'] = threshold_results
        
        for threshold, res in threshold_results.items():
            if 'error' not in res:
                print(f"  {threshold}: {res['accuracy']:.1%} acc, {res['roi_pct']:.1%} ROI ({res['total_bets']} bets)")
        
        self.results = results
        
        print(f"\n{'='*60}")
        print(f"VALIDATION COMPLETE: {self.domain_name.upper()}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_backtest_simulation(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model
    ) -> Dict[str, Any]:
        """Simulate real money betting with standard -110 odds"""
        
        unit_size = 100  # $100 per bet
        confidence_threshold = 0.70
        
        # Get predictions with confidence
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            confidence = np.max(y_proba, axis=1)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            confidence = np.ones(len(y_pred))  # No confidence available
        
        # Bet only when confident
        bet_mask = confidence >= confidence_threshold
        n_bets = bet_mask.sum()
        
        if n_bets == 0:
            return {'error': 'No bets meet confidence threshold'}
        
        # Calculate results
        bet_predictions = y_pred[bet_mask]
        bet_actuals = y_test[bet_mask]
        correct = (bet_predictions == bet_actuals).sum()
        accuracy = correct / n_bets
        
        # ROI calculation (assumes -110 odds)
        total_wagered = n_bets * unit_size
        winnings = correct * unit_size * 0.91  # -110 odds = win $91 on $100 bet
        losses = (n_bets - correct) * unit_size
        net_return = winnings - losses
        roi = (net_return / total_wagered) * 100
        
        return {
            'total_bets': int(n_bets),
            'correct': int(correct),
            'accuracy': float(accuracy),
            'total_wagered': float(total_wagered),
            'net_return': float(net_return),
            'roi_pct': float(roi),
            'unit_size': unit_size,
            'confidence_threshold': confidence_threshold
        }
    
    def _run_backtest_with_odds(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model,
        odds: np.ndarray
    ) -> Dict[str, Any]:
        """Simulate real money betting with actual odds"""
        
        unit_size = 100  # $100 per bet
        confidence_threshold = 0.70
        
        # Get predictions with confidence
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            confidence = np.max(y_proba, axis=1)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            confidence = np.ones(len(y_pred))
        
        # Bet only when confident
        bet_mask = confidence >= confidence_threshold
        n_bets = bet_mask.sum()
        
        if n_bets == 0:
            return {'error': 'No bets meet confidence threshold'}
        
        # Calculate results with actual odds
        bet_predictions = y_pred[bet_mask]
        bet_actuals = y_test[bet_mask]
        bet_odds = odds[bet_mask]
        
        correct = (bet_predictions == bet_actuals)
        accuracy = correct.sum() / n_bets
        
        # Calculate winnings with real odds
        total_wagered = n_bets * unit_size
        winnings = 0
        losses = 0
        
        for i, is_correct in enumerate(correct):
            if is_correct:
                # Convert American odds to decimal and calculate winnings
                if bet_odds[i] > 0:
                    win_amount = unit_size * (bet_odds[i] / 100)
                else:
                    win_amount = unit_size * (100 / abs(bet_odds[i]))
                winnings += win_amount
            else:
                losses += unit_size
        
        net_return = winnings - losses
        roi = (net_return / total_wagered) * 100
        
        return {
            'total_bets': int(n_bets),
            'correct': int(correct.sum()),
            'accuracy': float(accuracy),
            'total_wagered': float(total_wagered),
            'net_return': float(net_return),
            'roi_pct': float(roi),
            'unit_size': unit_size,
            'confidence_threshold': confidence_threshold,
            'uses_real_odds': True
        }
    
    def _test_confidence_thresholds(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model,
        odds: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test multiple confidence thresholds"""
        
        thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
        results = {}
        
        # Get predictions with confidence
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            confidence = np.max(y_proba, axis=1)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            confidence = np.ones(len(y_pred))
        
        unit_size = 100
        
        for threshold in thresholds:
            bet_mask = confidence >= threshold
            n_bets = bet_mask.sum()
            
            if n_bets == 0:
                results[f'threshold_{int(threshold*100)}'] = {'error': 'No bets meet threshold'}
                continue
            
            bet_predictions = y_pred[bet_mask]
            bet_actuals = y_test[bet_mask]
            correct = (bet_predictions == bet_actuals)
            accuracy = correct.sum() / n_bets
            
            # Calculate ROI
            total_wagered = n_bets * unit_size
            
            if odds is not None:
                bet_odds = odds[bet_mask]
                winnings = 0
                for i, is_correct in enumerate(correct):
                    if is_correct:
                        if bet_odds[i] > 0:
                            winnings += unit_size * (bet_odds[i] / 100)
                        else:
                            winnings += unit_size * (100 / abs(bet_odds[i]))
                losses = (~correct).sum() * unit_size
            else:
                # Standard -110 odds
                winnings = correct.sum() * unit_size * 0.91
                losses = (~correct).sum() * unit_size
            
            net_return = winnings - losses
            roi = (net_return / total_wagered) * 100
            
            results[f'threshold_{int(threshold*100)}'] = {
                'threshold': float(threshold),
                'total_bets': int(n_bets),
                'correct': int(correct.sum()),
                'accuracy': float(accuracy),
                'total_wagered': float(total_wagered),
                'net_return': float(net_return),
                'roi_pct': float(roi),
                'unit_size': unit_size
            }
        
        return results
    
    def save_results(self, filepath: str):
        """Save validation results to JSON"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {filepath}")
    
    def print_summary(self):
        """Print a formatted summary of validation results"""
        if not self.results:
            print("No results to display. Run validation first.")
            return
        
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY: {self.domain_name.upper()}")
        print(f"{'='*60}\n")
        
        # Test Performance
        print("TEST PERFORMANCE:")
        print(f"  Accuracy: {self.results['test_accuracy']:.1%}")
        print(f"  95% CI: [{self.results['confidence_interval_95']['lower']:.1%}, "
              f"{self.results['confidence_interval_95']['upper']:.1%}]")
        print(f"  P-value: {self.results['p_value']:.6f}")
        print(f"  Significant: {'✓ YES' if self.results['significant'] else '✗ NO'}")
        
        # Cross-validation
        print("\nCROSS-VALIDATION:")
        print(f"  Mean: {self.results['cross_validation']['mean']:.1%} ± "
              f"{self.results['cross_validation']['std']:.1%}")
        
        # Backtesting
        if 'backtesting' in self.results and 'error' not in self.results['backtesting']:
            bt = self.results['backtesting']
            print("\nBACKTESTING (70% Confidence):")
            print(f"  Total Bets: {bt['total_bets']}")
            print(f"  Accuracy: {bt['accuracy']:.1%}")
            print(f"  ROI: {bt['roi_pct']:.1%}")
            print(f"  Net Return: ${bt['net_return']:,.2f}")
        
        # Best threshold
        if 'confidence_thresholds' in self.results:
            valid_thresholds = {k: v for k, v in self.results['confidence_thresholds'].items() 
                              if 'error' not in v}
            if valid_thresholds:
                best = max(valid_thresholds.items(), key=lambda x: x[1]['roi_pct'])
                print("\nBEST CONFIDENCE THRESHOLD:")
                print(f"  Threshold: {best[1]['threshold']:.0%}")
                print(f"  Accuracy: {best[1]['accuracy']:.1%}")
                print(f"  ROI: {best[1]['roi_pct']:.1%}")
                print(f"  Bets: {best[1]['total_bets']}")
        
        print(f"\n{'='*60}\n")


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for betting returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of bet returns (as percentages)
    risk_free_rate : float
        Annual risk-free rate (default 2%)
        
    Returns
    -------
    sharpe : float
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Parameters
    ----------
    cumulative_returns : np.ndarray
        Array of cumulative returns over time
        
    Returns
    -------
    max_drawdown : float
        Maximum drawdown as percentage
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return float(max_drawdown)

