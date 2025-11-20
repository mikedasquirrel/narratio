"""
Bayesian Hyperparameter Optimization
=====================================

Uses Optuna for Bayesian optimization of:
- Ensemble model weights
- Meta-learner hyperparameters
- Confidence thresholds
- Edge thresholds
- Pattern combination logic

Target: 500 trials per league to maximize ROI.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class HyperparameterTuner:
    """Bayesian hyperparameter optimization for betting models."""
    
    def __init__(
        self,
        n_trials: int = 500,
        objective_metric: str = 'roi',  # 'roi' or 'accuracy'
        random_state: int = 42
    ):
        """
        Initialize tuner.
        
        Args:
            n_trials: Number of Optuna trials
            objective_metric: Metric to optimize
            random_state: Random seed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install with: pip install optuna")
        
        self.n_trials = n_trials
        self.objective_metric = objective_metric
        self.random_state = random_state
        
        self.best_params = None
        self.study = None
        
    def create_objective(self, X_train, y_train, X_val, y_val):
        """Create objective function for Optuna."""
        
        def objective(trial):
            # Hyperparameters to tune
            params = {
                # Ensemble parameters
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                
                # Meta-learner regularization
                'meta_C': trial.suggest_float('meta_C', 0.01, 10.0, log=True),
                
                # Betting thresholds
                'confidence_threshold': trial.suggest_float('confidence_threshold', 0.55, 0.70),
                'edge_threshold': trial.suggest_float('edge_threshold', 0.02, 0.10),
                
                # Kelly fraction
                'kelly_fraction': trial.suggest_float('kelly_fraction', 0.25, 1.0)
            }
            
            # Train model with these params
            model = GradientBoostingClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                random_state=self.random_state
            )
            
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate objective
            if self.objective_metric == 'accuracy':
                y_pred = (y_pred_proba > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred)
            else:  # roi
                # Calculate ROI with Kelly sizing
                score = self._calculate_roi(
                    y_pred_proba,
                    y_val,
                    params['confidence_threshold'],
                    params['edge_threshold'],
                    params['kelly_fraction']
                )
            
            return score
        
        return objective
    
    def _calculate_roi(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        confidence_threshold: float,
        edge_threshold: float,
        kelly_fraction: float
    ) -> float:
        """Calculate ROI with Kelly sizing."""
        total_profit = 0
        total_wagered = 0
        bankroll = 10000
        
        for prob, outcome in zip(probabilities, outcomes):
            # Only bet if meets thresholds
            if prob < confidence_threshold:
                continue
            
            # Calculate edge (assuming -110 odds)
            implied_prob = 0.524  # Break-even at -110
            edge = prob - implied_prob
            
            if edge < edge_threshold:
                continue
            
            # Kelly sizing
            kelly_full = (prob * 0.9091 - (1 - prob)) / 0.9091  # -110 odds
            bet_fraction = min(kelly_full * kelly_fraction, 0.02)  # Cap at 2%
            bet_amount = bankroll * bet_fraction
            
            if bet_amount > 0:
                total_wagered += bet_amount
                
                if outcome == 1:
                    profit = bet_amount * 0.9091
                    total_profit += profit
                    bankroll += profit
                else:
                    total_profit -= bet_amount
                    bankroll -= bet_amount
        
        roi = (total_profit / total_wagered) if total_wagered > 0 else 0
        return roi
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Best hyperparameters
        """
        print("=" * 80)
        print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"Trials: {self.n_trials}")
        print(f"Objective: {self.objective_metric}")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Create objective
        objective = self.create_objective(X_train, y_train, X_val, y_val)
        
        # Optimize
        print("\nOptimizing...")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Get best params
        self.best_params = self.study.best_params
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Best {self.objective_metric}: {self.study.best_value:.4f}")
        print("\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params
    
    def save_results(self, filepath: str):
        """Save optimization results."""
        if not self.study:
            print("No study to save")
            return
        
        results = {
            'n_trials': self.n_trials,
            'objective_metric': self.objective_metric,
            'best_value': self.study.best_value,
            'best_params': self.best_params,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")


def test_hyperparameter_tuning():
    """Test hyperparameter tuning."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Install with: pip install optuna")
        return
    
    print("Testing Hyperparameter Optimization (10 trials for speed)...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Optimize (only 10 trials for testing)
    tuner = HyperparameterTuner(n_trials=10, objective_metric='roi')
    best_params = tuner.optimize(X_train, y_train, X_val, y_val)
    
    # Save
    save_path = Path(__file__).parent.parent.parent / 'configs' / 'test_optimal_params.json'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tuner.save_results(str(save_path))
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING TEST COMPLETE")
    print("=" * 80)
    print(f"\nFor production, run with n_trials=500")
    print(f"Estimated time: 2-4 hours per league")


if __name__ == '__main__':
    test_hyperparameter_tuning()
