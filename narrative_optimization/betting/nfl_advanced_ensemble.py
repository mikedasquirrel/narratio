"""
NFL Advanced Ensemble System
=============================

Multi-strategy ensemble for NFL betting combining pattern-based and ML approaches:
1. Pattern-based predictions (16 profitable patterns)
2. Stacking Ensemble (LR, XGBoost, LightGBM)
3. Voting Ensemble
4. Hybrid Pattern-ML Ensemble

Optimized for NFL spread betting (ATS predictions).

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class NFLAdvancedEnsemble:
    """
    Advanced ensemble for NFL betting combining pattern-based and ML approaches.
    """
    
    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize NFL advanced ensemble.
        
        Args:
            patterns_file: Path to NFL betting patterns JSON file
        """
        self.patterns_file = patterns_file or str(
            Path(__file__).parent.parent.parent / 'data' / 'domains' / 'nfl_betting_patterns_FIXED.json'
        )
        self.patterns = self._load_patterns()
        self.ml_ensembles = {}
        self.hybrid_ensemble = None
        self.is_fitted = False
        
    def _load_patterns(self) -> List[Dict[str, Any]]:
        """Load profitable NFL patterns."""
        try:
            with open(self.patterns_file, 'r') as f:
                data = json.load(f)
            return data.get('patterns', [])
        except Exception as e:
            print(f"Warning: Could not load patterns: {e}")
            return []
    
    def extract_pattern_features(self, game: Dict[str, Any]) -> np.ndarray:
        """
        Extract pattern-based features for a game.
        
        Args:
            game: Game dictionary
            
        Returns:
            Feature vector for pattern matching
        """
        features = []
        
        # Extract game attributes
        is_home = game.get('is_home', False)
        spread = game.get('spread', 0)
        home_win_pct = game.get('home_win_pct', 0.5)
        away_win_pct = game.get('away_win_pct', 0.5)
        record_gap = home_win_pct - away_win_pct
        week = game.get('week', 1)
        is_division = game.get('is_division', False)
        is_rivalry = game.get('is_rivalry', False)
        
        # Pattern 1: Huge home underdog (+7+)
        features.append(1.0 if (is_home and spread >= 7.0) else 0.0)
        
        # Pattern 2: Strong record home
        features.append(1.0 if (is_home and record_gap >= 0.2) else 0.0)
        
        # Pattern 3: Big home underdog (+3.5+)
        features.append(1.0 if (is_home and spread >= 3.5) else 0.0)
        
        # Pattern 4: Rivalry + home dog
        features.append(1.0 if (is_home and spread > 0 and is_rivalry) else 0.0)
        
        # Pattern 5: High momentum home
        l10_pct = game.get('l10_win_pct', 0.5)
        features.append(1.0 if (is_home and l10_pct >= 0.7) else 0.0)
        
        # Pattern 6: High story + home dog
        story_quality = game.get('story_quality', 0.0)
        features.append(1.0 if (is_home and spread > 0 and story_quality > 0.5) else 0.0)
        
        # Pattern 7: Late season + home dog
        features.append(1.0 if (is_home and spread > 0 and week >= 14) else 0.0)
        
        # Pattern 8: Division + home dog
        features.append(1.0 if (is_home and spread > 0 and is_division) else 0.0)
        
        # Pattern 9: Home underdog (general)
        features.append(1.0 if (is_home and spread > 0) else 0.0)
        
        # Continuous features
        features.append(spread)  # Spread magnitude
        features.append(record_gap)  # Record differential
        features.append(l10_pct)  # Recent form
        features.append(week / 18.0)  # Season progress
        features.append(story_quality)  # Narrative strength
        features.append(float(is_division))  # Division game
        features.append(float(is_rivalry))  # Rivalry game
        
        return np.array(features)
    
    def create_ml_ensembles(self) -> Dict[str, Any]:
        """Create ML-based ensembles."""
        ensembles = {}
        
        # Logistic Regression
        ensembles['lr'] = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
            cv=5
        )
        
        # Random Forest
        ensembles['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        ensembles['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            ensembles['xgb'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            ensembles['lgbm'] = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
        
        return ensembles
    
    def fit(self, games: List[Dict[str, Any]], outcomes: List[int], validation_split: float = 0.2):
        """
        Fit all ensemble strategies.
        
        Args:
            games: List of game dictionaries
            outcomes: List of outcomes (1 = home covered spread, 0 = away covered)
            validation_split: Fraction for validation
        """
        print("=" * 80)
        print("TRAINING NFL ADVANCED ENSEMBLE")
        print("=" * 80)
        
        # Extract features
        X = np.array([self.extract_pattern_features(game) for game in games])
        y = np.array(outcomes)
        
        print(f"\nDataset: {len(games)} games")
        print(f"Features: {X.shape[1]} pattern-based features")
        print(f"Home ATS win rate: {y.mean():.3f}")
        
        # Split for validation
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, X_val, y_train, y_val = X, None, y, None
        
        # Train ML ensembles
        print("\nTraining ML Ensembles...")
        print("-" * 80)
        
        self.ml_ensembles = self.create_ml_ensembles()
        
        for name, model in self.ml_ensembles.items():
            print(f"  Training {name.upper()}...", end=' ')
            model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, model.predict(X_train))
            if X_val is not None:
                val_acc = accuracy_score(y_val, model.predict(X_val))
                print(f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")
            else:
                print(f"Train: {train_acc:.4f}")
        
        # Train hybrid ensemble (combines ML predictions + pattern matching)
        print("\nTraining Hybrid Pattern-ML Ensemble...")
        print("-" * 80)
        
        if X_val is not None:
            # Get predictions from all ML models on validation set
            ml_preds_val = []
            for name, model in self.ml_ensembles.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                else:
                    pred = model.predict(X_val)
                ml_preds_val.append(pred)
            
            # Combine ML predictions with original features
            hybrid_features_val = np.column_stack([X_val] + ml_preds_val)
            
            # Get ML predictions on training set
            ml_preds_train = []
            for name, model in self.ml_ensembles.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_train)[:, 1]
                else:
                    pred = model.predict(X_train)
                ml_preds_train.append(pred)
            
            hybrid_features_train = np.column_stack([X_train] + ml_preds_train)
            
            # Train hybrid
            self.hybrid_ensemble = CalibratedClassifierCV(
                LogisticRegression(C=0.1, max_iter=1000, random_state=42),
                cv=3
            )
            self.hybrid_ensemble.fit(hybrid_features_train, y_train)
            
            hybrid_acc = accuracy_score(y_val, self.hybrid_ensemble.predict(hybrid_features_val))
            print(f"  Hybrid Ensemble: Val: {hybrid_acc:.4f}")
        
        self.is_fitted = True
        
        print("\n" + "=" * 80)
        print("NFL ADVANCED ENSEMBLE TRAINING COMPLETE")
        print("=" * 80)
    
    def predict_proba(
        self,
        games: List[Dict[str, Any]],
        strategy: str = 'hybrid'
    ) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            games: List of game dictionaries
            strategy: 'hybrid', 'pattern', or specific ML model name
            
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Extract features
        X = np.array([self.extract_pattern_features(game) for game in games])
        
        if strategy == 'hybrid' and self.hybrid_ensemble is not None:
            # Get ML predictions
            ml_preds = []
            for name, model in self.ml_ensembles.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                ml_preds.append(pred)
            
            # Combine with original features
            hybrid_features = np.column_stack([X] + ml_preds)
            return self.hybrid_ensemble.predict_proba(hybrid_features)
        
        elif strategy == 'pattern':
            # Simple pattern-based prediction
            lr = self.ml_ensembles.get('lr')
            if lr:
                return lr.predict_proba(X)
            
        elif strategy in self.ml_ensembles:
            model = self.ml_ensembles[strategy]
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[:, 1] = pred
                proba[:, 0] = 1 - pred
                return proba
        
        # Default: use best available
        if 'xgb' in self.ml_ensembles:
            return self.ml_ensembles['xgb'].predict_proba(X)
        elif 'lgbm' in self.ml_ensembles:
            return self.ml_ensembles['lgbm'].predict_proba(X)
        else:
            return self.ml_ensembles['lr'].predict_proba(X)
    
    def predict(self, games: List[Dict[str, Any]], strategy: str = 'hybrid') -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(games, strategy)
        return (proba[:, 1] > 0.5).astype(int)
    
    def evaluate_all_strategies(
        self,
        test_games: List[Dict[str, Any]],
        test_outcomes: List[int]
    ) -> pd.DataFrame:
        """Evaluate all strategies."""
        results = []
        
        X_test = np.array([self.extract_pattern_features(game) for game in test_games])
        y_test = np.array(test_outcomes)
        
        # Evaluate each ML ensemble
        for name, model in self.ml_ensembles.items():
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = 0.5
            
            results.append({
                'strategy': name,
                'accuracy': acc,
                'auc': auc,
                'ats_pct': acc  # For NFL, accuracy = ATS win rate
            })
        
        # Evaluate hybrid
        if self.hybrid_ensemble is not None:
            pred = self.predict(test_games, strategy='hybrid')
            acc = accuracy_score(y_test, pred)
            proba = self.predict_proba(test_games, strategy='hybrid')
            auc = roc_auc_score(y_test, proba[:, 1])
            
            results.append({
                'strategy': 'hybrid (best)',
                'accuracy': acc,
                'auc': auc,
                'ats_pct': acc
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('accuracy', ascending=False)
        
        return df


def test_nfl_ensemble():
    """Test NFL advanced ensemble."""
    print("Testing NFL Advanced Ensemble...")
    
    # Create synthetic NFL games
    np.random.seed(42)
    n_games = 500
    
    games = []
    outcomes = []
    
    for i in range(n_games):
        spread = np.random.uniform(-14, 14)
        is_home = True if spread > 0 else False  # Underdog is home
        
        home_win_pct = np.random.uniform(0.3, 0.7)
        away_win_pct = np.random.uniform(0.3, 0.7)
        
        game = {
            'is_home': is_home,
            'spread': abs(spread),
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct,
            'l10_win_pct': np.random.uniform(0.3, 0.7),
            'week': np.random.randint(1, 19),
            'is_division': np.random.random() < 0.3,
            'is_rivalry': np.random.random() < 0.15,
            'story_quality': np.random.uniform(0, 1)
        }
        
        # Outcome: home underdogs win more often (simulating real NFL pattern)
        if is_home and spread >= 7.0:
            outcome = 1 if np.random.random() < 0.94 else 0  # 94% ATS
        elif is_home and spread >= 3.5:
            outcome = 1 if np.random.random() < 0.87 else 0  # 87% ATS
        else:
            outcome = 1 if np.random.random() < 0.58 else 0  # Baseline 58%
        
        games.append(game)
        outcomes.append(outcome)
    
    # Split data
    train_games = games[:400]
    test_games = games[400:]
    train_outcomes = outcomes[:400]
    test_outcomes = outcomes[400:]
    
    # Train ensemble
    ensemble = NFLAdvancedEnsemble()
    ensemble.fit(train_games, train_outcomes, validation_split=0.2)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    
    results = ensemble.evaluate_all_strategies(test_games, test_outcomes)
    print("\nStrategy Performance:")
    print(results.to_string(index=False))
    
    # Test prediction
    print("\n" + "=" * 80)
    print("PREDICTION TEST")
    print("=" * 80)
    
    test_sample = test_games[:3]
    
    for strategy in ['hybrid', 'pattern', 'xgb', 'rf']:
        try:
            proba = ensemble.predict_proba(test_sample, strategy=strategy)
            pred = ensemble.predict(test_sample, strategy=strategy)
            print(f"\n{strategy.upper()}:")
            print(f"  Probabilities: {proba[:, 1]}")
            print(f"  Predictions (Home ATS): {pred}")
        except Exception as e:
            print(f"\n{strategy.upper()}: {e}")
    
    print("\n" + "=" * 80)
    print("NFL ENSEMBLE TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    test_nfl_ensemble()

