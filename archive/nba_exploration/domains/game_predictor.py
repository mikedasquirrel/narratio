"""
NBA Game Outcome Prediction Models

Implements multiple prediction approaches:
- Narrative-only model
- Traditional stats baseline
- Hybrid narrative + stats model
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path


class NBAGamePredictor:
    """
    Predicts NBA game outcomes using narrative features.
    
    Can operate in three modes:
    - Narrative-only: Uses narrative features exclusively
    - Traditional: Uses stats and betting lines only
    - Hybrid: Combines both for optimal performance
    """
    
    def __init__(self, model_type: str = 'hybrid'):
        """
        Initialize NBA game predictor.
        
        Parameters
        ----------
        model_type : str
            'narrative', 'traditional', or 'hybrid'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.fitted = False
    
    def train(self, X: np.ndarray, y: np.ndarray, model_class: str = 'gradient_boosting'):
        """
        Train the prediction model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_games, n_features)
        y : np.ndarray
            Outcomes (1 = home wins, 0 = away wins)
        model_class : str
            'gradient_boosting', 'random_forest', or 'logistic'
        """
        print(f"Training {self.model_type} model with {model_class}...")
        print(f"  Training samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Home win rate: {y.mean():.3f}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Select model
        if model_class == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
        elif model_class == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42
            )
        else:  # logistic
            self.model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            )
        
        # Train
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data
        train_accuracy = self.model.score(X_scaled, y)
        print(f"  Training accuracy: {train_accuracy:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"  CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            print(f"  Top feature importance: {self.feature_importance.max():.3f}")
        
        self.fitted = True
        print("✅ Model training complete")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict game outcomes.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray
            Binary predictions (1 = home wins)
        probabilities : np.ndarray
            Win probabilities for each team (n_games, 2)
        """
        if not self.fitted:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_game(self, home_features: np.ndarray, away_features: np.ndarray, 
                     differential: np.ndarray) -> Dict[str, Any]:
        """
        Predict a single game with detailed output.
        
        Parameters
        ----------
        home_features : np.ndarray
            Home team narrative features  
        away_features : np.ndarray
            Away team narrative features
        differential : np.ndarray
            Feature differentials (home - away)
        
        Returns
        -------
        prediction : dict
            Contains probabilities, prediction, confidence
        """
        # Use only differential for prediction (model was trained on differentials)
        X = differential.reshape(1, -1)
        
        # Predict
        pred, prob = self.predict(X)
        
        # Calculate confidence
        home_prob = prob[0][1]  # Probability of home win
        away_prob = prob[0][0]  # Probability of away win
        confidence = abs(home_prob - 0.5) * 2  # 0-1 scale
        
        return {
            'home_win_probability': float(home_prob),
            'away_win_probability': float(away_prob),
            'predicted_winner': 'home' if pred[0] == 1 else 'away',
            'confidence': float(confidence),
            'confidence_level': 'HIGH' if confidence > 0.3 else 'MODERATE' if confidence > 0.15 else 'LOW'
        }
    
    def get_top_features(self, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get top N most important features.
        
        Returns
        -------
        top_features : list of (index, importance) tuples
        """
        if self.feature_importance is None:
            return []
        
        indices = np.argsort(self.feature_importance)[::-1][:n]
        importances = self.feature_importance[indices]
        
        return list(zip(indices.tolist(), importances.tolist()))
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted model")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data.get('model_type', 'unknown')
        self.feature_importance = model_data.get('feature_importance')
        self.fitted = True
        
        print(f"✅ Model loaded from {filepath}")


class TraditionalNBAPredictor(NBAGamePredictor):
    """
    Baseline predictor using traditional stats and betting lines only.
    No narrative features.
    """
    
    def __init__(self):
        super().__init__(model_type='traditional')
    
    def create_traditional_features(self, game_data: Dict) -> np.ndarray:
        """
        Create traditional statistical features.
        
        Parameters
        ----------
        game_data : dict
            Game information with stats
        
        Returns
        -------
        features : np.ndarray
            Traditional feature vector
        """
        # For demo: simulate traditional features
        features = [
            game_data.get('home_win_pct', 0.5),
            game_data.get('away_win_pct', 0.5),
            game_data.get('home_ppg', 110.0) / 130.0,  # Normalize
            game_data.get('away_ppg', 110.0) / 130.0,
            game_data.get('home_def_rating', 110.0) / 130.0,
            game_data.get('away_def_rating', 110.0) / 130.0,
            1.0 if game_data.get('is_home') else 0.0,  # Home court
            game_data.get('rest_days_home', 2) / 7.0,
            game_data.get('rest_days_away', 2) / 7.0,
            game_data.get('betting_line', 0) / 20.0  # Normalize to -1 to 1
        ]
        
        return np.array(features)

