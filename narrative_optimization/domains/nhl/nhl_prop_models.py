"""
NHL Prop Betting Models

Trains and manages models for NHL player prop predictions:
- Goals (over/under 0.5, 1.5)
- Assists (over/under 0.5, 1.5)  
- Shots (over/under lines)
- Points (over/under 0.5, 1.5, 2.5)
- Saves (goalie props)

Each prop type has its own ensemble model combining:
1. Logistic Regression (interpretable baseline)
2. Gradient Boosting (captures non-linear patterns)
3. Neural Network (deep narrative interactions)

Models are trained on player narrative features + game context.

Author: Prop Betting Model System
Date: November 20, 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


class NHLPropModel:
    """
    Base class for NHL prop betting models.
    
    Each prop type (goals, assists, etc.) gets its own instance.
    """
    
    def __init__(self, prop_type: str, line: float):
        """
        Parameters
        ----------
        prop_type : str
            Type of prop (goals, assists, shots, points, saves)
        line : float
            The line to predict over/under for (0.5, 1.5, etc.)
        """
        self.prop_type = prop_type
        self.line = line
        self.model_name = f"{prop_type}_o{line}"
        
        # Initialize models
        self.models = {
            'logistic': LogisticRegression(
                max_iter=1000,
                C=0.5,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'neural_net': MLPClassifier(
                hidden_layer_sizes=(50, 30),
                activation='relu',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        # Model weights (will be optimized during training)
        self.weights = {
            'logistic': 0.3,
            'gradient_boost': 0.4,
            'neural_net': 0.3
        }
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        
    def prepare_training_data(self, player_features: np.ndarray, 
                            game_results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        Parameters
        ----------
        player_features : np.ndarray
            Narrative features from NHLPlayerPerformanceTransformer
        game_results : list of dict
            Actual game results with player stats
            
        Returns
        -------
        X : np.ndarray
            Training features
        y : np.ndarray
            Binary labels (1 = over, 0 = under)
        """
        # Extract actual prop results
        y = []
        
        for result in game_results:
            if self.prop_type == 'goals':
                actual = result.get('goals', 0)
            elif self.prop_type == 'assists':
                actual = result.get('assists', 0)
            elif self.prop_type == 'shots':
                actual = result.get('shots', 0)
            elif self.prop_type == 'points':
                actual = result.get('goals', 0) + result.get('assists', 0)
            elif self.prop_type == 'saves':
                actual = result.get('saves', 0)
            else:
                raise ValueError(f"Unknown prop type: {self.prop_type}")
                
            # Binary classification: over = 1, under = 0
            y.append(1 if actual > self.line else 0)
            
        return player_features, np.array(y)
        
    def train(self, X: np.ndarray, y: np.ndarray, optimize_weights: bool = True):
        """
        Train the ensemble model.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Binary labels
        optimize_weights : bool
            Whether to optimize ensemble weights
        """
        print(f"\nTraining {self.model_name} model...")
        print(f"Data shape: {X.shape}")
        print(f"Label distribution: {np.mean(y):.3f} over rate")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train each model
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            
            val_scores[name] = {
                'accuracy': acc,
                'auc': auc,
                'log_loss': log_loss(y_val, y_proba)
            }
            
            print(f"    Accuracy: {acc:.3f}")
            print(f"    AUC: {auc:.3f}")
            
        # Optimize weights if requested
        if optimize_weights:
            self._optimize_weights(X_val_scaled, y_val)
            
        # Calculate feature importance
        self._calculate_feature_importance()
        
        self.is_fitted = True
        
        print(f"\n✓ {self.model_name} training complete")
        print(f"  Final weights: {self.weights}")
        
    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Optimize ensemble weights using validation set"""
        from scipy.optimize import minimize
        
        def ensemble_loss(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Get predictions from each model
            ensemble_proba = np.zeros(len(y_val))
            
            for i, (name, model) in enumerate(self.models.items()):
                proba = model.predict_proba(X_val)[:, 1]
                ensemble_proba += weights[i] * proba
                
            # Return log loss
            return log_loss(y_val, ensemble_proba)
            
        # Initial weights
        x0 = np.array([1/3, 1/3, 1/3])
        
        # Optimize
        result = minimize(
            ensemble_loss,
            x0,
            method='SLSQP',
            bounds=[(0, 1)] * 3,
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
            self.weights = {
                'logistic': optimal_weights[0],
                'gradient_boost': optimal_weights[1],
                'neural_net': optimal_weights[2]
            }
            
    def _calculate_feature_importance(self):
        """Calculate feature importance from gradient boosting model"""
        gb_model = self.models['gradient_boost']
        
        if hasattr(gb_model, 'feature_importances_'):
            self.feature_importance = gb_model.feature_importances_
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of going over the line.
        
        Parameters
        ----------
        X : np.ndarray
            Player narrative features
            
        Returns
        -------
        proba : np.ndarray
            Probability of going over the line
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        ensemble_proba = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.weights[name]
            proba = model.predict_proba(X_scaled)[:, 1]
            ensemble_proba += weight * proba
            
        return ensemble_proba
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_proba = self.predict_proba(X)
        y_pred = (y_proba > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'auc': roc_auc_score(y, y_proba),
            'log_loss': log_loss(y, y_proba),
            'over_rate_actual': np.mean(y),
            'over_rate_predicted': np.mean(y_pred)
        }
        
    def save(self, model_dir: Path):
        """Save model to disk"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, model_dir / f"{self.model_name}_{name}.pkl")
            
        # Save scaler
        joblib.dump(self.scaler, model_dir / f"{self.model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'prop_type': self.prop_type,
            'line': self.line,
            'weights': self.weights,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(model_dir / f"{self.model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load(self, model_dir: Path):
        """Load model from disk"""
        model_dir = Path(model_dir)
        
        # Load models
        for name in self.models:
            self.models[name] = joblib.load(model_dir / f"{self.model_name}_{name}.pkl")
            
        # Load scaler
        self.scaler = joblib.load(model_dir / f"{self.model_name}_scaler.pkl")
        
        # Load metadata
        with open(model_dir / f"{self.model_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        self.weights = metadata['weights']
        self.is_fitted = metadata['is_fitted']


class NHLPropModelSuite:
    """
    Complete suite of prop models for NHL betting.
    
    Manages training and prediction for all prop types and lines.
    """
    
    def __init__(self):
        """Initialize model suite"""
        # Define all prop models
        self.prop_configs = [
            # Goals
            ('goals', 0.5),
            ('goals', 1.5),
            ('goals', 2.5),
            
            # Assists
            ('assists', 0.5),
            ('assists', 1.5),
            
            # Shots
            ('shots', 2.5),
            ('shots', 3.5),
            ('shots', 4.5),
            
            # Points
            ('points', 0.5),
            ('points', 1.5),
            ('points', 2.5),
            
            # Saves (goalie)
            ('saves', 25.5),
            ('saves', 30.5),
            ('saves', 35.5),
        ]
        
        # Initialize models
        self.models = {}
        for prop_type, line in self.prop_configs:
            model_key = f"{prop_type}_o{line}"
            self.models[model_key] = NHLPropModel(prop_type, line)
            
    def train_all_models(self, training_data: Dict):
        """
        Train all prop models.
        
        Parameters
        ----------
        training_data : dict
            {
                'features': np.ndarray,  # Player narrative features
                'results': List[Dict],   # Game results
                'player_ids': List[int], # Player IDs for tracking
            }
        """
        print("\nNHL PROP MODEL TRAINING")
        print("=" * 80)
        
        features = training_data['features']
        results = training_data['results']
        
        # Train each model
        training_summary = {}
        
        for prop_type, line in self.prop_configs:
            model_key = f"{prop_type}_o{line}"
            model = self.models[model_key]
            
            # Skip goalie props for skaters
            if prop_type == 'saves':
                # Filter to goalies only
                goalie_mask = [r.get('position') == 'G' for r in results]
                if sum(goalie_mask) < 100:
                    print(f"\nSkipping {model_key} - insufficient goalie data")
                    continue
                    
                goalie_features = features[goalie_mask]
                goalie_results = [r for r, m in zip(results, goalie_mask) if m]
                
                X, y = model.prepare_training_data(goalie_features, goalie_results)
            else:
                # Filter out goalies for skater props
                skater_mask = [r.get('position') != 'G' for r in results]
                skater_features = features[skater_mask]
                skater_results = [r for r, m in zip(results, skater_mask) if m]
                
                X, y = model.prepare_training_data(skater_features, skater_results)
                
            # Only train if we have enough data
            if len(y) < 500:
                print(f"\nSkipping {model_key} - insufficient data ({len(y)} samples)")
                continue
                
            # Train model
            model.train(X, y)
            
            # Evaluate
            metrics = model.evaluate(X, y)
            training_summary[model_key] = metrics
            
        # Display summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        
        for model_key, metrics in training_summary.items():
            print(f"\n{model_key}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Over rate: {metrics['over_rate_actual']:.3f} actual, "
                  f"{metrics['over_rate_predicted']:.3f} predicted")
                  
        return training_summary
        
    def predict_props(self, player_features: np.ndarray, 
                     player_info: List[Dict]) -> List[Dict]:
        """
        Generate prop predictions for players.
        
        Parameters
        ----------
        player_features : np.ndarray
            Narrative features for each player
        player_info : list of dict
            Player metadata (name, position, etc.)
            
        Returns
        -------
        predictions : list of dict
            Prop predictions with probabilities
        """
        predictions = []
        
        for i, (features, info) in enumerate(zip(player_features, player_info)):
            player_name = info['player_name']
            position = info['position']
            
            # Skip inappropriate props
            if position == 'G':
                # Goalies only get save props
                prop_types = [('saves', line) for _, line in self.prop_configs 
                             if _ == 'saves']
            else:
                # Skaters get all except saves
                prop_types = [(t, l) for t, l in self.prop_configs 
                             if t != 'saves']
                             
            # Generate predictions
            for prop_type, line in prop_types:
                model_key = f"{prop_type}_o{line}"
                
                if model_key in self.models and self.models[model_key].is_fitted:
                    model = self.models[model_key]
                    
                    # Predict probability of going over
                    prob_over = model.predict_proba(features.reshape(1, -1))[0]
                    
                    predictions.append({
                        'player_name': player_name,
                        'player_id': info.get('player_id'),
                        'position': position,
                        'prop_type': prop_type,
                        'line': line,
                        'prob_over': float(prob_over),
                        'prob_under': float(1 - prob_over),
                        'model': model_key,
                        'confidence': abs(prob_over - 0.5) * 2,  # 0-1 scale
                    })
                    
        return predictions
        
    def calculate_edges(self, predictions: List[Dict], 
                       prop_odds: List[Dict]) -> List[Dict]:
        """
        Calculate betting edges vs book odds.
        
        Parameters
        ----------
        predictions : list of dict
            Model predictions
        prop_odds : list of dict
            Current prop odds from books
            
        Returns
        -------
        edges : list of dict
            Predictions with calculated edges
        """
        # Create lookup for odds
        odds_lookup = {}
        for prop in prop_odds:
            key = (
                prop['player_name'],
                prop['market'].replace('player_', '').replace('_over_under', ''),
                prop['line'],
                prop['side']
            )
            odds_lookup[key] = prop
            
        # Calculate edges
        edges = []
        
        for pred in predictions:
            # Look up odds for over
            over_key = (
                pred['player_name'],
                pred['prop_type'],
                pred['line'],
                'over'
            )
            
            under_key = (
                pred['player_name'],
                pred['prop_type'],
                pred['line'],
                'under'
            )
            
            over_odds = odds_lookup.get(over_key)
            under_odds = odds_lookup.get(under_key)
            
            if over_odds and under_odds:
                # Calculate implied probabilities
                over_implied = self._american_to_probability(over_odds['odds'])
                under_implied = self._american_to_probability(under_odds['odds'])
                
                # Calculate edges
                over_edge = pred['prob_over'] - over_implied
                under_edge = pred['prob_under'] - under_implied
                
                # Determine best side
                if over_edge > under_edge and over_edge > 0:
                    best_side = 'over'
                    edge = over_edge
                    odds = over_odds['odds']
                    implied = over_implied
                elif under_edge > 0:
                    best_side = 'under'
                    edge = under_edge
                    odds = under_odds['odds']
                    implied = under_implied
                else:
                    continue  # No edge
                    
                edges.append({
                    **pred,
                    'side': best_side,
                    'odds': odds,
                    'implied_prob': implied,
                    'edge': edge,
                    'edge_pct': edge * 100,
                    'bookmaker': over_odds['bookmaker'],
                    'expected_value': edge * self._calculate_payout(odds),
                })
                
        # Sort by edge
        edges.sort(key=lambda x: x['edge'], reverse=True)
        
        return edges
        
    def _american_to_probability(self, odds: int) -> float:
        """Convert American odds to probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
            
    def _calculate_payout(self, odds: int) -> float:
        """Calculate payout multiplier for American odds"""
        if odds > 0:
            return odds / 100
        else:
            return 100 / abs(odds)
            
    def save_all_models(self, model_dir: Path):
        """Save all trained models"""
        model_dir = Path(model_dir)
        
        for model_key, model in self.models.items():
            if model.is_fitted:
                model.save(model_dir)
                
        print(f"\n✓ Saved all models to {model_dir}")
        
    def load_all_models(self, model_dir: Path):
        """Load all models"""
        model_dir = Path(model_dir)
        
        for model_key, model in self.models.items():
            model_path = model_dir / f"{model_key}_metadata.json"
            
            if model_path.exists():
                model.load(model_dir)
                
        print(f"\n✓ Loaded models from {model_dir}")


def test_prop_models():
    """Test prop model training with synthetic data"""
    print("NHL PROP MODELS TEST")
    print("=" * 80)
    
    # Create synthetic training data
    n_samples = 1000
    n_features = 35  # From player transformer
    
    # Random features
    features = np.random.randn(n_samples, n_features)
    
    # Synthetic results (correlated with first few features)
    results = []
    for i in range(n_samples):
        # Goals correlate with offensive features
        goal_prob = 1 / (1 + np.exp(-features[i, 0]))
        goals = np.random.binomial(3, goal_prob * 0.3)
        
        # Assists correlate with different features
        assist_prob = 1 / (1 + np.exp(-features[i, 1]))
        assists = np.random.binomial(4, assist_prob * 0.25)
        
        # Shots
        shot_mean = 3 + features[i, 2] * 1.5
        shots = max(0, int(np.random.normal(shot_mean, 1.5)))
        
        results.append({
            'player_id': i,
            'goals': goals,
            'assists': assists,
            'shots': shots,
            'points': goals + assists,
            'position': 'C' if i % 10 != 0 else 'G',
            'saves': np.random.poisson(30) if i % 10 == 0 else 0,
        })
        
    # Create model suite
    suite = NHLPropModelSuite()
    
    # Train models
    training_data = {
        'features': features,
        'results': results,
        'player_ids': list(range(n_samples)),
    }
    
    summary = suite.train_all_models(training_data)
    
    # Test prediction
    print("\n" + "=" * 80)
    print("TEST PREDICTIONS")
    print("=" * 80)
    
    test_features = np.random.randn(5, n_features)
    test_players = [
        {'player_name': 'Test Player 1', 'player_id': 1001, 'position': 'C'},
        {'player_name': 'Test Player 2', 'player_id': 1002, 'position': 'LW'},
        {'player_name': 'Test Player 3', 'player_id': 1003, 'position': 'D'},
        {'player_name': 'Test Player 4', 'player_id': 1004, 'position': 'RW'},
        {'player_name': 'Test Goalie', 'player_id': 1005, 'position': 'G'},
    ]
    
    predictions = suite.predict_props(test_features, test_players)
    
    # Show sample predictions
    for player in test_players:
        print(f"\n{player['player_name']} ({player['position']}):")
        
        player_preds = [p for p in predictions if p['player_name'] == player['player_name']]
        
        # Show top props by confidence
        player_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        for pred in player_preds[:3]:
            print(f"  {pred['prop_type']} o{pred['line']}: "
                  f"{pred['prob_over']:.3f} over "
                  f"(confidence: {pred['confidence']:.3f})")
            

if __name__ == "__main__":
    test_prop_models()
