"""
MLB Context-Specific Models
Separate models for high-journey contexts following transformer insights

Key Contexts (where journey completion > 15%):
1. September games (quest climax)
2. Playoff race games (high stakes)
3. Rivalry games (enhanced narrative)
4. Historic stadiums (nominative amplification)

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
from typing import Dict, List
from mlb_betting_model import MLBBettingModel
from mlb_feature_pipeline import MLBFeaturePipeline


class MLBContextModels:
    """Train separate models for high-journey contexts"""
    
    def __init__(self):
        self.models = {}
        self.pipeline = MLBFeaturePipeline()
        
        # Define contexts following transformer insights
        self.contexts = {
            'september': 'September games (quest climax stage)',
            'playoff_race': 'Teams within 5 games of wild card',
            'rivalry': 'Major rivalry matchups',
            'historic_stadium': 'Wrigley, Fenway, etc.',
            'high_journey': 'Journey completion > 20%'
        }
    
    def train_context_model(self, context_name: str, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str]) -> Dict:
        """
        Train model for specific context
        
        Returns:
            Performance metrics
        """
        print(f"\nTraining model for context: {context_name}")
        print(f"  Games in context: {len(X)}")
        
        model = MLBBettingModel()
        metrics = model.train(X, y, feature_names)
        
        self.models[context_name] = model
        
        return metrics
    
    def train_all_contexts(self, games: List[Dict], stats_dict: Dict, 
                          roster_dict: Dict) -> Dict:
        """
        Train models for all defined contexts
        
        Returns:
            Dictionary of context performances
        """
        results = {}
        
        # Extract all features first
        X_all, game_ids, feature_names = self.pipeline.extract_batch(games, stats_dict, roster_dict)
        y_all = np.array([1 if g.get('home_wins', False) else 0 for g in games])
        
        # September context
        september_mask = np.array([g.get('month', 0) == 9 for g in games])
        if september_mask.sum() > 100:
            metrics = self.train_context_model(
                'september',
                X_all[september_mask],
                y_all[september_mask],
                feature_names
            )
            results['september'] = {
                'games': september_mask.sum(),
                'accuracy': metrics['ensemble']['val_accuracy'],
                'auc': metrics['ensemble']['val_auc']
            }
        
        # Playoff race context
        playoff_race_mask = np.array([
            X_all[i, feature_names.index('in_playoff_race')] > 0 
            for i in range(len(X_all)) if 'in_playoff_race' in feature_names
        ] + [False] * (len(X_all) - len([i for i in range(len(X_all)) if 'in_playoff_race' in feature_names])))[:len(X_all)]
        
        if playoff_race_mask.sum() > 100:
            metrics = self.train_context_model(
                'playoff_race',
                X_all[playoff_race_mask],
                y_all[playoff_race_mask],
                feature_names
            )
            results['playoff_race'] = {
                'games': playoff_race_mask.sum(),
                'accuracy': metrics['ensemble']['val_accuracy'],
                'auc': metrics['ensemble']['val_auc']
            }
        
        # Rivalry context
        rivalry_mask = np.array([g.get('is_rivalry', False) for g in games])
        if rivalry_mask.sum() > 50:
            metrics = self.train_context_model(
                'rivalry',
                X_all[rivalry_mask],
                y_all[rivalry_mask],
                feature_names
            )
            results['rivalry'] = {
                'games': rivalry_mask.sum(),
                'accuracy': metrics['ensemble']['val_accuracy'],
                'auc': metrics['ensemble']['val_auc']
            }
        
        # High-journey context (transformer target)
        if 'high_journey_game' in feature_names:
            high_journey_mask = X_all[:, feature_names.index('high_journey_game')] > 0
            if high_journey_mask.sum() > 100:
                metrics = self.train_context_model(
                    'high_journey',
                    X_all[high_journey_mask],
                    y_all[high_journey_mask],
                    feature_names
                )
                results['high_journey'] = {
                    'games': int(high_journey_mask.sum()),
                    'accuracy': metrics['ensemble']['val_accuracy'],
                    'auc': metrics['ensemble']['val_auc'],
                    'note': 'TRANSFORMER TARGET - Journey completion > 15%'
                }
        
        return results
    
    def get_best_model_for_game(self, game_features: Dict) -> str:
        """
        Select best model for a game based on context
        
        Returns:
            Model name to use
        """
        # Check contexts in priority order
        if game_features.get('high_journey_game', 0) > 0 and 'high_journey' in self.models:
            return 'high_journey'
        
        if game_features.get('month', 0) == 9 and 'september' in self.models:
            return 'september'
        
        if game_features.get('is_rivalry', 0) > 0 and 'rivalry' in self.models:
            return 'rivalry'
        
        if game_features.get('in_playoff_race', 0) > 0 and 'playoff_race' in self.models:
            return 'playoff_race'
        
        # Default to general model
        return 'general'
    
    def predict_with_context(self, game_features: Dict) -> Dict:
        """
        Predict using best context-specific model
        
        Returns:
            Prediction with context information
        """
        best_context = self.get_best_model_for_game(game_features)
        
        if best_context in self.models:
            model = self.models[best_context]
            prediction = model.predict_game(game_features)
            prediction['context_model'] = best_context
            prediction['journey_optimized'] = True
        else:
            prediction = {
                'error': 'No model trained for this context',
                'context': best_context
            }
        
        return prediction


if __name__ == '__main__':
    print("MLB Context Models - Transformer-Guided Optimization")
    print("=" * 80)
    print("\nContexts defined:")
    
    context_models = MLBContextModels()
    for name, desc in context_models.contexts.items():
        print(f"  - {name}: {desc}")
    
    print("\nUse train_context_models.py to train all contexts")
    print("=" * 80)

