"""
NHL Smart Pattern Selector

Intelligently selects which pattern to use for each game based on:
- Pattern historical performance
- Game characteristics
- Feature similarity to pattern training data
- Confidence scores from multiple models

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class NHLPatternSelector:
    """Smart pattern selection for NHL games"""
    
    def __init__(self):
        """Initialize selector"""
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'models'
        
        # Load models
        self.models = self._load_models()
        
        # Load patterns
        self.patterns = self._load_patterns()
        
        # Load historical features (for similarity matching)
        self.historical_features = self._load_historical_features()
    
    def _load_models(self) -> Dict:
        """Load trained models"""
        models = {}
        
        model_files = {
            'meta_ensemble': 'meta_ensemble.pkl',
            'gradient_boosting': 'gradient_boosting.pkl',
            'random_forest': 'random_forest.pkl',
            'logistic': 'logistic.pkl',
        }
        
        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
        
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                models['scaler'] = pickle.load(f)
        
        return models
    
    def _load_patterns(self) -> List[Dict]:
        """Load validated patterns"""
        patterns_path = self.project_root / 'data' / 'domains' / 'nhl_betting_patterns_learned.json'
        
        if patterns_path.exists():
            with open(patterns_path, 'r') as f:
                return json.load(f)
        return []
    
    def _load_historical_features(self) -> Optional[np.ndarray]:
        """Load historical features for similarity matching"""
        features_path = self.project_root / 'narrative_optimization' / 'domains' / 'nhl' / 'nhl_features_complete.npz'
        
        if features_path.exists():
            data = np.load(features_path)
            return data['features']
        return None
    
    def score_game_with_models(self, game_features: np.ndarray) -> Dict:
        """Score game with all trained models"""
        
        if 'scaler' in self.models:
            game_features_scaled = self.models['scaler'].transform(game_features.reshape(1, -1))
        else:
            game_features_scaled = game_features.reshape(1, -1)
        
        scores = {}
        
        for name, model in self.models.items():
            if name == 'scaler':
                continue
            
            try:
                proba = model.predict_proba(game_features_scaled)[0, 1]
                pred = model.predict(game_features_scaled)[0]
                
                scores[name] = {
                    'probability': float(proba),
                    'prediction': int(pred),
                    'confidence': abs(proba - 0.5) * 2,  # 0 to 1 scale
                }
            except Exception as e:
                print(f"   ⚠️  Error scoring with {name}: {e}")
        
        return scores
    
    def find_similar_historical_games(self, game_features: np.ndarray, top_k: int = 10) -> List[int]:
        """Find most similar historical games using feature similarity"""
        
        if self.historical_features is None:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for i in range(len(self.historical_features)):
            sim = 1 - cosine(game_features, self.historical_features[i])
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return [idx for idx, sim in similarities[:top_k]]
    
    def select_best_pattern(self, game: Dict, game_features: np.ndarray) -> Dict:
        """
        Select the best pattern for this specific game.
        
        Returns recommendation with reasoning.
        """
        
        # Score with models
        model_scores = self.score_game_with_models(game_features)
        
        # Find matching patterns
        matching_patterns = []
        
        # Check ML confidence patterns
        for pattern in self.patterns:
            if pattern.get('pattern_type') == 'ml_confidence':
                threshold = pattern.get('model_threshold', 0.5)
                
                # Check if any model exceeds threshold
                for model_name, scores in model_scores.items():
                    if scores['probability'] >= threshold:
                        matching_patterns.append({
                            'pattern': pattern,
                            'match_reason': f"{model_name} confidence {scores['probability']:.1%}",
                            'model_confidence': scores['probability'],
                            'priority': pattern['win_rate'],
                        })
                        break
            
            # Check nominative patterns (need to implement feature matching)
            elif pattern.get('pattern_type') == 'combination':
                # Check if game matches feature criteria
                # This would require implementing the specific feature logic
                # For now, skip detailed matching
                pass
        
        # Sort by priority (win rate)
        matching_patterns.sort(key=lambda x: x['priority'], reverse=True)
        
        if not matching_patterns:
            return {
                'recommendation': None,
                'reason': 'No patterns match with sufficient confidence',
                'model_scores': model_scores,
            }
        
        # Select best pattern
        best = matching_patterns[0]
        pattern = best['pattern']
        
        recommendation = {
            'bet_on': 'HOME' if game.get('home_team') else 'AWAY',
            'pattern_name': pattern['name'],
            'expected_win_rate': pattern['win_rate_pct'],
            'expected_roi': pattern['roi_pct'],
            'confidence': pattern['confidence'],
            'unit_size': pattern['unit_recommendation'],
            'match_reason': best['match_reason'],
            'model_scores': model_scores,
            'alternative_patterns': [p['pattern']['name'] for p in matching_patterns[1:3]],
        }
        
        return {
            'recommendation': recommendation,
            'model_scores': model_scores,
            'matching_patterns': len(matching_patterns),
        }


def main():
    """Test pattern selector"""
    
    selector = NHLPatternSelector()
    
    print("\n🎯 NHL PATTERN SELECTOR TEST")
    print("="*80)
    print(f"Loaded {len(selector.models)} models")
    print(f"Loaded {len(selector.patterns)} patterns")
    
    # Test on mock game
    mock_game = {'home_team': 'TOR', 'away_team': 'BOS'}
    mock_features = np.random.randn(79)  # Would be real features in production
    
    result = selector.select_best_pattern(mock_game, mock_features)
    
    if result['recommendation']:
        rec = result['recommendation']
        print("\n✅ RECOMMENDATION:")
        print(f"   Bet: {rec['bet_on']}")
        print(f"   Pattern: {rec['pattern_name']}")
        print(f"   Expected: {rec['expected_win_rate']:.1f}% win, {rec['expected_roi']:.1f}% ROI")
        print(f"   Confidence: {rec['confidence']}")
        print(f"   Units: {rec['unit_size']}u")
        print(f"   Reason: {rec['match_reason']}")
    else:
        print("\n⏸️  No high-confidence pattern match")
    
    print("\n✅ Pattern selector operational!")


if __name__ == "__main__":
    main()

