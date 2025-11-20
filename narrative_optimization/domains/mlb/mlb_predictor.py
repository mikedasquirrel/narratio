"""
MLB Predictor - Production Interface
Loads trained model and provides predictions for web interface

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional

from mlb_betting_model import MLBBettingModel
from mlb_feature_pipeline import MLBFeaturePipeline
from mlb_betting_strategy import MLBBettingStrategy


class MLBPredictor:
    """Production predictor for MLB games"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(Path(__file__).parent / 'trained_models' / 'mlb_betting_model.pkl')
        self.config_path = Path(__file__).parent / 'trained_models' / 'deployment_config.json'
        
        self.model = None
        self.pipeline = MLBFeaturePipeline()
        self.config = None
        
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """Load trained model"""
        try:
            self.model = MLBBettingModel.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not load model - {e}")
            print("Run train_mlb_complete.py first to train the model")
    
    def _load_config(self):
        """Load deployment config"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
            print(f"✓ Config loaded - Model version {self.config['model_version']}")
        except Exception as e:
            print(f"Warning: Could not load config - {e}")
            self.config = {
                'model_version': 'unknown',
                'status': 'NOT_READY'
            }
    
    def predict_game(self, home_team: str, away_team: str, 
                     home_stats: Dict = None, away_stats: Dict = None,
                     home_roster: list = None, away_roster: list = None,
                     game_context: Dict = None) -> Dict:
        """
        Predict a game outcome
        
        Args:
            home_team: Home team code (e.g., 'NYY')
            away_team: Away team code (e.g., 'BOS')
            home_stats: Home team stats
            away_stats: Away team stats
            home_roster: Home team roster
            away_roster: Away team roster
            game_context: Additional context (venue, date, etc.)
            
        Returns:
            Prediction dictionary with probabilities and betting recommendation
        """
        if not self.model or not self.model.is_fitted:
            return {
                'error': 'Model not loaded or not trained',
                'message': 'Run train_mlb_complete.py to train the model'
            }
        
        # Build game dict
        game = {
            'game_id': 'prediction',
            'home_team': home_team,
            'away_team': away_team,
            'is_rivalry': self._check_rivalry(home_team, away_team),
            'is_historic_stadium': game_context.get('is_historic_stadium', False) if game_context else False,
            'month': game_context.get('month', 6) if game_context else 6,
            'venue': game_context.get('venue', 'Stadium') if game_context else 'Stadium',
            'home_pitcher': game_context.get('home_pitcher', '') if game_context else '',
            'away_pitcher': game_context.get('away_pitcher', '') if game_context else ''
        }
        
        # Use defaults if not provided
        if not home_stats:
            home_stats = {'wins': 81, 'losses': 81, 'win_pct': 0.500}
        if not away_stats:
            away_stats = {'wins': 81, 'losses': 81, 'win_pct': 0.500}
        if not home_roster:
            home_roster = self._generate_default_roster()
        if not away_roster:
            away_roster = self._generate_default_roster()
        
        # Extract features
        features = self.pipeline.extract_all_features(
            game, home_stats, away_stats, home_roster, away_roster
        )
        
        # Get prediction
        prediction = self.model.predict_game(features)
        
        # Add betting recommendation
        strategy = MLBBettingStrategy(
            bankroll=1000,
            kelly_fraction=0.25,
            min_edge=0.05
        )
        
        # Use default odds if not provided
        home_odds = game_context.get('home_odds', -110) if game_context else -110
        away_odds = game_context.get('away_odds', -110) if game_context else -110
        
        bet_rec = strategy.get_bet_recommendation(prediction, home_odds, away_odds)
        
        result = {
            'matchup': f"{away_team} @ {home_team}",
            'prediction': prediction,
            'betting_recommendation': bet_rec,
            'features_used': len(features),
            'model_version': self.config.get('model_version', 'unknown'),
            'nominative_features': {
                'total_players': features.get('total_players', 0),
                'home_international_names': features.get('home_international_names', 0),
                'away_international_names': features.get('away_international_names', 0)
            },
            'context': {
                'is_rivalry': game['is_rivalry'],
                'is_historic_stadium': game['is_historic_stadium']
            }
        }
        
        return result
    
    def _check_rivalry(self, team1: str, team2: str) -> bool:
        """Check if matchup is a rivalry"""
        rivalries = [
            ('NYY', 'BOS'), ('LAD', 'SF'), ('CHC', 'STL'), ('HOU', 'TEX'),
            ('NYM', 'PHI'), ('BAL', 'WSH'), ('OAK', 'SF'), ('CWS', 'CHC')
        ]
        return (team1, team2) in rivalries or (team2, team1) in rivalries
    
    def _generate_default_roster(self) -> list:
        """Generate default roster for demo"""
        roster = []
        for i in range(25):
            roster.append({
                'full_name': f'Player {i+1}',
                'position_code': 'P' if i < 10 else 'POS'
            })
        return roster
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'status': self.config.get('status', 'UNKNOWN'),
            'version': self.config.get('model_version', 'unknown'),
            'trained_date': self.config.get('trained_date', 'unknown'),
            'feature_count': self.config.get('feature_count', 0),
            'backtest_performance': self.config.get('backtest_performance', {}),
            'is_loaded': self.model is not None and self.model.is_fitted
        }


# Global predictor instance
_predictor = None

def get_predictor() -> MLBPredictor:
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = MLBPredictor()
    return _predictor


if __name__ == '__main__':
    # Test predictor
    print("=" * 80)
    print("MLB PREDICTOR TEST")
    print("=" * 80)
    
    predictor = MLBPredictor()
    
    # Get model info
    info = predictor.get_model_info()
    print("\nModel Info:")
    print(f"  Status: {info['status']}")
    print(f"  Version: {info['version']}")
    print(f"  Features: {info['feature_count']}")
    print(f"  Loaded: {info['is_loaded']}")
    
    if info['is_loaded']:
        print("\nBacktest Performance:")
        perf = info['backtest_performance']
        print(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
        print(f"  ROI: {perf.get('roi', 0):.1f}%")
        print(f"  Total Bets: {perf.get('total_bets', 0)}")
        
        # Test prediction
        print("\nTest Prediction:")
        print("  Matchup: NYY @ BOS")
        
        result = predictor.predict_game(
            'BOS', 'NYY',
            home_stats={'wins': 85, 'losses': 65, 'win_pct': 0.567},
            away_stats={'wins': 90, 'losses': 60, 'win_pct': 0.600},
            game_context={'is_rivalry': True, 'is_historic_stadium': True}
        )
        
        print(f"\n  Predicted Winner: {result['prediction']['predicted_winner']}")
        print(f"  Home Win Probability: {result['prediction']['home_win_probability']:.3f}")
        print(f"  Confidence: {result['prediction']['confidence']:.3f}")
        print(f"  Edge: {result['prediction']['edge']:.3f}")
        
        if result['betting_recommendation']:
            rec = result['betting_recommendation']
            print(f"\n  Betting Recommendation:")
            print(f"    Side: {rec['side']}")
            print(f"    Amount: ${rec['bet_amount']:.2f}")
            print(f"    Expected Value: ${rec['expected_value']:.2f}")
        else:
            print(f"\n  No bet recommended (insufficient edge)")
    
    print("\n" + "=" * 80)

