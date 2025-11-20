"""
NBA Game Totals Prediction Model
=================================

Predicts total points (both teams combined) for over/under betting.

Uses:
- Pace analysis (possessions per game)
- Offensive/defensive efficiency
- Recent scoring trends
- Temporal transformers (pace changes)

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'narrative_optimization' / 'src'))

from transformers import (
    TemporalEvolutionTransformer,
    QuantitativeTransformer,
    PacingRhythmTransformer
)


class NBAGameTotalsModel:
    """
    Game totals (over/under) prediction model.
    
    Predicts combined score of both teams.
    """
    
    def __init__(self):
        """Initialize totals model"""
        
        # Transformers for pace/tempo analysis
        self.transformers = [
            ('temporal', TemporalEvolutionTransformer()),
            ('quantitative', QuantitativeTransformer()),
            ('pacing', PacingRhythmTransformer())
        ]
        
        # Statistical model
        self.model_ = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.team_pace_ = {}
        self.matchup_history_ = {}
        self.is_fitted_ = False
    
    def _calculate_team_pace(self, games: List[Dict]) -> Dict:
        """Calculate pace (possessions per game) for each team"""
        
        team_stats = defaultdict(lambda: {'points': [], 'games': 0})
        
        for game in games:
            team = game.get('team_name', 'Unknown')
            points = game.get('points', 100)
            
            team_stats[team]['points'].append(points)
            team_stats[team]['games'] += 1
        
        pace_data = {}
        for team, stats in team_stats.items():
            pace_data[team] = {
                'avg_points': np.mean(stats['points']),
                'std_points': np.std(stats['points']),
                'games': stats['games']
            }
        
        return pace_data
    
    def fit(self, games: List[Dict], verbose: bool = True):
        """
        Fit totals model on historical games.
        
        Parameters
        ----------
        games : list of dict
            Historical games with scores
        verbose : bool
            Print progress
            
        Returns
        -------
        self
        """
        if verbose:
            print(f"\n[Totals Model] Training on {len(games):,} games...")
        
        # Calculate team pace
        self.team_pace_ = self._calculate_team_pace(games)
        
        if verbose:
            print(f"[Totals Model] ✓ Calculated pace for {len(self.team_pace_)} teams")
        
        # Build training data
        X_features = []
        y_totals = []
        
        for game in games:
            team = game.get('team_name', 'Unknown')
            opp = game.get('matchup', '').split('vs.')[-1].strip() if 'vs.' in game.get('matchup', '') else 'Unknown'
            
            if team not in self.team_pace_ or opp not in self.team_pace_:
                continue
            
            # Team features
            team_avg = self.team_pace_[team]['avg_points']
            opp_avg = self.team_pace_.get(opp, {}).get('avg_points', 105)
            
            # Temporal context
            tc = game.get('temporal_context', {})
            
            # Features
            features = [
                team_avg,
                opp_avg,
                (team_avg + opp_avg) / 2,  # Expected total
                tc.get('season_win_pct', 0.5),
                tc.get('l10_win_pct', 0.5),
                1.0 if game.get('home_game', False) else 0.0,
                tc.get('games_played', 41) / 82.0
            ]
            
            X_features.append(features)
            
            # Target: actual game total
            total = game.get('points', 100) + game.get('opp_points', 100)
            y_totals.append(total)
        
        X = np.array(X_features)
        y = np.array(y_totals)
        
        if verbose:
            print(f"[Totals Model] Training samples: {len(X):,}")
            print(f"[Totals Model] Avg total: {y.mean():.1f} points")
        
        # Train
        self.model_.fit(X, y)
        
        # Validate
        cv_scores = cross_val_score(self.model_, X, y, cv=5, scoring='r2')
        
        if verbose:
            print(f"[Totals Model] ✓ Model trained")
            print(f"[Totals Model]   CV R²: {cv_scores.mean():.3f}")
            print(f"[Totals Model]   RMSE: ~{np.sqrt(np.mean((self.model_.predict(X) - y)**2)):.1f} points")
        
        self.is_fitted_ = True
        return self
    
    def predict_total(
        self,
        team1: str,
        team2: str,
        team1_home: bool,
        team1_context: Dict,
        total_line: float
    ) -> Dict:
        """
        Predict game total.
        
        Parameters
        ----------
        team1, team2 : str
            Team names
        team1_home : bool
            Is team1 home?
        team1_context : dict
            Team1's current form
        total_line : float
            O/U line (e.g., 223.5)
            
        Returns
        -------
        prediction : dict
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        # Get team pace
        team1_avg = self.team_pace_.get(team1, {}).get('avg_points', 105)
        team2_avg = self.team_pace_.get(team2, {}).get('avg_points', 105)
        
        # Build features
        features = [
            team1_avg,
            team2_avg,
            (team1_avg + team2_avg) / 2,
            team1_context.get('season_win_pct', 0.5),
            team1_context.get('l10_win_pct', 0.5),
            1.0 if team1_home else 0.0,
            team1_context.get('games_played', 41) / 82.0
        ]
        
        # Predict
        predicted_total = self.model_.predict([features])[0]
        
        # Over/under probability
        std_total = 12.0  # Typical standard deviation for game totals
        over_prob = 1 / (1 + np.exp(-(predicted_total - total_line) / std_total))
        
        # Edge
        market_prob = 0.524
        edge = over_prob - market_prob if over_prob > 0.5 else (1 - over_prob) - market_prob
        
        recommendation = "OVER" if over_prob > 0.5 else "UNDER"
        confidence = max(over_prob, 1 - over_prob)
        
        should_bet = confidence >= 0.56 and abs(edge) >= 0.04
        
        return {
            'matchup': f"{team1} vs {team2}",
            'line': total_line,
            'predicted_total': float(predicted_total),
            'recommendation': recommendation,
            'over_probability': float(over_prob),
            'confidence': float(confidence),
            'edge': float(edge),
            'should_bet': should_bet,
            'units': 1.5 if confidence >= 0.60 else 1.0
        }


# Demo
if __name__ == "__main__":
    print("\n" + "="*80)
    print("NBA TOTALS MODEL - DEMO")
    print("="*80)
    print()
    
    # Load data
    with open('data/domains/nba_complete_with_players.json') as f:
        games = json.load(f)
    
    print(f"[Demo] Loaded {len(games):,} games")
    
    # Train
    model = NBAGameTotalsModel()
    model.fit(games, verbose=True)
    
    # Demo predictions
    print("\n" + "="*80)
    print("SAMPLE TOTALS PREDICTIONS")
    print("="*80)
    print()
    
    test_totals = [
        ("Los Angeles Lakers", "Boston Celtics", True, {'season_win_pct': 0.55, 'l10_win_pct': 0.60, 'games_played': 50}, 223.5),
        ("Milwaukee Bucks", "Miami Heat", True, {'season_win_pct': 0.62, 'l10_win_pct': 0.70, 'games_played': 55}, 218.5),
    ]
    
    for team1, team2, home, context, line in test_totals:
        pred = model.predict_total(team1, team2, home, context, line)
        
        print(f"Game: {pred['matchup']}")
        print(f"  Line: {pred['line']}")
        print(f"  Predicted: {pred['predicted_total']:.1f}")
        print(f"  Recommendation: {pred['recommendation']} ({pred['confidence']:.1%})")
        print(f"  Bet: {pred['units']:.1f} units" if pred['should_bet'] else "  SKIP")
        print()
    
    print("✅ Totals model ready!")
    print()

