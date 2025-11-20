"""
Opponent Context Transformer

Captures how opponent's narrative situation affects the game.
Every story needs both protagonist and antagonist.

The opponent's desperation can be your danger.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class OpponentContextTransformer(BaseEstimator, TransformerMixin):
    """
    Extract opponent narrative context features.
    
    Philosophy:
    - Opponent's narrative needs affect game dynamics
    - Desperation creates dangerous opponents
    - Mutual narratives can amplify or cancel
    - Spoiler roles activate special energy
    - Asymmetric motivations create imbalance
    
    Features (25 total):
    - Opponent milestone proximities (5)
    - Opponent schedule situation (4)
    - Opponent emotional state (5)
    - Mutual narrative amplification (5)
    - Spoiler role activation (3)
    - Motivation asymmetry (3)
    """
    
    # Emotional states and their contagion effects
    EMOTIONAL_STATES = {
        'desperate': {'intensity': 0.9, 'contagion': 0.7},
        'confident': {'intensity': 0.7, 'contagion': 0.5},
        'frustrated': {'intensity': 0.8, 'contagion': 0.6},
        'motivated': {'intensity': 0.8, 'contagion': 0.4},
        'deflated': {'intensity': 0.6, 'contagion': 0.8},
        'vengeful': {'intensity': 0.9, 'contagion': 0.3},
        'relaxed': {'intensity': 0.4, 'contagion': 0.6}
    }
    
    # Spoiler role intensities
    SPOILER_SCENARIOS = {
        'playoff_elimination': 1.0,      # Can eliminate from playoffs
        'division_rival_spoil': 0.8,     # Hurt rival's chances
        'streak_breaker': 0.7,           # End opponent's streak
        'milestone_blocker': 0.6,        # Prevent milestone
        'pride_game': 0.5,               # Nothing to play for but pride
        'draft_position': 0.4            # Playing for draft position
    }
    
    def __init__(
        self,
        include_mutual_history: bool = True,
        emotional_contagion_weight: float = 0.7
    ):
        """
        Initialize opponent context analyzer.
        
        Parameters
        ----------
        include_mutual_history : bool
            Include shared narrative history
        emotional_contagion_weight : float
            How much opponent emotions affect game
        """
        self.include_mutual_history = include_mutual_history
        self.emotional_contagion_weight = emotional_contagion_weight
        
    def fit(self, X, y=None):
        """No fitting required for opponent analysis."""
        return self
        
    def transform(self, X):
        """
        Extract opponent context features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with opponent information
            
        Returns
        -------
        np.ndarray
            Opponent features (n_samples, 25)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Opponent milestones (5)
            milestone_features = self._extract_opponent_milestones(item)
            feature_vec.extend(milestone_features)
            
            # Opponent schedule (4)
            schedule_features = self._extract_opponent_schedule(item)
            feature_vec.extend(schedule_features)
            
            # Opponent emotional state (5)
            emotional_features = self._extract_opponent_emotions(item)
            feature_vec.extend(emotional_features)
            
            # Mutual narratives (5)
            mutual_features = self._extract_mutual_narratives(item)
            feature_vec.extend(mutual_features)
            
            # Spoiler role (3)
            spoiler_features = self._extract_spoiler_dynamics(item)
            feature_vec.extend(spoiler_features)
            
            # Motivation asymmetry (3)
            asymmetry_features = self._extract_motivation_asymmetry(item)
            feature_vec.extend(asymmetry_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_opponent_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract opponent milestone proximity features.
        
        Returns 5 features for opponent milestones.
        """
        features = []
        
        # Opponent approaching milestone
        opp_milestone_proximity = item.get('opponent_milestone_proximity', 0)
        if opp_milestone_proximity > 0 and opp_milestone_proximity <= 5:
            features.append(1.0 - opp_milestone_proximity / 5.0)
        else:
            features.append(0.0)
            
        # Type of milestone
        opp_milestone_type = item.get('opponent_milestone_type', 'none')
        
        milestone_weights = {
            'career_games': 0.7,
            'career_points': 0.8,
            'career_goals': 0.85,
            'career_wins': 0.75,
            'team_record': 0.9,
            'none': 0.0
        }
        
        features.append(milestone_weights.get(opp_milestone_type, 0.5))
        
        # Multiple opponent milestones
        opp_milestone_count = item.get('opponent_milestone_count', 0)
        features.append(min(1.0, opp_milestone_count * 0.3))
        
        # Opponent's first (debut narratives)
        opp_player_debut = item.get('opponent_player_debut', False)
        opp_coach_debut = item.get('opponent_coach_debut', False)
        
        debut_factor = 0.0
        if opp_player_debut:
            debut_factor += 0.7
        if opp_coach_debut:
            debut_factor += 0.8
            
        features.append(min(1.0, debut_factor))
        
        # Opponent streak at stake
        opp_win_streak = item.get('opponent_win_streak', 0)
        if opp_win_streak >= 5:
            streak_pressure = min(1.0, (opp_win_streak - 4) / 6.0)
            features.append(streak_pressure)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_opponent_schedule(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract opponent schedule situation features.
        
        Returns 4 features for opponent's schedule context.
        """
        features = []
        
        # Opponent rest advantage/disadvantage
        our_rest = item.get('days_since_last_game', 2)
        opp_rest = item.get('opponent_days_rest', 2)
        
        rest_differential = opp_rest - our_rest
        if rest_differential > 2:
            features.append(0.8)  # They're more rested
        elif rest_differential < -2:
            features.append(-0.8)  # We're more rested
        else:
            features.append(rest_differential / 5.0)
            
        # Opponent back-to-back
        opp_b2b = item.get('opponent_back_to_back', False)
        features.append(0.7 if opp_b2b else 0.0)
        
        # Opponent schedule density
        opp_games_week = item.get('opponent_games_past_week', 3)
        if opp_games_week >= 4:
            density_fatigue = min(1.0, (opp_games_week - 3) / 3.0)
            features.append(density_fatigue)
        else:
            features.append(0.0)
            
        # Opponent travel situation
        opp_road_trip_game = item.get('opponent_road_trip_game', 0)
        if opp_road_trip_game >= 4:
            travel_fatigue = min(1.0, (opp_road_trip_game - 3) / 3.0)
            features.append(travel_fatigue)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_opponent_emotions(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract opponent emotional state features.
        
        Returns 5 features for opponent psychology.
        """
        features = []
        
        # Recent opponent results
        opp_last_5_record = item.get('opponent_last_5_games', '2-3-0')
        
        # Parse record
        try:
            parts = opp_last_5_record.split('-')
            wins = int(parts[0])
            losses = int(parts[1]) if len(parts) > 1 else 0
        except:
            wins, losses = 2, 3
            
        # Momentum indicator
        if wins >= 4:
            features.append(0.8)  # Hot opponent
        elif wins <= 1:
            features.append(0.7)  # Desperate opponent
        else:
            features.append(0.3)  # Neutral
            
        # Opponent emotional state
        opp_emotional_state = item.get('opponent_emotional_state', 'neutral')
        
        if opp_emotional_state in self.EMOTIONAL_STATES:
            state_info = self.EMOTIONAL_STATES[opp_emotional_state]
            features.append(state_info['intensity'])
            features.append(state_info['contagion'] * self.emotional_contagion_weight)
        else:
            features.extend([0.5, 0.5])
            
        # Opponent coming off big win/loss
        opp_last_result = item.get('opponent_last_game_result', 'normal')
        
        result_effects = {
            'blowout_win': 0.7,    # Confident
            'comeback_win': 0.8,   # Momentum
            'blowout_loss': 0.9,   # Angry/desperate
            'collapse_loss': 0.85, # Frustrated
            'overtime_loss': 0.6,  # Disappointed
            'normal': 0.3
        }
        
        features.append(result_effects.get(opp_last_result, 0.5))
        
        # Opponent injury situation
        opp_key_injuries = item.get('opponent_key_injuries', 0)
        if opp_key_injuries >= 2:
            features.append(min(1.0, opp_key_injuries * 0.3))
        else:
            features.append(opp_key_injuries * 0.2)
            
        return features
        
    def _extract_mutual_narratives(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract mutual narrative amplification features.
        
        Returns 5 features for shared storylines.
        """
        features = []
        
        # Revenge game for both teams
        mutual_revenge = (item.get('revenge_game', False) and 
                         item.get('opponent_revenge_game', False))
        features.append(0.9 if mutual_revenge else 0.0)
        
        # Both teams need win (playoff race)
        our_must_win = item.get('must_win', False)
        opp_must_win = item.get('opponent_must_win', False)
        
        if our_must_win and opp_must_win:
            features.append(1.0)  # Maximum intensity
        elif our_must_win or opp_must_win:
            features.append(0.6)  # Asymmetric pressure
        else:
            features.append(0.0)
            
        # Historical rivalry intensity
        if self.include_mutual_history:
            rivalry_score = item.get('historical_rivalry_score', 0.5)
            recent_controversy = item.get('recent_controversy', False)
            
            if recent_controversy:
                rivalry_score = min(1.0, rivalry_score + 0.3)
                
            features.append(rivalry_score)
        else:
            features.append(0.5)
            
        # Trade/revenge narrative
        players_facing_former = item.get('players_facing_former_team', 0)
        features.append(min(1.0, players_facing_former * 0.4))
        
        # Narrative collision score
        our_narratives = item.get('active_narrative_count', 0)
        opp_narratives = item.get('opponent_narrative_count', 0)
        
        if our_narratives > 0 and opp_narratives > 0:
            collision = min(our_narratives, opp_narratives) / max(our_narratives, opp_narratives)
            features.append(collision)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_spoiler_dynamics(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract spoiler role activation features.
        
        Returns 3 features for spoiler narratives.
        """
        features = []
        
        # Check if opponent is eliminated
        opp_eliminated = item.get('opponent_eliminated', False)
        opp_games_since_elimination = item.get('opponent_games_since_elimination', 0)
        
        if opp_eliminated:
            # Can they spoil our playoffs?
            our_playoff_implications = item.get('playoff_implications', False)
            
            spoiler_intensity = 0.0
            
            if our_playoff_implications:
                spoiler_intensity = self.SPOILER_SCENARIOS['playoff_elimination']
            elif item.get('division_rival', False):
                spoiler_intensity = self.SPOILER_SCENARIOS['division_rival_spoil']
            elif item.get('streak_on_line', False):
                spoiler_intensity = self.SPOILER_SCENARIOS['streak_breaker']
            else:
                spoiler_intensity = self.SPOILER_SCENARIOS['pride_game']
                
            # Fresh elimination anger
            if opp_games_since_elimination <= 3:
                spoiler_intensity *= 1.2
                
            features.append(min(1.0, spoiler_intensity))
        else:
            features.append(0.0)
            
        # Reverse spoiler (we can spoil them)
        we_eliminated = item.get('team_eliminated', False)
        they_need_win = item.get('opponent_must_win', False)
        
        if we_eliminated and they_need_win:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Nothing to lose vs everything to lose
        opp_pressure = item.get('opponent_pressure_level', 0.5)
        our_pressure = item.get('team_pressure_level', 0.5)
        
        if opp_pressure < 0.3 and our_pressure > 0.7:
            features.append(0.7)  # Dangerous dynamic
        else:
            features.append(0.0)
            
        return features
        
    def _extract_motivation_asymmetry(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract motivation asymmetry features.
        
        Returns 3 features for differential motivation.
        """
        features = []
        
        # Calculate motivation scores
        our_motivation = 0.0
        opp_motivation = 0.0
        
        # Our motivators
        if item.get('must_win', False):
            our_motivation += 0.4
        if item.get('revenge_game', False):
            our_motivation += 0.3
        if item.get('milestone_approaching', False):
            our_motivation += 0.2
        if item.get('home_finale', False):
            our_motivation += 0.2
            
        # Opponent motivators
        if item.get('opponent_must_win', False):
            opp_motivation += 0.4
        if item.get('opponent_revenge_game', False):
            opp_motivation += 0.3
        if item.get('opponent_milestone_approaching', False):
            opp_motivation += 0.2
        if item.get('opponent_home_finale', False):
            opp_motivation += 0.2
            
        # Motivation differential
        motivation_diff = our_motivation - opp_motivation
        features.append(np.tanh(motivation_diff * 2))  # -1 to 1
        
        # Extreme asymmetry
        if abs(motivation_diff) > 0.6:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Meaning mismatch
        game_meaning_us = item.get('game_importance_team', 0.5)
        game_meaning_opp = item.get('game_importance_opponent', 0.5)
        
        meaning_diff = abs(game_meaning_us - game_meaning_opp)
        features.append(meaning_diff)
        
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Opponent milestone features
        names.extend([
            'opp_milestone_proximity',
            'opp_milestone_weight',
            'opp_multiple_milestones',
            'opp_debut_narrative',
            'opp_streak_pressure'
        ])
        
        # Opponent schedule features
        names.extend([
            'opp_rest_differential',
            'opp_back_to_back',
            'opp_schedule_density',
            'opp_travel_fatigue'
        ])
        
        # Opponent emotional features
        names.extend([
            'opp_momentum_state',
            'opp_emotional_intensity',
            'opp_emotional_contagion',
            'opp_last_game_effect',
            'opp_injury_desperation'
        ])
        
        # Mutual narrative features
        names.extend([
            'mutual_revenge_game',
            'mutual_must_win',
            'mutual_rivalry_intensity',
            'mutual_trade_narrative',
            'mutual_narrative_collision'
        ])
        
        # Spoiler features
        names.extend([
            'spoiler_activation',
            'spoiler_reverse',
            'spoiler_pressure_mismatch'
        ])
        
        # Asymmetry features
        names.extend([
            'motivation_differential',
            'motivation_extreme_asymmetry',
            'motivation_meaning_mismatch'
        ])
        
        return names
