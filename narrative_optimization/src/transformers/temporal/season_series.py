"""
Season Series Narrative Transformer

Tracks narrative evolution within season series between teams.
Each meeting writes a new chapter in their annual story.

Familiarity breeds contempt, but also understanding.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class SeasonSeriesNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract season series narrative evolution features.
    
    Philosophy:
    - First meetings have uncertainty
    - Middle meetings build patterns
    - Final meetings carry weight
    - Previous results create momentum
    - Venue patterns matter
    
    Features (20 total):
    - Meeting number effects (4)
    - Previous meeting outcomes (4)
    - Venue pattern effects (3)
    - Rivalry intensification (4)
    - Season series stakes (3)
    - Pattern recognition (2)
    """
    
    # Meeting narrative weights
    MEETING_NARRATIVES = {
        1: {'uncertainty': 1.0, 'intensity': 0.7, 'type': 'discovery'},
        2: {'uncertainty': 0.6, 'intensity': 0.8, 'type': 'adjustment'},
        3: {'uncertainty': 0.4, 'intensity': 0.85, 'type': 'pattern'},
        4: {'uncertainty': 0.3, 'intensity': 0.9, 'type': 'culmination'},
        5: {'uncertainty': 0.5, 'intensity': 0.95, 'type': 'decider'},
        6: {'uncertainty': 0.7, 'intensity': 0.8, 'type': 'fatigue'}
    }
    
    # Series momentum patterns
    MOMENTUM_PATTERNS = {
        'sweep': {'momentum': 1.0, 'pressure': 0.9},
        'dominant': {'momentum': 0.8, 'pressure': 0.7},
        'split': {'momentum': 0.0, 'pressure': 0.5},
        'comeback': {'momentum': 0.7, 'pressure': 0.8},
        'alternating': {'momentum': 0.3, 'pressure': 0.6}
    }
    
    def __init__(
        self,
        include_goal_differential: bool = True,
        venue_memory_weight: float = 0.8
    ):
        """
        Initialize season series analyzer.
        
        Parameters
        ----------
        include_goal_differential : bool
            Track cumulative goal differential
        venue_memory_weight : float
            How much venue history matters
        """
        self.include_goal_differential = include_goal_differential
        self.venue_memory_weight = venue_memory_weight
        
    def fit(self, X, y=None):
        """No fitting required for series analysis."""
        return self
        
    def transform(self, X):
        """
        Extract season series features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with series context
            
        Returns
        -------
        np.ndarray
            Series features (n_samples, 20)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Meeting number effects (4)
            meeting_features = self._extract_meeting_effects(item)
            feature_vec.extend(meeting_features)
            
            # Previous outcomes (4)
            outcome_features = self._extract_previous_outcomes(item)
            feature_vec.extend(outcome_features)
            
            # Venue patterns (3)
            venue_features = self._extract_venue_patterns(item)
            feature_vec.extend(venue_features)
            
            # Rivalry intensification (4)
            rivalry_features = self._extract_rivalry_intensification(item)
            feature_vec.extend(rivalry_features)
            
            # Series stakes (3)
            stakes_features = self._extract_series_stakes(item)
            feature_vec.extend(stakes_features)
            
            # Pattern recognition (2)
            pattern_features = self._extract_pattern_recognition(item)
            feature_vec.extend(pattern_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_meeting_effects(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract meeting number narrative effects.
        
        Returns 4 features for meeting position.
        """
        features = []
        
        meeting_number = item.get('season_series_meeting', 1)
        total_meetings = item.get('season_series_total', 4)
        
        # Get meeting narrative
        if meeting_number in self.MEETING_NARRATIVES:
            meeting_info = self.MEETING_NARRATIVES[meeting_number]
        else:
            # Default for 6+ meetings
            meeting_info = self.MEETING_NARRATIVES[6]
            
        features.append(meeting_info['uncertainty'])
        features.append(meeting_info['intensity'])
        
        # Final meeting indicator
        if meeting_number == total_meetings:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Series position (early/middle/late)
        if total_meetings > 0:
            position = meeting_number / total_meetings
            
            if position <= 0.33:
                features.append(0.7)  # Early series
            elif position <= 0.67:
                features.append(0.5)  # Mid series
            else:
                features.append(0.9)  # Late series
        else:
            features.append(0.5)
            
        return features
        
    def _extract_previous_outcomes(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract features from previous meeting outcomes.
        
        Returns 4 features for series momentum.
        """
        features = []
        
        # Series record so far
        series_record = item.get('season_series_record', '0-0-0')
        
        try:
            parts = series_record.split('-')
            wins = int(parts[0])
            losses = int(parts[1])
            ot_losses = int(parts[2]) if len(parts) > 2 else 0
        except:
            wins, losses, ot_losses = 0, 0, 0
            
        games_played = wins + losses + ot_losses
        
        # Win percentage in series
        if games_played > 0:
            win_pct = (wins + 0.5 * ot_losses) / games_played
            features.append(win_pct)
        else:
            features.append(0.5)
            
        # Momentum pattern
        last_meeting_result = item.get('last_series_meeting_result', 'none')
        
        if games_played == 0:
            momentum = 0.5
        elif wins == games_played:
            momentum = self.MOMENTUM_PATTERNS['sweep']['momentum']
        elif wins > losses * 2:
            momentum = self.MOMENTUM_PATTERNS['dominant']['momentum']
        elif abs(wins - losses) <= 1:
            momentum = self.MOMENTUM_PATTERNS['split']['momentum']
        else:
            momentum = 0.3
            
        features.append(momentum)
        
        # Consecutive wins/losses in series
        consecutive_results = item.get('series_consecutive_results', 0)
        if abs(consecutive_results) >= 2:
            features.append(min(1.0, abs(consecutive_results) * 0.3))
        else:
            features.append(0.0)
            
        # Goal differential momentum
        if self.include_goal_differential:
            series_goal_diff = item.get('season_series_goal_differential', 0)
            features.append(np.tanh(series_goal_diff / 10.0))
        else:
            features.append(0.0)
            
        return features
        
    def _extract_venue_patterns(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract venue-based series patterns.
        
        Returns 3 features for home/away dynamics.
        """
        features = []
        
        is_home = item.get('is_home', True)
        
        # Home record in series
        home_series_record = item.get('home_record_vs_opponent', '0-0-0')
        away_series_record = item.get('away_record_vs_opponent', '0-0-0')
        
        # Parse home record
        try:
            if is_home:
                parts = home_series_record.split('-')
            else:
                parts = away_series_record.split('-')
                
            venue_wins = int(parts[0])
            venue_losses = int(parts[1])
            venue_games = venue_wins + venue_losses
            
            if venue_games > 0:
                venue_success = venue_wins / venue_games
            else:
                venue_success = 0.5
                
        except:
            venue_success = 0.5
            
        features.append(venue_success * self.venue_memory_weight)
        
        # Venue dominance pattern
        home_dominance = item.get('series_home_team_dominance', False)
        if (is_home and home_dominance) or (not is_home and not home_dominance):
            features.append(0.8)  # Following pattern
        elif home_dominance:
            features.append(0.3)  # Against pattern
        else:
            features.append(0.5)  # No pattern
            
        # Last game at this venue
        last_venue_result = item.get('last_result_at_venue_vs_opponent', 'none')
        
        venue_memory = {
            'win': 0.7,
            'overtime_win': 0.6,
            'overtime_loss': 0.4,
            'loss': 0.3,
            'none': 0.5
        }
        
        features.append(venue_memory.get(last_venue_result, 0.5))
        
        return features
        
    def _extract_rivalry_intensification(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract rivalry intensification features.
        
        Returns 4 features for growing animosity.
        """
        features = []
        
        # Physical play escalation
        previous_pim_avg = item.get('series_avg_penalty_minutes', 10)
        season_pim_avg = item.get('season_avg_penalty_minutes', 12)
        
        if previous_pim_avg > season_pim_avg * 1.5:
            features.append(min(1.0, (previous_pim_avg / season_pim_avg - 1) / 2))
        else:
            features.append(0.0)
            
        # Close games pattern
        one_goal_games = item.get('series_one_goal_games', 0)
        overtime_games = item.get('series_overtime_games', 0)
        total_games = item.get('series_games_played', 1)
        
        if total_games > 0:
            close_game_rate = (one_goal_games + overtime_games) / total_games
            features.append(close_game_rate)
        else:
            features.append(0.0)
            
        # Controversy history
        series_controversies = item.get('series_controversial_incidents', 0)
        features.append(min(1.0, series_controversies * 0.3))
        
        # Division rival multiplier
        division_rival = item.get('division_rival', False)
        if division_rival:
            # Division rivals play more, intensity builds
            meeting_number = item.get('season_series_meeting', 1)
            intensity_growth = min(1.0, meeting_number * 0.2)
            features.append(intensity_growth)
        else:
            features.append(0.3)  # Non-division baseline
            
        return features
        
    def _extract_series_stakes(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract season series stakes features.
        
        Returns 3 features for what's at stake.
        """
        features = []
        
        meeting_number = item.get('season_series_meeting', 1)
        total_meetings = item.get('season_series_total', 4)
        
        # Can clinch series
        series_record = item.get('season_series_record', '0-0-0')
        
        try:
            parts = series_record.split('-')
            wins = int(parts[0])
            losses = int(parts[1])
        except:
            wins, losses = 0, 0
            
        games_remaining = total_meetings - (wins + losses)
        
        # Clinch scenarios
        can_clinch = False
        can_be_clinched = False
        
        if games_remaining > 0:
            if wins > total_meetings // 2:
                can_clinch = True
            elif losses > total_meetings // 2:
                can_be_clinched = True
            elif wins == losses and games_remaining == 1:
                can_clinch = True  # Deciding game
                
        features.append(1.0 if can_clinch else 0.0)
        features.append(0.8 if can_be_clinched else 0.0)
        
        # Tiebreaker implications
        standings_tied = item.get('teams_tied_in_standings', False)
        playoff_race = item.get('both_in_playoff_race', False)
        
        tiebreaker_importance = 0.0
        if standings_tied:
            tiebreaker_importance += 0.5
        if playoff_race:
            tiebreaker_importance += 0.5
            
        features.append(min(1.0, tiebreaker_importance))
        
        return features
        
    def _extract_pattern_recognition(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract pattern recognition features.
        
        Teams learn each other's tendencies.
        
        Returns 2 features for tactical evolution.
        """
        features = []
        
        # Scoring pattern changes
        series_avg_total_goals = item.get('series_avg_total_goals', 5.5)
        league_avg_goals = item.get('league_avg_goals', 6.0)
        
        # Teams figuring each other out (lower scoring)
        if series_avg_total_goals < league_avg_goals * 0.8:
            pattern_strength = 1.0 - (series_avg_total_goals / league_avg_goals)
            features.append(pattern_strength)
        else:
            features.append(0.0)
            
        # Strategic adjustments
        meeting_number = item.get('season_series_meeting', 1)
        
        # Coaching adjustments increase with meetings
        if meeting_number >= 3:
            adjustment_factor = min(1.0, (meeting_number - 2) * 0.25)
            features.append(adjustment_factor)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Meeting effect features
        names.extend([
            'meeting_uncertainty',
            'meeting_intensity',
            'meeting_is_final',
            'meeting_series_position'
        ])
        
        # Previous outcome features
        names.extend([
            'series_win_percentage',
            'series_momentum',
            'series_consecutive_streak',
            'series_goal_differential'
        ])
        
        # Venue features
        names.extend([
            'venue_success_rate',
            'venue_dominance_pattern',
            'venue_last_memory'
        ])
        
        # Rivalry features
        names.extend([
            'rivalry_physical_escalation',
            'rivalry_close_games',
            'rivalry_controversy_history',
            'rivalry_division_intensity'
        ])
        
        # Stakes features
        names.extend([
            'stakes_can_clinch',
            'stakes_can_be_clinched',
            'stakes_tiebreaker'
        ])
        
        # Pattern features
        names.extend([
            'pattern_defensive_trend',
            'pattern_coaching_adjustment'
        ])
        
        return names
