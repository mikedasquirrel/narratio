"""
Temporal Narrative Cycles Detector

Identifies and quantifies temporal patterns in sports narratives,
including revenge games, milestone effects, and momentum mathematics.

This transformer detects when timing creates narrative significance
and measures the resulting pressure on outcomes.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import math


class TemporalNarrativeCyclesTransformer(BaseEstimator, TransformerMixin):
    """
    Extract temporal narrative pattern features.
    
    Philosophy:
    - Timing is everything in narrative
    - Certain temporal contexts amplify drama
    - Momentum follows mathematical patterns
    - Cycles create predictable pressure points
    - Time transforms meaning
    
    Features (40 total):
    - Revenge game amplification (8)
    - Milestone proximity effects (8)
    - Season narrative arc positioning (6)
    - Series momentum mathematics (8)
    - Elimination game psychology (5)
    - Temporal clustering effects (5)
    """
    
    def __init__(
        self,
        momentum_decay_rate: float = 0.85,
        milestone_radius: int = 5,  # games
        include_advanced_momentum: bool = True
    ):
        """
        Initialize temporal narrative analyzer.
        
        Parameters
        ----------
        momentum_decay_rate : float
            How quickly momentum effects fade
        milestone_radius : int
            Games before/after milestone with effects
        include_advanced_momentum : bool
            Include complex momentum calculations
        """
        self.momentum_decay_rate = momentum_decay_rate
        self.milestone_radius = milestone_radius
        self.include_advanced_momentum = include_advanced_momentum
        
        # Revenge game windows
        self.revenge_windows = {
            'immediate': (0, 7),      # Same week
            'short': (7, 30),         # Same month
            'medium': (30, 180),      # Same season
            'playoff': (180, 365),    # Next playoff meeting
            'annual': (365, 730)      # Next season
        }
        
        # Milestone types and their gravity
        self.milestone_gravity = {
            'career_goals': {
                100: 0.6, 200: 0.7, 300: 0.8,
                400: 0.85, 500: 0.9, 600: 0.95,
                700: 0.97, 800: 0.98, 900: 0.99,
                1000: 1.0
            },
            'career_wins': {
                100: 0.5, 200: 0.6, 300: 0.7,
                400: 0.8, 500: 0.9, 600: 0.95,
                700: 0.98, 800: 0.99, 900: 1.0,
                1000: 1.0
            },
            'team_wins': {
                500: 0.6, 1000: 0.8, 1500: 0.9,
                2000: 0.95, 2500: 0.98, 3000: 1.0
            },
            'consecutive': {
                5: 0.5, 10: 0.7, 15: 0.8,
                20: 0.9, 25: 0.95, 30: 1.0
            }
        }
        
        # Season arc stages
        self.season_stages = {
            'opening': (0, 0.1),      # First 10%
            'establishing': (0.1, 0.25),
            'building': (0.25, 0.5),
            'turning': (0.5, 0.65),
            'intensifying': (0.65, 0.8),
            'climax': (0.8, 0.9),
            'resolution': (0.9, 1.0)
        }
        
    def fit(self, X, y=None):
        """
        Learn temporal patterns from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical game data with temporal info
        y : ignored
        
        Returns
        -------
        self
        """
        # Build temporal pattern database
        self.temporal_patterns_ = self._build_temporal_patterns(X)
        
        # Learn momentum decay curves
        self.momentum_curves_ = self._learn_momentum_curves(X)
        
        return self
        
    def transform(self, X):
        """
        Extract temporal narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with temporal context
            
        Returns
        -------
        np.ndarray
            Temporal features (n_samples, 40)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Revenge game amplification (8)
            revenge_features = self._extract_revenge_amplification(item)
            feature_vec.extend(revenge_features)
            
            # Milestone proximity effects (8)
            milestone_features = self._extract_milestone_effects(item)
            feature_vec.extend(milestone_features)
            
            # Season arc positioning (6)
            arc_features = self._extract_season_arc_position(item)
            feature_vec.extend(arc_features)
            
            # Series momentum mathematics (8)
            if self.include_advanced_momentum:
                momentum_features = self._extract_series_momentum(item)
            else:
                momentum_features = self._extract_basic_momentum(item)
            feature_vec.extend(momentum_features)
            
            # Elimination game psychology (5)
            elimination_features = self._extract_elimination_psychology(item)
            feature_vec.extend(elimination_features)
            
            # Temporal clustering effects (5)
            clustering_features = self._extract_temporal_clustering(item)
            feature_vec.extend(clustering_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _build_temporal_patterns(self, X):
        """Build database of temporal patterns from training data."""
        patterns = {
            'revenge_success_rate': {},
            'milestone_achievement_rate': {},
            'momentum_sequences': [],
            'elimination_outcomes': {}
        }
        
        # This would analyze historical data for patterns
        # For now, return defaults
        patterns['revenge_success_rate'] = {
            'immediate': 0.58,
            'short': 0.54,
            'medium': 0.52,
            'playoff': 0.61,
            'annual': 0.51
        }
        
        return patterns
        
    def _learn_momentum_curves(self, X):
        """Learn momentum decay and amplification curves."""
        # Would analyze actual momentum patterns
        # For now, return theoretical curves
        return {
            'win_streak': lambda x: 1.0 - np.exp(-x / 3.0),
            'loss_streak': lambda x: -1.0 + np.exp(-x / 3.0),
            'series_momentum': lambda x, y: np.tanh((x - y) / 2.0),
            'comeback_momentum': lambda x: x ** 1.5 if x > 0 else 0
        }
        
    def _extract_revenge_amplification(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract revenge game narrative amplification.
        
        Returns 8 features measuring revenge narrative strength.
        """
        features = []
        
        # Get revenge context
        last_meeting = item.get('last_meeting_date')
        if last_meeting:
            if isinstance(last_meeting, str):
                last_meeting = pd.to_datetime(last_meeting)
                
            current_date = item.get('game_date', datetime.now())
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
                
            days_since = (current_date - last_meeting).days
        else:
            days_since = 999  # No recent meeting
            
        # Revenge window activation
        revenge_type = None
        revenge_strength = 0.0
        
        for window_type, (min_days, max_days) in self.revenge_windows.items():
            if min_days <= days_since <= max_days:
                revenge_type = window_type
                # Strength peaks in middle of window
                window_position = (days_since - min_days) / (max_days - min_days)
                revenge_strength = np.sin(window_position * np.pi)
                break
                
        features.append(revenge_strength)
        
        # Revenge multipliers
        last_result = item.get('last_meeting_result', {})
        
        # Blowout revenge (lost by 5+ goals)
        if last_result.get('goal_diff', 0) <= -5:
            features.append(1.0)
        elif last_result.get('goal_diff', 0) <= -3:
            features.append(0.6)
        else:
            features.append(0.0)
            
        # Controversial loss revenge
        if last_result.get('controversial', False):
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Playoff elimination revenge
        if last_result.get('eliminated_by', False):
            if revenge_type == 'playoff':
                features.append(1.0)  # Maximum revenge
            else:
                features.append(0.7)  # Still remembers
        else:
            features.append(0.0)
            
        # Injury revenge (player returns against team that injured them)
        if item.get('injury_revenge_game', False):
            features.append(0.9)
        else:
            features.append(0.0)
            
        # Coaching revenge (against former team)
        if item.get('coach_revenge_game', False):
            games_since_departure = item.get('coach_games_since_departure', 100)
            if games_since_departure < 10:
                features.append(1.0)  # Fresh wound
            elif games_since_departure < 50:
                features.append(0.7)
            else:
                features.append(0.4)
        else:
            features.append(0.0)
            
        # Trade revenge (player against team that traded them)
        if item.get('trade_revenge_game', False):
            trade_bitterness = item.get('trade_bitterness_score', 0.5)
            features.append(trade_bitterness)
        else:
            features.append(0.0)
            
        # Cumulative revenge score
        revenge_factors = sum(features[1:])  # Skip base strength
        cumulative_revenge = min(1.0, revenge_strength * (1 + revenge_factors * 0.2))
        features.append(cumulative_revenge)
        
        return features
        
    def _extract_milestone_effects(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract milestone proximity narrative effects.
        
        Returns 8 features measuring milestone pressure.
        """
        features = []
        
        # Player milestone proximity
        player_milestones = item.get('player_milestones', [])
        
        # Goals milestone
        goal_milestone_pressure = 0.0
        for milestone in player_milestones:
            if milestone['type'] == 'goals':
                current = milestone['current']
                for threshold, gravity in self.milestone_gravity['career_goals'].items():
                    if current < threshold <= current + 5:
                        distance = threshold - current
                        pressure = gravity * np.exp(-distance / 2.0)
                        goal_milestone_pressure = max(goal_milestone_pressure, pressure)
                        
        features.append(goal_milestone_pressure)
        
        # Assists/points milestone
        point_milestone_pressure = 0.0
        for milestone in player_milestones:
            if milestone['type'] in ['assists', 'points']:
                proximity = milestone.get('games_to_milestone', 999)
                if proximity < self.milestone_radius:
                    pressure = 0.8 * np.exp(-proximity / 3.0)
                    point_milestone_pressure = max(point_milestone_pressure, pressure)
                    
        features.append(point_milestone_pressure)
        
        # Goalie wins milestone
        goalie_milestone = item.get('goalie_milestone', None)
        if goalie_milestone:
            wins_away = goalie_milestone.get('wins_from_milestone', 999)
            if wins_away <= 3:
                features.append(0.9 * np.exp(-wins_away / 2.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Team milestone proximity
        team_milestone = item.get('team_milestone', None)
        if team_milestone:
            milestone_type = team_milestone['type']
            distance = team_milestone.get('games_away', 999)
            
            if milestone_type in self.milestone_gravity['team_wins']:
                base_gravity = 0.7
                features.append(base_gravity * np.exp(-distance / 5.0))
            else:
                features.append(0.3 * np.exp(-distance / 5.0))
        else:
            features.append(0.0)
            
        # Consecutive streak milestones
        active_streak = item.get('active_streak', None)
        if active_streak:
            streak_length = active_streak['length']
            streak_type = active_streak['type']
            
            # Check proximity to round numbers
            milestone_pressure = 0.0
            for threshold, gravity in self.milestone_gravity['consecutive'].items():
                if streak_length == threshold - 1:
                    # One away from milestone
                    if streak_type == 'win':
                        milestone_pressure = gravity
                    else:
                        # Negative milestones (losing streaks) create pressure to break
                        milestone_pressure = -gravity
                        
            features.append(milestone_pressure)
        else:
            features.append(0.0)
            
        # Record-breaking proximity
        record_proximity = item.get('record_breaking_proximity', None)
        if record_proximity:
            record_importance = record_proximity.get('importance', 0.5)
            games_away = record_proximity.get('games_away', 999)
            
            if games_away <= 3:
                features.append(record_importance * (1.0 - games_away / 4.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Milestone convergence (multiple milestones align)
        milestone_count = sum(1 for f in features if f > 0.5)
        if milestone_count >= 2:
            features.append(min(1.0, milestone_count * 0.3))
        else:
            features.append(0.0)
            
        # Anti-milestone pressure (999 goals, etc.)
        anti_milestone = item.get('anti_milestone', False)
        if anti_milestone:
            features.append(0.8)  # Pressure to get past awkward number
        else:
            features.append(0.0)
            
        return features
        
    def _extract_season_arc_position(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract season narrative arc positioning features.
        
        Returns 6 features measuring narrative timing in season.
        """
        features = []
        
        # Calculate season progress
        games_played = item.get('team_games_played', 0)
        total_games = item.get('season_length', 82)
        season_progress = games_played / total_games
        
        # Identify current stage
        current_stage = None
        for stage, (start, end) in self.season_stages.items():
            if start <= season_progress < end:
                current_stage = stage
                stage_progress = (season_progress - start) / (end - start)
                break
                
        # Stage-specific pressure
        stage_pressure = {
            'opening': 0.3,          # Low pressure, finding identity
            'establishing': 0.5,     # Building patterns
            'building': 0.6,         # Momentum matters
            'turning': 0.8,          # Critical juncture
            'intensifying': 0.9,     # Every game matters
            'climax': 1.0,          # Maximum pressure
            'resolution': 0.7       # Playing out string
        }
        
        features.append(stage_pressure.get(current_stage, 0.5))
        
        # Narrative arc alignment
        team_narrative = item.get('season_narrative_type', 'standard')
        expected_position = item.get('expected_standings_position', 10)
        current_position = item.get('current_standings_position', 10)
        
        # Are they following expected arc?
        position_differential = current_position - expected_position
        
        if team_narrative == 'underdog_rise' and position_differential < -3:
            features.append(0.9)  # Exceeding expectations
        elif team_narrative == 'dynasty_defense' and position_differential > 3:
            features.append(-0.8)  # Failing expectations
        else:
            features.append(np.tanh(-position_differential / 5.0))
            
        # Turning point detection
        recent_trajectory = item.get('last_10_games_trajectory', 0.0)
        season_trajectory = item.get('season_trajectory', 0.0)
        
        if abs(recent_trajectory - season_trajectory) > 0.3:
            features.append(0.8)  # Potential turning point
        else:
            features.append(0.0)
            
        # Playoff race positioning
        playoff_buffer = item.get('points_from_playoff_line', 0)
        games_remaining = total_games - games_played
        
        if -5 <= playoff_buffer <= 5 and games_remaining < 20:
            # In the hunt with time running out
            urgency = 1.0 - (games_remaining / 20.0)
            features.append(urgency)
        else:
            features.append(0.0)
            
        # Narrative climax proximity
        if current_stage in ['intensifying', 'climax']:
            climax_factors = 0.0
            
            # Division race
            if item.get('games_behind_division', 0) <= 3:
                climax_factors += 0.3
                
            # Wild card race
            if item.get('wild_card_race', False):
                climax_factors += 0.3
                
            # Rivalry game in crucial time
            if item.get('is_rival', False):
                climax_factors += 0.2
                
            features.append(min(1.0, climax_factors))
        else:
            features.append(0.0)
            
        # Season arc completion pressure
        narrative_completion = item.get('narrative_completion_percentage', 0.0)
        if narrative_completion > 0.8 and current_stage == 'climax':
            features.append(1.0)  # Must complete the story
        else:
            features.append(narrative_completion * stage_pressure.get(current_stage, 0.5))
            
        return features
        
    def _extract_series_momentum(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract advanced series momentum mathematics.
        
        Returns 8 features measuring momentum dynamics.
        """
        features = []
        
        # Get series context
        is_playoffs = item.get('is_playoffs', False)
        series_score = item.get('series_score', [0, 0])
        last_game_score = item.get('last_game_score', [0, 0])
        home_team = item.get('is_home', True)
        
        if not is_playoffs or sum(series_score) == 0:
            # Regular season momentum
            return self._extract_basic_momentum(item)
            
        # Series position momentum
        total_games = sum(series_score)
        if total_games > 0:
            # Basic series momentum
            series_differential = series_score[0] - series_score[1]
            basic_momentum = np.tanh(series_differential / 2.0)
            features.append(basic_momentum)
            
            # Momentum shift from last game
            last_winner = 'home' if last_game_score[0] > last_game_score[1] else 'away'
            last_margin = abs(last_game_score[0] - last_game_score[1])
            
            if last_winner == 'home' and not home_team:
                # Away team lost last game at home
                shift = -0.3 * (1 + last_margin / 5.0)
            elif last_winner == 'away' and home_team:
                # Home team lost last game away
                shift = 0.2  # Less severe
            else:
                shift = 0.1 * np.sign(series_differential)
                
            features.append(np.tanh(shift))
            
            # 2-3-2 format home advantage momentum
            if total_games <= 2:  # First two at home
                if home_team:
                    features.append(0.3)
                else:
                    features.append(-0.2)
            elif total_games <= 5:  # Middle games away
                if home_team:
                    features.append(-0.2)
                else:
                    features.append(0.3)
            else:  # Final games at home
                if home_team:
                    features.append(0.4)
                else:
                    features.append(-0.3)
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Elimination game momentum
        if series_score[0] == 3 or series_score[1] == 3:
            # Someone facing elimination
            if (home_team and series_score[1] == 3) or (not home_team and series_score[0] == 3):
                features.append(0.8)  # Desperation momentum
            else:
                features.append(-0.6)  # Pressure to close out
        else:
            features.append(0.0)
            
        # Comeback momentum (down 3-0 or 3-1)
        if min(series_score) == 0 and max(series_score) == 3:
            # Down 3-0
            features.append(0.0)  # Usually doom
        elif series_score == [3, 1] or series_score == [1, 3]:
            # Down 3-1
            if (series_score[0] == 1 and home_team) or (series_score[1] == 1 and not home_team):
                features.append(0.6)  # Comeback starting
            else:
                features.append(0.0)
        elif series_score == [3, 2] or series_score == [2, 3]:
            # Comeback in progress
            if (series_score[0] == 2 and home_team) or (series_score[1] == 2 and not home_team):
                features.append(0.9)  # Maximum comeback momentum
            else:
                features.append(-0.7)  # Choking pressure
        else:
            features.append(0.0)
            
        # Game 7 momentum
        if series_score == [3, 3]:
            # Home team historically wins ~65% of Game 7s
            if home_team:
                features.append(0.4)
            else:
                features.append(-0.2)
        else:
            features.append(0.0)
            
        # Momentum volatility (how swingy the series has been)
        game_margins = item.get('series_game_margins', [])
        if len(game_margins) > 1:
            volatility = np.std(game_margins)
            features.append(np.tanh(volatility / 3.0))
        else:
            features.append(0.0)
            
        # Projected momentum (where series is heading)
        if self.momentum_curves_:
            momentum_func = self.momentum_curves_['series_momentum']
            projected = momentum_func(series_score[0], series_score[1])
            features.append(projected)
        else:
            features.append(basic_momentum if 'basic_momentum' in locals() else 0.0)
            
        return features
        
    def _extract_basic_momentum(self, item: Dict[str, Any]) -> List[float]:
        """Extract basic momentum features for regular season."""
        features = []
        
        # Recent form momentum
        last_5 = item.get('last_5_record', [2, 3])
        last_10 = item.get('last_10_record', [5, 5])
        
        short_momentum = (last_5[0] - last_5[1]) / 5.0
        medium_momentum = (last_10[0] - last_10[1]) / 10.0
        
        features.append(short_momentum)
        features.append(medium_momentum)
        
        # Streak momentum
        current_streak = item.get('current_streak', {'type': 'none', 'length': 0})
        if current_streak['type'] == 'win':
            streak_momentum = min(1.0, current_streak['length'] / 7.0)
        elif current_streak['type'] == 'loss':
            streak_momentum = max(-1.0, -current_streak['length'] / 7.0)
        else:
            streak_momentum = 0.0
            
        features.append(streak_momentum)
        
        # Home/away momentum differential
        home_record = item.get('home_record', [20, 20])
        away_record = item.get('away_record', [20, 20])
        
        if item.get('is_home', True):
            location_momentum = (home_record[0] - home_record[1]) / sum(home_record)
        else:
            location_momentum = (away_record[0] - away_record[1]) / sum(away_record)
            
        features.append(location_momentum)
        
        # Fill remaining with zeros to match expected size
        features.extend([0.0] * (8 - len(features)))
        
        return features
        
    def _extract_elimination_psychology(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract elimination game psychological features.
        
        Returns 5 features measuring elimination pressure dynamics.
        """
        features = []
        
        is_elimination = item.get('is_elimination_game', False)
        
        if not is_elimination:
            return [0.0] * 5
            
        # Which team faces elimination
        home_elimination = item.get('home_faces_elimination', False)
        away_elimination = item.get('away_faces_elimination', False)
        is_home = item.get('is_home', True)
        
        # Desperation factor
        if (home_elimination and is_home) or (away_elimination and not is_home):
            features.append(1.0)  # Maximum desperation
        else:
            features.append(0.0)
            
        # Closeout pressure
        if (home_elimination and not is_home) or (away_elimination and is_home):
            features.append(0.8)  # Pressure to finish
        else:
            features.append(0.0)
            
        # Experience differential
        playoff_experience_diff = item.get('playoff_experience_differential', 0.0)
        if abs(playoff_experience_diff) > 0.3:
            # Experience matters more in elimination
            features.append(playoff_experience_diff)
        else:
            features.append(0.0)
            
        # Previous elimination game record
        elim_record = item.get('franchise_elimination_record', [10, 10])
        if sum(elim_record) > 0:
            elim_success = elim_record[0] / sum(elim_record)
            # Historical success creates confidence/doubt
            features.append((elim_success - 0.5) * 2)  # -1 to 1
        else:
            features.append(0.0)
            
        # Do or die amplification
        series_game = item.get('series_game_number', 5)
        if series_game == 7:
            features.append(1.0)  # Maximum drama
        elif series_game >= 5:
            features.append(0.7)
        else:
            features.append(0.4)
            
        return features
        
    def _extract_temporal_clustering(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract temporal clustering effect features.
        
        Returns 5 features measuring clustered event impacts.
        """
        features = []
        
        # Back-to-back game effects
        is_back_to_back = item.get('is_back_to_back', False)
        if is_back_to_back:
            if item.get('is_home', True):
                features.append(0.3)  # Slight disadvantage
            else:
                features.append(0.5)  # Bigger disadvantage on road
        else:
            features.append(0.0)
            
        # Games in X days clustering
        games_last_week = item.get('games_in_last_7_days', 2)
        if games_last_week >= 4:
            features.append(0.8)  # Fatigue factor
        elif games_last_week >= 3:
            features.append(0.4)
        else:
            features.append(0.0)
            
        # Schedule density pressure
        upcoming_games_week = item.get('games_next_7_days', 2)
        if upcoming_games_week >= 4:
            # Might rest players
            features.append(0.6)
        else:
            features.append(0.0)
            
        # Rivalry clustering (multiple games against rival)
        rival_games_month = item.get('rival_games_this_month', 0)
        if rival_games_month >= 3:
            features.append(0.8)  # Intensity builds
        elif rival_games_month == 2:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Event confluence (multiple narratives converging)
        narrative_count = item.get('active_narrative_count', 1)
        if narrative_count >= 4:
            features.append(1.0)  # Narrative overload
        elif narrative_count >= 3:
            features.append(0.7)
        elif narrative_count >= 2:
            features.append(0.4)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Revenge features
        names.extend([
            'revenge_window_activation',
            'revenge_blowout_multiplier',
            'revenge_controversial_loss',
            'revenge_playoff_elimination',
            'revenge_injury_return',
            'revenge_coaching',
            'revenge_trade',
            'revenge_cumulative_score'
        ])
        
        # Milestone features
        names.extend([
            'milestone_goals_proximity',
            'milestone_points_proximity',
            'milestone_goalie_wins',
            'milestone_team_proximity',
            'milestone_streak',
            'milestone_record_breaking',
            'milestone_convergence',
            'milestone_anti_pressure'
        ])
        
        # Season arc features
        names.extend([
            'arc_stage_pressure',
            'arc_narrative_alignment',
            'arc_turning_point',
            'arc_playoff_race_urgency',
            'arc_climax_proximity',
            'arc_completion_pressure'
        ])
        
        # Momentum features
        names.extend([
            'momentum_series_basic',
            'momentum_last_game_shift',
            'momentum_home_advantage_phase',
            'momentum_elimination',
            'momentum_comeback',
            'momentum_game_7',
            'momentum_volatility',
            'momentum_projected'
        ])
        
        # Elimination features
        names.extend([
            'elimination_desperation',
            'elimination_closeout_pressure',
            'elimination_experience_factor',
            'elimination_historical_record',
            'elimination_do_or_die_amplification'
        ])
        
        # Clustering features
        names.extend([
            'clustering_back_to_back',
            'clustering_week_density',
            'clustering_upcoming_density',
            'clustering_rivalry_intensity',
            'clustering_narrative_confluence'
        ])
        
        return names
