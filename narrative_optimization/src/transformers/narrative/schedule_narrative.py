"""
Schedule Narrative Transformer

Extracts invisible narrative pressure from schedule positioning,
revealing how the structure of the season creates outcome pressure.

The schedule itself contains narrative DNA - this transformer decodes it.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from .invisible_base import InvisibleNarrativeTransformer


class ScheduleNarrativeTransformer(InvisibleNarrativeTransformer):
    """
    Extract narrative pressure from schedule positioning.
    
    Philosophy:
    - Schedule makers unconsciously encode narratives
    - Rest patterns create performance narratives
    - Schedule density affects team psychology
    - Position in season creates different pressures
    - Travel and homestands have narrative arcs
    
    Features (40 total):
    - Day of season positioning (8)
    - Days between games patterns (6)
    - Schedule density metrics (8)
    - Road trip/homestand positioning (8)
    - Back-to-back game effects (5)
    - Season arc positioning (5)
    """
    
    def __init__(
        self,
        season_length: int = 82,
        include_future_games: bool = True
    ):
        """
        Initialize schedule narrative analyzer.
        
        Parameters
        ----------
        season_length : int
            Number of games in season
        include_future_games : bool
            Whether to analyze future schedule pressure
        """
        super().__init__(
            narrative_id='schedule_narrative',
            description='Extracts narrative from schedule structure'
        )
        self.season_length = season_length
        self.include_future_games = include_future_games
        
        # Rest narrative thresholds
        self.rest_categories = {
            'exhausted': (0, 0),      # Back-to-back
            'tired': (1, 1),          # One day rest
            'normal': (2, 3),         # Standard rest
            'rested': (4, 6),         # Well rested
            'rusty': (7, float('inf')) # Too much rest
        }
        
        # Schedule density thresholds
        self.density_thresholds = {
            'light': (0, 2),
            'normal': (3, 4),
            'heavy': (5, 6),
            'brutal': (7, float('inf'))
        }
        
    def fit(self, X, y=None):
        """No fitting required for schedule analysis."""
        return self
        
    def transform(self, X):
        """
        Extract schedule narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with schedule context
            
        Returns
        -------
        np.ndarray
            Schedule features (n_samples, 40)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Day of season positioning (8)
            season_features = self._extract_season_position(item)
            feature_vec.extend(season_features)
            
            # Days between games (6)
            rest_features = self._extract_rest_patterns(item)
            feature_vec.extend(rest_features)
            
            # Schedule density (8)
            density_features = self._extract_schedule_density(item)
            feature_vec.extend(density_features)
            
            # Road trip/homestand (8)
            travel_features = self._extract_travel_patterns(item)
            feature_vec.extend(travel_features)
            
            # Back-to-back effects (5)
            b2b_features = self._extract_back_to_back_effects(item)
            feature_vec.extend(b2b_features)
            
            # Season arc position (5)
            arc_features = self._extract_season_arc_position(item)
            feature_vec.extend(arc_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_season_position(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract day of season positioning features.
        
        Returns 8 features measuring season position pressure.
        """
        features = []
        
        # Get game number
        game_number = item.get('game_number', 41)
        season_progress = game_number / self.season_length
        
        # Opening game pressure
        if game_number <= 5:
            features.append(1.0 - (game_number - 1) / 5.0)
        else:
            features.append(0.0)
            
        # Early season identity phase
        if 6 <= game_number <= 20:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Midpoint symmetry
        distance_from_midpoint = abs(game_number - self.season_length / 2)
        midpoint_pressure = 1.0 - (distance_from_midpoint / (self.season_length / 2))
        features.append(max(0.0, midpoint_pressure))
        
        # Playoff push activation (last 20 games)
        if game_number > self.season_length - 20:
            push_intensity = (game_number - (self.season_length - 20)) / 20.0
            features.append(push_intensity)
        else:
            features.append(0.0)
            
        # Season finale pressure
        if game_number >= self.season_length - 3:
            finale_pressure = (game_number - (self.season_length - 3)) / 3.0
            features.append(min(1.0, finale_pressure))
        else:
            features.append(0.0)
            
        # Position after key dates
        # Post-Christmas reset
        if 35 <= game_number <= 45:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Trade deadline proximity
        if 55 <= game_number <= 65:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Overall season progress pressure
        # Pressure peaks at 75% (playoff race)
        if season_progress < 0.75:
            progress_pressure = season_progress / 0.75
        else:
            progress_pressure = 1.0 - ((season_progress - 0.75) / 0.25) * 0.5
        features.append(progress_pressure)
        
        return features
        
    def _extract_rest_patterns(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract rest/fatigue narrative features.
        
        Returns 6 features measuring rest patterns.
        """
        features = []
        
        # Days since last game
        days_rest = item.get('days_since_last_game', 2)
        
        # Rest category encoding
        rest_category = 'normal'
        for category, (min_days, max_days) in self.rest_categories.items():
            if min_days <= days_rest <= max_days:
                rest_category = category
                break
                
        # Binary features for each rest category
        for category in ['exhausted', 'tired', 'normal', 'rested', 'rusty']:
            features.append(1.0 if rest_category == category else 0.0)
            
        # Continuous rest pressure
        # Optimal rest is 2-3 days
        if days_rest <= 3:
            rest_pressure = 1.0 - (abs(days_rest - 2.5) / 2.5)
        else:
            # Rust sets in after 3 days
            rest_pressure = max(0.0, 1.0 - (days_rest - 3) / 7.0)
            
        features.append(rest_pressure)
        
        return features
        
    def _extract_schedule_density(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract schedule density pressure features.
        
        Returns 8 features measuring game clustering.
        """
        features = []
        
        # Games in different windows
        games_last_7 = item.get('games_in_last_7_days', 2)
        games_last_14 = item.get('games_in_last_14_days', 5)
        games_last_21 = item.get('games_in_last_21_days', 8)
        
        # 7-day density
        density_7 = games_last_7 / 7.0
        features.append(min(1.0, density_7 * 2))  # Normalize to ~0-1
        
        # 14-day density
        density_14 = games_last_14 / 14.0
        features.append(min(1.0, density_14 * 2))
        
        # 21-day density
        density_21 = games_last_21 / 21.0
        features.append(min(1.0, density_21 * 2))
        
        # Density acceleration (getting busier)
        if games_last_14 > 0:
            recent_density = games_last_7 / 7.0
            prior_density = (games_last_14 - games_last_7) / 7.0
            acceleration = recent_density - prior_density
            features.append(np.tanh(acceleration * 5))  # -1 to 1
        else:
            features.append(0.0)
            
        # Future density pressure
        if self.include_future_games:
            games_next_7 = item.get('games_in_next_7_days', 2)
            games_next_14 = item.get('games_in_next_14_days', 5)
            
            future_pressure = min(1.0, games_next_7 / 5.0)
            features.append(future_pressure)
            
            # Density sandwich (busy-break-busy)
            if games_last_7 >= 3 and games_next_7 >= 3:
                features.append(0.8)  # Precious rest opportunity
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
            
        # Brutal stretch indicator
        if games_last_14 >= 8:
            brutal_factor = min(1.0, (games_last_14 - 8) / 4.0)
            features.append(brutal_factor)
        else:
            features.append(0.0)
            
        # Month game count pressure
        games_this_month = item.get('games_in_current_month', 12)
        month_pressure = min(1.0, games_this_month / 16.0)
        features.append(month_pressure)
        
        return features
        
    def _extract_travel_patterns(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract road trip and homestand narrative features.
        
        Returns 8 features measuring travel narratives.
        """
        features = []
        
        # Current trip/stand position
        is_home = item.get('is_home', True)
        consecutive_road = item.get('consecutive_road_games', 0)
        consecutive_home = item.get('consecutive_home_games', 0)
        
        # Road trip narrative
        if not is_home:
            # Trip game number
            trip_game = consecutive_road + 1
            
            # First game of trip (adjustment)
            features.append(1.0 if trip_game == 1 else 0.0)
            
            # Deep into trip (fatigue)
            if trip_game >= 4:
                fatigue = min(1.0, (trip_game - 3) / 4.0)
                features.append(fatigue)
            else:
                features.append(0.0)
                
            # Last game of trip (almost home)
            next_game_home = item.get('next_game_is_home', False)
            features.append(0.8 if next_game_home else 0.0)
            
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Homestand narrative  
        if is_home:
            # Stand game number
            stand_game = consecutive_home + 1
            
            # First game back home (energy)
            features.append(1.0 if stand_game == 1 else 0.0)
            
            # Long homestand (comfort)
            if stand_game >= 4:
                comfort = min(1.0, (stand_game - 3) / 3.0)
                features.append(comfort)
            else:
                features.append(0.0)
                
            # Last game before trip (anxiety)
            next_game_away = item.get('next_game_is_away', False)
            features.append(0.7 if next_game_away else 0.0)
            
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Extended road trip pressure
        total_road_games = item.get('current_road_trip_length', 3)
        if total_road_games >= 5:
            extended_pressure = min(1.0, (total_road_games - 4) / 4.0)
            features.append(extended_pressure)
        else:
            features.append(0.0)
            
        # Home cooking advantage
        days_at_home = item.get('consecutive_days_at_home', 0)
        if days_at_home >= 7:
            home_comfort = min(1.0, days_at_home / 14.0)
            features.append(home_comfort)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_back_to_back_effects(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract back-to-back game narrative effects.
        
        Returns 5 features measuring B2B pressure.
        """
        features = []
        
        is_b2b = item.get('is_back_to_back', False)
        
        if not is_b2b:
            return [0.0] * 5
            
        # First or second game of B2B
        is_second_game = item.get('is_second_of_b2b', False)
        
        features.append(0.7 if not is_second_game else 0.0)  # First game
        features.append(1.0 if is_second_game else 0.0)      # Second game
        
        # Travel between B2B games
        b2b_travel = item.get('b2b_travel_miles', 0)
        if b2b_travel > 1000:
            travel_factor = min(1.0, b2b_travel / 2000.0)
            features.append(travel_factor)
        else:
            features.append(0.0)
            
        # Opponent rest advantage in B2B
        opponent_b2b = item.get('opponent_is_back_to_back', False)
        if is_second_game and not opponent_b2b:
            features.append(0.9)  # Significant disadvantage
        else:
            features.append(0.0)
            
        # B2B frequency fatigue
        b2b_count_month = item.get('b2b_sets_this_month', 1)
        if b2b_count_month >= 3:
            features.append(0.8)  # Too many B2Bs
        else:
            features.append(b2b_count_month * 0.25)
            
        return features
        
    def _extract_season_arc_position(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract season narrative arc positioning.
        
        Returns 5 features measuring arc position.
        """
        features = []
        
        # Determine season phase
        game_date = item.get('game_date', datetime.now())
        season_start = item.get('season_start_date', datetime.now() - timedelta(days=60))
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        if isinstance(season_start, str):
            season_start = pd.to_datetime(season_start)
            
        days_elapsed, phase = self.calculate_season_position(game_date, season_start)
        
        # Key phase indicators
        key_phases = ['opening_week', 'thanksgiving_eval', 'trade_deadline_approach', 
                     'playoff_race', 'season_finale']
        
        for key_phase in key_phases:
            features.append(1.0 if phase == key_phase else 0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Season position features
        names.extend([
            'season_opening_game',
            'season_identity_phase',
            'season_midpoint_symmetry',
            'season_playoff_push',
            'season_finale_pressure',
            'season_post_christmas',
            'season_trade_deadline',
            'season_progress_pressure'
        ])
        
        # Rest pattern features
        names.extend([
            'rest_exhausted',
            'rest_tired',
            'rest_normal',
            'rest_rested',
            'rest_rusty',
            'rest_optimal_pressure'
        ])
        
        # Schedule density features
        names.extend([
            'density_7_day',
            'density_14_day',
            'density_21_day',
            'density_acceleration',
            'density_future_pressure',
            'density_sandwich_rest',
            'density_brutal_stretch',
            'density_monthly_pressure'
        ])
        
        # Travel pattern features
        names.extend([
            'travel_road_trip_start',
            'travel_road_trip_deep',
            'travel_road_trip_end',
            'travel_homestand_start',
            'travel_homestand_comfort',
            'travel_homestand_end',
            'travel_extended_pressure',
            'travel_home_cooking'
        ])
        
        # Back-to-back features
        names.extend([
            'b2b_first_game',
            'b2b_second_game',
            'b2b_travel_fatigue',
            'b2b_opponent_advantage',
            'b2b_frequency_fatigue'
        ])
        
        # Season arc features
        names.extend([
            'arc_opening_week',
            'arc_thanksgiving_eval',
            'arc_trade_deadline',
            'arc_playoff_race',
            'arc_season_finale'
        ])
        
        return names
