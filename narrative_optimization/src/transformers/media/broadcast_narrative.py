"""
Broadcast Narrative Transformer

Infers broadcast narrative pressure from game time/day patterns.
Primetime games carry different narrative weight than afternoon tilts.

The invisible hand of the broadcast schedule shapes outcomes.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, time
from sklearn.base import BaseEstimator, TransformerMixin


class BroadcastNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract broadcast narrative pressure features.
    
    Philosophy:
    - Primetime slots create performance pressure
    - National broadcasts amplify narratives
    - Special windows have special energy
    - Matinee games have different psychology
    - Late starts affect both teams differently
    
    Features (25 total):
    - Primetime slot detection (5)
    - National vs regional inference (4)
    - Special broadcast windows (5)
    - Matinee game shifts (4)
    - Late night exhaustion (3)
    - Broadcast pressure composite (4)
    """
    
    # Broadcast slot definitions
    BROADCAST_SLOTS = {
        'early_matinee': {
            'start_hour': 12,
            'end_hour': 14,
            'days': [0, 1, 2, 3, 4, 5, 6],
            'type': 'matinee',
            'pressure': 0.6
        },
        'afternoon': {
            'start_hour': 14,
            'end_hour': 17,
            'days': [5, 6],  # Weekend afternoons
            'type': 'afternoon',
            'pressure': 0.7
        },
        'early_evening': {
            'start_hour': 17,
            'end_hour': 19,
            'days': [0, 1, 2, 3, 4],
            'type': 'regional',
            'pressure': 0.5
        },
        'primetime': {
            'start_hour': 19,
            'end_hour': 21,
            'days': [0, 1, 2, 3, 4, 5, 6],
            'type': 'primetime',
            'pressure': 0.9
        },
        'late_night': {
            'start_hour': 21,
            'end_hour': 23,
            'days': [0, 1, 2, 3, 4, 5, 6],
            'type': 'late',
            'pressure': 0.6
        },
        'west_coast_late': {
            'start_hour': 22,
            'end_hour': 24,
            'days': [0, 1, 2, 3, 4],
            'type': 'west_coast',
            'pressure': 0.5
        }
    }
    
    # Special broadcast windows
    SPECIAL_WINDOWS = {
        'hockey_night_canada': {
            'day': 5,  # Saturday
            'hour': 19,
            'teams': ['Toronto', 'Montreal', 'Ottawa', 'Calgary', 
                     'Edmonton', 'Vancouver', 'Winnipeg'],
            'impact': 1.0
        },
        'wednesday_rivalry': {
            'day': 2,  # Wednesday
            'hour': 20,
            'type': 'rivalry',
            'impact': 0.9
        },
        'sunday_showcase': {
            'day': 6,  # Sunday
            'hour': 15,
            'type': 'national',
            'impact': 0.8
        },
        'thanksgiving_classic': {
            'month': 11,
            'day_range': (22, 28),
            'impact': 0.9
        },
        'new_years_day': {
            'month': 1,
            'day': 1,
            'impact': 1.0
        }
    }
    
    def __init__(
        self,
        include_canadian_broadcasts: bool = True,
        timezone_adjust: bool = True
    ):
        """
        Initialize broadcast narrative analyzer.
        
        Parameters
        ----------
        include_canadian_broadcasts : bool
            Include Hockey Night in Canada patterns
        timezone_adjust : bool
            Adjust for local timezone effects
        """
        self.include_canadian_broadcasts = include_canadian_broadcasts
        self.timezone_adjust = timezone_adjust
        
    def fit(self, X, y=None):
        """No fitting required for broadcast analysis."""
        return self
        
    def transform(self, X):
        """
        Extract broadcast narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with time/date information
            
        Returns
        -------
        np.ndarray
            Broadcast features (n_samples, 25)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Primetime slot detection (5)
            slot_features = self._extract_broadcast_slot(item)
            feature_vec.extend(slot_features)
            
            # National vs regional (4)
            coverage_features = self._extract_coverage_type(item)
            feature_vec.extend(coverage_features)
            
            # Special windows (5)
            special_features = self._extract_special_windows(item)
            feature_vec.extend(special_features)
            
            # Matinee shifts (4)
            matinee_features = self._extract_matinee_effects(item)
            feature_vec.extend(matinee_features)
            
            # Late night factors (3)
            late_features = self._extract_late_night_effects(item)
            feature_vec.extend(late_features)
            
            # Broadcast pressure composite (4)
            pressure_features = self._extract_broadcast_pressure(item)
            feature_vec.extend(pressure_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_broadcast_slot(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract broadcast slot features.
        
        Returns 5 features for time slot categorization.
        """
        features = []
        
        game_time = item.get('game_time', '19:00')
        game_date = item.get('game_date', datetime.now())
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Parse game time
        if isinstance(game_time, str):
            try:
                hour = int(game_time.split(':')[0])
            except:
                hour = 19  # Default 7 PM
        else:
            hour = game_time.hour if hasattr(game_time, 'hour') else 19
            
        day_of_week = game_date.weekday()
        
        # Check each broadcast slot
        slot_found = False
        for slot_name, slot_info in self.BROADCAST_SLOTS.items():
            if (slot_info['start_hour'] <= hour < slot_info['end_hour'] and
                day_of_week in slot_info['days']):
                
                # Primetime indicator
                if slot_name == 'primetime':
                    features.append(1.0)
                else:
                    features.append(0.0)
                    
                # Matinee indicator
                if slot_info['type'] == 'matinee':
                    features.append(1.0)
                else:
                    features.append(0.0)
                    
                # Late night indicator
                if slot_info['type'] in ['late', 'west_coast']:
                    features.append(1.0)
                else:
                    features.append(0.0)
                    
                # Slot pressure
                features.append(slot_info['pressure'])
                
                slot_found = True
                break
                
        if not slot_found:
            # Default values
            features.extend([0.0, 0.0, 0.0, 0.5])
            
        # Weekend primetime bonus
        if day_of_week in [4, 5] and 18 <= hour <= 21:
            features.append(0.8)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_coverage_type(self, item: Dict[str, Any]) -> List[float]:
        """
        Infer national vs regional broadcast type.
        
        Returns 4 features for coverage inference.
        """
        features = []
        
        # Direct broadcast info if available
        broadcast_type = item.get('broadcast_type', None)
        
        if broadcast_type:
            features.append(1.0 if broadcast_type == 'national' else 0.0)
            features.append(1.0 if broadcast_type == 'regional' else 0.0)
            features.append(1.0 if broadcast_type == 'local' else 0.0)
        else:
            # Infer from context
            game_time = item.get('game_time', '19:00')
            game_date = item.get('game_date', datetime.now())
            home_team = item.get('home_team', '')
            away_team = item.get('away_team', '')
            
            if isinstance(game_date, str):
                game_date = pd.to_datetime(game_date)
                
            # Parse hour
            if isinstance(game_time, str):
                try:
                    hour = int(game_time.split(':')[0])
                except:
                    hour = 19
            else:
                hour = game_time.hour if hasattr(game_time, 'hour') else 19
                
            day_of_week = game_date.weekday()
            
            # National broadcast indicators
            national_score = 0.0
            
            # Saturday night in Canada
            if (day_of_week == 5 and hour == 19 and
                self.include_canadian_broadcasts):
                canadian_teams = {'Toronto', 'Montreal', 'Ottawa', 'Calgary',
                                'Edmonton', 'Vancouver', 'Winnipeg'}
                if any(team in canadian_teams for team in [home_team, away_team]):
                    national_score = 1.0
                    
            # Wednesday rivalry night
            elif day_of_week == 2 and hour == 20:
                rivalry_matchups = [
                    {'Boston', 'Montreal'},
                    {'Toronto', 'Montreal'},
                    {'Pittsburgh', 'Philadelphia'},
                    {'New York Rangers', 'New York Islanders'}
                ]
                teams = {home_team, away_team}
                if any(teams == matchup for matchup in rivalry_matchups):
                    national_score = 0.9
                    
            # Sunday afternoon
            elif day_of_week == 6 and 12 <= hour <= 16:
                national_score = 0.7
                
            features.append(national_score)  # National
            features.append(0.5 if national_score < 0.5 else 0.0)  # Regional
            features.append(0.0 if national_score > 0.3 else 0.3)  # Local
            
        # Broadcast pressure multiplier
        max_broadcast = max(features[:3])
        features.append(max_broadcast)
        
        return features
        
    def _extract_special_windows(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract special broadcast window features.
        
        Returns 5 features for special broadcasts.
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        game_time = item.get('game_time', '19:00')
        home_team = item.get('home_team', '')
        away_team = item.get('away_team', '')
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Parse hour
        if isinstance(game_time, str):
            try:
                hour = int(game_time.split(':')[0])
            except:
                hour = 19
        else:
            hour = game_time.hour if hasattr(game_time, 'hour') else 19
            
        day_of_week = game_date.weekday()
        
        # Hockey Night in Canada
        hnc_score = 0.0
        if (self.include_canadian_broadcasts and
            day_of_week == 5 and hour == 19):
            canadian_teams = {'Toronto', 'Montreal', 'Ottawa', 'Calgary',
                            'Edmonton', 'Vancouver', 'Winnipeg'}
            if any(team in canadian_teams for team in [home_team, away_team]):
                hnc_score = 1.0
                
        features.append(hnc_score)
        
        # Wednesday Rivalry Night
        rivalry_score = 0.0
        if day_of_week == 2 and 19 <= hour <= 21:
            # Check for rivalry indicators
            division_rival = item.get('division_rival', False)
            games_played = item.get('season_series_games_played', 0)
            if division_rival or games_played >= 2:
                rivalry_score = 0.8
                
        features.append(rivalry_score)
        
        # Holiday games
        holiday_score = 0.0
        if game_date.month == 1 and game_date.day == 1:
            holiday_score = 1.0  # New Year's Day
        elif game_date.month == 12 and 24 <= game_date.day <= 26:
            holiday_score = 0.9  # Christmas period
        elif (game_date.month == 11 and 
              22 <= game_date.day <= 28 and
              day_of_week in [3, 4]):  # Thanksgiving
            holiday_score = 0.8
            
        features.append(holiday_score)
        
        # Stadium Series / Winter Classic
        outdoor_game = item.get('outdoor_game', False)
        special_event = item.get('special_event_game', False)
        
        features.append(1.0 if outdoor_game else 0.0)
        
        # Maximum special window impact
        max_special = max(hnc_score, rivalry_score, holiday_score,
                         1.0 if outdoor_game else 0.0)
        features.append(max_special)
        
        return features
        
    def _extract_matinee_effects(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract matinee game narrative effects.
        
        Returns 4 features for afternoon games.
        """
        features = []
        
        game_time = item.get('game_time', '19:00')
        game_date = item.get('game_date', datetime.now())
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Parse hour
        if isinstance(game_time, str):
            try:
                hour = int(game_time.split(':')[0])
            except:
                hour = 19
        else:
            hour = game_time.hour if hasattr(game_time, 'hour') else 19
            
        day_of_week = game_date.weekday()
        
        # Is matinee?
        is_matinee = hour < 17
        features.append(1.0 if is_matinee else 0.0)
        
        if is_matinee:
            # Weekend vs weekday matinee
            if day_of_week in [5, 6]:
                features.append(0.8)  # Weekend matinee
            else:
                features.append(0.6)  # Weekday matinee (unusual)
                
            # Early bird special (before 2 PM)
            if hour < 14:
                features.append(0.7)
            else:
                features.append(0.0)
                
            # Matinee after night game
            b2b_matinee = item.get('back_to_back_matinee', False)
            features.append(0.9 if b2b_matinee else 0.0)
            
        else:
            features.extend([0.0, 0.0, 0.0])
            
        return features
        
    def _extract_late_night_effects(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract late night game effects.
        
        Returns 3 features for late starts.
        """
        features = []
        
        game_time = item.get('game_time', '19:00')
        home_team = item.get('home_team', '')
        away_team = item.get('away_team', '')
        
        # Parse hour
        if isinstance(game_time, str):
            try:
                hour = int(game_time.split(':')[0])
            except:
                hour = 19
        else:
            hour = game_time.hour if hasattr(game_time, 'hour') else 19
            
        # Late start indicator
        is_late = hour >= 21
        features.append(1.0 if is_late else 0.0)
        
        # West coast late start for eastern team
        west_coast_teams = {'Los Angeles', 'Anaheim', 'San Jose', 'Vegas',
                           'Seattle', 'Vancouver', 'Calgary', 'Edmonton'}
        east_coast_teams = {'Boston', 'Buffalo', 'Detroit', 'Florida',
                           'Montreal', 'Ottawa', 'Tampa Bay', 'Toronto',
                           'Carolina', 'Columbus', 'New Jersey', 'NY Islanders',
                           'NY Rangers', 'Philadelphia', 'Pittsburgh', 'Washington'}
        
        if (is_late and 
            home_team in west_coast_teams and 
            away_team in east_coast_teams):
            features.append(0.9)  # Significant disadvantage
        else:
            features.append(0.0)
            
        # Extra late (10:30 PM or later local)
        if hour >= 22.5:
            features.append(0.8)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_broadcast_pressure(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract composite broadcast pressure features.
        
        Returns 4 features for overall broadcast narrative.
        """
        features = []
        
        # Aggregate all pressure indicators
        game_time = item.get('game_time', '19:00')
        game_date = item.get('game_date', datetime.now())
        
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Parse hour
        if isinstance(game_time, str):
            try:
                hour = int(game_time.split(':')[0])
            except:
                hour = 19
        else:
            hour = game_time.hour if hasattr(game_time, 'hour') else 19
            
        day_of_week = game_date.weekday()
        
        # Calculate composite pressure
        pressure_score = 0.0
        
        # Time slot pressure
        if 19 <= hour <= 21:
            pressure_score += 0.3
        if day_of_week in [4, 5, 6]:
            pressure_score += 0.2
            
        # Special circumstances
        if item.get('playoff_implications', False):
            pressure_score += 0.3
        if item.get('rivalry_game', False):
            pressure_score += 0.2
            
        features.append(min(1.0, pressure_score))
        
        # Viewership estimate (based on slot and teams)
        big_market_teams = {'Toronto', 'Montreal', 'New York Rangers',
                           'Boston', 'Chicago', 'Philadelphia', 'Detroit',
                           'Los Angeles', 'Pittsburgh'}
        
        home_team = item.get('home_team', '')
        away_team = item.get('away_team', '')
        
        viewership_score = 0.5  # Base
        if home_team in big_market_teams:
            viewership_score += 0.25
        if away_team in big_market_teams:
            viewership_score += 0.25
            
        features.append(min(1.0, viewership_score))
        
        # Narrative crystallization
        # How much the broadcast enhances existing narratives
        milestone_game = item.get('milestone_approaching', False)
        streak_game = item.get('streak_on_line', False)
        elimination_game = item.get('must_win', False)
        
        narrative_enhancement = 0.0
        if milestone_game:
            narrative_enhancement += 0.4
        if streak_game:
            narrative_enhancement += 0.3
        if elimination_game:
            narrative_enhancement += 0.5
            
        features.append(min(1.0, narrative_enhancement))
        
        # Overall broadcast narrative intensity
        overall_intensity = np.mean([
            features[0],  # Pressure
            features[1],  # Viewership
            features[2]   # Enhancement
        ])
        features.append(overall_intensity)
        
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Broadcast slot features
        names.extend([
            'slot_primetime',
            'slot_matinee',
            'slot_late_night',
            'slot_pressure',
            'slot_weekend_primetime'
        ])
        
        # Coverage type features
        names.extend([
            'coverage_national',
            'coverage_regional',
            'coverage_local',
            'coverage_max_level'
        ])
        
        # Special window features
        names.extend([
            'special_hockey_night_canada',
            'special_rivalry_night',
            'special_holiday_game',
            'special_outdoor_game',
            'special_max_impact'
        ])
        
        # Matinee features
        names.extend([
            'matinee_game',
            'matinee_weekend_vs_weekday',
            'matinee_early_bird',
            'matinee_after_night'
        ])
        
        # Late night features
        names.extend([
            'late_night_start',
            'late_east_in_west',
            'late_extra_late'
        ])
        
        # Broadcast pressure features
        names.extend([
            'pressure_composite',
            'pressure_viewership',
            'pressure_narrative_enhancement',
            'pressure_overall_intensity'
        ])
        
        return names
