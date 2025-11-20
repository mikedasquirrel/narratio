"""
Calendar Rhythm Transformer

Detects narrative patterns in calendar positioning - the rhythms of the week,
month, and season that create invisible pressure on outcomes.

Time itself has narrative weight. Some days matter more than others.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin


class CalendarRhythmTransformer(BaseEstimator, TransformerMixin):
    """
    Extract calendar-based narrative rhythms.
    
    Philosophy:
    - Days of week have different narrative energy
    - Months have psychological arcs
    - Holidays create special narrative contexts
    - Anniversary dates echo with power
    - Season timing affects urgency
    
    Features (30 total):
    - Day of week effects (7)
    - Day of month patterns (5)
    - Holiday proximity effects (5)
    - Season arc positioning (6)
    - Anniversary date power (4)
    - Monthly rhythm (3)
    """
    
    # Day narratives
    DAY_NARRATIVES = {
        0: {'name': 'Monday', 'energy': 0.7, 'type': 'fresh_start'},
        1: {'name': 'Tuesday', 'energy': 0.5, 'type': 'routine'},
        2: {'name': 'Wednesday', 'energy': 0.6, 'type': 'rivalry_night'},
        3: {'name': 'Thursday', 'energy': 0.5, 'type': 'routine'},
        4: {'name': 'Friday', 'energy': 0.8, 'type': 'showcase'},
        5: {'name': 'Saturday', 'energy': 1.0, 'type': 'primetime'},
        6: {'name': 'Sunday', 'energy': 0.7, 'type': 'varied'}
    }
    
    # Key holidays and their narrative impact
    HOLIDAYS = {
        'new_years': {'date': (1, 1), 'impact': 1.0, 'window': 3},
        'mlk_day': {'date': (1, 15), 'impact': 0.6, 'window': 1},  # Approximate
        'valentines': {'date': (2, 14), 'impact': 0.7, 'window': 1},
        'st_patricks': {'date': (3, 17), 'impact': 0.8, 'window': 1},
        'easter': {'date': (4, 1), 'impact': 0.6, 'window': 2},  # Variable
        'mothers_day': {'date': (5, 10), 'impact': 0.5, 'window': 1},  # Approximate
        'memorial_day': {'date': (5, 25), 'impact': 0.7, 'window': 3},  # Approximate
        'fathers_day': {'date': (6, 15), 'impact': 0.5, 'window': 1},  # Approximate
        'independence_day': {'date': (7, 4), 'impact': 0.8, 'window': 2},
        'labor_day': {'date': (9, 1), 'impact': 0.7, 'window': 3},  # Approximate
        'thanksgiving_us': {'date': (11, 24), 'impact': 0.9, 'window': 4},  # Approximate
        'black_friday': {'date': (11, 25), 'impact': 0.8, 'window': 1},  # Approximate
        'christmas': {'date': (12, 25), 'impact': 1.0, 'window': 5},
        'new_years_eve': {'date': (12, 31), 'impact': 0.9, 'window': 1}
    }
    
    # Canadian holidays for NHL
    CANADIAN_HOLIDAYS = {
        'canada_day': {'date': (7, 1), 'impact': 0.9, 'window': 2},
        'thanksgiving_ca': {'date': (10, 10), 'impact': 0.8, 'window': 2},  # Approximate
        'remembrance_day': {'date': (11, 11), 'impact': 0.7, 'window': 1},
        'boxing_day': {'date': (12, 26), 'impact': 0.8, 'window': 1}
    }
    
    def __init__(
        self,
        include_canadian_holidays: bool = True,
        anniversary_lookback_years: int = 10
    ):
        """
        Initialize calendar rhythm analyzer.
        
        Parameters
        ----------
        include_canadian_holidays : bool
            Include Canadian holidays for NHL games
        anniversary_lookback_years : int
            Years to look back for anniversary games
        """
        self.include_canadian_holidays = include_canadian_holidays
        self.anniversary_lookback_years = anniversary_lookback_years
        
        # Combine holiday lists if needed
        self.all_holidays = self.HOLIDAYS.copy()
        if include_canadian_holidays:
            self.all_holidays.update(self.CANADIAN_HOLIDAYS)
            
    def fit(self, X, y=None):
        """No fitting required for calendar analysis."""
        return self
        
    def transform(self, X):
        """
        Extract calendar rhythm features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with date information
            
        Returns
        -------
        np.ndarray
            Calendar features (n_samples, 30)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Day of week effects (7)
            dow_features = self._extract_day_of_week(item)
            feature_vec.extend(dow_features)
            
            # Day of month patterns (5)
            dom_features = self._extract_day_of_month(item)
            feature_vec.extend(dom_features)
            
            # Holiday proximity (5)
            holiday_features = self._extract_holiday_proximity(item)
            feature_vec.extend(holiday_features)
            
            # Season arc positioning (6)
            arc_features = self._extract_season_arc(item)
            feature_vec.extend(arc_features)
            
            # Anniversary power (4)
            anniversary_features = self._extract_anniversary_power(item)
            feature_vec.extend(anniversary_features)
            
            # Monthly rhythm (3)
            monthly_features = self._extract_monthly_rhythm(item)
            feature_vec.extend(monthly_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_day_of_week(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract day of week narrative features.
        
        Returns 7 features (one-hot encoding).
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        dow = game_date.weekday()
        
        # One-hot encode the day
        for day in range(7):
            features.append(1.0 if dow == day else 0.0)
            
        return features
        
    def _extract_day_of_month(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract day of month narrative patterns.
        
        Returns 5 features for monthly positioning.
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        day_of_month = game_date.day
        days_in_month = self._days_in_month(game_date.year, game_date.month)
        
        # Beginning of month (fresh start)
        if day_of_month <= 3:
            features.append(1.0 - (day_of_month - 1) / 3.0)
        else:
            features.append(0.0)
            
        # End of month (urgency)
        if day_of_month >= days_in_month - 2:
            features.append((day_of_month - (days_in_month - 3)) / 3.0)
        else:
            features.append(0.0)
            
        # Mid-month doldrums
        mid_point = days_in_month // 2
        if mid_point - 2 <= day_of_month <= mid_point + 2:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # First Friday (payday games)
        if 1 <= day_of_month <= 7 and game_date.weekday() == 4:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Monthly position (0-1)
        features.append(day_of_month / days_in_month)
        
        return features
        
    def _extract_holiday_proximity(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract holiday proximity effects.
        
        Returns 5 features for holiday narratives.
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Find nearest holidays
        holiday_impacts = []
        
        for holiday_name, holiday_info in self.all_holidays.items():
            month, day = holiday_info['date']
            impact = holiday_info['impact']
            window = holiday_info['window']
            
            # Create holiday date for this year
            try:
                holiday_date = datetime(game_date.year, month, day)
                
                # Check if holiday is in previous year (for early season)
                if holiday_date > game_date + timedelta(days=180):
                    holiday_date = datetime(game_date.year - 1, month, day)
                # Check if holiday is in next year (for late season)
                elif holiday_date < game_date - timedelta(days=180):
                    holiday_date = datetime(game_date.year + 1, month, day)
                    
                days_away = abs((game_date - holiday_date).days)
                
                if days_away <= window:
                    # Impact decreases with distance
                    adjusted_impact = impact * (1.0 - days_away / (window + 1))
                    holiday_impacts.append(adjusted_impact)
                    
            except ValueError:
                # Invalid date (e.g., Feb 29 in non-leap year)
                continue
                
        # Maximum holiday impact
        max_impact = max(holiday_impacts) if holiday_impacts else 0.0
        features.append(max_impact)
        
        # Multiple holidays
        features.append(min(1.0, len(holiday_impacts) * 0.3))
        
        # Special holiday categories
        # Christmas season (Dec 20-Jan 2)
        if (game_date.month == 12 and game_date.day >= 20) or \
           (game_date.month == 1 and game_date.day <= 2):
            features.append(0.9)
        else:
            features.append(0.0)
            
        # American Thanksgiving week
        if game_date.month == 11 and 22 <= game_date.day <= 28:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Weekend holiday (extra energy)
        if max_impact > 0.5 and game_date.weekday() in [4, 5, 6]:
            features.append(0.7)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_season_arc(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract season arc narrative positioning.
        
        Returns 6 features for season narrative phases.
        """
        features = []
        
        # Use game number or calculate from date
        game_number = item.get('game_number', 41)
        total_games = item.get('total_season_games', 82)
        
        season_progress = game_number / total_games
        
        # Season phases
        # Opening statement games
        if game_number <= 10:
            features.append(1.0 - game_number / 10.0)
        else:
            features.append(0.0)
            
        # Pre-All-Star push
        if 0.4 <= season_progress <= 0.5:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Post-All-Star reset
        if 0.5 <= season_progress <= 0.55:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Trade deadline drama
        if 0.65 <= season_progress <= 0.7:
            features.append(0.9)
        else:
            features.append(0.0)
            
        # Playoff push
        if season_progress >= 0.8:
            intensity = (season_progress - 0.8) / 0.2
            features.append(min(1.0, intensity * 1.2))
        else:
            features.append(0.0)
            
        # Dog days (mid-February slog)
        if 0.55 <= season_progress <= 0.65:
            features.append(0.6)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_anniversary_power(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract anniversary date narrative power.
        
        Returns 4 features for anniversary effects.
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        # Check for anniversaries of significant events
        anniversary_matches = []
        
        # Team-specific anniversaries
        significant_dates = item.get('team_significant_dates', [])
        
        # Add some default significant dates if not provided
        if not significant_dates:
            # Examples: Stanley Cup wins, franchise founding, etc.
            significant_dates = [
                {'date': '1967-06-05', 'importance': 1.0},  # Example Cup win
                {'date': '1972-09-28', 'importance': 0.9},  # Summit Series
                {'date': '1980-02-22', 'importance': 1.0},  # Miracle on Ice
            ]
            
        for sig_date in significant_dates:
            if isinstance(sig_date, dict):
                date_str = sig_date.get('date', '')
                importance = sig_date.get('importance', 0.5)
            else:
                date_str = sig_date
                importance = 0.5
                
            try:
                sig_datetime = pd.to_datetime(date_str)
                
                # Check if same month/day
                if sig_datetime.month == game_date.month and \
                   sig_datetime.day == game_date.day:
                    years_ago = game_date.year - sig_datetime.year
                    if 0 < years_ago <= self.anniversary_lookback_years:
                        # Round anniversaries more powerful
                        if years_ago % 25 == 0:
                            power = 1.0
                        elif years_ago % 10 == 0:
                            power = 0.9
                        elif years_ago % 5 == 0:
                            power = 0.7
                        else:
                            power = 0.5
                            
                        anniversary_matches.append(power * importance)
                        
            except:
                continue
                
        # Maximum anniversary power
        features.append(max(anniversary_matches) if anniversary_matches else 0.0)
        
        # Multiple anniversaries
        features.append(min(1.0, len(anniversary_matches) * 0.4))
        
        # Special date categories
        # Season opener anniversary
        is_season_opener = item.get('is_season_opener', False)
        features.append(1.0 if is_season_opener else 0.0)
        
        # Rivalry anniversary game
        rivalry_anniversary = item.get('rivalry_anniversary_game', False)
        features.append(0.8 if rivalry_anniversary else 0.0)
        
        return features
        
    def _extract_monthly_rhythm(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract monthly rhythm patterns.
        
        Returns 3 features for monthly narratives.
        """
        features = []
        
        game_date = item.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
            
        month = game_date.month
        
        # Month-specific energy levels (NHL season)
        month_energy = {
            10: 1.0,  # October - fresh start
            11: 0.8,  # November - finding rhythm
            12: 0.9,  # December - holiday games
            1: 0.7,   # January - midwinter grind
            2: 0.6,   # February - dog days
            3: 0.8,   # March - playoff push begins
            4: 1.0,   # April - playoff intensity
            5: 1.0,   # May - playoffs
            6: 1.0    # June - Finals
        }
        
        features.append(month_energy.get(month, 0.5))
        
        # Games played in month so far
        games_this_month = item.get('games_played_this_month', 0)
        
        # Monthly fatigue
        if games_this_month > 12:
            fatigue = min(1.0, (games_this_month - 12) / 5.0)
        else:
            fatigue = 0.0
        features.append(fatigue)
        
        # Fresh month energy (first 3 games)
        if games_this_month <= 3:
            features.append(0.7)
        else:
            features.append(0.0)
            
        return features
        
    def _days_in_month(self, year: int, month: int) -> int:
        """Get number of days in a month."""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
            
        this_month = datetime(year, month, 1)
        return (next_month - this_month).days
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Day of week features
        names.extend([
            'dow_monday',
            'dow_tuesday',
            'dow_wednesday_rivalry',
            'dow_thursday',
            'dow_friday_showcase',
            'dow_saturday_primetime',
            'dow_sunday'
        ])
        
        # Day of month features
        names.extend([
            'dom_beginning_fresh',
            'dom_end_urgency',
            'dom_midmonth_doldrums',
            'dom_first_friday',
            'dom_position'
        ])
        
        # Holiday features
        names.extend([
            'holiday_max_impact',
            'holiday_multiple',
            'holiday_christmas_season',
            'holiday_thanksgiving_week',
            'holiday_weekend_energy'
        ])
        
        # Season arc features
        names.extend([
            'arc_opening_statement',
            'arc_allstar_push',
            'arc_allstar_reset',
            'arc_trade_deadline',
            'arc_playoff_push',
            'arc_dog_days'
        ])
        
        # Anniversary features
        names.extend([
            'anniversary_max_power',
            'anniversary_multiple',
            'anniversary_season_opener',
            'anniversary_rivalry_game'
        ])
        
        # Monthly rhythm features
        names.extend([
            'month_energy_level',
            'month_fatigue',
            'month_fresh_start'
        ])
        
        return names
