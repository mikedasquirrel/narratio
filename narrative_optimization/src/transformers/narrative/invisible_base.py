"""
Base class for Invisible Narrative Transformers

These transformers extract narrative pressure from scheduling, timing, 
and structural patterns that operate below conscious awareness.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from ..base import NarrativeTransformer


class InvisibleNarrativeTransformer(NarrativeTransformer):
    """
    Base class for transformers that detect invisible narrative patterns.
    
    Provides utility methods for:
    - Calendar analysis
    - Schedule pattern detection
    - Milestone calculations
    - Season arc positioning
    """
    
    # Season arc milestones
    SEASON_MILESTONES = {
        'opening_week': (0, 7),
        'identity_formation': (8, 20),
        'thanksgiving_eval': (35, 45),  # American Thanksgiving
        'pre_christmas': (46, 55),
        'christmas_break': (56, 60),
        'new_year_reset': (61, 65),
        'all_star_approach': (66, 75),
        'all_star_break': (76, 80),
        'trade_deadline_approach': (81, 90),
        'final_push': (91, 110),
        'playoff_race': (111, 125),
        'season_finale': (126, 132)
    }
    
    # Day of week narrative weights
    DAY_WEIGHTS = {
        0: 0.7,  # Monday - fresh start
        1: 0.5,  # Tuesday - routine
        2: 0.6,  # Wednesday - rivalry night potential
        3: 0.5,  # Thursday - routine
        4: 0.8,  # Friday - showcase
        5: 1.0,  # Saturday - primetime
        6: 0.7   # Sunday - varied (afternoon/evening)
    }
    
    def __init__(self, narrative_id: str, description: str):
        super().__init__(narrative_id, description)
        
    def calculate_season_position(self, game_date: datetime, season_start: datetime) -> Tuple[int, str]:
        """
        Calculate day of season and narrative arc phase.
        
        Returns
        -------
        day_of_season : int
            Number of days since season start
        phase : str
            Current season narrative phase
        """
        days_elapsed = (game_date - season_start).days
        
        # Determine phase
        phase = 'regular'
        for phase_name, (start, end) in self.SEASON_MILESTONES.items():
            if start <= days_elapsed <= end:
                phase = phase_name
                break
                
        return days_elapsed, phase
        
    def calculate_days_between_games(self, current_date: datetime, 
                                   previous_date: Optional[datetime]) -> int:
        """Calculate rest days between games."""
        if previous_date is None:
            return 7  # Assume well-rested for first game
            
        return (current_date - previous_date).days
        
    def detect_back_to_back(self, days_rest: int) -> bool:
        """Detect back-to-back game situation."""
        return days_rest <= 1
        
    def calculate_schedule_density(self, game_dates: List[datetime], 
                                 current_date: datetime,
                                 window_days: int = 7) -> int:
        """
        Calculate games played in recent window.
        
        Parameters
        ----------
        game_dates : list of datetime
            All game dates in season
        current_date : datetime
            Current game date
        window_days : int
            Number of days to look back
            
        Returns
        -------
        int
            Number of games in window
        """
        window_start = current_date - timedelta(days=window_days)
        games_in_window = sum(1 for date in game_dates 
                            if window_start <= date < current_date)
        return games_in_window
        
    def infer_broadcast_type(self, game_datetime: datetime, 
                           home_team: str, away_team: str) -> str:
        """
        Infer broadcast type from game time and teams.
        
        Returns
        -------
        str
            'national', 'regional', or 'local'
        """
        hour = game_datetime.hour
        day_of_week = game_datetime.weekday()
        
        # Saturday night in Canada
        canadian_teams = {'Toronto', 'Montreal', 'Ottawa', 'Winnipeg', 
                         'Calgary', 'Edmonton', 'Vancouver'}
        if day_of_week == 5 and 18 <= hour <= 20:  # Saturday 6-8 PM
            if home_team in canadian_teams or away_team in canadian_teams:
                return 'national'
                
        # Sunday afternoon US
        if day_of_week == 6 and 12 <= hour <= 16:  # Sunday noon-4 PM
            return 'national'
            
        # Wednesday rivalry night
        if day_of_week == 2 and hour == 20:  # Wednesday 8 PM
            return 'national'
            
        # Late games are usually regional/local
        if hour >= 22 or hour <= 10:
            return 'local'
            
        return 'regional'
        
    def calculate_milestone_proximity(self, current_value: int, 
                                    milestones: List[int]) -> Tuple[int, int]:
        """
        Calculate distance to nearest milestone.
        
        Returns
        -------
        distance_to_next : int
            Distance to next milestone (0 if past all)
        distance_from_last : int
            Distance from last milestone (0 if before all)
        """
        future_milestones = [m for m in milestones if m > current_value]
        past_milestones = [m for m in milestones if m <= current_value]
        
        distance_to_next = min(future_milestones) - current_value if future_milestones else 0
        distance_from_last = current_value - max(past_milestones) if past_milestones else 0
        
        return distance_to_next, distance_from_last
        
    def detect_elimination_proximity(self, current_record: Tuple[int, int],
                                   games_remaining: int,
                                   playoff_line: int) -> Dict[str, Any]:
        """
        Calculate elimination and playoff proximity.
        
        Parameters
        ----------
        current_record : tuple
            (wins, losses)
        games_remaining : int
            Games left in season
        playoff_line : int
            Points needed for playoffs (estimated)
            
        Returns
        -------
        dict
            Elimination proximity metrics
        """
        wins, losses = current_record
        current_points = wins * 2  # NHL points system
        max_possible_points = current_points + (games_remaining * 2)
        
        # Simple elimination detection
        if max_possible_points < playoff_line:
            games_since_elimination = 0  # Already eliminated
        else:
            # How many losses until elimination?
            points_needed = playoff_line - current_points
            games_to_elimination = max(0, games_remaining - (points_needed // 2))
            games_since_elimination = -games_to_elimination
            
        return {
            'eliminated': max_possible_points < playoff_line,
            'games_to_elimination': max(0, -games_since_elimination),
            'games_since_elimination': max(0, games_since_elimination),
            'playoff_probability': min(1.0, current_points / playoff_line),
            'must_win_percentage': max(0.0, points_needed / (games_remaining * 2))
        }
        
    def calculate_season_series_position(self, meetings_so_far: int,
                                       total_meetings: int) -> Dict[str, float]:
        """
        Calculate narrative position within season series.
        
        Returns
        -------
        dict
            Series position features
        """
        if total_meetings == 0:
            return {'series_position': 0.0, 'series_fatigue': 0.0}
            
        position = meetings_so_far / total_meetings
        
        # Fatigue increases with meetings
        fatigue = min(1.0, meetings_so_far / 4.0)
        
        # Uncertainty highest in first meeting, lowest in middle
        if meetings_so_far == 0:
            uncertainty = 1.0
        elif meetings_so_far == total_meetings:
            uncertainty = 0.7  # Final meeting has some uncertainty
        else:
            uncertainty = 0.3
            
        return {
            'series_position': position,
            'series_fatigue': fatigue,
            'series_uncertainty': uncertainty
        }
