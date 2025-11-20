"""
Milestone Proximity Transformer

Calculates proximity to narrative-significant numbers that create
unconscious pressure on outcomes. Round numbers matter more than we think.

The approach of milestones creates invisible narrative gravity.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from .invisible_base import InvisibleNarrativeTransformer


class MilestoneProximityTransformer(InvisibleNarrativeTransformer):
    """
    Extract milestone proximity narrative features.
    
    Philosophy:
    - Round numbers create unconscious pressure
    - Proximity to milestones affects performance
    - Just-passed milestones create relief
    - Multiple approaching milestones compound pressure
    - Some numbers are more "milestone-y" than others
    
    Features (35 total):
    - Game milestones (6)
    - Point milestones (6)
    - Goal milestones (6)
    - Win milestones (5)
    - Save milestones (5)
    - Team record proximities (4)
    - Milestone convergence (3)
    """
    
    def __init__(
        self,
        milestone_window: int = 10,
        include_team_milestones: bool = True
    ):
        """
        Initialize milestone proximity analyzer.
        
        Parameters
        ----------
        milestone_window : int
            Games/points to look ahead for milestones
        include_team_milestones : bool
            Include team-level milestone tracking
        """
        super().__init__(
            narrative_id='milestone_proximity',
            description='Detects proximity to round number milestones'
        )
        self.milestone_window = milestone_window
        self.include_team_milestones = include_team_milestones
        
        # Define milestone thresholds
        self.milestones = {
            'games': [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
                     1100, 1200, 1300, 1400, 1500],
            'points': [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 
                      1000, 1100, 1200, 1300, 1400, 1500],
            'goals': [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                     550, 600, 650, 700, 750, 800],
            'assists': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'wins': [100, 200, 300, 400, 500, 600, 700],
            'saves': [10000, 15000, 20000, 25000, 30000, 35000, 40000],
            'team_wins': [500, 1000, 1500, 2000, 2500, 3000, 3500],
            'team_points': [50, 60, 70, 80, 90, 100, 110, 120]
        }
        
        # Milestone gravity (how much pull they have)
        self.milestone_gravity = {
            1: 1.0,       # First goal/point/win
            100: 0.9,     # Century marks
            500: 0.95,    # Half millenniums
            1000: 1.0,    # Millenniums
            'round_50': 0.7,   # Multiples of 50
            'round_100': 0.85, # Multiples of 100
            'default': 0.5
        }
        
    def fit(self, X, y=None):
        """No fitting required for milestone analysis."""
        return self
        
    def transform(self, X):
        """
        Extract milestone proximity features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with player/team statistics
            
        Returns
        -------
        np.ndarray
            Milestone features (n_samples, 35)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Game milestones (6)
            game_features = self._extract_game_milestones(item)
            feature_vec.extend(game_features)
            
            # Point milestones (6)
            point_features = self._extract_point_milestones(item)
            feature_vec.extend(point_features)
            
            # Goal milestones (6)
            goal_features = self._extract_goal_milestones(item)
            feature_vec.extend(goal_features)
            
            # Win milestones (5)
            win_features = self._extract_win_milestones(item)
            feature_vec.extend(win_features)
            
            # Save milestones (5)
            save_features = self._extract_save_milestones(item)
            feature_vec.extend(save_features)
            
            # Team record proximities (4)
            if self.include_team_milestones:
                team_features = self._extract_team_milestones(item)
            else:
                team_features = [0.0] * 4
            feature_vec.extend(team_features)
            
            # Milestone convergence (3)
            convergence_features = self._extract_milestone_convergence(item)
            feature_vec.extend(convergence_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _calculate_milestone_pressure(self, current_value: int, 
                                    milestone_list: List[int]) -> Dict[str, float]:
        """
        Calculate pressure from nearest milestones.
        
        Returns dict with proximity metrics.
        """
        distance_to, distance_from = self.calculate_milestone_proximity(
            current_value, milestone_list
        )
        
        # No pressure if far from milestones
        if distance_to > self.milestone_window and distance_from > self.milestone_window:
            return {
                'approach_pressure': 0.0,
                'recent_relief': 0.0,
                'milestone_gravity': 0.0
            }
            
        # Calculate approach pressure
        if 0 < distance_to <= self.milestone_window:
            next_milestone = current_value + distance_to
            gravity = self._get_milestone_gravity(next_milestone)
            # Pressure increases as we get closer
            approach_pressure = gravity * (1.0 - distance_to / self.milestone_window)
        else:
            approach_pressure = 0.0
            
        # Calculate recent milestone relief
        if 0 < distance_from <= 5:  # Just passed
            prev_milestone = current_value - distance_from
            gravity = self._get_milestone_gravity(prev_milestone)
            # Relief fades quickly
            recent_relief = gravity * (1.0 - distance_from / 5.0)
        else:
            recent_relief = 0.0
            
        return {
            'approach_pressure': approach_pressure,
            'recent_relief': recent_relief,
            'milestone_gravity': gravity if distance_to <= self.milestone_window else 0.0
        }
        
    def _get_milestone_gravity(self, milestone_value: int) -> float:
        """Get the narrative gravity of a specific milestone number."""
        # Special milestones
        if milestone_value in self.milestone_gravity:
            return self.milestone_gravity[milestone_value]
            
        # Round number categories
        if milestone_value % 1000 == 0:
            return 1.0
        elif milestone_value % 500 == 0:
            return 0.95
        elif milestone_value % 100 == 0:
            return self.milestone_gravity['round_100']
        elif milestone_value % 50 == 0:
            return self.milestone_gravity['round_50']
        else:
            return self.milestone_gravity['default']
            
    def _extract_game_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract career game milestone features.
        
        Returns 6 features for game milestones.
        """
        features = []
        
        # Get key players' game counts
        players = []
        
        # Home team stars
        if 'home_team_leaders' in item:
            for player in item['home_team_leaders']:
                players.append({
                    'games': player.get('career_games', 0),
                    'is_home': True
                })
                
        # Away team stars
        if 'away_team_leaders' in item:
            for player in item['away_team_leaders']:
                players.append({
                    'games': player.get('career_games', 0),
                    'is_home': False
                })
                
        # If no detailed data, use summary
        if not players:
            players = [
                {'games': item.get('home_star_games', 450), 'is_home': True},
                {'games': item.get('away_star_games', 320), 'is_home': False}
            ]
            
        # Find most significant milestone approach
        max_pressure = 0.0
        home_pressure = 0.0
        away_pressure = 0.0
        milestone_count = 0
        
        for player in players[:6]:  # Top 6 players
            pressure_data = self._calculate_milestone_pressure(
                player['games'], self.milestones['games']
            )
            
            if pressure_data['approach_pressure'] > 0:
                milestone_count += 1
                if player['is_home']:
                    home_pressure = max(home_pressure, pressure_data['approach_pressure'])
                else:
                    away_pressure = max(away_pressure, pressure_data['approach_pressure'])
                    
            max_pressure = max(max_pressure, pressure_data['approach_pressure'])
            
        features.append(max_pressure)              # Maximum game milestone pressure
        features.append(home_pressure)             # Home team milestone pressure
        features.append(away_pressure)             # Away team milestone pressure
        features.append(min(1.0, milestone_count * 0.3))  # Multiple milestones
        
        # First NHL game (special milestone)
        first_game_home = item.get('home_player_first_game', False)
        first_game_away = item.get('away_player_first_game', False)
        
        features.append(1.0 if first_game_home else 0.0)
        features.append(1.0 if first_game_away else 0.0)
        
        return features
        
    def _extract_point_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract career point milestone features.
        
        Returns 6 features for point milestones.
        """
        features = []
        
        # Get top scorers' point totals
        scorers = []
        
        # Use leader data if available
        for key in ['home_team_leaders', 'away_team_leaders']:
            if key in item:
                is_home = 'home' in key
                for player in item[key][:3]:  # Top 3 scorers
                    scorers.append({
                        'points': player.get('career_points', 0),
                        'is_home': is_home
                    })
                    
        # Fallback to simple data
        if not scorers:
            scorers = [
                {'points': item.get('home_leader_points', 650), 'is_home': True},
                {'points': item.get('away_leader_points', 420), 'is_home': False}
            ]
            
        # Calculate milestone pressures
        max_pressure = 0.0
        max_gravity = 0.0
        approaching_count = 0
        
        for scorer in scorers:
            pressure_data = self._calculate_milestone_pressure(
                scorer['points'], self.milestones['points']
            )
            
            if pressure_data['approach_pressure'] > 0:
                approaching_count += 1
                
            max_pressure = max(max_pressure, pressure_data['approach_pressure'])
            max_gravity = max(max_gravity, pressure_data['milestone_gravity'])
            
        features.append(max_pressure)              # Maximum point milestone pressure
        features.append(max_gravity)               # Milestone importance
        features.append(min(1.0, approaching_count * 0.4))  # Multiple players approaching
        
        # Special point milestones
        first_point = item.get('player_seeking_first_point', False)
        features.append(1.0 if first_point else 0.0)
        
        # Points in specific timeframes (season milestones)
        season_50_proximity = item.get('player_near_50_point_season', False)
        season_100_proximity = item.get('player_near_100_point_season', False)
        
        features.append(0.8 if season_50_proximity else 0.0)
        features.append(1.0 if season_100_proximity else 0.0)
        
        return features
        
    def _extract_goal_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract goal milestone features.
        
        Returns 6 features for goal milestones.
        """
        features = []
        
        # Get goal scorers
        goal_scorers = item.get('key_goal_scorers', [])
        
        if not goal_scorers:
            # Use summary data
            goal_scorers = [
                {'goals': item.get('home_top_scorer_goals', 250)},
                {'goals': item.get('away_top_scorer_goals', 180)}
            ]
            
        # Find most significant goal milestone
        max_pressure = 0.0
        special_milestone = 0.0
        
        for scorer in goal_scorers[:4]:
            goals = scorer.get('goals', 0)
            pressure_data = self._calculate_milestone_pressure(
                goals, self.milestones['goals']
            )
            
            max_pressure = max(max_pressure, pressure_data['approach_pressure'])
            
            # Special milestone detection (500, 600, 700)
            distance_to_500 = 500 - goals if goals < 500 else 999
            distance_to_600 = 600 - goals if goals < 600 else 999
            distance_to_700 = 700 - goals if goals < 700 else 999
            
            if distance_to_500 <= 5:
                special_milestone = max(special_milestone, 1.0 - distance_to_500 / 5.0)
            elif distance_to_600 <= 5:
                special_milestone = max(special_milestone, 0.9 - distance_to_600 / 5.0)
            elif distance_to_700 <= 5:
                special_milestone = max(special_milestone, 0.95 - distance_to_700 / 5.0)
                
        features.append(max_pressure)              # Goal milestone pressure
        features.append(special_milestone)         # Special goal milestone
        
        # First career goal
        first_goal_seeker = item.get('player_seeking_first_goal', False)
        features.append(1.0 if first_goal_seeker else 0.0)
        
        # Hat trick milestone (3 goals in game)
        player_on_2_goals = item.get('player_has_2_goals_today', False)
        features.append(0.9 if player_on_2_goals else 0.0)
        
        # Season goal milestones
        near_50_goals = item.get('player_near_50_goal_season', False)
        near_60_goals = item.get('player_near_60_goal_season', False)
        
        features.append(0.9 if near_50_goals else 0.0)
        features.append(1.0 if near_60_goals else 0.0)
        
        return features
        
    def _extract_win_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract coach/goalie win milestone features.
        
        Returns 5 features for win milestones.
        """
        features = []
        
        # Coach wins
        home_coach_wins = item.get('home_coach_career_wins', 450)
        away_coach_wins = item.get('away_coach_career_wins', 320)
        
        coach_pressures = []
        for wins in [home_coach_wins, away_coach_wins]:
            pressure_data = self._calculate_milestone_pressure(
                wins, self.milestones['wins']
            )
            coach_pressures.append(pressure_data['approach_pressure'])
            
        features.append(max(coach_pressures))      # Max coach milestone pressure
        features.append(coach_pressures[0])        # Home coach pressure
        features.append(coach_pressures[1])        # Away coach pressure
        
        # Goalie wins
        home_goalie_wins = item.get('home_goalie_career_wins', 200)
        away_goalie_wins = item.get('away_goalie_career_wins', 150)
        
        goalie_pressures = []
        for wins in [home_goalie_wins, away_goalie_wins]:
            pressure_data = self._calculate_milestone_pressure(
                wins, self.milestones['wins']
            )
            goalie_pressures.append(pressure_data['approach_pressure'])
            
        features.append(max(goalie_pressures))     # Max goalie milestone pressure
        
        # First career win
        seeking_first_win = item.get('goalie_seeking_first_win', False)
        features.append(1.0 if seeking_first_win else 0.0)
        
        return features
        
    def _extract_save_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract goalie save milestone features.
        
        Returns 5 features for save milestones.
        """
        features = []
        
        # Career saves
        home_goalie_saves = item.get('home_goalie_career_saves', 15000)
        away_goalie_saves = item.get('away_goalie_career_saves', 12000)
        
        save_pressures = []
        for saves in [home_goalie_saves, away_goalie_saves]:
            pressure_data = self._calculate_milestone_pressure(
                saves, self.milestones['saves']
            )
            save_pressures.append(pressure_data['approach_pressure'])
            
        features.append(max(save_pressures))       # Max save milestone pressure
        features.append(save_pressures[0])         # Home goalie pressure
        features.append(save_pressures[1])         # Away goalie pressure
        
        # Shutout milestones
        home_goalie_shutouts = item.get('home_goalie_career_shutouts', 35)
        away_goalie_shutouts = item.get('away_goalie_career_shutouts', 25)
        
        # Check for round shutout numbers (10, 25, 50, 75, 100)
        shutout_milestones = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]
        
        shutout_pressure = 0.0
        for shutouts in [home_goalie_shutouts, away_goalie_shutouts]:
            pressure_data = self._calculate_milestone_pressure(
                shutouts, shutout_milestones
            )
            shutout_pressure = max(shutout_pressure, pressure_data['approach_pressure'])
            
        features.append(shutout_pressure)
        
        # Save percentage milestones (need current stats)
        chasing_930_sv = item.get('goalie_chasing_930_save_pct', False)
        features.append(0.8 if chasing_930_sv else 0.0)
        
        return features
        
    def _extract_team_milestones(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract team record proximity features.
        
        Returns 4 features for team milestones.
        """
        features = []
        
        # Team wins
        home_team_wins = item.get('home_team_total_wins', 1250)
        away_team_wins = item.get('away_team_total_wins', 980)
        
        team_win_pressure = 0.0
        for wins in [home_team_wins, away_team_wins]:
            pressure_data = self._calculate_milestone_pressure(
                wins, self.milestones['team_wins']
            )
            team_win_pressure = max(team_win_pressure, pressure_data['approach_pressure'])
            
        features.append(team_win_pressure)
        
        # Season points
        home_team_points = item.get('home_team_points', 65)
        away_team_points = item.get('away_team_points', 58)
        
        points_pressure = 0.0
        for points in [home_team_points, away_team_points]:
            pressure_data = self._calculate_milestone_pressure(
                points, self.milestones['team_points']
            )
            points_pressure = max(points_pressure, pressure_data['approach_pressure'])
            
        features.append(points_pressure)
        
        # Franchise records proximity
        near_franchise_record = item.get('near_franchise_record', False)
        features.append(0.9 if near_franchise_record else 0.0)
        
        # Win streak milestones (5, 10, 15, 20)
        current_win_streak = item.get('current_win_streak', 0)
        streak_milestones = [5, 7, 10, 12, 15, 17, 20]
        
        streak_pressure = self._calculate_milestone_pressure(
            current_win_streak, streak_milestones
        )
        features.append(streak_pressure['approach_pressure'])
        
        return features
        
    def _extract_milestone_convergence(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract features for multiple converging milestones.
        
        Returns 3 features for milestone convergence effects.
        """
        features = []
        
        # Count approaching milestones
        approaching_milestones = item.get('total_approaching_milestones', 0)
        
        # Single milestone vs multiple
        if approaching_milestones == 0:
            features.append(0.0)
        elif approaching_milestones == 1:
            features.append(0.5)
        elif approaching_milestones == 2:
            features.append(0.8)
        else:
            features.append(1.0)  # 3+ milestones create maximum pressure
            
        # Milestone collision (multiple in same game)
        same_game_milestones = item.get('potential_same_game_milestones', 0)
        features.append(min(1.0, same_game_milestones * 0.5))
        
        # Team vs individual milestone conflict
        team_milestone = item.get('team_approaching_milestone', False)
        individual_milestone = item.get('player_approaching_milestone', False)
        
        if team_milestone and individual_milestone:
            features.append(0.9)  # Competing narratives
        elif team_milestone or individual_milestone:
            features.append(0.5)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Game milestone features
        names.extend([
            'game_milestone_max_pressure',
            'game_milestone_home_pressure',
            'game_milestone_away_pressure',
            'game_multiple_milestones',
            'game_first_nhl_home',
            'game_first_nhl_away'
        ])
        
        # Point milestone features
        names.extend([
            'point_milestone_pressure',
            'point_milestone_gravity',
            'point_multiple_approaching',
            'point_first_career',
            'point_season_50',
            'point_season_100'
        ])
        
        # Goal milestone features
        names.extend([
            'goal_milestone_pressure',
            'goal_special_milestone',
            'goal_first_career',
            'goal_hat_trick_watch',
            'goal_season_50',
            'goal_season_60'
        ])
        
        # Win milestone features
        names.extend([
            'win_coach_max_pressure',
            'win_coach_home_pressure',
            'win_coach_away_pressure',
            'win_goalie_pressure',
            'win_goalie_first'
        ])
        
        # Save milestone features
        names.extend([
            'save_milestone_max',
            'save_home_goalie',
            'save_away_goalie',
            'save_shutout_milestone',
            'save_percentage_chase'
        ])
        
        # Team milestone features
        names.extend([
            'team_wins_milestone',
            'team_points_milestone',
            'team_franchise_record',
            'team_win_streak'
        ])
        
        # Convergence features
        names.extend([
            'convergence_multiple_milestones',
            'convergence_same_game',
            'convergence_team_individual'
        ])
        
        return names
