"""
Elimination Proximity Transformer

Detects death march and spoiler narratives as teams approach
mathematical elimination or clinching scenarios.

Hope dies slowly, then all at once.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from .invisible_base import InvisibleNarrativeTransformer


class EliminationProximityTransformer(InvisibleNarrativeTransformer):
    """
    Extract elimination proximity narrative features.
    
    Philosophy:
    - Approaching elimination creates desperation
    - Recent elimination breeds dangerous freedom
    - Clinching scenarios add pressure
    - Draft implications change motivations
    - Spoiler roles activate pride
    
    Features (25 total):
    - Games to mathematical elimination (4)
    - Games since elimination possible (4)
    - Playoff probability trajectory (5)
    - Draft lottery implications (4)
    - Spoiler role intensity (4)
    - Clinching proximity (4)
    """
    
    # Elimination narrative stages
    ELIMINATION_STAGES = {
        'safe': {'games_to_elim': float('inf'), 'pressure': 0.2},
        'comfortable': {'games_to_elim': 20, 'pressure': 0.3},
        'watching': {'games_to_elim': 15, 'pressure': 0.5},
        'concerned': {'games_to_elim': 10, 'pressure': 0.7},
        'desperate': {'games_to_elim': 5, 'pressure': 0.9},
        'death_march': {'games_to_elim': 3, 'pressure': 1.0},
        'eliminated': {'games_to_elim': 0, 'pressure': 0.0}
    }
    
    # Post-elimination psychology phases
    POST_ELIMINATION_PHASES = {
        'shock': (0, 3, 0.8),        # Days 0-3: Initial shock/anger
        'acceptance': (4, 7, 0.6),    # Days 4-7: Coming to terms
        'freedom': (8, 14, 0.7),      # Days 8-14: Playing loose
        'evaluation': (15, 30, 0.5),  # Days 15-30: Auditioning
        'vacation': (31, 999, 0.3)    # Days 31+: Going through motions
    }
    
    def __init__(
        self,
        playoff_line_method: str = 'historical',
        include_draft_narratives: bool = True
    ):
        """
        Initialize elimination proximity analyzer.
        
        Parameters
        ----------
        playoff_line_method : str
            Method to estimate playoff line ('historical' or 'dynamic')
        include_draft_narratives : bool
            Include draft lottery implications
        """
        super().__init__(
            narrative_id='elimination_proximity',
            description='Tracks approach to elimination and clinching'
        )
        self.playoff_line_method = playoff_line_method
        self.include_draft_narratives = include_draft_narratives
        
        # Historical playoff lines by conference
        self.historical_playoff_lines = {
            'Eastern': 95,  # Typical points needed
            'Western': 93,
            'default': 94
        }
        
    def fit(self, X, y=None):
        """No fitting required for elimination analysis."""
        return self
        
    def transform(self, X):
        """
        Extract elimination proximity features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with standings context
            
        Returns
        -------
        np.ndarray
            Elimination features (n_samples, 25)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Games to elimination (4)
            elimination_features = self._extract_elimination_proximity(item)
            feature_vec.extend(elimination_features)
            
            # Games since elimination (4)
            post_elim_features = self._extract_post_elimination(item)
            feature_vec.extend(post_elim_features)
            
            # Playoff probability (5)
            probability_features = self._extract_playoff_probability(item)
            feature_vec.extend(probability_features)
            
            # Draft implications (4)
            if self.include_draft_narratives:
                draft_features = self._extract_draft_implications(item)
            else:
                draft_features = [0.0] * 4
            feature_vec.extend(draft_features)
            
            # Spoiler intensity (4)
            spoiler_features = self._extract_spoiler_intensity(item)
            feature_vec.extend(spoiler_features)
            
            # Clinching proximity (4)
            clinch_features = self._extract_clinching_proximity(item)
            feature_vec.extend(clinch_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_elimination_proximity(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract games to elimination features.
        
        Returns 4 features for elimination approach.
        """
        features = []
        
        # Get standings info
        current_points = item.get('team_points', 50)
        games_remaining = item.get('games_remaining', 41)
        conference = item.get('conference', 'default')
        
        # Estimate playoff line
        if self.playoff_line_method == 'historical':
            playoff_line = self.historical_playoff_lines.get(conference, 94)
        else:
            # Dynamic based on current season
            playoff_line = item.get('projected_playoff_line', 94)
            
        # Calculate elimination proximity
        max_possible_points = current_points + (games_remaining * 2)
        
        if max_possible_points < playoff_line:
            # Already eliminated
            games_to_elimination = 0
            elimination_stage = 'eliminated'
        else:
            # How many losses until elimination?
            points_needed = playoff_line - current_points
            affordable_losses = games_remaining - (points_needed // 2)
            games_to_elimination = max(1, affordable_losses)
            
            # Determine stage
            for stage_name, stage_info in self.ELIMINATION_STAGES.items():
                if games_to_elimination >= stage_info['games_to_elim']:
                    elimination_stage = stage_name
                    break
                    
        # Games to elimination (normalized)
        if games_to_elimination > 0:
            features.append(1.0 / games_to_elimination)
        else:
            features.append(1.0)  # Eliminated
            
        # Elimination pressure
        pressure = self.ELIMINATION_STAGES[elimination_stage]['pressure']
        features.append(pressure)
        
        # Rate of approach (how fast elimination is coming)
        recent_form = item.get('last_10_games_points_percentage', 0.5)
        if recent_form < 0.5 and games_to_elimination < 10:
            approach_rate = (0.5 - recent_form) * 2 * pressure
            features.append(approach_rate)
        else:
            features.append(0.0)
            
        # Must-win indicator
        if 0 < games_to_elimination <= 5:
            must_win = min(1.0, 1.0 / games_to_elimination)
            features.append(must_win)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_post_elimination(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract post-elimination psychology features.
        
        Returns 4 features for eliminated teams.
        """
        features = []
        
        is_eliminated = item.get('team_eliminated', False)
        games_since_elimination = item.get('games_since_elimination', 0)
        
        if not is_eliminated:
            return [0.0] * 4
            
        # Find current phase
        current_phase = None
        phase_intensity = 0.0
        
        for phase_name, (start_day, end_day, intensity) in self.POST_ELIMINATION_PHASES.items():
            if start_day <= games_since_elimination <= end_day:
                current_phase = phase_name
                phase_intensity = intensity
                break
                
        # Phase indicators
        features.append(1.0 if current_phase == 'shock' else 0.0)
        features.append(1.0 if current_phase == 'freedom' else 0.0)
        
        # Overall post-elimination intensity
        features.append(phase_intensity)
        
        # Young player showcase mode
        prospect_games = item.get('prospects_in_lineup', 0)
        if prospect_games > 0 and games_since_elimination > 7:
            features.append(min(1.0, prospect_games * 0.3))
        else:
            features.append(0.0)
            
        return features
        
    def _extract_playoff_probability(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract playoff probability trajectory features.
        
        Returns 5 features for playoff chances.
        """
        features = []
        
        # Current probability
        playoff_prob = item.get('playoff_probability', 0.5)
        features.append(playoff_prob)
        
        # Probability trajectory
        prob_7_days_ago = item.get('playoff_prob_week_ago', 0.5)
        prob_14_days_ago = item.get('playoff_prob_2weeks_ago', 0.5)
        
        # Recent change
        recent_change = playoff_prob - prob_7_days_ago
        features.append(np.tanh(recent_change * 5))  # -1 to 1
        
        # Acceleration
        if prob_7_days_ago > 0:
            week1_change = prob_7_days_ago - prob_14_days_ago
            week2_change = playoff_prob - prob_7_days_ago
            acceleration = week2_change - week1_change
            features.append(np.tanh(acceleration * 10))
        else:
            features.append(0.0)
            
        # Critical threshold proximity
        # Teams care most about 20%, 50%, 80% thresholds
        critical_thresholds = [0.2, 0.5, 0.8]
        min_distance = min([abs(playoff_prob - t) for t in critical_thresholds])
        
        if min_distance < 0.1:
            threshold_pressure = 1.0 - (min_distance / 0.1)
            features.append(threshold_pressure)
        else:
            features.append(0.0)
            
        # Collapse indicator (was safe, now in danger)
        if prob_14_days_ago > 0.8 and playoff_prob < 0.5:
            features.append(0.9)  # Panic mode
        else:
            features.append(0.0)
            
        return features
        
    def _extract_draft_implications(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract draft lottery implication features.
        
        Returns 4 features for tanking incentives.
        """
        features = []
        
        # Current draft position
        reverse_standing = item.get('reverse_standing_position', 15)
        
        # Top 3 pick proximity
        if reverse_standing <= 5:
            pick_value = 1.0 - (reverse_standing - 1) / 5.0
            features.append(pick_value)
        else:
            features.append(0.0)
            
        # Lottery odds improvement potential
        games_remaining = item.get('games_remaining', 20)
        current_points = item.get('team_points', 50)
        
        # Teams within 6 points below
        teams_within_reach = item.get('teams_within_6_points_below', 0)
        if teams_within_reach > 0 and games_remaining > 10:
            tank_potential = min(1.0, teams_within_reach * 0.3)
            features.append(tank_potential)
        else:
            features.append(0.0)
            
        # Draft year strength
        draft_class_strength = item.get('draft_class_rating', 'average')
        
        draft_values = {
            'generational': 1.0,
            'strong': 0.8,
            'average': 0.5,
            'weak': 0.3
        }
        
        features.append(draft_values.get(draft_class_strength, 0.5))
        
        # Organizational direction clarity
        # Teams clearly rebuilding vs trying to compete
        rebuilding_mode = item.get('team_rebuilding', False)
        eliminated = item.get('team_eliminated', False)
        
        if rebuilding_mode and eliminated:
            features.append(0.8)  # Clear tank incentive
        else:
            features.append(0.0)
            
        return features
        
    def _extract_spoiler_intensity(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract spoiler role intensity features.
        
        Returns 4 features for spoiler narratives.
        """
        features = []
        
        team_eliminated = item.get('team_eliminated', False)
        opponent_playoff_race = item.get('opponent_in_playoff_race', False)
        
        if not team_eliminated:
            return [0.0] * 4
            
        # Basic spoiler activation
        if opponent_playoff_race:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Division rival spoiler
        division_rival = item.get('division_rival', False)
        if division_rival and opponent_playoff_race:
            features.append(1.0)  # Maximum satisfaction
        else:
            features.append(0.0)
            
        # Spoiler with history
        recent_playoff_loss_to_opponent = item.get('lost_to_opponent_in_playoffs_last_3_years', False)
        if recent_playoff_loss_to_opponent and opponent_playoff_race:
            features.append(0.9)  # Revenge spoiler
        else:
            features.append(0.0)
            
        # Pride game intensity
        # Last home game, senior night, etc.
        special_circumstances = sum([
            item.get('last_home_game', False),
            item.get('senior_night', False),
            item.get('jersey_retirement', False),
            item.get('franchise_anniversary', False)
        ])
        
        if special_circumstances > 0:
            pride_intensity = min(1.0, special_circumstances * 0.4)
            features.append(pride_intensity)
        else:
            features.append(0.3)  # Base pride level
            
        return features
        
    def _extract_clinching_proximity(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract clinching scenario features.
        
        Returns 4 features for clinching pressure.
        """
        features = []
        
        # Playoff clinching
        can_clinch_playoffs = item.get('can_clinch_playoffs_today', False)
        games_to_clinch_playoffs = item.get('min_games_to_clinch_playoffs', 999)
        
        if can_clinch_playoffs:
            features.append(1.0)
        elif games_to_clinch_playoffs <= 5:
            features.append(1.0 - games_to_clinch_playoffs / 5.0)
        else:
            features.append(0.0)
            
        # Division clinching
        can_clinch_division = item.get('can_clinch_division_today', False)
        games_to_clinch_division = item.get('min_games_to_clinch_division', 999)
        
        if can_clinch_division:
            features.append(0.9)
        elif games_to_clinch_division <= 5:
            features.append(0.9 - games_to_clinch_division / 5.0)
        else:
            features.append(0.0)
            
        # Home ice clinching
        can_clinch_home_ice = item.get('can_clinch_home_ice_today', False)
        
        features.append(0.7 if can_clinch_home_ice else 0.0)
        
        # Multiple clinching scenarios
        clinch_scenarios = sum([
            can_clinch_playoffs,
            can_clinch_division,
            can_clinch_home_ice,
            item.get('can_clinch_conference', False),
            item.get('can_clinch_presidents_trophy', False)
        ])
        
        if clinch_scenarios >= 2:
            features.append(min(1.0, clinch_scenarios * 0.4))
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Elimination proximity features
        names.extend([
            'elim_games_to_elimination',
            'elim_pressure_level',
            'elim_approach_rate',
            'elim_must_win_indicator'
        ])
        
        # Post-elimination features
        names.extend([
            'post_elim_shock_phase',
            'post_elim_freedom_phase',
            'post_elim_intensity',
            'post_elim_youth_showcase'
        ])
        
        # Playoff probability features
        names.extend([
            'playoff_prob_current',
            'playoff_prob_trend',
            'playoff_prob_acceleration',
            'playoff_prob_threshold',
            'playoff_prob_collapse'
        ])
        
        # Draft features
        names.extend([
            'draft_top_pick_proximity',
            'draft_tank_potential',
            'draft_class_strength',
            'draft_organizational_clarity'
        ])
        
        # Spoiler features
        names.extend([
            'spoiler_basic_activation',
            'spoiler_division_rival',
            'spoiler_revenge_narrative',
            'spoiler_pride_intensity'
        ])
        
        # Clinching features
        names.extend([
            'clinch_playoffs_proximity',
            'clinch_division_proximity',
            'clinch_home_ice_today',
            'clinch_multiple_scenarios'
        ])
        
        return names
