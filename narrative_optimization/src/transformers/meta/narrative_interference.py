"""
Narrative Interference Transformer

Measures narrative density and interference patterns - when too many
stories collide, they can cancel out or amplify unpredictably.

Sometimes the absence of narrative creates the strongest story.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin


class NarrativeInterferenceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative interference and density features.
    
    Philosophy:
    - Multiple narratives can interfere destructively
    - Narrative fatigue affects performance
    - Some stories need space to breathe
    - Emotional games leave residue
    - Too much attention can curse outcomes
    
    Features (30 total):
    - Previous N games emotional weight (6)
    - Narrative exhaustion indicators (5)
    - Multiple storyline collision (6)
    - Publicity burden accumulation (5)
    - Emotional game recovery (4)
    - Narrative void detection (4)
    """
    
    # Emotional game categories and their residual effects
    EMOTIONAL_GAME_TYPES = {
        'revenge_game': {'weight': 0.9, 'decay_rate': 0.7},
        'milestone_game': {'weight': 0.8, 'decay_rate': 0.8},
        'rivalry_game': {'weight': 0.7, 'decay_rate': 0.6},
        'elimination_game': {'weight': 1.0, 'decay_rate': 0.9},
        'comeback_win': {'weight': 0.8, 'decay_rate': 0.5},
        'blowout_loss': {'weight': 0.7, 'decay_rate': 0.6},
        'overtime_game': {'weight': 0.6, 'decay_rate': 0.5},
        'controversy_game': {'weight': 0.8, 'decay_rate': 0.8},
        'injury_game': {'weight': 0.7, 'decay_rate': 0.7},
        'ceremony_game': {'weight': 0.6, 'decay_rate': 0.4}
    }
    
    # Narrative density thresholds
    DENSITY_LEVELS = {
        'void': (0.0, 0.2),      # No narratives
        'light': (0.2, 0.4),     # Normal flow
        'moderate': (0.4, 0.6),  # Building stories
        'heavy': (0.6, 0.8),     # Multiple narratives
        'saturated': (0.8, 1.0)  # Overwhelming
    }
    
    def __init__(
        self,
        lookback_games: int = 5,
        emotional_decay_rate: float = 0.7,
        include_opponent_narratives: bool = True
    ):
        """
        Initialize narrative interference analyzer.
        
        Parameters
        ----------
        lookback_games : int
            Number of previous games to analyze
        emotional_decay_rate : float
            How quickly emotional weight fades
        include_opponent_narratives : bool
            Consider opponent's narrative burden
        """
        self.lookback_games = lookback_games
        self.emotional_decay_rate = emotional_decay_rate
        self.include_opponent_narratives = include_opponent_narratives
        
    def fit(self, X, y=None):
        """No fitting required for interference analysis."""
        return self
        
    def transform(self, X):
        """
        Extract narrative interference features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with narrative context
            
        Returns
        -------
        np.ndarray
            Interference features (n_samples, 30)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Previous games emotional weight (6)
            emotional_features = self._extract_emotional_residue(item)
            feature_vec.extend(emotional_features)
            
            # Narrative exhaustion (5)
            exhaustion_features = self._extract_narrative_exhaustion(item)
            feature_vec.extend(exhaustion_features)
            
            # Storyline collision (6)
            collision_features = self._extract_storyline_collision(item)
            feature_vec.extend(collision_features)
            
            # Publicity burden (5)
            publicity_features = self._extract_publicity_burden(item)
            feature_vec.extend(publicity_features)
            
            # Emotional recovery (4)
            recovery_features = self._extract_emotional_recovery(item)
            feature_vec.extend(recovery_features)
            
            # Narrative void (4)
            void_features = self._extract_narrative_void(item)
            feature_vec.extend(void_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_emotional_residue(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract emotional residue from previous games.
        
        Returns 6 features measuring lingering effects.
        """
        features = []
        
        # Get previous games data
        previous_games = item.get('previous_games', [])
        
        if not previous_games:
            # Use summary data
            previous_games = []
            for i in range(self.lookback_games):
                game_key = f'game_minus_{i+1}'
                if game_key in item:
                    previous_games.append(item[game_key])
                    
        # Calculate emotional weights
        total_weight = 0.0
        max_single_weight = 0.0
        emotional_types = set()
        days_since_emotional = 999
        
        for i, game in enumerate(previous_games[:self.lookback_games]):
            if isinstance(game, dict):
                # Calculate game emotional weight
                game_weight = 0.0
                
                for emotion_type, emotion_info in self.EMOTIONAL_GAME_TYPES.items():
                    if game.get(emotion_type, False):
                        # Apply decay based on games ago
                        decay_factor = emotion_info['decay_rate'] ** i
                        weight = emotion_info['weight'] * decay_factor
                        game_weight += weight
                        emotional_types.add(emotion_type)
                        
                        if i == 0 and weight > 0:
                            days_since_emotional = game.get('days_ago', 2)
                            
                total_weight += game_weight
                max_single_weight = max(max_single_weight, game_weight)
                
        # Total emotional burden
        features.append(min(1.0, total_weight))
        
        # Maximum single game impact
        features.append(max_single_weight)
        
        # Variety of emotional types
        features.append(min(1.0, len(emotional_types) * 0.2))
        
        # Recent emotional game (last 3 days)
        if days_since_emotional <= 3:
            features.append(1.0 - days_since_emotional / 3.0)
        else:
            features.append(0.0)
            
        # Consecutive emotional games
        consecutive_emotional = 0
        for game in previous_games:
            if isinstance(game, dict):
                has_emotion = any(game.get(etype, False) 
                                for etype in self.EMOTIONAL_GAME_TYPES)
                if has_emotion:
                    consecutive_emotional += 1
                else:
                    break
                    
        features.append(min(1.0, consecutive_emotional * 0.3))
        
        # Unprocessed trauma (big loss not followed by win)
        last_game = previous_games[0] if previous_games else {}
        if isinstance(last_game, dict):
            if (last_game.get('blowout_loss', False) or 
                last_game.get('elimination_loss', False)):
                features.append(0.9)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_narrative_exhaustion(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract narrative exhaustion indicators.
        
        Returns 5 features measuring story fatigue.
        """
        features = []
        
        # Media mentions saturation
        media_mentions_recent = item.get('media_mentions_past_week', 0)
        media_mentions_baseline = item.get('media_mentions_average', 1)
        
        if media_mentions_baseline > 0:
            media_saturation = media_mentions_recent / media_mentions_baseline
            features.append(min(1.0, media_saturation / 3.0))
        else:
            features.append(0.5)
            
        # Primetime game frequency
        primetime_games_month = item.get('primetime_games_past_month', 0)
        if primetime_games_month >= 5:
            features.append(min(1.0, (primetime_games_month - 4) / 4.0))
        else:
            features.append(0.0)
            
        # National broadcast fatigue
        national_games_recent = item.get('national_games_past_10', 0)
        features.append(min(1.0, national_games_recent / 5.0))
        
        # Storyline duration (how long current narrative has persisted)
        current_narrative_days = item.get('current_storyline_days', 0)
        if current_narrative_days > 14:
            staleness = min(1.0, (current_narrative_days - 14) / 14.0)
            features.append(staleness)
        else:
            features.append(0.0)
            
        # Controversy fatigue
        controversy_mentions = item.get('controversy_mentions_week', 0)
        features.append(min(1.0, controversy_mentions / 10.0))
        
        return features
        
    def _extract_storyline_collision(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract features for multiple colliding narratives.
        
        Returns 6 features for narrative interference.
        """
        features = []
        
        # Count active narratives
        active_narratives = 0
        narrative_weights = []
        
        # Individual milestones
        if item.get('milestone_approaching', False):
            active_narratives += 1
            narrative_weights.append(0.8)
            
        # Team milestones
        if item.get('team_milestone_game', False):
            active_narratives += 1
            narrative_weights.append(0.7)
            
        # Revenge narrative
        if item.get('revenge_game', False):
            active_narratives += 1
            narrative_weights.append(0.9)
            
        # Streak narrative
        if item.get('streak_on_line', False):
            active_narratives += 1
            narrative_weights.append(0.7)
            
        # Playoff implications
        if item.get('playoff_implications', False):
            active_narratives += 1
            narrative_weights.append(0.85)
            
        # Rivalry
        if item.get('rivalry_game', False):
            active_narratives += 1
            narrative_weights.append(0.6)
            
        # Return from injury
        if item.get('star_return_game', False):
            active_narratives += 1
            narrative_weights.append(0.8)
            
        # Trade narrative
        if item.get('facing_former_team', False):
            active_narratives += 1
            narrative_weights.append(0.85)
            
        # Number of active narratives
        features.append(min(1.0, active_narratives / 5.0))
        
        # Collision intensity (multiple high-weight narratives)
        if len(narrative_weights) >= 2:
            sorted_weights = sorted(narrative_weights, reverse=True)
            collision_intensity = sorted_weights[0] * sorted_weights[1]
            features.append(collision_intensity)
        else:
            features.append(0.0)
            
        # Narrative diversity (different types)
        narrative_types = set()
        if item.get('milestone_approaching', False):
            narrative_types.add('achievement')
        if item.get('revenge_game', False):
            narrative_types.add('emotional')
        if item.get('playoff_implications', False):
            narrative_types.add('stakes')
        if item.get('rivalry_game', False):
            narrative_types.add('historical')
            
        features.append(len(narrative_types) / 4.0)
        
        # Conflicting narratives (e.g., milestone + revenge)
        has_positive = item.get('milestone_approaching', False) or \
                      item.get('streak_on_line', False)
        has_negative = item.get('revenge_game', False) or \
                      item.get('elimination_threat', False)
        
        if has_positive and has_negative:
            features.append(0.8)  # Conflicting energy
        else:
            features.append(0.0)
            
        # Over-determination (too many reasons to win)
        if active_narratives >= 4:
            features.append(min(1.0, (active_narratives - 3) / 3.0))
        else:
            features.append(0.0)
            
        # Maximum narrative weight
        max_weight = max(narrative_weights) if narrative_weights else 0.0
        features.append(max_weight)
        
        return features
        
    def _extract_publicity_burden(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract publicity burden accumulation features.
        
        Returns 5 features for media pressure.
        """
        features = []
        
        # Pre-game hype level
        pregame_stories = item.get('pregame_story_count', 0)
        features.append(min(1.0, pregame_stories / 10.0))
        
        # Social media buzz
        social_mentions = item.get('social_media_mentions_24h', 0)
        social_baseline = item.get('social_media_baseline', 100)
        
        if social_baseline > 0:
            social_amplification = social_mentions / social_baseline
            features.append(min(1.0, social_amplification / 5.0))
        else:
            features.append(0.5)
            
        # Betting action indicator
        betting_volume = item.get('betting_volume_vs_average', 1.0)
        if betting_volume > 2.0:
            features.append(min(1.0, (betting_volume - 2.0) / 2.0))
        else:
            features.append(0.0)
            
        # Celebrity attention
        celebrity_attendance = item.get('celebrity_attendance_expected', False)
        special_guests = item.get('special_ceremony_guests', 0)
        
        celebrity_factor = 0.0
        if celebrity_attendance:
            celebrity_factor += 0.6
        if special_guests > 0:
            celebrity_factor += min(0.4, special_guests * 0.2)
            
        features.append(celebrity_factor)
        
        # Narrative crystallization in media
        unanimous_prediction = item.get('media_prediction_consensus', 0.5)
        if unanimous_prediction > 0.8 or unanimous_prediction < 0.2:
            features.append(0.8)  # Too much agreement
        else:
            features.append(0.0)
            
        return features
        
    def _extract_emotional_recovery(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract emotional game recovery patterns.
        
        Returns 4 features for recovery needs.
        """
        features = []
        
        # Days since last emotional game
        last_emotional_game = item.get('days_since_emotional_game', 7)
        
        if last_emotional_game <= 2:
            recovery_need = 1.0 - (last_emotional_game / 2.0)
            features.append(recovery_need)
        else:
            features.append(0.0)
            
        # Type of last emotional game
        last_emotional_type = item.get('last_emotional_game_type', 'none')
        
        heavy_types = ['elimination_game', 'revenge_game', 'controversy_game']
        if last_emotional_type in heavy_types:
            features.append(0.8)
        elif last_emotional_type != 'none':
            features.append(0.4)
        else:
            features.append(0.0)
            
        # Emotional whiplash (win after devastating loss)
        previous_result = item.get('previous_game_result', 'none')
        two_games_ago = item.get('two_games_ago_result', 'none')
        
        if (previous_result == 'big_win' and two_games_ago == 'devastating_loss') or \
           (previous_result == 'devastating_loss' and two_games_ago == 'big_win'):
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Unresolved narrative tension
        unresolved_storyline = item.get('unresolved_storyline', False)
        if unresolved_storyline:
            features.append(0.6)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_narrative_void(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract narrative void detection features.
        
        Sometimes the absence of narrative is the story.
        
        Returns 4 features for narrative absence.
        """
        features = []
        
        # Check for narrative density
        active_narratives = sum([
            item.get('milestone_approaching', False),
            item.get('revenge_game', False),
            item.get('rivalry_game', False),
            item.get('streak_on_line', False),
            item.get('playoff_implications', False),
            item.get('special_ceremony', False)
        ])
        
        # True void (no narratives)
        if active_narratives == 0:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Relative void (fewer narratives than usual)
        team_average_narratives = item.get('team_average_narrative_density', 2.0)
        if active_narratives < team_average_narratives * 0.5:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Schedule void (mundane positioning)
        is_midweek = item.get('game_date', datetime.now()).weekday() in [1, 2, 3]
        is_midseason = 0.3 < item.get('season_progress', 0.5) < 0.7
        non_rival = not item.get('division_rival', False)
        
        if is_midweek and is_midseason and non_rival:
            features.append(0.6)
        else:
            features.append(0.0)
            
        # Trap game indicator (void before important game)
        next_game_big = item.get('next_game_importance', 0.5)
        if active_narratives <= 1 and next_game_big > 0.8:
            features.append(0.8)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Emotional residue features
        names.extend([
            'residue_total_weight',
            'residue_max_single',
            'residue_type_variety',
            'residue_recent_emotional',
            'residue_consecutive',
            'residue_unprocessed_trauma'
        ])
        
        # Narrative exhaustion features
        names.extend([
            'exhaustion_media_saturation',
            'exhaustion_primetime_frequency',
            'exhaustion_national_games',
            'exhaustion_storyline_staleness',
            'exhaustion_controversy_fatigue'
        ])
        
        # Storyline collision features
        names.extend([
            'collision_active_count',
            'collision_intensity',
            'collision_type_diversity',
            'collision_conflicting',
            'collision_overdetermined',
            'collision_max_weight'
        ])
        
        # Publicity burden features
        names.extend([
            'publicity_pregame_hype',
            'publicity_social_buzz',
            'publicity_betting_action',
            'publicity_celebrity_factor',
            'publicity_media_consensus'
        ])
        
        # Emotional recovery features
        names.extend([
            'recovery_days_needed',
            'recovery_heavy_type',
            'recovery_emotional_whiplash',
            'recovery_unresolved_tension'
        ])
        
        # Narrative void features
        names.extend([
            'void_true_absence',
            'void_relative_quiet',
            'void_schedule_mundane',
            'void_trap_game'
        ])
        
        return names
