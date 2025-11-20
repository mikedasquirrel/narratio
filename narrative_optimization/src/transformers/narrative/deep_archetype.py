"""
Deep Narrative Archetype Transformer

Detects and quantifies classic narrative archetypes in sports contexts,
focusing on the Hero's Journey and other fundamental story patterns.

This transformer identifies where teams/players are in their narrative arc
and amplifies features based on archetypal story momentum.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
import re
from datetime import datetime, timedelta


class DeepArchetypeTransformer(BaseEstimator, TransformerMixin):
    """
    Extract deep narrative archetype features from sports data.
    
    Philosophy:
    - Every team/player follows archetypal narrative patterns
    - Hero's Journey has predictable stages that create momentum
    - Underdog stories have quantifiable power
    - Dynasty narratives follow rise/peak/decline patterns
    - Redemption arcs create measurable pressure
    
    Features (40 total):
    - Hero's Journey stage indicators (12)
    - Underdog narrative amplification (8)
    - Dynasty lifecycle positioning (6)
    - Redemption arc progression (6)
    - Comeback story momentum (4)
    - Rivalry archetype intensity (4)
    """
    
    def __init__(
        self,
        include_text_analysis: bool = True,
        temporal_weight: float = 0.8,
        momentum_decay: float = 0.9
    ):
        """
        Initialize deep archetype analyzer.
        
        Parameters
        ----------
        include_text_analysis : bool
            Analyze text fields for narrative language
        temporal_weight : float
            How much recent events matter vs historical
        momentum_decay : float
            How quickly narrative momentum fades
        """
        self.include_text_analysis = include_text_analysis
        self.temporal_weight = temporal_weight
        self.momentum_decay = momentum_decay
        
        # Hero's Journey stages
        self.heros_journey_stages = [
            'ordinary_world',      # Regular season start
            'call_to_adventure',   # Playoff contention
            'refusal_of_call',     # Early struggles
            'meeting_mentor',      # Key acquisition/coach
            'crossing_threshold',  # Making playoffs
            'tests_and_allies',    # Playoff battles
            'approach',            # Conference finals
            'ordeal',              # Championship series
            'reward',              # Victory
            'road_back',           # Next season pressure
            'resurrection',        # Dynasty building
            'return_transformed'   # Legacy cemented
        ]
        
        # Narrative patterns to detect
        self.narrative_patterns = {
            'underdog': {
                'indicators': ['upset', 'david', 'goliath', 'unlikely', 'surprise'],
                'stat_thresholds': {'win_pct_diff': -0.2, 'odds_diff': 200}
            },
            'dynasty': {
                'indicators': ['dynasty', 'empire', 'dominant', 'reign'],
                'phases': ['rise', 'peak', 'decline', 'fall']
            },
            'redemption': {
                'indicators': ['redemption', 'comeback', 'return', 'vindication'],
                'triggers': ['injury_return', 'scandal_recovery', 'failure_bounce']
            },
            'rivalry': {
                'indicators': ['rival', 'nemesis', 'blood', 'hatred', 'classic'],
                'intensity_factors': ['history', 'geography', 'playoffs', 'controversy']
            }
        }
        
    def fit(self, X, y=None):
        """
        Learn narrative baselines from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Sports data with narrative elements
        y : ignored
        
        Returns
        -------
        self
        """
        # Convert to DataFrame if needed
        if isinstance(X, list):
            self.training_data_ = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            self.training_data_ = X.copy()
        else:
            self.training_data_ = pd.DataFrame()
            
        # Learn typical patterns
        self._learn_narrative_baselines()
        
        return self
        
    def transform(self, X):
        """
        Extract deep archetype features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Sports data to analyze
            
        Returns
        -------
        np.ndarray
            Archetype features (n_samples, 40)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Hero's Journey stage features (12)
            journey_features = self._extract_heros_journey(item)
            feature_vec.extend(journey_features)
            
            # Underdog narrative (8)
            underdog_features = self._extract_underdog_narrative(item)
            feature_vec.extend(underdog_features)
            
            # Dynasty lifecycle (6)
            dynasty_features = self._extract_dynasty_lifecycle(item)
            feature_vec.extend(dynasty_features)
            
            # Redemption arc (6)
            redemption_features = self._extract_redemption_arc(item)
            feature_vec.extend(redemption_features)
            
            # Comeback momentum (4)
            comeback_features = self._extract_comeback_momentum(item)
            feature_vec.extend(comeback_features)
            
            # Rivalry archetype (4)
            rivalry_features = self._extract_rivalry_archetype(item)
            feature_vec.extend(rivalry_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _learn_narrative_baselines(self):
        """Learn typical narrative patterns from training data."""
        if self.training_data_.empty:
            # Set defaults if no training data
            self.narrative_baselines_ = {
                'avg_upset_margin': 0.15,
                'dynasty_duration': 5.0,
                'redemption_window': 2.0,
                'rivalry_frequency': 0.1
            }
            return
            
        # Calculate actual baselines from data
        self.narrative_baselines_ = {}
        
        # Upset patterns
        if 'win_pct' in self.training_data_.columns:
            upsets = self.training_data_[
                self.training_data_['win_pct'] < 0.4
            ]
            self.narrative_baselines_['avg_upset_margin'] = (
                upsets['win_pct'].std() if len(upsets) > 0 else 0.15
            )
            
        # Add more baseline calculations as needed
        
    def _extract_heros_journey(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract Hero's Journey stage indicators.
        
        Returns 12 features representing journey position and momentum.
        """
        features = []
        
        # Determine current stage based on season position
        season_progress = item.get('games_played', 0) / 82.0  # NHL season
        playoff_status = item.get('playoff_position', None)
        recent_performance = item.get('last_10_record', [5, 5])
        
        # Stage activation based on context
        stage_activations = np.zeros(12)
        
        # Ordinary world (early season)
        if season_progress < 0.2:
            stage_activations[0] = 1.0 - (season_progress * 5)
            
        # Call to adventure (playoff race begins)
        if 0.4 < season_progress < 0.7 and playoff_status is not None:
            stage_activations[1] = 1.0 if 6 <= playoff_status <= 10 else 0.3
            
        # Crossing threshold (clinching playoffs)
        if item.get('playoffs_clinched', False):
            stage_activations[4] = 1.0
            
        # Tests and allies (playoff rounds)
        playoff_round = item.get('playoff_round', 0)
        if playoff_round > 0:
            stage_activations[5] = 0.25 * playoff_round
            
        # Ordeal (finals)
        if playoff_round == 4:  # NHL finals
            stage_activations[7] = 1.0
            
        # Add momentum weighting
        momentum = self._calculate_narrative_momentum(item)
        stage_activations *= momentum
        
        features.extend(stage_activations)
        
        return features
        
    def _extract_underdog_narrative(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract underdog story amplification features.
        
        Returns 8 features capturing underdog narrative strength.
        """
        features = []
        
        # Basic underdog indicators
        win_pct = item.get('win_pct', 0.5)
        opponent_win_pct = item.get('opponent_win_pct', 0.5)
        odds = item.get('moneyline', 100)
        
        # Underdog score (0-1)
        underdog_score = 0.0
        
        # Win percentage differential
        if opponent_win_pct - win_pct > 0.2:
            underdog_score += 0.3
            
        # Odds differential  
        if odds > 200:  # Heavy underdog
            underdog_score += 0.3
            
        # David vs Goliath markers
        if item.get('market_size', 'medium') == 'small' and \
           item.get('opponent_market_size', 'medium') == 'large':
            underdog_score += 0.2
            
        # Recent upset history amplifies narrative
        recent_upsets = item.get('recent_upset_wins', 0)
        underdog_score += min(0.2, recent_upsets * 0.05)
        
        features.append(min(1.0, underdog_score))
        
        # Underdog momentum (hot streak despite odds)
        if win_pct < 0.4 and recent_performance[0] > 7:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Public sentiment alignment
        if self.include_text_analysis:
            text = item.get('description', '').lower()
            underdog_words = sum(1 for word in self.narrative_patterns['underdog']['indicators']
                               if word in text)
            features.append(min(1.0, underdog_words / 3.0))
        else:
            features.append(0.0)
            
        # Historical underdog success rate in similar situations
        features.append(item.get('similar_underdog_success', 0.3))
        
        # Injury/absence creating underdog status
        key_player_out = item.get('star_player_injured', False)
        features.append(1.0 if key_player_out else 0.0)
        
        # Playoff underdog multiplier
        if item.get('is_playoffs', False) and underdog_score > 0.5:
            features.append(1.5)
        else:
            features.append(1.0)
            
        # Road underdog bonus
        if item.get('is_away', False) and underdog_score > 0.3:
            features.append(1.2)
        else:
            features.append(1.0)
            
        # Cinderella run indicator
        playoff_seed = item.get('playoff_seed', 0)
        if playoff_seed >= 7 and playoff_round > 1:
            features.append(2.0)
        else:
            features.append(1.0)
            
        return features
        
    def _extract_dynasty_lifecycle(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract dynasty narrative lifecycle features.
        
        Returns 6 features capturing dynasty phase and momentum.
        """
        features = []
        
        # Dynasty indicators
        championships = item.get('recent_championships', 0)
        consecutive_playoffs = item.get('consecutive_playoff_years', 0)
        core_age = item.get('core_player_avg_age', 27)
        
        # Determine dynasty phase
        dynasty_phase = 'none'
        phase_strength = 0.0
        
        if championships == 0 and consecutive_playoffs < 3:
            if item.get('young_core', False) and win_pct > 0.55:
                dynasty_phase = 'rise'
                phase_strength = min(1.0, (win_pct - 0.5) * 5)
        elif championships >= 1 and consecutive_playoffs >= 3:
            if core_age < 30:
                dynasty_phase = 'peak'
                phase_strength = min(1.0, championships / 3.0)
            else:
                dynasty_phase = 'decline'
                phase_strength = max(0.3, 1.0 - (core_age - 30) / 5.0)
        elif championships >= 2 and core_age > 32:
            dynasty_phase = 'fall'
            phase_strength = 1.0
            
        # Phase encoding (one-hot style but with strength)
        for phase in ['rise', 'peak', 'decline', 'fall']:
            if dynasty_phase == phase:
                features.append(phase_strength)
            else:
                features.append(0.0)
                
        # Dynasty momentum (are they following expected trajectory?)
        expected_win_pct = {
            'none': 0.5,
            'rise': 0.58,
            'peak': 0.65,
            'decline': 0.55,
            'fall': 0.45
        }
        
        win_pct = item.get('win_pct', 0.5)  # Get win_pct from item
        actual_vs_expected = win_pct - expected_win_pct.get(dynasty_phase, 0.5)
        features.append(np.tanh(actual_vs_expected * 10))  # Smooth activation
        
        # Dynasty narrative pressure (expectations)
        if dynasty_phase == 'peak':
            features.append(2.0)  # Maximum pressure
        elif dynasty_phase == 'rise':
            features.append(1.2)  # Building pressure
        else:
            features.append(1.0)
            
        return features
        
    def _extract_redemption_arc(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract redemption narrative features.
        
        Returns 6 features capturing redemption arc progression.
        """
        features = []
        
        # Redemption triggers
        injury_return = item.get('key_player_return_from_injury', False)
        previous_failure = item.get('eliminated_last_year_same_round', False)
        scandal_recovery = item.get('franchise_scandal_recovery', False)
        coaching_return = item.get('coach_revenge_game', False)
        
        # Calculate redemption score
        redemption_score = 0.0
        
        if injury_return:
            redemption_score += 0.4
        if previous_failure:
            redemption_score += 0.3
        if scandal_recovery:
            redemption_score += 0.2
        if coaching_return:
            redemption_score += 0.3
            
        features.append(min(1.0, redemption_score))
        
        # Redemption arc stage (beginning, middle, climax)
        if redemption_score > 0:
            games_since_trigger = item.get('games_since_redemption_trigger', 0)
            if games_since_trigger < 10:
                features.extend([1.0, 0.0, 0.0])  # Beginning
            elif games_since_trigger < 30:
                features.extend([0.0, 1.0, 0.0])  # Middle
            else:
                features.extend([0.0, 0.0, 1.0])  # Climax
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Redemption momentum
        if redemption_score > 0 and recent_performance[0] > 6:
            features.append(1.5)
        else:
            features.append(1.0)
            
        # Public sympathy factor
        features.append(item.get('fan_sympathy_rating', 0.5))
        
        return features
        
    def _extract_comeback_momentum(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract comeback story momentum features.
        
        Returns 4 features capturing comeback narrative strength.
        """
        features = []
        
        # In-game comeback potential
        current_deficit = item.get('current_score_deficit', 0)
        time_remaining = item.get('period_time_remaining', 1.0)
        
        comeback_potential = 0.0
        if current_deficit > 0:
            # Higher deficit + less time = more dramatic
            comeback_potential = min(1.0, current_deficit / 3.0) * (1.0 - time_remaining)
            
        features.append(comeback_potential)
        
        # Series comeback (down 3-1, etc.)
        series_deficit = item.get('series_deficit', 0)
        if series_deficit >= 2:
            features.append(min(1.0, series_deficit / 3.0))
        else:
            features.append(0.0)
            
        # Season comeback (from bad start)
        season_low_point = item.get('season_worst_position', 15)
        current_position = item.get('current_standings_position', 10)
        
        if season_low_point > 10 and current_position <= 8:
            season_comeback = (season_low_point - current_position) / 10.0
            features.append(min(1.0, season_comeback))
        else:
            features.append(0.0)
            
        # Historical comeback precedent
        features.append(item.get('franchise_comeback_history_score', 0.3))
        
        return features
        
    def _extract_rivalry_archetype(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract rivalry archetype intensity features.
        
        Returns 4 features capturing rivalry narrative elements.
        """
        features = []
        
        # Base rivalry score
        rivalry_score = 0.0
        
        # Geographic rivalry
        if item.get('division_rival', False):
            rivalry_score += 0.3
        if item.get('geographic_rival', False):
            rivalry_score += 0.2
            
        # Historical rivalry
        playoff_meetings = item.get('playoff_meetings_last_10_years', 0)
        rivalry_score += min(0.3, playoff_meetings * 0.1)
        
        # Recent controversy
        if item.get('recent_controversial_game', False):
            rivalry_score += 0.2
            
        features.append(min(1.0, rivalry_score))
        
        # Rivalry game importance multiplier
        if item.get('is_playoffs', False) and rivalry_score > 0.5:
            features.append(2.0)
        elif item.get('standings_implications', False) and rivalry_score > 0.3:
            features.append(1.5)
        else:
            features.append(1.0)
            
        # Blood feud indicator (extreme rivalry)
        fights_last_game = item.get('fights_last_meeting', 0)
        suspensions_involved = item.get('players_suspended_from_rivalry', 0)
        
        blood_feud_score = min(1.0, (fights_last_game * 0.3 + suspensions_involved * 0.2))
        features.append(blood_feud_score)
        
        # Fan intensity metric
        features.append(item.get('social_media_rivalry_heat', 0.5))
        
        return features
        
    def _calculate_narrative_momentum(self, item: Dict[str, Any]) -> float:
        """
        Calculate overall narrative momentum multiplier.
        
        Combines recency, intensity, and public awareness.
        """
        base_momentum = 1.0
        
        # Recency boost
        days_since_event = item.get('days_since_narrative_event', 30)
        recency_factor = np.exp(-days_since_event / 30.0)  # Exponential decay
        base_momentum *= (0.5 + 0.5 * recency_factor)
        
        # Intensity boost
        media_mentions = item.get('media_mention_count', 0)
        if media_mentions > 100:
            base_momentum *= 1.5
        elif media_mentions > 50:
            base_momentum *= 1.2
            
        # Playoff/primetime boost
        if item.get('is_playoffs', False):
            base_momentum *= 1.5
        elif item.get('is_primetime', False):
            base_momentum *= 1.2
            
        return min(3.0, base_momentum)  # Cap at 3x
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Hero's Journey stages
        for stage in self.heros_journey_stages:
            names.append(f'heros_journey_{stage}')
            
        # Underdog features
        names.extend([
            'underdog_score',
            'underdog_hot_streak',
            'underdog_sentiment',
            'underdog_historical_success',
            'underdog_key_absence',
            'underdog_playoff_multiplier',
            'underdog_road_bonus',
            'underdog_cinderella_run'
        ])
        
        # Dynasty features
        names.extend([
            'dynasty_rise',
            'dynasty_peak',
            'dynasty_decline',
            'dynasty_fall',
            'dynasty_trajectory_alignment',
            'dynasty_pressure'
        ])
        
        # Redemption features
        names.extend([
            'redemption_score',
            'redemption_arc_beginning',
            'redemption_arc_middle', 
            'redemption_arc_climax',
            'redemption_momentum',
            'redemption_sympathy'
        ])
        
        # Comeback features
        names.extend([
            'comeback_in_game',
            'comeback_series',
            'comeback_season',
            'comeback_precedent'
        ])
        
        # Rivalry features
        names.extend([
            'rivalry_intensity',
            'rivalry_importance_multiplier',
            'rivalry_blood_feud',
            'rivalry_fan_heat'
        ])
        
        return names
