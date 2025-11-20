"""
Ritual and Ceremony Impact Analyzer

Detects and quantifies the narrative impact of pre-game ceremonies,
rituals, and special events on game outcomes.

This transformer identifies when ceremonial elements create narrative
pressure or momentum that influences competitive dynamics.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta


class RitualCeremonyTransformer(BaseEstimator, TransformerMixin):
    """
    Extract ritual and ceremony impact features.
    
    Philosophy:
    - Rituals create narrative weight and expectation
    - Ceremonies can inspire or burden teams
    - Timing of honors affects performance
    - Collective rituals build momentum
    - Sacred moments demand appropriate outcomes
    
    Features (30 total):
    - Jersey retirement proximity (5)
    - Banner raising momentum (5)
    - Anniversary game pressure (5)
    - Tribute night dynamics (5)
    - Pre-game ceremony types (5)
    - Ritual disruption effects (5)
    """
    
    def __init__(
        self,
        ceremony_weight: float = 0.8,
        proximity_decay: float = 0.9,
        include_fan_rituals: bool = True
    ):
        """
        Initialize ritual and ceremony analyzer.
        
        Parameters
        ----------
        ceremony_weight : float
            Base weight for ceremonial impacts
        proximity_decay : float
            How quickly ceremony effects fade
        include_fan_rituals : bool
            Include fan-driven ritual analysis
        """
        self.ceremony_weight = ceremony_weight
        self.proximity_decay = proximity_decay
        self.include_fan_rituals = include_fan_rituals
        
        # Ceremony impact mappings
        self.ceremony_impacts = {
            'jersey_retirement': {
                'home_boost': 0.8,
                'emotional_weight': 0.9,
                'pressure': 0.7
            },
            'banner_raising': {
                'home_boost': 0.9,
                'emotional_weight': 0.8,
                'pressure': 0.6
            },
            'ring_ceremony': {
                'home_boost': 0.7,
                'emotional_weight': 0.7,
                'pressure': 0.5
            },
            'memorial_tribute': {
                'home_boost': 0.6,
                'emotional_weight': 1.0,
                'pressure': 0.8
            },
            'milestone_recognition': {
                'home_boost': 0.5,
                'emotional_weight': 0.6,
                'pressure': 0.6
            },
            'hall_of_fame': {
                'home_boost': 0.7,
                'emotional_weight': 0.8,
                'pressure': 0.7
            }
        }
        
        # Ritual patterns
        self.ritual_types = {
            'pre_game': ['anthem', 'moment_silence', 'special_guest'],
            'fan_driven': ['chants', 'traditions', 'superstitions'],
            'team_specific': ['entrance', 'warmup', 'huddle'],
            'seasonal': ['opener', 'holidays', 'playoffs']
        }
        
    def fit(self, X, y=None):
        """
        Learn ritual pattern impacts from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical game data with ceremony info
        y : ignored
        
        Returns
        -------
        self
        """
        # Learn ceremony outcome patterns
        self.ceremony_patterns_ = self._learn_ceremony_patterns(X)
        
        # Build ritual effectiveness database
        self.ritual_effectiveness_ = self._analyze_ritual_effectiveness(X)
        
        return self
        
    def transform(self, X):
        """
        Extract ritual and ceremony features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with ceremony context
            
        Returns
        -------
        np.ndarray
            Ritual features (n_samples, 30)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Jersey retirement proximity (5)
            retirement_features = self._extract_jersey_retirement(item)
            feature_vec.extend(retirement_features)
            
            # Banner raising momentum (5)
            banner_features = self._extract_banner_raising(item)
            feature_vec.extend(banner_features)
            
            # Anniversary pressure (5)
            anniversary_features = self._extract_anniversary_pressure(item)
            feature_vec.extend(anniversary_features)
            
            # Tribute night dynamics (5)
            tribute_features = self._extract_tribute_dynamics(item)
            feature_vec.extend(tribute_features)
            
            # Pre-game ceremony types (5)
            ceremony_features = self._extract_ceremony_types(item)
            feature_vec.extend(ceremony_features)
            
            # Ritual disruption effects (5)
            disruption_features = self._extract_ritual_disruption(item)
            feature_vec.extend(disruption_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _learn_ceremony_patterns(self, X):
        """Learn how ceremonies affect game outcomes."""
        patterns = {
            'ceremony_win_rates': {},
            'emotional_carryover': {},
            'opponent_response': {}
        }
        
        # Would analyze historical ceremony games
        # For now, theoretical patterns
        patterns['ceremony_win_rates'] = {
            'jersey_retirement': 0.65,
            'banner_raising': 0.70,
            'ring_ceremony': 0.62,
            'memorial_tribute': 0.58,
            'milestone_recognition': 0.55
        }
        
        return patterns
        
    def _analyze_ritual_effectiveness(self, X):
        """Analyze effectiveness of various rituals."""
        return {
            'anthem_variations': {'special_performer': 0.6, 'standard': 0.5},
            'fan_chants': {'unified': 0.7, 'sporadic': 0.4},
            'team_rituals': {'consistent': 0.6, 'modified': 0.3}
        }
        
    def _extract_jersey_retirement(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract jersey retirement ceremony impact.
        
        Returns 5 features measuring retirement ceremony effects.
        """
        features = []
        
        # Check for jersey retirement
        has_retirement = item.get('jersey_retirement_ceremony', False)
        
        if not has_retirement:
            return [0.0] * 5
            
        # Retiree significance
        player_significance = item.get('retiree_franchise_rank', 10)
        if player_significance <= 3:
            features.append(1.0)  # Top 3 all-time player
        elif player_significance <= 5:
            features.append(0.8)
        elif player_significance <= 10:
            features.append(0.6)
        else:
            features.append(0.4)
            
        # Emotional connection strength
        years_since_retirement = item.get('years_since_player_retired', 5)
        fan_favorite_score = item.get('fan_favorite_score', 0.7)
        
        if years_since_retirement < 2:
            emotional_factor = 0.9  # Fresh memories
        elif years_since_retirement < 5:
            emotional_factor = 0.7
        elif years_since_retirement < 10:
            emotional_factor = 0.5
        else:
            emotional_factor = 0.3
            
        features.append(emotional_factor * fan_favorite_score)
        
        # Current team connection
        former_teammates_active = item.get('former_teammates_on_roster', 0)
        if former_teammates_active >= 5:
            features.append(1.0)  # Strong connection
        elif former_teammates_active >= 2:
            features.append(0.6)
        else:
            features.append(0.2)
            
        # Ceremony timing pressure
        is_home = item.get('is_home', True)
        if is_home:
            features.append(self.ceremony_impacts['jersey_retirement']['home_boost'])
        else:
            features.append(0.0)  # Weird if away team
            
        # Legacy game importance
        playoff_implications = item.get('playoff_implications', False)
        rival_opponent = item.get('is_rival', False)
        
        importance_multiplier = 1.0
        if playoff_implications:
            importance_multiplier *= 1.3
        if rival_opponent:
            importance_multiplier *= 1.2
            
        features.append(min(1.0, 0.7 * importance_multiplier))
        
        return features
        
    def _extract_banner_raising(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract banner raising ceremony momentum.
        
        Returns 5 features measuring championship banner effects.
        """
        features = []
        
        has_banner_raising = item.get('banner_raising_ceremony', False)
        
        if not has_banner_raising:
            return [0.0] * 5
            
        # Championship recency
        months_since_championship = item.get('months_since_championship', 6)
        
        if months_since_championship < 6:
            features.append(1.0)  # Season opener typically
        elif months_since_championship < 12:
            features.append(0.7)  # Delayed ceremony
        else:
            features.append(0.4)  # Historical banner
            
        # Team continuity
        roster_retention = item.get('championship_roster_retention', 0.7)
        features.append(roster_retention)
        
        # Opponent psychology
        opponent_eliminated = item.get('opponent_was_eliminated_by', False)
        if opponent_eliminated:
            features.append(0.9)  # Maximum salt in wound
        else:
            features.append(0.3)
            
        # Dynasty potential
        recent_championships = item.get('championships_last_5_years', 1)
        if recent_championships >= 3:
            features.append(0.9)  # Dynasty affirmation
        elif recent_championships >= 2:
            features.append(0.7)
        else:
            features.append(0.5)
            
        # Ceremonial momentum
        ceremony_length = item.get('ceremony_duration_minutes', 15)
        if ceremony_length > 30:
            features.append(0.8)  # Extended celebration
        elif ceremony_length > 20:
            features.append(0.6)
        else:
            features.append(0.4)
            
        return features
        
    def _extract_anniversary_pressure(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract anniversary game pressure features.
        
        Returns 5 features measuring anniversary significance.
        """
        features = []
        
        # Anniversary type and significance
        anniversary_type = item.get('anniversary_type', None)
        
        if not anniversary_type:
            return [0.0] * 5
            
        # Anniversary importance hierarchy
        importance_map = {
            'franchise_founding': 0.8,
            'championship': 0.9,
            'arena_opening': 0.6,
            'historic_game': 0.7,
            'tragedy_memorial': 1.0
        }
        
        features.append(importance_map.get(anniversary_type, 0.5))
        
        # Round number significance
        anniversary_years = item.get('anniversary_years', 0)
        if anniversary_years % 50 == 0:
            features.append(1.0)  # Golden anniversaries
        elif anniversary_years % 25 == 0:
            features.append(0.8)
        elif anniversary_years % 10 == 0:
            features.append(0.6)
        elif anniversary_years % 5 == 0:
            features.append(0.4)
        else:
            features.append(0.2)
            
        # Living connections
        participants_present = item.get('original_participants_attending', 0)
        if participants_present > 20:
            features.append(1.0)
        elif participants_present > 10:
            features.append(0.7)
        elif participants_present > 5:
            features.append(0.5)
        else:
            features.append(0.2)
            
        # Ceremonial weight
        has_special_ceremony = item.get('special_anniversary_ceremony', False)
        invited_legends = item.get('legends_in_attendance', 0)
        
        ceremony_weight = 0.0
        if has_special_ceremony:
            ceremony_weight += 0.5
        ceremony_weight += min(0.5, invited_legends * 0.05)
        
        features.append(ceremony_weight)
        
        # Historical parallel pressure
        anniversary_outcome = item.get('original_event_outcome', None)
        if anniversary_outcome == 'victory':
            features.append(0.8)  # Pressure to repeat
        elif anniversary_outcome == 'defeat':
            features.append(0.7)  # Pressure to reverse
        else:
            features.append(0.5)
            
        return features
        
    def _extract_tribute_dynamics(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract tribute night dynamics features.
        
        Returns 5 features measuring tribute event impact.
        """
        features = []
        
        tribute_type = item.get('tribute_type', None)
        
        if not tribute_type:
            return [0.0] * 5
            
        # Tribute emotional weight
        emotional_weights = {
            'memorial': 1.0,          # Deceased person
            'career_ending_injury': 0.9,
            'retirement_tour': 0.8,
            'milestone': 0.6,
            'charity': 0.5,
            'military': 0.7,
            'first_responders': 0.7
        }
        
        features.append(emotional_weights.get(tribute_type, 0.5))
        
        # Personal connection strength
        if tribute_type == 'memorial':
            connection_to_team = item.get('deceased_connection_to_team', 'none')
            if connection_to_team == 'player':
                features.append(1.0)
            elif connection_to_team == 'coach':
                features.append(0.9)
            elif connection_to_team == 'staff':
                features.append(0.7)
            elif connection_to_team == 'fan':
                features.append(0.6)
            else:
                features.append(0.3)
        else:
            features.append(item.get('tribute_personal_connection', 0.5))
            
        # Community involvement
        community_participation = item.get('community_participation_level', 'low')
        participation_map = {
            'city_wide': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        features.append(participation_map.get(community_participation, 0.3))
        
        # Timing within game
        tribute_timing = item.get('tribute_timing', 'pre_game')
        if tribute_timing == 'throughout':
            features.append(0.9)  # Constant reminder
        elif tribute_timing == 'pre_game':
            features.append(0.6)
        elif tribute_timing == 'intermission':
            features.append(0.4)
        else:
            features.append(0.2)
            
        # Opponent respect level
        opponent_participation = item.get('opponent_participates', False)
        if opponent_participation:
            features.append(0.8)  # Unified tribute
        else:
            features.append(0.3)
            
        return features
        
    def _extract_ceremony_types(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract pre-game ceremony type effects.
        
        Returns 5 features measuring ceremony impacts.
        """
        features = []
        
        ceremonies = item.get('pre_game_ceremonies', [])
        
        if not ceremonies:
            return [0.0] * 5
            
        # Ceremony count pressure
        ceremony_count = len(ceremonies)
        if ceremony_count >= 3:
            features.append(0.9)  # Overload
        elif ceremony_count == 2:
            features.append(0.6)
        elif ceremony_count == 1:
            features.append(0.4)
        else:
            features.append(0.0)
            
        # Special anthem performer
        anthem_performer = item.get('anthem_performer_type', 'standard')
        if anthem_performer == 'celebrity':
            features.append(0.7)
        elif anthem_performer == 'hero':
            features.append(0.8)  # Military, first responder
        elif anthem_performer == 'child':
            features.append(0.6)  # Emotional appeal
        elif anthem_performer == 'group':
            features.append(0.5)
        else:
            features.append(0.3)
            
        # Moment of silence impact
        has_moment_silence = item.get('moment_of_silence', False)
        if has_moment_silence:
            silence_reason = item.get('moment_silence_reason', 'tragedy')
            if silence_reason == 'tragedy':
                features.append(0.9)
            elif silence_reason == 'remembrance':
                features.append(0.7)
            else:
                features.append(0.5)
        else:
            features.append(0.0)
            
        # Special guests/drops
        special_elements = {
            'celebrity_puck_drop': 0.6,
            'legend_appearance': 0.8,
            'championship_team_reunion': 0.9,
            'youth_team': 0.5,
            'none': 0.0
        }
        
        special_element = item.get('special_ceremony_element', 'none')
        features.append(special_elements.get(special_element, 0.3))
        
        # Ceremony timing efficiency
        total_ceremony_time = item.get('total_ceremony_minutes', 10)
        if total_ceremony_time > 45:
            features.append(0.3)  # Too long, energy drain
        elif total_ceremony_time > 30:
            features.append(0.5)
        elif 15 <= total_ceremony_time <= 25:
            features.append(0.8)  # Optimal
        else:
            features.append(0.6)
            
        return features
        
    def _extract_ritual_disruption(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract ritual disruption effect features.
        
        Returns 5 features measuring broken ritual impacts.
        """
        features = []
        
        # Regular ritual disruption
        ritual_disrupted = item.get('regular_ritual_disrupted', False)
        
        if ritual_disrupted:
            disruption_type = item.get('disruption_type', 'minor')
            if disruption_type == 'major':
                features.append(0.8)  # Significant negative impact
            elif disruption_type == 'moderate':
                features.append(0.5)
            else:
                features.append(0.3)
        else:
            features.append(0.0)
            
        # Superstition breaking
        superstition_broken = item.get('team_superstition_broken', False)
        if superstition_broken:
            superstition_importance = item.get('superstition_importance', 0.5)
            features.append(superstition_importance)
        else:
            features.append(0.0)
            
        # Fan ritual participation
        if self.include_fan_rituals:
            fan_ritual_level = item.get('fan_ritual_participation', 'normal')
            if fan_ritual_level == 'exceptional':
                features.append(0.8)
            elif fan_ritual_level == 'high':
                features.append(0.6)
            elif fan_ritual_level == 'normal':
                features.append(0.4)
            elif fan_ritual_level == 'low':
                features.append(0.2)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Venue tradition adherence
        venue_traditions = item.get('venue_traditions_followed', True)
        if not venue_traditions:
            tradition_importance = item.get('tradition_importance_level', 0.5)
            features.append(-tradition_importance)  # Negative impact
        else:
            features.append(0.3)  # Slight positive
            
        # Ritual innovation success
        new_ritual_attempted = item.get('new_ritual_introduced', False)
        if new_ritual_attempted:
            reception = item.get('new_ritual_reception', 'neutral')
            if reception == 'enthusiastic':
                features.append(0.7)
            elif reception == 'positive':
                features.append(0.5)
            elif reception == 'neutral':
                features.append(0.0)
            else:
                features.append(-0.5)  # Rejected
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Jersey retirement features
        names.extend([
            'retirement_player_significance',
            'retirement_emotional_connection',
            'retirement_team_connection',
            'retirement_ceremony_pressure',
            'retirement_legacy_importance'
        ])
        
        # Banner raising features
        names.extend([
            'banner_championship_recency',
            'banner_roster_continuity',
            'banner_opponent_psychology',
            'banner_dynasty_potential',
            'banner_ceremonial_momentum'
        ])
        
        # Anniversary features
        names.extend([
            'anniversary_type_importance',
            'anniversary_round_number',
            'anniversary_living_connections',
            'anniversary_ceremonial_weight',
            'anniversary_historical_pressure'
        ])
        
        # Tribute features
        names.extend([
            'tribute_emotional_weight',
            'tribute_personal_connection',
            'tribute_community_involvement',
            'tribute_game_timing',
            'tribute_opponent_respect'
        ])
        
        # Ceremony type features
        names.extend([
            'ceremony_count_pressure',
            'ceremony_anthem_special',
            'ceremony_moment_silence',
            'ceremony_special_elements',
            'ceremony_timing_efficiency'
        ])
        
        # Ritual disruption features
        names.extend([
            'ritual_regular_disruption',
            'ritual_superstition_breaking',
            'ritual_fan_participation',
            'ritual_venue_tradition',
            'ritual_innovation_reception'
        ])
        
        return names
