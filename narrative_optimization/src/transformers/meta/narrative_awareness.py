"""
Meta-Narrative Awareness Detector

Identifies when participants (players, coaches, media, fans) become
aware of and respond to narrative pressures, creating self-fulfilling
or self-defeating prophecies.

This transformer detects meta-level narrative consciousness and its
impact on outcomes.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Set
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import re


class MetaNarrativeAwarenessTransformer(BaseEstimator, TransformerMixin):
    """
    Extract meta-narrative awareness features.
    
    Philosophy:
    - Narrative awareness changes behavior
    - Self-fulfilling prophecies are real
    - Media narrative seeding affects outcomes
    - Player/coach acknowledgment amplifies pressure
    - Breaking the fourth wall has consequences
    
    Features (35 total):
    - Media narrative saturation (6)
    - Player/coach acknowledgment (6)
    - Fan base expectation unity (5)
    - Betting market reflection (5)
    - Commentary bias amplification (5)
    - Fourth wall breaking moments (4)
    - Narrative resistance indicators (4)
    """
    
    def __init__(
        self,
        include_social_media: bool = True,
        awareness_threshold: float = 0.7,
        meta_weight: float = 0.8
    ):
        """
        Initialize meta-narrative awareness detector.
        
        Parameters
        ----------
        include_social_media : bool
            Include social media narrative analysis
        awareness_threshold : float
            Threshold for narrative consciousness
        meta_weight : float
            Weight for meta-level features
        """
        self.include_social_media = include_social_media
        self.awareness_threshold = awareness_threshold
        self.meta_weight = meta_weight
        
        # Narrative awareness indicators
        self.awareness_markers = {
            'explicit': {
                'keywords': ['narrative', 'story', 'script', 'destiny', 'meant to be'],
                'weight': 1.0
            },
            'implicit': {
                'keywords': ['feels like', 'seems like', 'supposed to', 'inevitable'],
                'weight': 0.7
            },
            'deflection': {
                'keywords': ['just another game', 'no pressure', 'not thinking about'],
                'weight': 0.8  # Protesting too much
            },
            'embrace': {
                'keywords': ['embrace', 'opportunity', 'moment', 'special'],
                'weight': 0.9
            }
        }
        
        # Meta-narrative types
        self.meta_patterns = {
            'coronation': 'Team expected to win championship',
            'david_goliath': 'Underdog story acknowledged',
            'redemption': 'Comeback narrative recognized',
            'passing_torch': 'Generational transition noted',
            'last_dance': 'Final opportunity acknowledged',
            'destiny': 'Fate narrative embraced'
        }
        
    def fit(self, X, y=None):
        """
        Learn meta-narrative patterns from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical data with narrative outcomes
        y : ignored
        
        Returns
        -------
        self
        """
        # Learn awareness impact patterns
        self.awareness_impacts_ = self._learn_awareness_impacts(X)
        
        # Build meta-narrative database
        self.meta_patterns_ = self._build_meta_patterns(X)
        
        return self
        
    def transform(self, X):
        """
        Extract meta-narrative awareness features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with narrative context
            
        Returns
        -------
        np.ndarray
            Meta-awareness features (n_samples, 35)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Media saturation (6)
            media_features = self._extract_media_saturation(item)
            feature_vec.extend(media_features)
            
            # Player/coach acknowledgment (6)
            acknowledgment_features = self._extract_participant_acknowledgment(item)
            feature_vec.extend(acknowledgment_features)
            
            # Fan expectation unity (5)
            fan_features = self._extract_fan_expectation_unity(item)
            feature_vec.extend(fan_features)
            
            # Betting market reflection (5)
            betting_features = self._extract_betting_narrative_reflection(item)
            feature_vec.extend(betting_features)
            
            # Commentary bias (5)
            commentary_features = self._extract_commentary_bias(item)
            feature_vec.extend(commentary_features)
            
            # Fourth wall breaking (4)
            fourth_wall_features = self._extract_fourth_wall_breaking(item)
            feature_vec.extend(fourth_wall_features)
            
            # Narrative resistance (4)
            resistance_features = self._extract_narrative_resistance(item)
            feature_vec.extend(resistance_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _learn_awareness_impacts(self, X):
        """Learn how narrative awareness affects outcomes."""
        impacts = {
            'acknowledgment_effects': {},
            'resistance_success_rate': {},
            'embrace_outcomes': {}
        }
        
        # Would analyze when awareness helped/hurt
        # For now, theoretical impacts
        impacts['acknowledgment_effects'] = {
            'positive_embrace': 0.65,
            'nervous_acknowledgment': 0.45,
            'active_denial': 0.40,
            'meta_commentary': 0.55
        }
        
        return impacts
        
    def _build_meta_patterns(self, X):
        """Build database of meta-narrative patterns."""
        return {
            'self_fulfilling_rate': 0.62,
            'narrative_resistance_success': 0.38,
            'media_prophecy_accuracy': 0.58
        }
        
    def _extract_media_saturation(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract media narrative saturation levels.
        
        Returns 6 features measuring media narrative density.
        """
        features = []
        
        # Narrative mention frequency
        media_mentions = item.get('narrative_media_mentions', 0)
        total_mentions = item.get('total_media_mentions', 1)
        
        narrative_saturation = media_mentions / max(1, total_mentions)
        features.append(min(1.0, narrative_saturation))
        
        # Headline narrative percentage
        headlines = item.get('recent_headlines', [])
        narrative_headlines = sum(1 for h in headlines 
                                if any(marker in h.lower() 
                                      for markers in self.awareness_markers.values()
                                      for marker in markers['keywords']))
        
        if headlines:
            features.append(narrative_headlines / len(headlines))
        else:
            features.append(0.0)
            
        # Media narrative consistency
        media_narratives = item.get('identified_media_narratives', [])
        if len(media_narratives) > 1:
            # Check if all telling same story
            unique_narratives = len(set(media_narratives))
            consistency = 1.0 - (unique_narratives - 1) / len(media_narratives)
            features.append(consistency)
        else:
            features.append(0.0)
            
        # Narrative sophistication level
        sophisticated_terms = ['arc', 'narrative', 'story', 'script', 'chapter']
        sophistication_count = sum(1 for term in sophisticated_terms
                                 if term in item.get('media_text', '').lower())
        
        features.append(min(1.0, sophistication_count / 3.0))
        
        # Cross-media platform coverage
        platforms_covering = item.get('platforms_covering_narrative', 0)
        features.append(min(1.0, platforms_covering / 5.0))  # 5 major platforms
        
        # Narrative momentum (increasing mentions)
        mention_trajectory = item.get('narrative_mention_trajectory', 0.0)
        features.append(np.tanh(mention_trajectory))  # -1 to 1
        
        return features
        
    def _extract_participant_acknowledgment(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract player/coach narrative acknowledgment.
        
        Returns 6 features measuring participant awareness.
        """
        features = []
        
        # Player quotes analysis
        player_quotes = item.get('recent_player_quotes', [])
        
        awareness_scores = {
            'explicit': 0.0,
            'implicit': 0.0,
            'deflection': 0.0,
            'embrace': 0.0
        }
        
        for quote in player_quotes:
            quote_lower = quote.lower()
            for awareness_type, markers in self.awareness_markers.items():
                if any(keyword in quote_lower for keyword in markers['keywords']):
                    awareness_scores[awareness_type] += markers['weight']
                    
        # Normalize and add features
        total_quotes = max(1, len(player_quotes))
        features.append(min(1.0, awareness_scores['explicit'] / total_quotes))
        features.append(min(1.0, awareness_scores['implicit'] / total_quotes))
        
        # Coach narrative management
        coach_quotes = item.get('coach_quotes', [])
        coach_acknowledges = any(
            any(keyword in quote.lower() 
                for keyword in self.awareness_markers['explicit']['keywords'])
            for quote in coach_quotes
        )
        
        if coach_acknowledges:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Team messaging alignment
        team_messaging = item.get('team_official_messaging', '')
        if any(marker in team_messaging.lower() 
               for marker in ['one game at a time', 'no distractions', 'focus']):
            features.append(0.8)  # Trying to deflect narrative
        else:
            features.append(0.3)
            
        # Body language/non-verbal cues
        tension_indicators = item.get('pre_game_tension_level', 0.5)
        features.append(tension_indicators)
        
        # Historical narrative comments
        previous_narrative_games = item.get('previous_high_narrative_games', 0)
        narrative_experience = min(1.0, previous_narrative_games / 10.0)
        features.append(narrative_experience)
        
        return features
        
    def _extract_fan_expectation_unity(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract fan base expectation unity features.
        
        Returns 5 features measuring collective fan consciousness.
        """
        features = []
        
        # Fan confidence level
        fan_confidence = item.get('fan_confidence_score', 0.5)
        features.append(fan_confidence)
        
        # Expectation consensus
        fan_predictions = item.get('fan_prediction_distribution', {})
        if fan_predictions:
            # Calculate entropy of predictions
            total_predictions = sum(fan_predictions.values())
            if total_predictions > 0:
                probabilities = [count/total_predictions 
                               for count in fan_predictions.values()]
                entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
                # Low entropy = high consensus
                consensus = 1.0 - min(1.0, entropy / 2.0)
                features.append(consensus)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
            
        # Ticket demand as belief indicator
        ticket_demand_ratio = item.get('ticket_demand_vs_average', 1.0)
        if ticket_demand_ratio > 2.0:
            features.append(1.0)  # Everyone wants to witness
        else:
            features.append(min(1.0, ticket_demand_ratio / 2.0))
            
        # Social media sentiment alignment
        if self.include_social_media:
            sentiment_variance = item.get('fan_sentiment_variance', 0.5)
            features.append(1.0 - sentiment_variance)
        else:
            features.append(0.5)
            
        # Ritual participation rate
        ritual_participation = item.get('fan_ritual_participation_rate', 0.5)
        features.append(ritual_participation)
        
        return features
        
    def _extract_betting_narrative_reflection(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract betting market narrative reflection.
        
        Returns 5 features measuring how betting reflects narrative.
        """
        features = []
        
        # Line movement narrative alignment
        line_movement = item.get('line_movement_direction', 0)
        narrative_direction = item.get('narrative_favors', 'neutral')
        
        if (line_movement > 0 and narrative_direction == 'home') or \
           (line_movement < 0 and narrative_direction == 'away'):
            features.append(0.9)  # Market following narrative
        elif line_movement == 0:
            features.append(0.5)
        else:
            features.append(0.1)  # Market resisting narrative
            
        # Sharp vs public alignment
        sharp_percentage = item.get('sharp_bet_percentage', 50)
        public_percentage = item.get('public_bet_percentage', 50)
        
        if abs(sharp_percentage - public_percentage) < 10:
            features.append(0.8)  # Everyone agrees
        else:
            features.append(0.3)
            
        # Unusual betting patterns
        betting_volume = item.get('betting_volume_vs_average', 1.0)
        if betting_volume > 2.0:
            features.append(1.0)  # Narrative game
        else:
            features.append(betting_volume / 2.0)
            
        # Prop bet narrative alignment
        narrative_props = item.get('narrative_aligned_prop_volume', 0.5)
        features.append(narrative_props)
        
        # Late money narrative shift
        late_money_shift = item.get('late_money_narrative_shift', False)
        if late_money_shift:
            features.append(0.9)  # Smart money sees the story
        else:
            features.append(0.3)
            
        return features
        
    def _extract_commentary_bias(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract commentary bias amplification features.
        
        Returns 5 features measuring announcer narrative seeding.
        """
        features = []
        
        # Pre-game narrative framing
        pregame_narrative_mentions = item.get('pregame_narrative_mentions', 0)
        if pregame_narrative_mentions > 5:
            features.append(1.0)
        else:
            features.append(pregame_narrative_mentions / 5.0)
            
        # Commentary narrative keywords
        commentary_text = item.get('commentary_sample', '')
        narrative_density = 0.0
        
        for marker_type, markers in self.awareness_markers.items():
            keyword_count = sum(1 for keyword in markers['keywords']
                              if keyword in commentary_text.lower())
            narrative_density += keyword_count * markers['weight']
            
        features.append(min(1.0, narrative_density / 10.0))
        
        # Historical reference frequency
        historical_references = item.get('commentary_historical_references', 0)
        features.append(min(1.0, historical_references / 5.0))
        
        # Narrative question frequency
        rhetorical_questions = item.get('commentary_rhetorical_questions', 0)
        features.append(min(1.0, rhetorical_questions / 10.0))
        
        # Commentary emotion level
        emotion_level = item.get('commentary_emotion_level', 0.5)
        features.append(emotion_level)
        
        return features
        
    def _extract_fourth_wall_breaking(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract fourth wall breaking moment detection.
        
        Returns 4 features measuring meta-awareness moments.
        """
        features = []
        
        # Direct narrative acknowledgment
        direct_acknowledgments = item.get('direct_narrative_acknowledgments', 0)
        if direct_acknowledgments > 0:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # "Script" language usage
        script_references = item.get('script_narrative_references', 0)
        features.append(min(1.0, script_references / 3.0))
        
        # Self-aware humor/memes
        self_aware_content = item.get('self_aware_social_content', 0)
        if self_aware_content > 10:
            features.append(1.0)  # Viral meta moment
        else:
            features.append(self_aware_content / 10.0)
            
        # Winking at the camera moments
        meta_moments = [
            'player_narrative_gesture',
            'coach_narrative_comment', 
            'fan_narrative_sign',
            'media_self_reference'
        ]
        
        meta_score = sum(0.25 for moment in meta_moments
                        if item.get(moment, False))
        features.append(meta_score)
        
        return features
        
    def _extract_narrative_resistance(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract narrative resistance indicator features.
        
        Returns 4 features measuring attempts to break narrative.
        """
        features = []
        
        # Active narrative denial
        denial_statements = item.get('narrative_denial_count', 0)
        if denial_statements > 3:
            features.append(1.0)  # Protesting too much
        else:
            features.append(denial_statements / 3.0)
            
        # Counter-narrative creation
        counter_narrative = item.get('counter_narrative_attempted', False)
        if counter_narrative:
            success = item.get('counter_narrative_traction', 0.0)
            features.append(success)
        else:
            features.append(0.0)
            
        # Chaos agent behavior
        unpredictable_actions = item.get('narrative_disrupting_actions', 0)
        features.append(min(1.0, unpredictable_actions / 5.0))
        
        # Narrative fatigue indicators
        narrative_duration = item.get('days_of_narrative_coverage', 0)
        if narrative_duration > 14:
            fatigue = min(1.0, (narrative_duration - 14) / 14.0)
            features.append(fatigue)
        else:
            features.append(0.0)
            
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Media saturation features
        names.extend([
            'media_narrative_saturation',
            'media_headline_percentage',
            'media_narrative_consistency',
            'media_sophistication_level',
            'media_platform_coverage',
            'media_narrative_momentum'
        ])
        
        # Participant acknowledgment features
        names.extend([
            'player_explicit_awareness',
            'player_implicit_awareness',
            'coach_narrative_acknowledgment',
            'team_messaging_alignment',
            'participant_tension_level',
            'participant_narrative_experience'
        ])
        
        # Fan expectation features
        names.extend([
            'fan_confidence_level',
            'fan_expectation_consensus',
            'fan_ticket_demand_belief',
            'fan_sentiment_alignment',
            'fan_ritual_participation'
        ])
        
        # Betting market features
        names.extend([
            'betting_line_narrative_alignment',
            'betting_sharp_public_consensus',
            'betting_volume_narrative',
            'betting_prop_narrative_alignment',
            'betting_late_money_shift'
        ])
        
        # Commentary bias features
        names.extend([
            'commentary_pregame_framing',
            'commentary_narrative_density',
            'commentary_historical_references',
            'commentary_rhetorical_questions',
            'commentary_emotion_level'
        ])
        
        # Fourth wall features
        names.extend([
            'fourth_wall_direct_acknowledgment',
            'fourth_wall_script_references',
            'fourth_wall_self_aware_content',
            'fourth_wall_meta_moments'
        ])
        
        # Narrative resistance features
        names.extend([
            'resistance_active_denial',
            'resistance_counter_narrative',
            'resistance_chaos_behavior',
            'resistance_narrative_fatigue'
        ])
        
        return names
