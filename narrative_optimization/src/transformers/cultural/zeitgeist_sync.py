"""
Cultural Zeitgeist Resonance Tracker

Detects and quantifies how sports narratives align with broader
cultural themes, social movements, and collective consciousness.

This transformer identifies when games transcend sports to become
cultural moments by resonating with contemporary themes.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Set
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
from collections import Counter


class CulturalZeitgeistTransformer(BaseEstimator, TransformerMixin):
    """
    Extract cultural zeitgeist resonance features.
    
    Philosophy:
    - Sports narratives gain power when aligned with cultural moments
    - City/region narrative needs create outcome pressure
    - Media crystallization reflects collective consciousness
    - Social movements can infuse games with meaning
    - Championship windows align with generational moments
    
    Features (35 total):
    - Current event thematic alignment (7)
    - City narrative need detection (6)
    - Media narrative crystallization (6)
    - Social sentiment convergence (6)
    - Championship window psychology (5)
    - Cultural symbolism amplification (5)
    """
    
    def __init__(
        self,
        include_social_analysis: bool = True,
        cultural_weight: float = 0.7,
        local_vs_national: float = 0.6  # Weight local culture more
    ):
        """
        Initialize cultural zeitgeist analyzer.
        
        Parameters
        ----------
        include_social_analysis : bool
            Include social media sentiment analysis
        cultural_weight : float
            How much to weight cultural factors
        local_vs_national : float
            Balance between local and national cultural themes
        """
        self.include_social_analysis = include_social_analysis
        self.cultural_weight = cultural_weight
        self.local_vs_national = local_vs_national
        
        # Cultural theme categories
        self.cultural_themes = {
            'resilience': {
                'keywords': ['comeback', 'overcome', 'rebuild', 'recover', 'rise'],
                'contexts': ['economic_recovery', 'natural_disaster', 'social_healing']
            },
            'unity': {
                'keywords': ['together', 'united', 'community', 'solidarity', 'one'],
                'contexts': ['divided_times', 'crisis_response', 'celebration']
            },
            'change': {
                'keywords': ['new', 'transformation', 'revolution', 'fresh', 'different'],
                'contexts': ['political_shift', 'generational_change', 'innovation']
            },
            'tradition': {
                'keywords': ['heritage', 'legacy', 'history', 'classic', 'original'],
                'contexts': ['anniversary', 'nostalgia', 'preservation']
            },
            'justice': {
                'keywords': ['fair', 'deserved', 'earned', 'righteous', 'vindication'],
                'contexts': ['social_movement', 'correction', 'balance']
            }
        }
        
        # City narrative archetypes
        self.city_narratives = {
            'industrial_revival': ['rust_belt', 'manufacturing', 'blue_collar'],
            'tech_disruption': ['innovation', 'startup', 'silicon'],
            'cultural_renaissance': ['arts', 'music', 'creative'],
            'sports_salvation': ['title_drought', 'championship_starved'],
            'david_goliath': ['small_market', 'underdog_city', 'overlooked']
        }
        
    def fit(self, X, y=None):
        """
        Learn cultural patterns from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical data with cultural context
        y : ignored
        
        Returns
        -------
        self
        """
        # Build cultural pattern database
        self.cultural_patterns_ = self._build_cultural_patterns(X)
        
        # Learn theme resonance strengths
        self.theme_resonance_ = self._learn_theme_resonance(X)
        
        return self
        
    def transform(self, X):
        """
        Extract cultural zeitgeist features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Game data with cultural context
            
        Returns
        -------
        np.ndarray
            Cultural features (n_samples, 35)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Current event alignment (7)
            event_features = self._extract_current_event_alignment(item)
            feature_vec.extend(event_features)
            
            # City narrative needs (6)
            city_features = self._extract_city_narrative_needs(item)
            feature_vec.extend(city_features)
            
            # Media crystallization (6)
            media_features = self._extract_media_crystallization(item)
            feature_vec.extend(media_features)
            
            # Social sentiment convergence (6)
            if self.include_social_analysis:
                social_features = self._extract_social_convergence(item)
            else:
                social_features = [0.0] * 6
            feature_vec.extend(social_features)
            
            # Championship window psychology (5)
            window_features = self._extract_championship_psychology(item)
            feature_vec.extend(window_features)
            
            # Cultural symbolism (5)
            symbolism_features = self._extract_cultural_symbolism(item)
            feature_vec.extend(symbolism_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _build_cultural_patterns(self, X):
        """Build database of cultural patterns from training data."""
        patterns = {
            'theme_success_rates': {},
            'city_narrative_completion': {},
            'social_amplification_factors': {}
        }
        
        # This would analyze when cultural alignment led to outcomes
        # For now, theoretical values
        patterns['theme_success_rates'] = {
            'resilience': 0.65,
            'unity': 0.60,
            'change': 0.55,
            'tradition': 0.58,
            'justice': 0.62
        }
        
        return patterns
        
    def _learn_theme_resonance(self, X):
        """Learn which themes resonate in different contexts."""
        # Would analyze historical data
        # For now, return theoretical resonance scores
        return {
            'economic_downturn': {'resilience': 0.9, 'unity': 0.7},
            'social_unrest': {'justice': 0.9, 'unity': 0.8},
            'celebration': {'tradition': 0.8, 'unity': 0.9},
            'change_period': {'change': 0.9, 'resilience': 0.6}
        }
        
    def _extract_current_event_alignment(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract alignment with current cultural events.
        
        Returns 7 features measuring thematic resonance.
        """
        features = []
        
        # Get current context
        current_events = item.get('current_cultural_events', [])
        team_narrative = item.get('team_narrative_theme', '')
        
        # Theme resonance scores
        theme_scores = {}
        for theme, info in self.cultural_themes.items():
            score = 0.0
            
            # Check keyword presence
            narrative_text = item.get('narrative_description', '').lower()
            keyword_matches = sum(1 for kw in info['keywords'] 
                                if kw in narrative_text)
            score += min(0.5, keyword_matches * 0.1)
            
            # Check context alignment
            for event in current_events:
                if any(ctx in event.lower() for ctx in info['contexts']):
                    score += 0.3
                    
            theme_scores[theme] = min(1.0, score)
            
        # Add top theme scores
        for theme in ['resilience', 'unity', 'change', 'tradition', 'justice']:
            features.append(theme_scores.get(theme, 0.0))
            
        # Meta-theme convergence (multiple themes align)
        active_themes = sum(1 for score in theme_scores.values() if score > 0.5)
        if active_themes >= 3:
            features.append(1.0)  # Strong cultural moment
        elif active_themes >= 2:
            features.append(0.6)
        else:
            features.append(0.0)
            
        # Timing synchronicity (game during cultural moment)
        cultural_moment = item.get('occurring_during_major_event', False)
        if cultural_moment:
            features.append(1.0)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_city_narrative_needs(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract city-specific narrative need features.
        
        Returns 6 features measuring local cultural pressure.
        """
        features = []
        
        # City characteristics
        city = item.get('city', 'Unknown')
        city_attributes = item.get('city_attributes', [])
        years_since_championship = item.get('city_championship_drought', 0)
        
        # Championship drought pressure
        if years_since_championship > 50:
            features.append(1.0)  # Desperate for success
        elif years_since_championship > 25:
            features.append(0.8)
        elif years_since_championship > 10:
            features.append(0.5)
        else:
            features.append(0.2)
            
        # City narrative type detection
        narrative_scores = {}
        for narrative_type, keywords in self.city_narratives.items():
            score = sum(0.3 for kw in keywords if kw in city_attributes)
            narrative_scores[narrative_type] = min(1.0, score)
            
        # Industrial revival narrative
        if narrative_scores.get('industrial_revival', 0) > 0.5:
            features.append(0.9)  # Sports as city renaissance symbol
        else:
            features.append(0.0)
            
        # Small market magic
        market_size = item.get('market_size', 'medium')
        if market_size == 'small' and narrative_scores.get('david_goliath', 0) > 0.5:
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Sports as sole bright spot
        city_struggles = item.get('city_economic_struggles', False)
        other_success = item.get('city_other_successes', True)
        
        if city_struggles and not other_success:
            features.append(0.9)  # Team is city's hope
        else:
            features.append(0.0)
            
        # Regional pride factor
        regional_rivalry = item.get('regional_rivalry_game', False)
        if regional_rivalry:
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Generational moment (city's time has come)
        demographic_shift = item.get('city_demographic_shift', False)
        new_generation = item.get('new_generation_emerging', False)
        
        if demographic_shift or new_generation:
            features.append(0.8)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_media_crystallization(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract media narrative crystallization features.
        
        Returns 6 features measuring media consensus building.
        """
        features = []
        
        # Media narrative analysis
        media_narratives = item.get('media_narratives', [])
        headline_themes = item.get('headline_themes', [])
        
        # Narrative consistency across outlets
        if len(media_narratives) > 3:
            # Count common themes
            theme_counter = Counter()
            for narrative in media_narratives:
                themes = narrative.get('themes', [])
                theme_counter.update(themes)
                
            # Check for dominant narrative
            if theme_counter:
                most_common_count = theme_counter.most_common(1)[0][1]
                consistency = most_common_count / len(media_narratives)
                features.append(min(1.0, consistency))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Headline convergence
        if len(headline_themes) > 5:
            unique_themes = len(set(headline_themes))
            convergence = 1.0 - (unique_themes / len(headline_themes))
            features.append(convergence)
        else:
            features.append(0.0)
            
        # National media attention
        national_coverage = item.get('national_media_coverage_level', 0)
        features.append(min(1.0, national_coverage / 10.0))
        
        # Story arc maturity
        narrative_days = item.get('days_since_narrative_began', 0)
        if 3 <= narrative_days <= 10:
            features.append(0.8)  # Peak narrative window
        elif narrative_days > 10:
            features.append(0.4)  # Fading
        else:
            features.append(0.2)  # Building
            
        # Cross-sport narrative (transcends hockey)
        mentioned_in_other_contexts = item.get('non_sports_media_mentions', 0)
        if mentioned_in_other_contexts > 5:
            features.append(1.0)
        else:
            features.append(mentioned_in_other_contexts / 5.0)
            
        # Documentary/feature potential
        human_interest_score = item.get('human_interest_score', 0.0)
        features.append(human_interest_score)
        
        return features
        
    def _extract_social_convergence(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract social media sentiment convergence features.
        
        Returns 6 features measuring collective consciousness.
        """
        features = []
        
        # Social sentiment metrics
        sentiment_variance = item.get('social_sentiment_variance', 1.0)
        engagement_rate = item.get('social_engagement_rate', 0.0)
        viral_coefficient = item.get('viral_spread_coefficient', 0.0)
        
        # Sentiment unity (everyone feels the same)
        unity_score = 1.0 - min(1.0, sentiment_variance)
        features.append(unity_score)
        
        # Engagement intensity
        features.append(min(1.0, engagement_rate))
        
        # Viral momentum
        if viral_coefficient > 2.0:
            features.append(1.0)  # Gone viral
        else:
            features.append(viral_coefficient / 2.0)
            
        # Meme crystallization
        meme_count = item.get('related_meme_count', 0)
        meme_spread = item.get('meme_spread_rate', 0.0)
        
        if meme_count > 10 and meme_spread > 0.5:
            features.append(1.0)  # Meme magic activated
        else:
            features.append(min(1.0, meme_count / 10.0 * meme_spread))
            
        # Hashtag convergence
        dominant_hashtag_percentage = item.get('dominant_hashtag_usage', 0.0)
        features.append(dominant_hashtag_percentage)
        
        # Cross-platform consensus
        platform_sentiment_alignment = item.get('cross_platform_alignment', 0.0)
        features.append(platform_sentiment_alignment)
        
        return features
        
    def _extract_championship_psychology(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract championship window psychological features.
        
        Returns 5 features measuring title window pressure.
        """
        features = []
        
        # Window status
        window_stage = item.get('championship_window_stage', 'closed')
        core_age_avg = item.get('core_players_avg_age', 27)
        
        # Window urgency based on stage
        urgency_map = {
            'opening': 0.3,
            'open': 0.6,
            'peak': 0.9,
            'closing': 1.0,
            'closed': 0.1
        }
        features.append(urgency_map.get(window_stage, 0.5))
        
        # Generational talent pressure
        has_generational_talent = item.get('has_generational_player', False)
        talent_age = item.get('superstar_age', 25)
        
        if has_generational_talent:
            if talent_age > 32:
                features.append(1.0)  # Running out of time
            elif talent_age > 28:
                features.append(0.7)
            else:
                features.append(0.4)
        else:
            features.append(0.0)
            
        # "This is our year" syndrome
        preseason_expectations = item.get('preseason_championship_odds', 20.0)
        current_performance = item.get('current_championship_odds', 20.0)
        
        if preseason_expectations < 10.0 and current_performance < 5.0:
            features.append(0.9)  # Living up to hype
        elif preseason_expectations < 10.0:
            features.append(0.6)  # Pressure building
        else:
            features.append(0.0)
            
        # All-in indicator (traded future for now)
        futures_traded = item.get('draft_picks_traded_away', 0)
        rental_players = item.get('rental_players_acquired', 0)
        
        all_in_score = min(1.0, (futures_traded + rental_players) * 0.2)
        features.append(all_in_score)
        
        # Last dance narrative
        key_players_expiring = item.get('key_players_contract_expiring', 0)
        coach_hot_seat = item.get('coach_job_security', 1.0)
        
        if key_players_expiring >= 3 or coach_hot_seat < 0.3:
            features.append(0.9)  # Now or never
        elif key_players_expiring >= 1:
            features.append(0.5)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_cultural_symbolism(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract cultural symbolism amplification features.
        
        Returns 5 features measuring symbolic narrative power.
        """
        features = []
        
        # David vs Goliath symbolism
        payroll_differential = item.get('payroll_ratio', 1.0)
        market_differential = item.get('market_size_ratio', 1.0)
        
        if payroll_differential < 0.5 or market_differential < 0.5:
            features.append(0.9)  # Clear underdog
        elif payroll_differential < 0.7 or market_differential < 0.7:
            features.append(0.5)
        else:
            features.append(0.0)
            
        # Old vs New symbolism
        franchise_age_diff = item.get('franchise_age_differential', 0)
        playing_style_contrast = item.get('style_contrast_score', 0.0)
        
        if abs(franchise_age_diff) > 50 and playing_style_contrast > 0.7:
            features.append(0.8)  # Clash of eras
        else:
            features.append(playing_style_contrast * 0.5)
            
        # Geographic/cultural divide
        geographic_distance = item.get('cities_distance_miles', 0)
        cultural_difference = item.get('cultural_difference_score', 0.0)
        
        if geographic_distance > 2000 and cultural_difference > 0.7:
            features.append(0.8)  # Coast vs coast, etc.
        else:
            features.append(cultural_difference * 0.5)
            
        # Redemption symbolism
        redemption_narratives = item.get('active_redemption_stories', 0)
        public_sympathy = item.get('public_sympathy_score', 0.0)
        
        if redemption_narratives > 0 and public_sympathy > 0.7:
            features.append(0.9)
        else:
            features.append(public_sympathy * redemption_narratives * 0.3)
            
        # Destiny symbolism
        anniversary = item.get('significant_anniversary', False)
        numerology = item.get('numerical_significance', False)
        historical_parallel = item.get('historical_parallel_strength', 0.0)
        
        destiny_score = 0.0
        if anniversary:
            destiny_score += 0.4
        if numerology:
            destiny_score += 0.3
        destiny_score += historical_parallel * 0.3
        
        features.append(min(1.0, destiny_score))
        
        return features
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Event alignment features
        names.extend([
            'event_theme_resilience',
            'event_theme_unity',
            'event_theme_change',
            'event_theme_tradition',
            'event_theme_justice',
            'event_meta_convergence',
            'event_timing_synchronicity'
        ])
        
        # City narrative features
        names.extend([
            'city_championship_drought',
            'city_industrial_revival',
            'city_small_market_magic',
            'city_sole_bright_spot',
            'city_regional_pride',
            'city_generational_moment'
        ])
        
        # Media features
        names.extend([
            'media_narrative_consistency',
            'media_headline_convergence',
            'media_national_attention',
            'media_story_arc_maturity',
            'media_cross_sport_narrative',
            'media_documentary_potential'
        ])
        
        # Social features
        names.extend([
            'social_sentiment_unity',
            'social_engagement_intensity',
            'social_viral_momentum',
            'social_meme_crystallization',
            'social_hashtag_convergence',
            'social_platform_consensus'
        ])
        
        # Championship psychology features
        names.extend([
            'championship_window_urgency',
            'championship_generational_pressure',
            'championship_expectations_alignment',
            'championship_all_in_indicator',
            'championship_last_dance'
        ])
        
        # Symbolism features
        names.extend([
            'symbolism_david_goliath',
            'symbolism_old_new',
            'symbolism_geographic_cultural',
            'symbolism_redemption',
            'symbolism_destiny'
        ])
        
        return names
