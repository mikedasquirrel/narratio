"""
Narrative Completion Pressure Transformer

The meta-feature that measures how badly the universe "needs" a particular
outcome to maintain story coherence and satisfy narrative expectations.

This transformer detects when stories demand resolution and quantifies
the pressure for specific outcomes based on narrative logic.

Author: Narrative Enhancement System
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import math


class NarrativeCompletionPressureTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative completion pressure meta-features.
    
    Philosophy:
    - Some outcomes are narratively "required" for coherence
    - Symmetry and patterns create expectation
    - Character arcs must resolve appropriately  
    - Collective consciousness converges on certain outcomes
    - Historical echoes demand repetition or breaking
    
    Features (45 total):
    - Symmetry requirements (8)
    - Character arc completion needs (8)
    - Franchise destiny timing (6)
    - Collective expectation crystallization (8)
    - Historical pattern matching (8)
    - Narrative coherence pressure (7)
    """
    
    def __init__(
        self,
        symmetry_weight: float = 0.8,
        include_media_analysis: bool = True,
        historical_depth: int = 50  # years to look back
    ):
        """
        Initialize narrative completion pressure analyzer.
        
        Parameters
        ----------
        symmetry_weight : float
            How much to weight symmetrical patterns
        include_media_analysis : bool
            Include media narrative analysis
        historical_depth : int
            Years of history to consider for patterns
        """
        self.symmetry_weight = symmetry_weight
        self.include_media_analysis = include_media_analysis
        self.historical_depth = historical_depth
        
        # Key narrative completion patterns
        self.completion_patterns = {
            'redemption': {
                'setup_time': 365,  # days
                'peak_pressure': 0.9,
                'decay_rate': 0.1
            },
            'dynasty_end': {
                'typical_duration': 5 * 365,
                'variance': 2 * 365,
                'pressure_curve': 'sigmoid'
            },
            'drought_breaking': {
                'pressure_per_year': 0.02,
                'acceleration_after': 25,
                'max_pressure': 1.0
            },
            'revenge': {
                'immediate_window': 7,
                'season_window': 180,
                'playoff_multiplier': 2.0
            }
        }
        
        # Symmetry patterns to detect
        self.symmetry_types = [
            'anniversary',      # 10, 25, 50 year marks
            'numerical',        # Same date, score, etc
            'mirror',           # Reverse of previous outcome
            'cyclical',         # Repeating patterns
            'karmic'           # What goes around comes around
        ]
        
    def fit(self, X, y=None):
        """
        Learn narrative pressure baselines from training data.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Historical sports data with outcomes
        y : ignored
        
        Returns
        -------
        self
        """
        # Convert to DataFrame if needed
        if isinstance(X, list):
            self.historical_data_ = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            self.historical_data_ = X.copy()
        else:
            self.historical_data_ = pd.DataFrame()
            
        # Learn typical pressure patterns
        self._learn_pressure_patterns()
        
        # Build historical pattern database
        self._build_pattern_database()
        
        return self
        
    def transform(self, X):
        """
        Extract narrative completion pressure features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Current game/matchup data
            
        Returns
        -------
        np.ndarray
            Pressure features (n_samples, 45)
        """
        # Handle single sample
        if isinstance(X, dict):
            X = [X]
            
        features = []
        for item in X:
            feature_vec = []
            
            # Symmetry requirements (8)
            symmetry_features = self._extract_symmetry_requirements(item)
            feature_vec.extend(symmetry_features)
            
            # Character arc completion (8)
            character_features = self._extract_character_arc_needs(item)
            feature_vec.extend(character_features)
            
            # Franchise destiny timing (6)
            destiny_features = self._extract_franchise_destiny(item)
            feature_vec.extend(destiny_features)
            
            # Collective expectation (8)
            expectation_features = self._extract_collective_expectation(item)
            feature_vec.extend(expectation_features)
            
            # Historical pattern matching (8)
            historical_features = self._extract_historical_patterns(item)
            feature_vec.extend(historical_features)
            
            # Narrative coherence pressure (7)
            coherence_features = self._extract_coherence_pressure(item)
            feature_vec.extend(coherence_features)
            
            features.append(feature_vec)
            
        return np.array(features, dtype=np.float32)
        
    def _learn_pressure_patterns(self):
        """Learn typical narrative pressure patterns from historical data."""
        self.pressure_baselines_ = {}
        
        if self.historical_data_.empty:
            # Set defaults
            self.pressure_baselines_ = {
                'avg_upset_pressure': 0.3,
                'playoff_pressure_multiplier': 1.5,
                'rivalry_pressure_base': 0.4,
                'milestone_pressure': 0.6
            }
            return
            
        # Calculate actual patterns from data
        # This would analyze when narrative pressure led to expected outcomes
        
    def _build_pattern_database(self):
        """Build database of historical patterns for matching."""
        self.pattern_database_ = {
            'anniversaries': {},
            'score_patterns': {},
            'series_patterns': {},
            'seasonal_cycles': {}
        }
        
        if self.historical_data_.empty:
            return
            
        # Extract significant historical patterns
        # This would identify recurring patterns in the data
        
    def _extract_symmetry_requirements(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract symmetry-based narrative requirements.
        
        Returns 8 features measuring symmetrical pattern pressure.
        """
        features = []
        
        current_date = item.get('game_date', datetime.now())
        if isinstance(current_date, str):
            try:
                current_date = pd.to_datetime(current_date)
            except:
                current_date = datetime.now()
                
        # Anniversary pressure
        team_founded = item.get('team_founded_year', 1970)
        years_old = current_date.year - team_founded
        
        anniversary_pressure = 0.0
        if years_old % 50 == 0:
            anniversary_pressure = 1.0
        elif years_old % 25 == 0:
            anniversary_pressure = 0.8
        elif years_old % 10 == 0:
            anniversary_pressure = 0.5
            
        features.append(anniversary_pressure)
        
        # Date symmetry (playing on same date as historic event)
        historic_date_match = item.get('matches_historic_date', False)
        if historic_date_match:
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Score symmetry potential
        previous_score = item.get('last_meeting_score', [0, 0])
        likely_scores = item.get('predicted_score_range', [[2, 4], [3, 5]])
        
        score_symmetry = 0.0
        for score in likely_scores:
            if score == previous_score or score == previous_score[::-1]:
                score_symmetry = 0.8
                break
                
        features.append(score_symmetry)
        
        # Series symmetry (2-2 ties create maximum pressure)
        series_score = item.get('series_score', [0, 0])
        if series_score == [2, 2]:
            features.append(1.0)  # Maximum symmetry pressure
        elif series_score == [3, 3]:
            features.append(0.9)  # Game 7 pressure
        elif series_score[0] == series_score[1]:
            features.append(0.6)  # Tied series
        else:
            features.append(0.0)
            
        # Mirror outcome pressure (reverse previous result)
        needs_reversal = item.get('narrative_needs_reversal', False)
        if needs_reversal:
            last_result = item.get('last_meeting_winner', 'home')
            if item.get('home_team') == last_result:
                features.append(1.0)  # Away team "should" win
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Cyclical pattern pressure
        pattern_position = self._detect_cyclical_position(item)
        features.append(pattern_position)
        
        # Numerical destiny (jersey numbers, dates aligning)
        numerical_alignment = item.get('numerical_alignment_score', 0.0)
        features.append(numerical_alignment)
        
        # Karmic balance pressure
        karma_imbalance = item.get('karma_imbalance_score', 0.0)
        features.append(np.tanh(karma_imbalance))  # -1 to 1
        
        return features
        
    def _extract_character_arc_needs(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract character arc completion pressure.
        
        Returns 8 features measuring story arc requirements.
        """
        features = []
        
        # Player retirement arc
        retiring_player = item.get('key_player_final_season', False)
        games_remaining = item.get('retiring_player_games_left', 82)
        
        if retiring_player:
            # Pressure increases as games decrease
            retirement_pressure = 1.0 - (games_remaining / 82.0)
            if item.get('is_playoffs'):
                retirement_pressure = min(1.0, retirement_pressure * 1.5)
            features.append(retirement_pressure)
        else:
            features.append(0.0)
            
        # Coach vindication arc
        coach_narrative = item.get('coach_narrative_type', None)
        if coach_narrative == 'fired_returning':
            features.append(0.9)  # High vindication pressure
        elif coach_narrative == 'first_year_proving':
            features.append(0.6)
        elif coach_narrative == 'legacy_building':
            features.append(0.7)
        else:
            features.append(0.0)
            
        # Rookie sensation arc
        rookie_story = item.get('rookie_storyline', None)
        if rookie_story == 'hometown_hero':
            features.append(0.8)
        elif rookie_story == 'draft_bust_redemption':
            features.append(0.7)
        elif rookie_story == 'unexpected_emergence':
            features.append(0.6)
        else:
            features.append(0.0)
            
        # Veteran last chance
        veteran_window = item.get('veteran_championship_window', 'open')
        if veteran_window == 'closing':
            features.append(0.8)
        elif veteran_window == 'final':
            features.append(1.0)
        else:
            features.append(0.0)
            
        # Team identity arc
        identity_stage = item.get('team_identity_stage', 'established')
        if identity_stage == 'forming':
            features.append(0.6)  # Need defining moment
        elif identity_stage == 'proving':
            features.append(0.8)  # Need validation
        elif identity_stage == 'cementing':
            features.append(0.7)  # Need confirmation
        else:
            features.append(0.3)
            
        # Redemption arc progress
        redemption_progress = item.get('redemption_arc_progress', 0.0)
        if redemption_progress > 0.7:
            features.append(0.9)  # Near completion, high pressure
        else:
            features.append(redemption_progress)
            
        # Dynasty arc stage  
        dynasty_stage = item.get('dynasty_narrative_stage', None)
        if dynasty_stage == 'coronation':
            features.append(0.9)  # Must win to confirm
        elif dynasty_stage == 'defense':
            features.append(0.7)  # Pressure to maintain
        elif dynasty_stage == 'last_stand':
            features.append(0.8)  # Final chance
        else:
            features.append(0.0)
            
        # Cinderella completion
        cinderella_stage = item.get('cinderella_run_stage', 0)
        if cinderella_stage > 0:
            # Pressure increases with each round
            features.append(min(1.0, cinderella_stage * 0.25))
        else:
            features.append(0.0)
            
        return features
        
    def _extract_franchise_destiny(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract franchise destiny timing features.
        
        Returns 6 features measuring franchise-level narrative timing.
        """
        features = []
        
        # Championship window alignment
        window_status = item.get('championship_window', 'closed')
        window_years_remaining = item.get('window_years_left', 0)
        
        if window_status == 'peak':
            features.append(0.9)  # Maximum pressure
        elif window_status == 'open':
            # Pressure increases as window closes
            features.append(max(0.3, 1.0 - window_years_remaining / 5.0))
        elif window_status == 'closing':
            features.append(0.8)
        else:
            features.append(0.0)
            
        # Drought pressure
        championship_drought = item.get('years_since_championship', 0)
        playoff_drought = item.get('years_since_playoffs', 0)
        
        # Championship drought pressure (accelerating)
        if championship_drought > 50:
            features.append(1.0)
        elif championship_drought > 25:
            features.append(0.8 + (championship_drought - 25) * 0.008)
        else:
            features.append(championship_drought * 0.02)
            
        # Playoff drought pressure
        features.append(min(1.0, playoff_drought * 0.1))
        
        # Generational timing
        last_generation_peak = item.get('years_since_last_peak', 0)
        if 15 <= last_generation_peak <= 20:
            features.append(0.8)  # New generation due
        elif 20 < last_generation_peak:
            features.append(0.9)  # Overdue
        else:
            features.append(0.3)
            
        # Market destiny pressure
        market_size = item.get('market_size', 'medium')
        market_years_waiting = item.get('market_championship_drought', 0)
        
        if market_size == 'large' and market_years_waiting > 20:
            features.append(0.9)  # Big market needs success
        elif market_size == 'small' and item.get('cinderella_potential', False):
            features.append(0.8)  # Small market magic
        else:
            features.append(0.4)
            
        # Franchise milestone pressure
        milestone_proximity = item.get('franchise_milestone_proximity', None)
        if milestone_proximity == '1000th_win':
            features.append(0.9)
        elif milestone_proximity == '500th_win':
            features.append(0.7)
        elif milestone_proximity:
            features.append(0.6)
        else:
            features.append(0.0)
            
        return features
        
    def _extract_collective_expectation(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract collective expectation crystallization features.
        
        Returns 8 features measuring consensus narrative pressure.
        """
        features = []
        
        # Media consensus score
        media_prediction_variance = item.get('media_prediction_variance', 1.0)
        media_consensus = 1.0 - min(1.0, media_prediction_variance)
        features.append(media_consensus)
        
        # Fan expectation unity
        fan_confidence = item.get('fan_confidence_score', 0.5)
        fan_unity = item.get('fanbase_unity_score', 0.5)
        
        features.append(fan_confidence * fan_unity)
        
        # Betting market crystallization
        odds_movement = item.get('odds_movement_stability', 0.0)
        betting_consensus = item.get('betting_consensus_score', 0.0)
        
        features.append((odds_movement + betting_consensus) / 2.0)
        
        # Social media convergence
        if self.include_media_analysis:
            sentiment_variance = item.get('social_sentiment_variance', 1.0)
            viral_narrative = item.get('viral_narrative_strength', 0.0)
            
            social_convergence = (1.0 - sentiment_variance) * (1 + viral_narrative)
            features.append(min(1.0, social_convergence))
        else:
            features.append(0.0)
            
        # Expert alignment
        expert_agreement = item.get('expert_pick_agreement', 0.5)
        expert_confidence = item.get('expert_confidence_avg', 0.5)
        
        features.append(expert_agreement * expert_confidence)
        
        # Narrative inevitability score
        inevitability_language = item.get('inevitability_language_score', 0.0)
        features.append(inevitability_language)
        
        # Collective memory activation
        # (when everyone remembers similar situation)
        memory_activation = item.get('collective_memory_score', 0.0)
        features.append(memory_activation)
        
        # Destiny language proliferation
        destiny_mentions = item.get('destiny_related_mentions', 0)
        features.append(min(1.0, destiny_mentions / 100.0))
        
        return features
        
    def _extract_historical_patterns(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract historical pattern matching pressure.
        
        Returns 8 features measuring historical echo requirements.
        """
        features = []
        
        current_year = item.get('year', datetime.now().year)
        
        # 10-year pattern matching
        pattern_10 = self._check_historical_pattern(item, 10)
        features.append(pattern_10)
        
        # 20-year pattern matching  
        pattern_20 = self._check_historical_pattern(item, 20)
        features.append(pattern_20)
        
        # 50-year pattern matching
        pattern_50 = self._check_historical_pattern(item, 50)
        features.append(pattern_50 * 1.2)  # Boost for round numbers
        
        # Exact date repetition
        historical_date_events = item.get('events_on_this_date', [])
        if len(historical_date_events) > 0:
            # Check if pattern suggests specific outcome
            date_pattern_strength = self._analyze_date_pattern(historical_date_events)
            features.append(date_pattern_strength)
        else:
            features.append(0.0)
            
        # Generational echo (25-30 years)
        gen_echo = self._check_generational_echo(item)
        features.append(gen_echo)
        
        # Reverse pattern pressure (time to break streak)
        streak_info = item.get('historical_streak', {})
        if streak_info:
            streak_length = streak_info.get('length', 0)
            streak_type = streak_info.get('type', 'neutral')
            
            if streak_length > 10 and streak_type == 'negative':
                # Long negative streak creates pressure to break
                features.append(min(1.0, streak_length / 20.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
            
        # Historical rivalry pattern
        rivalry_history = item.get('rivalry_historical_pattern', None)
        if rivalry_history == 'alternating':
            # If pattern is alternating wins, pressure to continue
            features.append(0.8)
        elif rivalry_history == 'dominance_shift_due':
            features.append(0.9)
        else:
            features.append(0.0)
            
        # Century mark pressure (100 years)
        team_founded = item.get('team_founded_year', 1970)
        if current_year % 100 == team_founded % 100:
            features.append(1.0)  # Century anniversary
        else:
            features.append(0.0)
            
        return features
        
    def _extract_coherence_pressure(self, item: Dict[str, Any]) -> List[float]:
        """
        Extract narrative coherence pressure features.
        
        Returns 7 features measuring story coherence requirements.
        """
        features = []
        
        # Story arc coherence
        current_narrative = item.get('dominant_narrative', 'neutral')
        narrative_progress = item.get('narrative_progress', 0.5)
        expected_outcome = item.get('coherent_outcome', 'either')
        
        if expected_outcome != 'either':
            # Strong coherence requirement
            if narrative_progress > 0.8:
                features.append(0.9)  # Must complete arc
            else:
                features.append(0.6)
        else:
            features.append(0.0)
            
        # Thematic consistency pressure
        season_theme = item.get('season_theme', None)
        if season_theme in ['resilience', 'dominance', 'chaos']:
            theme_consistency = item.get('theme_consistency_score', 0.5)
            features.append(theme_consistency)
        else:
            features.append(0.0)
            
        # Poetic justice requirement
        injustice_score = item.get('unresolved_injustice', 0.0)
        if injustice_score > 0:
            features.append(min(1.0, injustice_score * 1.5))
        else:
            features.append(0.0)
            
        # Narrative contradiction avoidance
        would_contradict = item.get('outcome_would_contradict_narrative', False)
        if would_contradict:
            features.append(-1.0)  # Negative pressure
        else:
            features.append(0.0)
            
        # Story momentum alignment
        momentum_direction = item.get('narrative_momentum_direction', 'neutral')
        if momentum_direction == 'strong_positive':
            features.append(0.8)
        elif momentum_direction == 'strong_negative':
            features.append(-0.8)
        else:
            features.append(0.0)
            
        # Character consistency requirement
        character_break = item.get('would_break_character', False)
        if character_break:
            features.append(-0.9)  # Strong pressure against
        else:
            features.append(0.0)
            
        # Meta-narrative completion
        # (larger story beyond single game)
        meta_narrative_stage = item.get('meta_narrative_stage', 0.0)
        if meta_narrative_stage > 0.9:
            features.append(1.0)  # Climax requires resolution
        else:
            features.append(meta_narrative_stage)
            
        return features
        
    def _detect_cyclical_position(self, item: Dict[str, Any]) -> float:
        """Detect position in cyclical patterns."""
        # Check various cycles
        cycles_detected = []
        
        # Weekly cycle (certain teams better on certain days)
        day_of_week = item.get('day_of_week', 'Wednesday')
        team_day_performance = item.get('team_day_stats', {})
        if day_of_week in team_day_performance:
            if team_day_performance[day_of_week] > 0.6:
                cycles_detected.append(0.7)
                
        # Monthly cycle
        day_of_month = item.get('day_of_month', 15)
        if day_of_month in [1, 15, 30, 31]:
            cycles_detected.append(0.5)  # Key dates
            
        # Seasonal cycle (hot/cold months)
        month = item.get('month', 6)
        team_monthly_pattern = item.get('team_monthly_performance', {})
        if month in team_monthly_pattern:
            cycles_detected.append(team_monthly_pattern[month])
            
        return max(cycles_detected) if cycles_detected else 0.0
        
    def _check_historical_pattern(self, item: Dict[str, Any], years_back: int) -> float:
        """Check for historical pattern at specific year interval."""
        current_year = item.get('year', datetime.now().year)
        target_year = current_year - years_back
        
        historical_event = item.get(f'event_in_{target_year}', None)
        if not historical_event:
            return 0.0
            
        # Check similarity
        similarity_factors = 0.0
        
        # Same teams
        if historical_event.get('teams') == item.get('teams'):
            similarity_factors += 0.3
            
        # Same round/situation  
        if historical_event.get('round') == item.get('round'):
            similarity_factors += 0.3
            
        # Similar stakes
        if historical_event.get('stakes') == item.get('stakes'):
            similarity_factors += 0.2
            
        # Pattern suggests repetition
        if historical_event.get('outcome_pattern') == 'repeat':
            similarity_factors += 0.2
            
        return min(1.0, similarity_factors)
        
    def _analyze_date_pattern(self, events: List[Dict]) -> float:
        """Analyze pattern in historical events on same date."""
        if not events:
            return 0.0
            
        # Look for consistent patterns
        outcomes = [e.get('outcome') for e in events]
        
        # If same outcome repeatedly
        if len(set(outcomes)) == 1:
            return 0.9
            
        # If clear pattern (alternating, etc)
        if len(outcomes) > 4:
            # Check for alternating pattern
            alternating = all(outcomes[i] != outcomes[i+1] 
                            for i in range(len(outcomes)-1))
            if alternating:
                return 0.8
                
        return 0.3  # Some historical significance
        
    def _check_generational_echo(self, item: Dict[str, Any]) -> float:
        """Check for generational echo patterns (25-30 years)."""
        echo_events = []
        
        for years in [25, 26, 27, 28, 29, 30]:
            event = item.get(f'event_{years}_years_ago', None)
            if event and event.get('significance', 0) > 0.7:
                echo_events.append(event)
                
        if not echo_events:
            return 0.0
            
        # Find most similar event
        max_similarity = 0.0
        for event in echo_events:
            similarity = self._calculate_event_similarity(item, event)
            max_similarity = max(max_similarity, similarity)
            
        # Generational echoes are powerful
        return min(1.0, max_similarity * 1.3)
        
    def _calculate_event_similarity(self, current: Dict, historical: Dict) -> float:
        """Calculate similarity between current situation and historical event."""
        similarity = 0.0
        
        # Team involvement
        if set(current.get('teams', [])) & set(historical.get('teams', [])):
            similarity += 0.3
            
        # Stakes similarity
        if current.get('stakes') == historical.get('stakes'):
            similarity += 0.3
            
        # Narrative similarity
        current_narrative = current.get('narrative_type')
        historical_narrative = historical.get('narrative_type')
        if current_narrative == historical_narrative:
            similarity += 0.4
            
        return similarity
        
    def get_feature_names(self) -> List[str]:
        """Return feature names for interpretability."""
        names = []
        
        # Symmetry features
        names.extend([
            'symmetry_anniversary_pressure',
            'symmetry_date_match',
            'symmetry_score_potential',
            'symmetry_series_balance',
            'symmetry_mirror_outcome',
            'symmetry_cyclical_position',
            'symmetry_numerical_destiny',
            'symmetry_karmic_balance'
        ])
        
        # Character arc features
        names.extend([
            'arc_player_retirement',
            'arc_coach_vindication',
            'arc_rookie_emergence',
            'arc_veteran_last_chance',
            'arc_team_identity',
            'arc_redemption_progress',
            'arc_dynasty_stage',
            'arc_cinderella_completion'
        ])
        
        # Franchise destiny features
        names.extend([
            'destiny_championship_window',
            'destiny_championship_drought',
            'destiny_playoff_drought',
            'destiny_generational_timing',
            'destiny_market_pressure',
            'destiny_milestone_proximity'
        ])
        
        # Collective expectation features
        names.extend([
            'expectation_media_consensus',
            'expectation_fan_unity',
            'expectation_betting_crystallization',
            'expectation_social_convergence',
            'expectation_expert_alignment',
            'expectation_inevitability_score',
            'expectation_collective_memory',
            'expectation_destiny_language'
        ])
        
        # Historical pattern features
        names.extend([
            'historical_10_year_pattern',
            'historical_20_year_pattern',
            'historical_50_year_pattern',
            'historical_date_repetition',
            'historical_generational_echo',
            'historical_streak_breaking',
            'historical_rivalry_pattern',
            'historical_century_mark'
        ])
        
        # Coherence pressure features
        names.extend([
            'coherence_arc_completion',
            'coherence_thematic_consistency',
            'coherence_poetic_justice',
            'coherence_contradiction_avoidance',
            'coherence_momentum_alignment',
            'coherence_character_consistency',
            'coherence_meta_narrative'
        ])
        
        return names
