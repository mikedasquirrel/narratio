"""
Statistical Narrative Transformer

**KEY INSIGHT**: Stats ARE narrative in quantitative domains.

Extracts narrativity from numerical records:
- Win percentages → dominance narrative
- Streaks → momentum narrative  
- Differentials → advantage narrative
- Consistency → reliability narrative

This transformer converts NUMBERS into NARRATIVE FEATURES.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class StatisticalNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #37: Extract narrativity from statistical records
    
    Converts win%, streaks, records into narrative features:
    - Dominance (win% >> 0.5)
    - Underdog (win% << 0.5)
    - Momentum (positive/negative streaks)
    - Volatility (consistency of performance)
    - Form (recent vs historical)
    """
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit transformer (stateless for this version)"""
        return self
    
    def transform(self, records: List[dict]) -> np.ndarray:
        """
        Transform statistical records into narrative features
        
        Args:
            records: List of dicts with keys like:
                - 'win_pct': float
                - 'opponent_win_pct': float
                - 'streak': int (positive = winning, negative = losing)
                - 'recent_form': float (last N games win%)
                - etc.
        
        Returns:
            features: Array of narrative features
        """
        features_list = []
        
        for record in records:
            feats = self._extract_from_record(record)
            features_list.append(feats)
        
        return np.array(features_list)
    
    def _extract_from_record(self, record: dict) -> np.ndarray:
        """Extract narrative features from one statistical record"""
        features = []
        
        # === DOMINANCE NARRATIVE ===
        win_pct = record.get('win_pct', 0.5)
        features.append(win_pct)  # Raw win%
        features.append(float(win_pct > 0.7))  # Strong team flag
        features.append(float(win_pct < 0.3))  # Weak team flag
        features.append((win_pct - 0.5) ** 2)  # Distance from mediocrity
        
        # === MOMENTUM NARRATIVE ===
        streak = record.get('streak', 0)
        features.append(streak / 10.0)  # Normalized streak
        features.append(float(streak >= 3))  # Hot streak flag
        features.append(float(streak <= -3))  # Cold streak flag
        features.append(abs(streak) / 10.0)  # Streak magnitude (unsigned)
        
        # === FORM NARRATIVE (recent vs historical) ===
        recent_form = record.get('recent_form', win_pct)
        features.append(recent_form)
        features.append(recent_form - win_pct)  # Form momentum (improving/declining)
        features.append(float(recent_form > win_pct + 0.1))  # Surging
        features.append(float(recent_form < win_pct - 0.1))  # Slumping
        
        # === CONSISTENCY NARRATIVE ===
        variance = record.get('performance_variance', 0.1)
        features.append(variance)  # Higher = more volatile
        features.append(1.0 / (1.0 + variance))  # Consistency score (inverse)
        features.append(float(variance < 0.05))  # Highly consistent flag
        
        # === ADVANTAGE NARRATIVE (if opponent stats available) ===
        opp_win_pct = record.get('opponent_win_pct', 0.5)
        if opp_win_pct is not None:
            features.append(win_pct - opp_win_pct)  # Win% differential (KEY!)
            features.append((win_pct - opp_win_pct) ** 2)  # Mismatch magnitude
            features.append(float(abs(win_pct - opp_win_pct) > 0.3))  # Mismatch flag
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # === UNDERDOG NARRATIVE ===
        features.append(float(win_pct < 0.4 and opp_win_pct > 0.6))  # True underdog
        features.append((0.5 - win_pct) * (opp_win_pct - 0.5))  # Upset potential
        
        # === STAKES NARRATIVE (if contextual) ===
        playoff_race = record.get('playoff_race', False)
        elimination_game = record.get('elimination_game', False)
        
        features.append(float(playoff_race))
        features.append(float(elimination_game))
        features.append(float(playoff_race) * win_pct)  # Good team in race = high stakes
        features.append(float(playoff_race) * (1 - win_pct))  # Bad team in race = desperation
        
        # Store feature names on first call
        if not self.feature_names_:
            self.feature_names_ = [
                'win_pct', 'strong_team', 'weak_team', 'dist_from_mediocre',
                'streak_norm', 'hot_streak', 'cold_streak', 'streak_magnitude',
                'recent_form', 'form_momentum', 'surging', 'slumping',
                'performance_variance', 'consistency', 'highly_consistent',
                'win_pct_diff', 'mismatch_magnitude', 'mismatch_flag',
                'true_underdog', 'upset_potential',
                'playoff_race', 'elimination_game', 'stakes_good_team', 'stakes_bad_team'
            ]
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return self.feature_names_


class MatchupNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #38: Extract narrativity from matchup-specific dynamics
    
    Captures:
    - Style matchups (strength vs strength, strength vs weakness)
    - Historical patterns (how teams typically match up)
    - Scheme advantages (tactical narratives)
    """
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, matchups: List[dict]) -> np.ndarray:
        """
        Transform matchup data into narrative features
        
        Args:
            matchups: List of dicts with:
                - 'home_offensive_strength': float
                - 'away_defensive_strength': float
                - 'home_defensive_strength': float
                - 'away_offensive_strength': float
                - 'style_similarity': float (0-1)
                - etc.
        
        Returns:
            features: Matchup narrative features
        """
        features_list = []
        
        for matchup in matchups:
            feats = self._extract_matchup_narrativity(matchup)
            features_list.append(feats)
        
        return np.array(features_list)
    
    def _extract_matchup_narrativity(self, matchup: dict) -> np.ndarray:
        """Extract narrativity from one matchup"""
        features = []
        
        # === STRENGTH VS STRENGTH ===
        home_off = matchup.get('home_offensive_strength', 0.5)
        away_off = matchup.get('away_offensive_strength', 0.5)
        home_def = matchup.get('home_defensive_strength', 0.5)
        away_def = matchup.get('away_defensive_strength', 0.5)
        
        # Offense vs defense matchups (KEY NARRATIVES!)
        features.append(home_off - away_def)  # Home offense vs away defense
        features.append(away_off - home_def)  # Away offense vs home defense
        features.append((home_off - away_def) - (away_off - home_def))  # Net advantage
        
        # === STYLE MATCHUP ===
        style_sim = matchup.get('style_similarity', 0.5)
        features.append(style_sim)  # Similar styles = more predictable
        features.append(1.0 - style_sim)  # Contrasting styles = upset potential
        
        # === ROCK-PAPER-SCISSORS ===
        # Some teams counter others regardless of records
        counter_advantage = matchup.get('counter_advantage', 0.0)
        features.append(counter_advantage)
        
        # === HISTORICAL MATCHUP ===
        h2h_record = matchup.get('h2h_win_pct', 0.5)
        games_played = matchup.get('h2h_games', 0)
        
        features.append(h2h_record)
        features.append(float(games_played > 0))  # History exists
        features.append(h2h_record * float(games_played >= 3))  # Reliable history
        
        # Store names
        if not self.feature_names_:
            self.feature_names_ = [
                'home_off_vs_away_def', 'away_off_vs_home_def', 'net_matchup_advantage',
                'style_similarity', 'style_contrast',
                'counter_advantage',
                'h2h_record', 'h2h_exists', 'h2h_reliable'
            ]
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_


class RestTravelNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #39: Extract narrativity from rest and travel
    
    Calculates from game dates and team locations:
    - Days since last game (rest advantage)
    - Travel distance (fatigue narrative)
    - Time zone changes (jet lag narrative)
    - Back-to-back games (exhaustion)
    """
    
    def __init__(self):
        self.feature_names_ = ['rest_days_diff', 'travel_burden_diff', 
                              'timezone_changes', 'back_to_back_disadvantage']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, schedule_data: List[dict]) -> np.ndarray:
        """
        Extract rest/travel narrativity
        
        Args:
            schedule_data: List of dicts with:
                - 'home_rest_days': int
                - 'away_rest_days': int
                - 'away_travel_distance': float (miles)
                - 'timezone_diff': int
        """
        features_list = []
        
        for game in schedule_data:
            home_rest = game.get('home_rest_days', 7)
            away_rest = game.get('away_rest_days', 7)
            travel_dist = game.get('away_travel_distance', 1000)
            tz_diff = game.get('timezone_diff', 0)
            
            features = [
                (home_rest - away_rest) / 7.0,  # Rest advantage
                travel_dist / 3000.0,  # Travel burden
                abs(tz_diff) / 3.0,  # Timezone changes
                float(away_rest <= 4)  # Away team on short rest
            ]
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_


# Export
__all__ = [
    'StatisticalNarrativeTransformer',
    'MatchupNarrativeTransformer',
    'RestTravelNarrativeTransformer'
]

