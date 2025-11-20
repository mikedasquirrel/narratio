"""
Game Flow Narrative Transformer (#40)

**CRITICAL INSIGHT**: Sequence matters, not just final outcome!

Extracts narrativity from HOW games unfold:
- Momentum swings
- Comeback narrativity  
- Scoring velocity
- Lead changes
- Clutch performance
- Drama intensity

This is where live betting edges exist!

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin


class GameFlowNarrativeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #40: Extract narrativity from game flow sequences
    
    Analyzes HOW the game unfolded, not just who won.
    """
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, game_flows: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform game flow data into narrative features
        
        Args:
            game_flows: List of dicts with:
                - 'score_sequence': [(0,0), (7,0), (7,7), (14,7), ...]
                - 'time_sequence': [0, 5, 12, 18, ...] (minutes)
                - 'quarter_ends': [0, 15, 30, 45, 60] (cumulative minutes)
        
        Returns:
            features: Flow narrative features
        """
        features_list = []
        
        for flow in game_flows:
            feats = self._extract_flow_narrativity(flow)
            features_list.append(feats)
        
        return np.array(features_list)
    
    def _extract_flow_narrativity(self, flow: Dict[str, Any]) -> np.ndarray:
        """Extract all flow narrative features"""
        features = []
        
        score_seq = flow.get('score_sequence', [(0, 0)])
        time_seq = flow.get('time_sequence', [0])
        
        # Convert to arrays
        home_scores = np.array([s[0] for s in score_seq])
        away_scores = np.array([s[1] for s in score_seq])
        margins = home_scores - away_scores
        times = np.array(time_seq)
        
        # === 1. LEAD CHANGES (Momentum swings) ===
        lead_changes = np.sum(np.diff(np.sign(margins)) != 0)
        features.append(lead_changes / 10.0)  # Normalize
        features.append(float(lead_changes >= 3))  # High volatility game
        features.append(float(lead_changes == 0))  # Wire-to-wire domination
        
        # === 2. COMEBACK NARRATIVITY ===
        max_home_lead = np.max(margins)
        max_away_lead = np.abs(np.min(margins))
        largest_deficit_overcome = 0
        
        # Check if team overcame deficit to win
        final_margin = margins[-1]
        if final_margin > 0:  # Home won
            largest_deficit_overcome = max_away_lead if max_away_lead > 0 else 0
        elif final_margin < 0:  # Away won
            largest_deficit_overcome = max_home_lead if max_home_lead > 0 else 0
        
        features.append(largest_deficit_overcome / 20.0)  # Comeback intensity
        features.append(float(largest_deficit_overcome >= 10))  # Dramatic comeback
        features.append(float(largest_deficit_overcome >= 14))  # Epic comeback (2 TD+)
        
        # === 3. SCORING VELOCITY ===
        total_points = home_scores[-1] + away_scores[-1]
        total_time = times[-1] if len(times) > 0 else 60
        
        scoring_rate = total_points / max(total_time, 1)  # Points per minute
        features.append(scoring_rate / 1.0)  # Normalize (avg ~0.7 pts/min)
        features.append(float(scoring_rate > 0.8))  # High-scoring game
        features.append(float(scoring_rate < 0.5))  # Low-scoring slugfest
        
        # === 4. RUNS/STREAKS (Unanswered scoring) ===
        # Detect runs (one team scoring multiple times)
        scoring_team = np.diff(home_scores + away_scores)  # Who scored each time
        home_scoring = np.diff(home_scores) > 0
        
        max_run = 1
        current_run = 1
        for i in range(1, len(home_scoring)):
            if home_scoring[i] == home_scoring[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        features.append(max_run / 5.0)  # Longest run
        features.append(float(max_run >= 3))  # Significant run
        
        # === 5. QUARTER-BY-QUARTER NARRATIVITY ===
        # Which quarters were high-scoring? (4th quarter drama!)
        if len(score_seq) >= 4:
            # Simple: just track if 4th quarter had scoring
            q4_started = len(score_seq) * 3 // 4
            q4_points = (home_scores[-1] - home_scores[q4_started] + 
                        away_scores[-1] - away_scores[q4_started])
            features.append(q4_points / 20.0)  # 4th quarter scoring
            features.append(float(q4_points >= 14))  # Dramatic finish
        else:
            features.extend([0.0, 0.0])
        
        # === 6. MARGIN STABILITY ===
        margin_std = np.std(margins)
        features.append(margin_std / 15.0)  # Margin volatility
        features.append(float(margin_std < 3))  # Stable margin (blowout)
        features.append(float(margin_std > 10))  # Wild swings
        
        # === 7. FINAL MARGIN (Drama indicator) ===
        final_margin_abs = abs(final_margin)
        features.append(final_margin_abs / 30.0)
        features.append(float(final_margin_abs <= 3))  # One-score game
        features.append(float(final_margin_abs <= 7))  # Within one TD
        features.append(float(final_margin_abs >= 21))  # Blowout (3+ TD)
        
        # === 8. NARRATIVE INTENSITY SCORE ===
        intensity = 0.0
        intensity += (lead_changes / 5.0) * 0.3  # More lead changes = more drama
        intensity += (largest_deficit_overcome / 15.0) * 0.4  # Comeback = high drama
        intensity += float(final_margin_abs <= 3) * 0.3  # Close game = drama
        
        features.append(min(intensity, 2.0))  # Cap at 2.0
        
        # Store feature names
        if not self.feature_names_:
            self.feature_names_ = [
                'lead_changes_norm', 'high_volatility', 'wire_to_wire',
                'comeback_intensity', 'dramatic_comeback', 'epic_comeback',
                'scoring_rate', 'high_scoring_game', 'low_scoring_game',
                'max_run_length', 'significant_run',
                'q4_scoring', 'dramatic_finish',
                'margin_volatility', 'stable_blowout', 'wild_swings',
                'final_margin_norm', 'one_score_game', 'within_one_td', 'blowout',
                'narrative_intensity'
            ]
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_


class LiveBettingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #41: Real-time narrative for live betting
    
    Given current game state, predict:
    - Win probability (dynamic)
    - Expected final score
    - Comeback probability
    - Next score probability
    """
    
    def __init__(self):
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, live_states: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform live game state into predictive features
        
        Args:
            live_states: List of dicts with:
                - 'current_score': (home, away)
                - 'time_remaining': minutes
                - 'quarter': 1-4
                - 'possession': 'home' or 'away'
                - 'field_position': int (yard line)
                - 'down': 1-4
                - 'distance': int (yards to go)
        
        Returns:
            features: Live prediction features
        """
        features_list = []
        
        for state in live_states:
            feats = self._extract_live_features(state)
            features_list.append(feats)
        
        return np.array(features_list)
    
    def _extract_live_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract features from current game state"""
        features = []
        
        current_score = state.get('current_score', (0, 0))
        home_score, away_score = current_score
        margin = home_score - away_score
        
        time_remaining = state.get('time_remaining', 60)
        quarter = state.get('quarter', 1)
        
        # === 1. CURRENT STATE ===
        features.append(home_score / 50.0)  # Normalize
        features.append(away_score / 50.0)
        features.append((margin + 30) / 60.0)  # Margin [-30, +30] → [0, 1]
        features.append(time_remaining / 60.0)
        features.append(quarter / 4.0)
        
        # === 2. GAME SITUATION ===
        features.append(float(abs(margin) <= 3))  # One-score game
        features.append(float(abs(margin) <= 7))  # Within one TD
        features.append(float(time_remaining <= 5))  # Final 5 minutes
        features.append(float(quarter == 4))  # 4th quarter
        
        # === 3. MOMENTUM INDICATORS ===
        last_score = state.get('last_score_by', 'home')
        features.append(float(last_score == 'home'))
        
        possession = state.get('possession', 'home')
        features.append(float(possession == 'home'))
        
        # === 4. SITUATIONAL PRESSURE ===
        # Team behind with ball = comeback potential
        features.append(float(margin < 0 and possession == 'home'))  # Home behind w/ ball
        features.append(float(margin > 0 and possession == 'away'))  # Away behind w/ ball
        
        # === 5. FIELD POSITION (if available) ===
        field_pos = state.get('field_position', 50)
        features.append(field_pos / 100.0)
        features.append(float(field_pos >= 80))  # Red zone
        features.append(float(field_pos <= 20))  # Backed up
        
        # === 6. DOWN/DISTANCE (if available) ===
        down = state.get('down', 1)
        distance = state.get('distance', 10)
        features.append(down / 4.0)
        features.append(distance / 20.0)
        features.append(float(down == 3 and distance > 7))  # 3rd and long (key situation)
        
        # === 7. WIN PROBABILITY INPUTS ===
        # Simple model inputs for live win prob
        time_pct = time_remaining / 60.0
        margin_impact = margin / max(time_remaining / 10, 1)  # Margin matters more late
        features.append(margin_impact / 10.0)
        
        # === 8. COMEBACK DIFFICULTY ===
        possessions_remaining = time_remaining / 5.0  # Rough estimate
        scores_needed = abs(margin) / 7.0  # TDs needed
        comeback_difficulty = scores_needed / max(possessions_remaining, 1)
        features.append(min(comeback_difficulty, 2.0))
        
        if not self.feature_names_:
            self.feature_names_ = [
                'home_score_norm', 'away_score_norm', 'margin_norm', 'time_remaining_norm', 'quarter_norm',
                'one_score_game', 'within_one_td', 'final_5min', 'fourth_quarter',
                'last_score_home', 'possession_home',
                'home_behind_w_ball', 'away_behind_w_ball',
                'field_position', 'red_zone', 'backed_up',
                'down_norm', 'distance_norm', 'third_and_long',
                'margin_impact', 'comeback_difficulty'
            ]
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_


class PropBettingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer #42: Player prop prediction
    
    Predicts specific player outcomes:
    - "Mahomes over 2.5 passing TDs"
    - "Kelce over 75.5 receiving yards"
    - "First TD scorer"
    
    Uses player-specific narrativity (nominative features matter here!)
    """
    
    def __init__(self, prop_type: str = 'player_performance'):
        """
        Args:
            prop_type: 'player_performance', 'team_total', 'first_score', 'anytime_scorer'
        """
        self.prop_type = prop_type
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, prop_contexts: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform prop contexts into features
        
        Args:
            prop_contexts: List of dicts with:
                - 'player_name': str
                - 'player_position': str
                - 'historical_avg': float (avg performance)
                - 'prop_line': float (over/under line)
                - 'opponent_defense_rank': int
                - 'game_script': str ('pass_heavy', 'run_heavy', 'balanced')
                - 'weather': dict
        """
        features_list = []
        
        for context in prop_contexts:
            feats = self._extract_prop_narrativity(context)
            features_list.append(feats)
        
        return np.array(features_list)
    
    def _extract_prop_narrativity(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract prop-specific narrative features"""
        features = []
        
        # === 1. PLAYER BASELINE ===
        historical_avg = context.get('historical_avg', 0)
        prop_line = context.get('prop_line', 0)
        
        features.append(historical_avg / 100.0)  # Normalize to context
        features.append(prop_line / 100.0)
        features.append((historical_avg - prop_line) / 100.0)  # Line vs average
        features.append(float(historical_avg > prop_line * 1.1))  # Line seems low
        features.append(float(historical_avg < prop_line * 0.9))  # Line seems high
        
        # === 2. MATCHUP-SPECIFIC ===
        opp_def_rank = context.get('opponent_defense_rank', 16)
        features.append(opp_def_rank / 32.0)  # Normalize
        features.append(float(opp_def_rank >= 25))  # Weak defense
        features.append(float(opp_def_rank <= 10))  # Strong defense
        
        # Matchup advantage
        features.append((16 - opp_def_rank) / 32.0)  # Better if facing bad defense
        
        # === 3. GAME SCRIPT ===
        game_script = context.get('game_script', 'balanced')
        position = context.get('player_position', 'unknown')
        
        # QB benefits from pass-heavy
        features.append(float(position == 'QB' and game_script == 'pass_heavy'))
        # RB benefits from run-heavy
        features.append(float(position == 'RB' and game_script == 'run_heavy'))
        # WR benefits from pass-heavy
        features.append(float(position == 'WR' and game_script == 'pass_heavy'))
        
        # === 4. SITUATIONAL ===
        is_home = context.get('is_home', True)
        features.append(float(is_home))  # Home players perform better
        
        primetime = context.get('primetime', False)
        features.append(float(primetime))  # Star players shine in primetime
        
        division_game = context.get('division_game', False)
        features.append(float(division_game))  # Familiarity affects performance
        
        # === 5. NOMINATIVE MOMENTUM ===
        recent_games = context.get('last_3_games_avg', historical_avg)
        features.append(recent_games / 100.0)
        features.append(recent_games - historical_avg)  # Hot hand
        features.append(float(recent_games > historical_avg * 1.2))  # On fire
        
        if not self.feature_names_:
            self.feature_names_ = [
                'historical_avg', 'prop_line', 'line_vs_avg', 'line_seems_low', 'line_seems_high',
                'opp_def_rank', 'weak_defense', 'strong_defense', 'matchup_advantage',
                'qb_pass_heavy', 'rb_run_heavy', 'wr_pass_heavy',
                'is_home', 'primetime', 'division_game',
                'recent_avg', 'hot_hand', 'on_fire'
            ]
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_


__all__ = [
    'GameFlowNarrativeTransformer',
    'LiveBettingTransformer',
    'PropBettingTransformer'
]

