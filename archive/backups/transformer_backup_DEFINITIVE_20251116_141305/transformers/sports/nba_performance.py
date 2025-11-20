"""
NBA Performance Transformer

Extracts NARRATIVITY FROM NBA STATISTICS.
Stats as story: scoring power, efficiency, playmaking, defensive impact.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin


class NBAPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from NBA performance statistics.
    
    Features: 35 total
    - Scoring narrative (8)
    - Playmaking narrative (6)
    - Defense & rebounding (6)
    - Efficiency & impact (7)
    - Context & momentum (8)
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.stat_means_ = {}
        self.stat_stds_ = {}
    
    def fit(self, X, y=None):
        """Learn normalization parameters"""
        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            return self
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.stat_means_[col] = df[col].mean()
            self.stat_stds_[col] = df[col].std() if df[col].std() > 0 else 1.0
        
        return self
    
    def transform(self, X):
        """Transform NBA stats to narrative features"""
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
        
        features = []
        for stats in X:
            feat = self._extract_nba_features(stats)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_nba_features(self, stats: Dict) -> List[float]:
        """Extract all NBA performance narrative features"""
        features = []
        
        # === SCORING NARRATIVE (8 features) ===
        
        # 1. Scoring power (PPG)
        ppg = stats.get('points_per_game', stats.get('ppg', 110.0))
        features.append(self._normalize_stat('ppg', ppg))
        
        # 2. Shooting efficiency (True Shooting %)
        ts_pct = stats.get('true_shooting_pct', stats.get('ts_pct', 0.55))
        features.append(ts_pct)
        
        # 3. Shot quality (Effective FG%)
        efg = stats.get('effective_fg_pct', stats.get('efg', 0.52))
        features.append(efg)
        
        # 4. Pressure performance (FT%)
        ft_pct = stats.get('free_throw_pct', stats.get('ft_pct', 0.75))
        features.append(ft_pct)
        
        # 5. Modern game (3P%)
        three_pt = stats.get('three_point_pct', stats.get('3p_pct', 0.35))
        features.append(three_pt)
        
        # 6. Inside game (points in paint per game)
        paint_ppg = stats.get('points_in_paint', 45.0)
        features.append(paint_ppg / 60.0)  # Normalize around max ~60
        
        # 7. Tempo narrative (fast break points)
        fastbreak = stats.get('fastbreak_points', 12.0)
        features.append(fastbreak / 20.0)
        
        # 8. Clutch scoring (last 5 min close games)
        clutch = stats.get('clutch_scoring', 0.5)
        features.append(clutch)
        
        # === PLAYMAKING NARRATIVE (6 features) ===
        
        # 9. Facilitator narrative (assists per game)
        apg = stats.get('assists_per_game', stats.get('apg', 24.0))
        features.append(self._normalize_stat('apg', apg))
        
        # 10. Decision-making (AST/TO ratio)
        ast_to = stats.get('ast_to_ratio', 1.5)
        features.append(min(1.0, ast_to / 2.5))
        
        # 11. Offensive load (usage rate)
        usage = stats.get('usage_rate', 0.20)
        features.append(usage)
        
        # 12. Ball security (turnover %)
        to_pct = stats.get('turnover_pct', 0.13)
        features.append(1.0 - to_pct)  # Invert
        
        # 13. Creation ability (potential assists)
        pot_ast = stats.get('potential_assists', 30.0)
        features.append(pot_ast / 40.0)
        
        # 14. Team play (secondary assists)
        sec_ast = stats.get('secondary_assists', 2.0)
        features.append(sec_ast / 5.0)
        
        # === DEFENSE & REBOUNDING (6 features) ===
        
        # 15. Defensive impact (defensive rating)
        def_rating = stats.get('defensive_rating', stats.get('def_rtg', 110.0))
        features.append(1.0 - self._normalize_stat('def_rating', def_rating))  # Lower is better
        
        # 16. Disruption (steals per game)
        spg = stats.get('steals_per_game', stats.get('spg', 7.5))
        features.append(self._normalize_stat('spg', spg))
        
        # 17. Rim protection (blocks per game)
        bpg = stats.get('blocks_per_game', stats.get('bpg', 5.0))
        features.append(self._normalize_stat('bpg', bpg))
        
        # 18. Board dominance (rebounds per game)
        rpg = stats.get('rebounds_per_game', stats.get('rpg', 44.0))
        features.append(self._normalize_stat('rpg', rpg))
        
        # 19. Defensive win shares
        def_ws = stats.get('defensive_win_shares', 0.0)
        features.append(min(1.0, def_ws / 5.0))
        
        # 20. Opponent FG% at rim
        opp_fg_rim = stats.get('opp_fg_pct_at_rim', 0.60)
        features.append(1.0 - opp_fg_rim)
        
        # === EFFICIENCY & IMPACT (7 features) ===
        
        # 21. Player Efficiency Rating
        per = stats.get('player_efficiency_rating', stats.get('per', 15.0))
        features.append(self._normalize_stat('per', per))
        
        # 22. Box Plus/Minus
        bpm = stats.get('box_plus_minus', stats.get('bpm', 0.0))
        features.append(self._normalize_stat('bpm', bpm))
        
        # 23. Value Over Replacement (VORP)
        vorp = stats.get('value_over_replacement', stats.get('vorp', 0.0))
        features.append(min(1.0, (vorp + 2) / 6))  # Scale to 0-1
        
        # 24. Win Shares
        win_shares = stats.get('win_shares', 0.0)
        features.append(min(1.0, win_shares / 10.0))
        
        # 25. Net rating (team impact)
        net_rating = stats.get('net_rating', 0.0)
        features.append(self._normalize_stat('net_rating', net_rating))
        
        # 26. Offensive rating
        off_rating = stats.get('offensive_rating', stats.get('off_rtg', 110.0))
        features.append(self._normalize_stat('off_rtg', off_rating))
        
        # 27. Plus/minus
        plus_minus = stats.get('plus_minus', 0.0)
        features.append(self._normalize_stat('plus_minus', plus_minus))
        
        # === CONTEXT & MOMENTUM (8 features) ===
        
        # 28. Home court advantage
        home_split = stats.get('home_win_pct', 0.6)
        away_split = stats.get('away_win_pct', 0.4)
        home_advantage = home_split - away_split
        features.append(home_advantage)
        
        # 29. Conference dominance
        conf_record = stats.get('conference_record', 0.5)
        features.append(conf_record)
        
        # 30. vs Elite teams
        vs_top10 = stats.get('record_vs_top10', 0.4)
        features.append(vs_top10)
        
        # 31. Fatigue factor (back-to-back performance)
        b2b_perf = stats.get('back_to_back_performance', 0.45)
        features.append(b2b_perf)
        
        # 32. Rest advantage
        rest_diff = stats.get('rest_differential', 0.0)
        features.append(np.clip(rest_diff / 2.0, -1, 1))
        
        # 33. Recent form (last 10 games)
        recent_10 = stats.get('last_10_win_pct', 0.5)
        features.append(recent_10)
        
        # 34. Momentum (win streak)
        streak = stats.get('win_streak', 0)
        features.append(min(1.0, streak / 5.0))
        
        # 35. Altitude performance (Denver effect)
        altitude = stats.get('altitude_performance', 0.5)
        features.append(altitude)
        
        return features
    
    def _normalize_stat(self, stat_name: str, value: float) -> float:
        """Normalize stat to z-score"""
        if not self.normalize or stat_name not in self.stat_means_:
            return value / 150.0 if value > 10 else value  # Simple scaling
        
        mean = self.stat_means_[stat_name]
        std = self.stat_stds_[stat_name]
        
        z_score = (value - mean) / std
        return np.clip((z_score + 3) / 6, 0, 1)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            # Scoring
            'nba_ppg', 'nba_ts_pct', 'nba_efg_pct', 'nba_ft_pct',
            'nba_three_pt_pct', 'nba_paint_points', 'nba_fastbreak_points', 'nba_clutch_scoring',
            
            # Playmaking
            'nba_apg', 'nba_ast_to_ratio', 'nba_usage_rate', 'nba_turnover_pct',
            'nba_potential_assists', 'nba_secondary_assists',
            
            # Defense & Rebounding
            'nba_def_rating', 'nba_steals', 'nba_blocks', 'nba_rebounds',
            'nba_def_win_shares', 'nba_opp_fg_at_rim',
            
            # Efficiency
            'nba_per', 'nba_bpm', 'nba_vorp', 'nba_win_shares',
            'nba_net_rating', 'nba_off_rating', 'nba_plus_minus',
            
            # Context
            'nba_home_advantage', 'nba_conference_record', 'nba_vs_top10',
            'nba_b2b_performance', 'nba_rest_advantage', 'nba_recent_form',
            'nba_win_streak', 'nba_altitude_performance'
        ])

