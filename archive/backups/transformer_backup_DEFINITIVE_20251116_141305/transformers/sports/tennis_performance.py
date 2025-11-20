"""
Tennis Performance Transformer

Extracts NARRATIVITY FROM TENNIS STATISTICS.
Stats as story: surface mastery, momentum, mental strength, rivalry context.

Critical insight: Numbers tell stories in tennis. A 85% clay court win rate
isn't just a stat - it's surface dominance (Nadal narrative). A 7-match 
win streak isn't just wins - it's momentum. This transformer translates
tennis numbers into narrative features.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict


class TennisPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from tennis performance statistics.
    
    Philosophy:
    - Stats ARE narrative (not separate from it)
    - Performance patterns reveal psychological states
    - Surface context transforms meaning (clay vs grass vs hard)
    - Recent form reveals momentum and confidence
    - H2H history reveals rivalry and psychological edge
    
    Accepts:
    - Dict with stat keys (player history, match context)
    - DataFrame with stat columns
    - JSON-like nested structures
    
    Features: 50 total
    - Surface specialization narrative (12)
    - Form & momentum narrative (12)
    - Tournament context narrative (8)
    - Physical & fatigue narrative (6)
    - Ranking & competitive position (6)
    - Head-to-head & rivalry narrative (6)
    """
    
    def __init__(
        self,
        normalize: bool = True,
        compute_differentials: bool = True,
        include_h2h: bool = True,
        lookback_matches: int = 20
    ):
        """
        Initialize tennis performance analyzer.
        
        Parameters
        ----------
        normalize : bool
            Normalize stats to z-scores (relative to tour)
        compute_differentials : bool
            Compute surface differentials, form changes
        include_h2h : bool
            Include head-to-head rivalry features
        lookback_matches : int
            Number of recent matches to consider for form
        """
        self.normalize = normalize
        self.compute_differentials = compute_differentials
        self.include_h2h = include_h2h
        self.lookback_matches = lookback_matches
        
        # Stats we'll track for normalization
        self.stat_means_ = {}
        self.stat_stds_ = {}
    
    def fit(self, X, y=None):
        """
        Learn normalization parameters from training set.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Tennis player/match statistics
        y : ignored
        
        Returns
        -------
        self
        """
        # Convert to DataFrame for easy stats
        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X
        else:
            return self
        
        # Learn means and stds for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.stat_means_[col] = df[col].mean()
            self.stat_stds_[col] = df[col].std() if df[col].std() > 0 else 1.0
        
        return self
    
    def transform(self, X):
        """
        Transform tennis stats to narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Tennis player/match statistics
            
        Returns
        -------
        features : ndarray of shape (n_samples, 50)
            Tennis performance narrative features
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
        
        features = []
        for stats in X:
            feat = self._extract_tennis_features(stats)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_tennis_features(self, stats: Dict) -> List[float]:
        """Extract all tennis performance narrative features"""
        features = []
        
        # === SURFACE SPECIALIZATION NARRATIVE (12 features) ===
        
        # 1-3. Surface win rates (clay/grass/hard)
        clay_win_rate = stats.get('clay_win_rate', stats.get('surface_clay_pct', 0.5))
        grass_win_rate = stats.get('grass_win_rate', stats.get('surface_grass_pct', 0.5))
        hard_win_rate = stats.get('hard_win_rate', stats.get('surface_hard_pct', 0.5))
        features.extend([clay_win_rate, grass_win_rate, hard_win_rate])
        
        # 4-6. Surface specialization (variance from mean)
        overall_win_rate = stats.get('overall_win_rate', 0.5)
        clay_spec = clay_win_rate - overall_win_rate
        grass_spec = grass_win_rate - overall_win_rate
        hard_spec = hard_win_rate - overall_win_rate
        features.extend([clay_spec, grass_spec, hard_spec])
        
        # 7. Surface dominance (max specialization)
        surface_dominance = max(abs(clay_spec), abs(grass_spec), abs(hard_spec))
        features.append(surface_dominance)
        
        # 8. Surface versatility (consistency across surfaces)
        surface_std = np.std([clay_win_rate, grass_win_rate, hard_win_rate])
        surface_versatility = 1.0 - min(1.0, surface_std / 0.3)  # Lower std = more versatile
        features.append(surface_versatility)
        
        # 9. Current surface mastery (performance on match surface)
        current_surface = stats.get('current_surface', 'hard')
        surface_map = {'clay': clay_win_rate, 'grass': grass_win_rate, 'hard': hard_win_rate}
        current_surface_mastery = surface_map.get(current_surface, hard_win_rate)
        features.append(current_surface_mastery)
        
        # 10. Surface advantage vs opponent
        opponent_surface_rate = stats.get('opponent_surface_win_rate', 0.5)
        surface_advantage = current_surface_mastery - opponent_surface_rate
        features.append(surface_advantage)
        
        # 11. Indoor vs outdoor performance
        indoor_rate = stats.get('indoor_win_rate', 0.5)
        outdoor_rate = stats.get('outdoor_win_rate', 0.5)
        features.append((indoor_rate + outdoor_rate) / 2)
        
        # 12. Altitude/conditions adaptation
        altitude_perf = stats.get('altitude_performance', 0.5)
        features.append(altitude_perf)
        
        # === FORM & MOMENTUM NARRATIVE (12 features) ===
        
        # 13. Recent form (last 5 matches)
        last_5_wins = stats.get('last_5_wins', stats.get('recent_form_5', 2.5))
        last_5_rate = last_5_wins / 5.0
        features.append(last_5_rate)
        
        # 14. Medium-term form (last 10 matches)
        last_10_wins = stats.get('last_10_wins', stats.get('recent_form_10', 5.0))
        last_10_rate = last_10_wins / 10.0
        features.append(last_10_rate)
        
        # 15. Long-term form (last 20 matches)
        last_20_wins = stats.get('last_20_wins', stats.get('recent_form_20', 10.0))
        last_20_rate = last_20_wins / 20.0
        features.append(last_20_rate)
        
        # 16. Momentum trajectory (short-term vs long-term)
        if self.compute_differentials:
            momentum = last_5_rate - last_20_rate
            features.append(momentum)
        else:
            features.append(0.0)
        
        # 17. Win streak length
        win_streak = stats.get('win_streak', stats.get('current_streak', 0))
        win_streak_normalized = min(1.0, win_streak / 10.0)  # 10+ match streak = max
        features.append(win_streak_normalized)
        
        # 18. Tournament progress (matches won this tournament)
        tourney_matches_won = stats.get('tourney_matches_won', 0)
        tourney_progress = min(1.0, tourney_matches_won / 7.0)  # 7 = win tournament
        features.append(tourney_progress)
        
        # 19. Confidence indicator (sets won rate in recent matches)
        recent_sets_won_rate = stats.get('recent_sets_won_rate', 0.6)
        features.append(recent_sets_won_rate)
        
        # 20. Mental strength (comeback ability - won after losing first set)
        comeback_rate = stats.get('comeback_win_rate', stats.get('down_set_recovery', 0.3))
        features.append(comeback_rate)
        
        # 21. Closing ability (win rate when ahead)
        closing_rate = stats.get('closing_rate', stats.get('up_set_win', 0.85))
        features.append(closing_rate)
        
        # 22. Tiebreak performance (mental toughness)
        tiebreak_rate = stats.get('tiebreak_win_rate', 0.5)
        features.append(tiebreak_rate)
        
        # 23. Decisive set performance (5th/3rd set win rate)
        decisive_set_rate = stats.get('decisive_set_win_rate', 0.5)
        features.append(decisive_set_rate)
        
        # 24. Break point conversion (clutch ability)
        bp_conversion = stats.get('break_point_conversion', 0.4)
        features.append(bp_conversion)
        
        # === TOURNAMENT CONTEXT NARRATIVE (8 features) ===
        
        # 25. Grand Slam performance
        grand_slam_rate = stats.get('grand_slam_win_rate', 0.4)
        features.append(grand_slam_rate)
        
        # 26. Masters 1000 performance
        masters_rate = stats.get('masters_win_rate', 0.45)
        features.append(masters_rate)
        
        # 27. ATP 500 performance
        atp500_rate = stats.get('atp500_win_rate', 0.5)
        features.append(atp500_rate)
        
        # 28. Current tournament level performance
        current_level = stats.get('tournament_level', 'atp_500')
        level_map = {
            'grand_slam': grand_slam_rate,
            'masters_1000': masters_rate,
            'atp_500': atp500_rate,
            'atp_250': stats.get('atp250_win_rate', 0.55)
        }
        current_level_perf = level_map.get(current_level, 0.5)
        features.append(current_level_perf)
        
        # 29. Big match experience (Grand Slam matches played)
        gs_matches = stats.get('grand_slam_matches', stats.get('big_match_experience', 10))
        big_match_exp = min(1.0, gs_matches / 50.0)  # 50+ GS matches = max experience
        features.append(big_match_exp)
        
        # 30. Finals record (championship mentality)
        finals_rate = stats.get('finals_win_rate', stats.get('final_record', 0.5))
        features.append(finals_rate)
        
        # 31. Semifinal conversion (pressure handling)
        semi_conversion = stats.get('semi_to_final_rate', 0.5)
        features.append(semi_conversion)
        
        # 32. Deep run consistency (% tournaments reaching QF+)
        deep_run_rate = stats.get('quarterfinal_plus_rate', 0.3)
        features.append(deep_run_rate)
        
        # === PHYSICAL & FATIGUE NARRATIVE (6 features) ===
        
        # 33. Match load (matches in last 7 days)
        matches_7d = stats.get('matches_last_7_days', stats.get('recent_matches', 0))
        fatigue_7d = min(1.0, matches_7d / 5.0)  # 5+ matches in week = high fatigue
        features.append(fatigue_7d)
        
        # 34. Match load (matches in last 14 days)
        matches_14d = stats.get('matches_last_14_days', matches_7d * 2)
        fatigue_14d = min(1.0, matches_14d / 10.0)
        features.append(fatigue_14d)
        
        # 35. Rest days before match
        rest_days = stats.get('rest_days', stats.get('days_since_last_match', 3))
        rest_advantage = min(1.0, rest_days / 7.0)  # 7+ days = full rest
        features.append(rest_advantage)
        
        # 36. Age factor (experience vs physical decline)
        age = stats.get('age', stats.get('player_age', 27))
        # Peak around 25-28, decline after 30
        age_factor = 1.0 - max(0, (age - 25) / 15.0)  # Linear decline from 25
        features.append(np.clip(age_factor, 0, 1))
        
        # 37. Match duration trend (fitness indicator)
        avg_match_minutes = stats.get('avg_match_minutes', stats.get('avg_duration', 120))
        duration_efficiency = 1.0 - min(1.0, (avg_match_minutes - 90) / 90)  # Shorter = more efficient
        features.append(np.clip(duration_efficiency, 0, 1))
        
        # 38. Five-set stamina (performance in long matches)
        five_set_rate = stats.get('five_set_win_rate', 0.5)
        features.append(five_set_rate)
        
        # === RANKING & COMPETITIVE POSITION (6 features) ===
        
        # 39. Current ranking (competitive position)
        ranking = stats.get('ranking', stats.get('current_rank', 50))
        rank_strength = 1.0 - min(1.0, (ranking - 1) / 99)  # Top 100 scale
        features.append(rank_strength)
        
        # 40. Ranking momentum (change in last 3 months)
        rank_change = stats.get('ranking_change_3m', stats.get('rank_momentum', 0))
        rank_momentum = np.clip(rank_change / 20.0, -1, 1)  # ±20 ranks = significant
        features.append(rank_momentum)
        
        # 41. Career high proximity (psychological confidence)
        career_high = stats.get('career_high_rank', 50)
        career_high_proximity = 1.0 - abs(ranking - career_high) / 100.0
        features.append(np.clip(career_high_proximity, 0, 1))
        
        # 42. Top 10 win rate (quality of competition)
        top10_rate = stats.get('vs_top10_win_rate', stats.get('quality_wins', 0.3))
        features.append(top10_rate)
        
        # 43. Top 50 win rate
        top50_rate = stats.get('vs_top50_win_rate', 0.4)
        features.append(top50_rate)
        
        # 44. Ranking differential vs opponent
        opp_ranking = stats.get('opponent_ranking', 50)
        rank_diff = (opp_ranking - ranking) / 100.0  # Positive = facing lower ranked
        features.append(np.clip(rank_diff, -1, 1))
        
        # === HEAD-TO-HEAD & RIVALRY NARRATIVE (6 features) ===
        
        if self.include_h2h:
            # 45. H2H record vs opponent
            h2h_wins = stats.get('h2h_wins', 0)
            h2h_total = stats.get('h2h_total', 0)
            h2h_rate = h2h_wins / h2h_total if h2h_total > 0 else 0.5
            features.append(h2h_rate)
            
            # 46. H2H on current surface
            h2h_surface_wins = stats.get('h2h_surface_wins', 0)
            h2h_surface_total = stats.get('h2h_surface_total', 0)
            h2h_surface_rate = h2h_surface_wins / h2h_surface_total if h2h_surface_total > 0 else 0.5
            features.append(h2h_surface_rate)
            
            # 47. Rivalry intensity (number of meetings)
            h2h_meetings = stats.get('h2h_meetings', h2h_total)
            rivalry_intensity = min(1.0, h2h_meetings / 20.0)  # 20+ meetings = major rivalry
            features.append(rivalry_intensity)
            
            # 48. Recent H2H (last 3 meetings)
            h2h_recent_wins = stats.get('h2h_recent_wins', h2h_wins)
            h2h_recent_rate = h2h_recent_wins / 3 if h2h_meetings >= 3 else h2h_rate
            features.append(h2h_recent_rate)
            
            # 49. H2H momentum (recent vs overall)
            h2h_momentum = h2h_recent_rate - h2h_rate
            features.append(h2h_momentum)
            
            # 50. Psychological edge (dominant H2H + recent form)
            psych_edge = (h2h_rate * 0.6 + last_5_rate * 0.4) - 0.5  # Centered at 0
            features.append(psych_edge)
        else:
            # If H2H not included, use neutral values
            features.extend([0.5, 0.5, 0.0, 0.5, 0.0, 0.0])
        
        return features
    
    def _normalize_stat(self, stat_name: str, value: float) -> float:
        """Normalize stat to z-score if normalization enabled"""
        if not self.normalize or stat_name not in self.stat_means_:
            return value
        
        mean = self.stat_means_[stat_name]
        std = self.stat_stds_[stat_name]
        
        z_score = (value - mean) / std
        # Clip to ±3 std devs and scale to roughly 0-1
        return np.clip((z_score + 3) / 6, 0, 1)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names"""
        return np.array([
            # Surface specialization
            'tennis_clay_win_rate',
            'tennis_grass_win_rate',
            'tennis_hard_win_rate',
            'tennis_clay_specialization',
            'tennis_grass_specialization',
            'tennis_hard_specialization',
            'tennis_surface_dominance',
            'tennis_surface_versatility',
            'tennis_current_surface_mastery',
            'tennis_surface_advantage',
            'tennis_indoor_outdoor_avg',
            'tennis_altitude_adaptation',
            
            # Form & momentum
            'tennis_form_last_5',
            'tennis_form_last_10',
            'tennis_form_last_20',
            'tennis_momentum_trajectory',
            'tennis_win_streak',
            'tennis_tournament_progress',
            'tennis_sets_won_confidence',
            'tennis_comeback_ability',
            'tennis_closing_ability',
            'tennis_tiebreak_performance',
            'tennis_decisive_set_performance',
            'tennis_break_point_conversion',
            
            # Tournament context
            'tennis_grand_slam_performance',
            'tennis_masters_performance',
            'tennis_atp500_performance',
            'tennis_current_level_performance',
            'tennis_big_match_experience',
            'tennis_finals_record',
            'tennis_semifinal_conversion',
            'tennis_deep_run_consistency',
            
            # Physical & fatigue
            'tennis_fatigue_7_days',
            'tennis_fatigue_14_days',
            'tennis_rest_advantage',
            'tennis_age_factor',
            'tennis_match_duration_efficiency',
            'tennis_five_set_stamina',
            
            # Ranking & position
            'tennis_ranking_strength',
            'tennis_ranking_momentum',
            'tennis_career_high_proximity',
            'tennis_vs_top10_rate',
            'tennis_vs_top50_rate',
            'tennis_ranking_differential',
            
            # H2H & rivalry
            'tennis_h2h_overall',
            'tennis_h2h_surface',
            'tennis_rivalry_intensity',
            'tennis_h2h_recent',
            'tennis_h2h_momentum',
            'tennis_psychological_edge'
        ])
