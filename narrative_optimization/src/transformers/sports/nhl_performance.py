"""
NHL Performance Transformer

Extracts NARRATIVITY FROM NHL STATISTICS.
Stats as story: scoring power, goalie performance, physicality, special teams.

Critical insight: Hockey stats tell stories. A .930 save percentage isn't just a stat -
it's a goalie dominance narrative. A 25% power play isn't just efficiency -
it's special teams superiority. This transformer translates numbers into narrative features.

Author: Narrative Integration System  
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class NHLPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from NHL performance statistics.
    
    Philosophy:
    - Stats ARE narrative (not separate from it)
    - Performance patterns reveal psychological states
    - Context transforms meaning (home vs away, playoffs, rivalries)
    - Goalie performance is THE critical hockey narrative
    - Physical play and special teams define game character
    
    Accepts:
    - Dict with stat keys
    - DataFrame with stat columns
    - JSON-like nested structures
    
    Features: 50 total
    - Offensive performance narrative (10)
    - Defensive performance narrative (10)
    - Goalie narrative (10) - HOCKEY-SPECIFIC
    - Physical/intensity narrative (5) - HOCKEY-SPECIFIC
    - Special teams narrative (5) - HOCKEY-SPECIFIC
    - Contextual performance (10)
    """
    
    def __init__(
        self,
        normalize: bool = True,
        include_goalie: bool = True,
        include_physical: bool = True,
        include_special_teams: bool = True
    ):
        """
        Initialize NHL performance analyzer.
        
        Parameters
        ----------
        normalize : bool
            Normalize stats to z-scores (relative to league)
        include_goalie : bool
            Include goalie-specific features (critical for NHL)
        include_physical : bool
            Include physical play metrics (hits, blocks, PIM)
        include_special_teams : bool
            Include power play and penalty kill features
        """
        self.normalize = normalize
        self.include_goalie = include_goalie
        self.include_physical = include_physical
        self.include_special_teams = include_special_teams
        
        # Stats we'll track for normalization
        self.stat_means_ = {}
        self.stat_stds_ = {}
    
    def fit(self, X, y=None):
        """
        Learn normalization parameters from training set.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            NHL team statistics
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
        Transform NHL stats to narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            NHL team statistics
            
        Returns
        -------
        features : ndarray of shape (n_samples, 50)
            NHL performance narrative features
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
        
        features = []
        for stats in X:
            feat = self._extract_nhl_features(stats)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_nhl_features(self, stats: Dict) -> List[float]:
        """Extract all NHL performance narrative features"""
        features = []
        
        # === OFFENSIVE PERFORMANCE NARRATIVE (10 features) ===
        
        # 1. Scoring power (Goals per game)
        gpg = stats.get('goals_per_game', stats.get('gpg', 2.8))
        features.append(self._normalize_stat('goals_per_game', gpg))
        
        # 2. Offensive pressure (Shots per game)
        spg = stats.get('shots_per_game', stats.get('spg', 30.0))
        features.append(self._normalize_stat('shots_per_game', spg))
        
        # 3. Shooting efficiency (Shooting percentage)
        shoot_pct = stats.get('shooting_pct', stats.get('sh_pct', 0.095))
        features.append(shoot_pct)
        
        # 4. 5v5 dominance (Even strength goals)
        ev_goals = stats.get('even_strength_goals', stats.get('es_goals', 2.0))
        features.append(self._normalize_stat('even_strength_goals', ev_goals))
        
        # 5. High-danger chances (scoring quality)
        hd_chances = stats.get('high_danger_chances', stats.get('hdc', 10.0))
        features.append(hd_chances / 20.0)  # Normalize to 0-1 range
        
        # 6. Expected goals (xG) - offensive threat
        xg = stats.get('expected_goals', stats.get('xg', 2.8))
        features.append(self._normalize_stat('expected_goals', xg))
        
        # 7. Offensive zone time (possession dominance)
        oz_time = stats.get('offensive_zone_time', stats.get('oz_time_pct', 0.50))
        features.append(oz_time)
        
        # 8. Faceoff win % (puck control)
        faceoff_pct = stats.get('faceoff_win_pct', stats.get('fo_pct', 0.50))
        features.append(faceoff_pct)
        
        # 9. Assists per game (playmaking)
        apg = stats.get('assists_per_game', stats.get('apg', 4.5))
        features.append(self._normalize_stat('assists_per_game', apg))
        
        # 10. Offensive momentum (recent goal trend)
        recent_gpg = stats.get('recent_goals_per_game', stats.get('l5_gpg', gpg))
        features.append(self._normalize_stat('recent_goals_per_game', recent_gpg))
        
        # === DEFENSIVE PERFORMANCE NARRATIVE (10 features) ===
        
        # 11. Goals against (defensive vulnerability)
        gaa = stats.get('goals_against_per_game', stats.get('gaa', 2.8))
        features.append(1.0 - self._normalize_stat('goals_against_per_game', gaa))  # Invert (lower is better)
        
        # 12. Shots against (defensive pressure)
        saa = stats.get('shots_against_per_game', stats.get('saa', 30.0))
        features.append(1.0 - self._normalize_stat('shots_against_per_game', saa))  # Invert
        
        # 13. Save percentage (goalie + defense quality)
        sv_pct = stats.get('save_percentage', stats.get('sv_pct', 0.905))
        features.append(sv_pct)
        
        # 14. Blocks per game (sacrifice/commitment)
        blocks = stats.get('blocks_per_game', stats.get('blocks', 15.0))
        features.append(self._normalize_stat('blocks_per_game', blocks))
        
        # 15. Takeaways (disruption ability)
        takeaways = stats.get('takeaways_per_game', stats.get('takeaways', 8.0))
        features.append(self._normalize_stat('takeaways_per_game', takeaways))
        
        # 16. Hits per game (physical defense)
        hits = stats.get('hits_per_game', stats.get('hits', 22.0))
        features.append(self._normalize_stat('hits_per_game', hits))
        
        # 17. High-danger chances against (defensive quality)
        hdca = stats.get('high_danger_chances_against', stats.get('hdca', 10.0))
        features.append(1.0 - (hdca / 20.0))  # Invert and normalize
        
        # 18. Expected goals against (xGA) - defensive threat
        xga = stats.get('expected_goals_against', stats.get('xga', 2.8))
        features.append(1.0 - self._normalize_stat('expected_goals_against', xga))  # Invert
        
        # 19. Defensive zone time (pressure endured)
        dz_time = stats.get('defensive_zone_time', stats.get('dz_time_pct', 0.50))
        features.append(1.0 - dz_time)  # Invert (less is better)
        
        # 20. Defensive consistency (recent GAA trend)
        recent_gaa = stats.get('recent_gaa', stats.get('l5_gaa', gaa))
        features.append(1.0 - self._normalize_stat('recent_gaa', recent_gaa))  # Invert
        
        # === GOALIE NARRATIVE (10 features) - HOCKEY-SPECIFIC ===
        
        if self.include_goalie:
            # 21. Starting goalie save % (elite vs average)
            goalie_sv = stats.get('goalie_save_pct', stats.get('g_sv_pct', 0.905))
            features.append(goalie_sv)
            
            # 22. Goalie GAA (goals against average)
            goalie_gaa = stats.get('goalie_gaa', stats.get('g_gaa', 2.8))
            features.append(1.0 - self._normalize_stat('goalie_gaa', goalie_gaa))  # Invert
            
            # 23. Goalie shutouts (dominance indicator)
            shutouts = stats.get('goalie_shutouts', stats.get('g_shutouts', 2))
            features.append(min(1.0, shutouts / 10.0))  # Normalize
            
            # 24. Goalie wins (success rate)
            goalie_wins = stats.get('goalie_wins', stats.get('g_wins', 20))
            features.append(min(1.0, goalie_wins / 40.0))  # Normalize
            
            # 25. Goalie recent form (hot goalie indicator)
            goalie_recent_sv = stats.get('goalie_recent_sv_pct', stats.get('g_l5_sv', goalie_sv))
            features.append(goalie_recent_sv)
            
            # 26. Goalie vs opponent history (matchup advantage)
            goalie_vs_opp = stats.get('goalie_vs_opponent_sv', stats.get('g_vs_opp', goalie_sv))
            features.append(goalie_vs_opp)
            
            # 27. Goalie home/road split (environment impact)
            goalie_home_sv = stats.get('goalie_home_sv', stats.get('g_home_sv', goalie_sv))
            goalie_road_sv = stats.get('goalie_road_sv', stats.get('g_road_sv', goalie_sv))
            home_road_diff = goalie_home_sv - goalie_road_sv
            features.append(home_road_diff + 0.5)  # Center at 0.5
            
            # 28. Goalie rest (games since last start)
            goalie_rest = stats.get('goalie_rest_days', stats.get('g_rest', 2))
            features.append(min(1.0, goalie_rest / 5.0))  # Normalize
            
            # 29. Goalie playoff experience (pressure performance)
            goalie_playoff_games = stats.get('goalie_playoff_games', stats.get('g_playoff', 0))
            features.append(min(1.0, goalie_playoff_games / 50.0))  # Normalize
            
            # 30. Starter vs backup narrative (quality differential)
            is_starter = stats.get('goalie_is_starter', stats.get('g_starter', True))
            features.append(1.0 if is_starter else 0.5)
        else:
            # Add placeholder features if goalie disabled
            features.extend([0.5] * 10)
        
        # === PHYSICAL/INTENSITY NARRATIVE (5 features) - HOCKEY-SPECIFIC ===
        
        if self.include_physical:
            # 31. Hits per game (physicality/aggression)
            hits_pg = stats.get('hits_per_game', stats.get('hits', 22.0))
            features.append(self._normalize_stat('hits_per_game', hits_pg))
            
            # 32. Penalty minutes (discipline vs aggression balance)
            pim = stats.get('penalty_minutes_per_game', stats.get('pim', 8.0))
            features.append(self._normalize_stat('penalty_minutes_per_game', pim))
            
            # 33. Fighting majors (enforcer presence)
            fights = stats.get('fighting_majors', stats.get('fights', 0))
            features.append(min(1.0, fights / 10.0))  # Normalize
            
            # 34. Playoff toughness score (grit indicator)
            playoff_hits = stats.get('playoff_hits_per_game', stats.get('po_hits', hits_pg))
            features.append(self._normalize_stat('playoff_hits_per_game', playoff_hits))
            
            # 35. Rivalry intensity multiplier
            is_rivalry = stats.get('is_rivalry', False)
            features.append(1.2 if is_rivalry else 1.0)
        else:
            features.extend([0.5] * 5)
        
        # === SPECIAL TEAMS NARRATIVE (5 features) - HOCKEY-SPECIFIC ===
        
        if self.include_special_teams:
            # 36. Power play efficiency (man-advantage skill)
            pp_pct = stats.get('power_play_pct', stats.get('pp_pct', 0.20))
            features.append(pp_pct)
            
            # 37. Penalty kill efficiency (shorthanded defense)
            pk_pct = stats.get('penalty_kill_pct', stats.get('pk_pct', 0.80))
            features.append(pk_pct)
            
            # 38. Power play goals (special teams offense)
            pp_goals = stats.get('pp_goals_per_game', stats.get('ppg', 0.6))
            features.append(pp_goals / 2.0)  # Normalize
            
            # 39. Shorthanded goals (chaos factor)
            sh_goals = stats.get('sh_goals_per_game', stats.get('shg', 0.05))
            features.append(sh_goals / 0.2)  # Normalize (rare events)
            
            # 40. Special teams differential (PP% - opponent PK%)
            st_diff = stats.get('special_teams_diff', 0.0)
            features.append((st_diff + 0.2) / 0.4)  # Normalize to 0-1
        else:
            features.extend([0.5] * 5)
        
        # === CONTEXTUAL PERFORMANCE (10 features) ===
        
        # 41. Home advantage (home vs away performance)
        home_win_pct = stats.get('home_win_pct', 0.55)
        away_win_pct = stats.get('away_win_pct', 0.45)
        features.append(home_win_pct - away_win_pct + 0.5)  # Center at 0.5
        
        # 42. Back-to-back performance (fatigue impact)
        b2b_win_pct = stats.get('back_to_back_win_pct', 0.40)
        features.append(b2b_win_pct)
        
        # 43. Rest advantage (performance with rest)
        rest_win_pct = stats.get('rest_win_pct', 0.55)
        features.append(rest_win_pct)
        
        # 44. Division game performance (familiarity)
        div_win_pct = stats.get('division_win_pct', 0.50)
        features.append(div_win_pct)
        
        # 45. Playoff performance (pressure situations)
        playoff_win_pct = stats.get('playoff_win_pct', 0.50)
        features.append(playoff_win_pct)
        
        # 46. Recent form (L5 win percentage)
        l5_win_pct = stats.get('l5_win_pct', 0.50)
        features.append(l5_win_pct)
        
        # 47. Recent form (L10 win percentage)
        l10_win_pct = stats.get('l10_win_pct', 0.50)
        features.append(l10_win_pct)
        
        # 48. Head-to-head record (vs current opponent)
        h2h_win_pct = stats.get('h2h_win_pct', 0.50)
        features.append(h2h_win_pct)
        
        # 49. Win streak indicator (momentum)
        win_streak = stats.get('win_streak', 0)
        features.append(min(1.0, win_streak / 5.0))  # Normalize
        
        # 50. Season progression (early vs late season form)
        games_played = stats.get('games_played', 41)
        season_phase = games_played / 82.0  # 82 games in NHL season
        features.append(min(1.0, season_phase))
        
        return features
    
    def _normalize_stat(self, stat_name: str, value: float) -> float:
        """Normalize a stat using learned mean and std"""
        if not self.normalize or stat_name not in self.stat_means_:
            return value
        
        mean = self.stat_means_[stat_name]
        std = self.stat_stds_[stat_name]
        
        if std == 0:
            return 0.5
        
        # Z-score normalization, then sigmoid to 0-1
        z = (value - mean) / std
        normalized = 1 / (1 + np.exp(-z))  # Sigmoid
        
        return float(normalized)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return [
            # Offensive (10)
            'goals_per_game', 'shots_per_game', 'shooting_pct',
            'even_strength_goals', 'high_danger_chances', 'expected_goals',
            'offensive_zone_time', 'faceoff_win_pct', 'assists_per_game',
            'offensive_momentum',
            
            # Defensive (10)
            'goals_against', 'shots_against', 'save_percentage',
            'blocks_per_game', 'takeaways', 'hits_defensive',
            'high_danger_chances_against', 'expected_goals_against',
            'defensive_zone_time', 'defensive_consistency',
            
            # Goalie (10)
            'goalie_save_pct', 'goalie_gaa', 'goalie_shutouts',
            'goalie_wins', 'goalie_recent_form', 'goalie_vs_opponent',
            'goalie_home_road_split', 'goalie_rest', 'goalie_playoff_exp',
            'goalie_starter_quality',
            
            # Physical (5)
            'hits_per_game', 'penalty_minutes', 'fighting_majors',
            'playoff_toughness', 'rivalry_intensity',
            
            # Special Teams (5)
            'power_play_pct', 'penalty_kill_pct', 'pp_goals',
            'shorthanded_goals', 'special_teams_diff',
            
            # Contextual (10)
            'home_advantage', 'back_to_back_performance', 'rest_advantage',
            'division_performance', 'playoff_performance', 'l5_form',
            'l10_form', 'h2h_record', 'win_streak', 'season_phase'
        ]


# Convenience function
def extract_nhl_features(games_data: List[Dict], fit: bool = True) -> np.ndarray:
    """
    Extract NHL performance features from game data.
    
    Parameters
    ----------
    games_data : list of dict
        NHL game statistics
    fit : bool
        Whether to fit the transformer (learn normalization parameters)
    
    Returns
    -------
    features : ndarray
        Extracted features
    """
    transformer = NHLPerformanceTransformer()
    
    if fit:
        transformer.fit(games_data)
    
    return transformer.transform(games_data)

