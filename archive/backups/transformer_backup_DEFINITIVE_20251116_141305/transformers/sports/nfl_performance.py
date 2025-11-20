"""
NFL Performance Transformer

Extracts NARRATIVITY FROM NFL STATISTICS.
Stats as story: momentum, dominance, clutch ability, context.

Critical insight: Numbers tell stories. 300 YPG isn't just a stat - 
it's an offensive power narrative. A 5-game win streak isn't just wins -
it's momentum. This transformer translates numbers into narrative features.

Author: Narrative Integration System
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class NFLPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from NFL performance statistics.
    
    Philosophy:
    - Stats ARE narrative (not separate from it)
    - Performance patterns reveal psychological states
    - Context transforms meaning (home vs away, vs winners, etc.)
    - Trends reveal momentum and trajectory
    
    Accepts:
    - Dict with stat keys
    - DataFrame with stat columns
    - JSON-like nested structures
    
    Features: 40 total
    - Offensive performance narrative (10)
    - Defensive performance narrative (10)
    - Contextual performance (10)
    - Situational performance (10)
    """
    
    def __init__(
        self,
        normalize: bool = True,
        compute_differentials: bool = True,
        include_derived: bool = True
    ):
        """
        Initialize NFL performance analyzer.
        
        Parameters
        ----------
        normalize : bool
            Normalize stats to z-scores (relative to league)
        compute_differentials : bool
            Compute home-away, recent-overall differentials
        include_derived : bool
            Include derived stats (efficiency, ratios)
        """
        self.normalize = normalize
        self.compute_differentials = compute_differentials
        self.include_derived = include_derived
        
        # Stats we'll track for normalization
        self.stat_means_ = {}
        self.stat_stds_ = {}
    
    def fit(self, X, y=None):
        """
        Learn normalization parameters from training set.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            NFL team statistics
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
        Transform NFL stats to narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            NFL team statistics
            
        Returns
        -------
        features : ndarray of shape (n_samples, 40)
            NFL performance narrative features
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
        
        features = []
        for stats in X:
            feat = self._extract_nfl_features(stats)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_nfl_features(self, stats: Dict) -> List[float]:
        """Extract all NFL performance narrative features"""
        features = []
        
        # === OFFENSIVE PERFORMANCE NARRATIVE (10 features) ===
        
        # 1. Scoring power (PPG)
        ppg = stats.get('points_per_game', stats.get('ppg', 20.0))
        features.append(self._normalize_stat('points_per_game', ppg))
        
        # 2. Total offensive power (yards per game)
        ypg = stats.get('yards_per_game', stats.get('total_yards', 300.0))
        features.append(self._normalize_stat('yards_per_game', ypg))
        
        # 3. Passing efficiency (YPA)
        ypa = stats.get('yards_per_attempt', stats.get('passing_ypa', 7.0))
        features.append(self._normalize_stat('yards_per_attempt', ypa))
        
        # 4. Ground game power (rushing YPC)
        ypc = stats.get('rushing_yards_per_carry', stats.get('rushing_ypc', 4.0))
        features.append(self._normalize_stat('rushing_yards_per_carry', ypc))
        
        # 5. Clutch ability (3rd down conversion %)
        third_down = stats.get('third_down_pct', stats.get('third_down_conversion', 0.40))
        features.append(third_down)
        
        # 6. Finishing ability (red zone %)
        red_zone = stats.get('red_zone_pct', stats.get('red_zone_efficiency', 0.55))
        features.append(red_zone)
        
        # 7. Ball security (turnover differential)
        to_diff = stats.get('turnover_differential', stats.get('to_diff', 0.0))
        features.append(self._normalize_stat('turnover_differential', to_diff))
        
        # 8. Control narrative (time of possession)
        top = stats.get('time_of_possession', stats.get('top', 30.0))
        features.append((top - 30.0) / 5.0)  # Normalize around 30 minutes
        
        # 9. Offensive ranking (competitive position)
        off_rank = stats.get('offensive_ranking', stats.get('off_rank', 16.0))
        features.append(1.0 - (off_rank / 32.0))  # Higher rank = better
        
        # 10. Explosive play rate (big play narrative)
        explosive = stats.get('explosive_play_rate', 0.10)
        features.append(explosive * 10)  # Scale to 0-1
        
        # === DEFENSIVE PERFORMANCE NARRATIVE (10 features) ===
        
        # 11. Defensive stinginess (points allowed per game)
        ppg_allowed = stats.get('points_allowed_per_game', stats.get('ppg_allowed', 22.0))
        features.append(1.0 - self._normalize_stat('points_allowed_per_game', ppg_allowed))  # Invert (lower is better)
        
        # 12. Defensive strength (yards allowed per game)
        ypg_allowed = stats.get('yards_allowed_per_game', stats.get('yards_allowed', 340.0))
        features.append(1.0 - self._normalize_stat('yards_allowed_per_game', ypg_allowed))
        
        # 13. Pass rush power (sacks per game)
        sacks = stats.get('sacks_per_game', stats.get('sacks', 2.0))
        features.append(self._normalize_stat('sacks_per_game', sacks))
        
        # 14. Opportunistic defense (takeaways per game)
        takeaways = stats.get('takeaways_per_game', stats.get('takeaways', 1.0))
        features.append(self._normalize_stat('takeaways_per_game', takeaways))
        
        # 15. Situational defense (3rd down defense %)
        third_down_def = stats.get('third_down_defense', 0.40)
        features.append(1.0 - third_down_def)  # Lower % is better for defense
        
        # 16. Goal line stands (red zone defense %)
        red_zone_def = stats.get('red_zone_defense', 0.55)
        features.append(1.0 - red_zone_def)
        
        # 17. Defensive ranking
        def_rank = stats.get('defensive_ranking', stats.get('def_rank', 16.0))
        features.append(1.0 - (def_rank / 32.0))
        
        # 18. Pass defense efficiency
        pass_def = stats.get('pass_defense_efficiency', 0.5)
        features.append(pass_def)
        
        # 19. Run defense efficiency
        run_def = stats.get('run_defense_efficiency', 0.5)
        features.append(run_def)
        
        # 20. Turnover creation rate
        to_rate = stats.get('turnover_creation_rate', 0.10)
        features.append(to_rate * 10)
        
        # === CONTEXTUAL PERFORMANCE (10 features) ===
        
        # 21. Home dominance (home win %)
        home_record = stats.get('home_win_pct', stats.get('home_record', 0.5))
        features.append(home_record)
        
        # 22. Road warrior (away win %)
        away_record = stats.get('away_win_pct', stats.get('away_record', 0.5))
        features.append(away_record)
        
        # 23. Home field advantage (home - away)
        if self.compute_differentials:
            home_advantage = home_record - away_record
            features.append(home_advantage)
        else:
            features.append(0.0)
        
        # 24. Division dominance (division record)
        div_record = stats.get('division_win_pct', stats.get('division_record', 0.5))
        features.append(div_record)
        
        # 25. Conference strength (conference record)
        conf_record = stats.get('conference_win_pct', stats.get('conference_record', 0.5))
        features.append(conf_record)
        
        # 26. Quality wins (record vs winning teams)
        vs_winners = stats.get('record_vs_winning_teams', 0.4)
        features.append(vs_winners)
        
        # 27. Clutch performance (record in one-score games)
        close_games = stats.get('record_close_games', 0.5)
        features.append(close_games)
        
        # 28. Big stage performance (primetime record)
        primetime = stats.get('primetime_record', 0.5)
        features.append(primetime)
        
        # 29. Momentum (recent form - last 5 games)
        recent_form = stats.get('last_5_games_pct', stats.get('recent_record', 0.5))
        features.append(recent_form)
        
        # 30. Hot streak (win streak length)
        win_streak = stats.get('win_streak', stats.get('current_streak', 0))
        features.append(min(1.0, win_streak / 5.0))  # Normalize (5+ game streak = max)
        
        # === SITUATIONAL PERFORMANCE (10 features) ===
        
        # 31. Dominance narrative (point differential)
        point_diff = stats.get('point_differential', stats.get('point_diff', 0.0))
        features.append(self._normalize_stat('point_differential', point_diff))
        
        # 32. Rest advantage (rest days differential)
        rest_diff = stats.get('rest_differential', 0.0)
        features.append(np.clip(rest_diff / 3.0, -1, 1))  # Normalize ±3 days
        
        # 33. Preparation narrative (post-bye performance)
        post_bye = stats.get('post_bye_record', 0.5)
        features.append(post_bye)
        
        # 34. Weather narrative (outdoor game performance)
        weather_perf = stats.get('weather_game_performance', 0.5)
        features.append(weather_perf)
        
        # 35. Venue adaptation (dome vs outdoor)
        dome_vs_outdoor = stats.get('dome_vs_outdoor_diff', 0.0)
        features.append(dome_vs_outdoor)
        
        # 36. Surface narrative (grass vs turf)
        grass_vs_turf = stats.get('grass_vs_turf_diff', 0.0)
        features.append(grass_vs_turf)
        
        # 37. Competition level (strength of schedule)
        sos = stats.get('strength_of_schedule', 0.5)
        features.append(sos)
        
        # 38. Health narrative (injury severity if available)
        injury_impact = stats.get('injury_severity', 0.0)
        features.append(np.clip(injury_impact, 0, 1))
        
        # 39. Chemistry narrative (roster continuity)
        continuity = stats.get('roster_continuity', 0.75)
        features.append(continuity)
        
        # 40. Experience narrative (playoff + coaching experience)
        experience = (stats.get('playoff_experience', 0.5) + stats.get('coaching_experience', 0.5)) / 2
        features.append(experience)
        
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
            # Offensive
            'nfl_points_per_game',
            'nfl_yards_per_game',
            'nfl_passing_efficiency',
            'nfl_rushing_power',
            'nfl_third_down_pct',
            'nfl_red_zone_efficiency',
            'nfl_turnover_differential',
            'nfl_time_of_possession',
            'nfl_offensive_ranking',
            'nfl_explosive_play_rate',
            
            # Defensive
            'nfl_points_allowed',
            'nfl_yards_allowed',
            'nfl_sacks',
            'nfl_takeaways',
            'nfl_third_down_defense',
            'nfl_red_zone_defense',
            'nfl_defensive_ranking',
            'nfl_pass_defense',
            'nfl_run_defense',
            'nfl_turnover_creation',
            
            # Contextual
            'nfl_home_performance',
            'nfl_away_performance',
            'nfl_home_field_advantage',
            'nfl_division_record',
            'nfl_conference_record',
            'nfl_vs_winning_teams',
            'nfl_close_game_record',
            'nfl_primetime_performance',
            'nfl_recent_form',
            'nfl_win_streak',
            
            # Situational
            'nfl_point_differential',
            'nfl_rest_advantage',
            'nfl_post_bye_performance',
            'nfl_weather_performance',
            'nfl_dome_vs_outdoor',
            'nfl_grass_vs_turf',
            'nfl_strength_of_schedule',
            'nfl_injury_impact',
            'nfl_roster_continuity',
            'nfl_experience_level'
        ])

