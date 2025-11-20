"""
NHL Focused Temporal Features

Implements ONLY the 8-10 temporal features that provide unique predictive signal
beyond the baseline 900 features.

These features are non-redundant and capture underpriced temporal narratives:
1. Expectation differential (vs preseason)
2. Playoff push intensity
3. Multi-window form divergence (L5 vs L20)
4. Venue-specific momentum
5. Scoring trend acceleration
6. Desperation index
7. Form trajectory (improving vs declining)
8. Situational motivation (division games, rivalry)

Author: Focused Temporal System
Date: November 19, 2025
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class FocusedTemporalExtractor:
    """Extract only high-value, non-redundant temporal features"""
    
    def __init__(self):
        """Initialize extractor"""
        # Preseason expectations (would come from betting markets)
        # For now, use historical averages
        self.preseason_expectations = self._load_preseason_expectations()
    
    def extract_features(self, game: Dict, season_context: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Extract 10 focused temporal features.
        
        Parameters
        ----------
        game : dict
            Current game
        season_context : list of dict, optional
            Season games for context
        
        Returns
        -------
        features : dict
            10 focused temporal features
        """
        if not season_context:
            return self._get_zero_template()
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        game_date = game.get('date', '')
        
        features = {}
        
        # 1. EXPECTATION DIFFERENTIAL
        # Most predictive single feature (r=0.154)
        home_expected = self.preseason_expectations.get(home_team, 41.0)
        away_expected = self.preseason_expectations.get(away_team, 41.0)
        
        home_wins = self._count_wins(home_team, season_context, game_date)
        away_wins = self._count_wins(away_team, season_context, game_date)
        
        home_games = self._count_games(home_team, season_context, game_date)
        away_games = self._count_games(away_team, season_context, game_date)
        
        home_vs_exp = (home_wins - home_expected * (home_games / 82)) / 10 if home_games > 0 else 0
        away_vs_exp = (away_wins - away_expected * (away_games / 82)) / 10 if away_games > 0 else 0
        
        features['expectation_differential'] = home_vs_exp - away_vs_exp
        
        # 2. PLAYOFF PUSH INTENSITY
        # Desperation factor for bubble teams
        home_position = self._get_playoff_position(home_team, season_context, game_date)
        away_position = self._get_playoff_position(away_team, season_context, game_date)
        
        games_remaining_home = 82 - home_games
        games_remaining_away = 82 - away_games
        
        home_push = self._calculate_playoff_push(home_position, games_remaining_home)
        away_push = self._calculate_playoff_push(away_position, games_remaining_away)
        
        features['playoff_push_differential'] = home_push - away_push
        
        # 3. FORM DIVERGENCE (L5 vs L20)
        # Captures acceleration/deceleration
        home_l5 = self._count_wins_window(home_team, season_context, game_date, 5)
        home_l20 = self._count_wins_window(home_team, season_context, game_date, 20)
        away_l5 = self._count_wins_window(away_team, season_context, game_date, 5)
        away_l20 = self._count_wins_window(away_team, season_context, game_date, 20)
        
        home_l5_rate = home_l5 / 5 if home_l5 is not None else 0.5
        home_l20_rate = home_l20 / 20 if home_l20 is not None else 0.5
        away_l5_rate = away_l5 / 5 if away_l5 is not None else 0.5
        away_l20_rate = away_l20 / 20 if away_l20 is not None else 0.5
        
        home_acceleration = home_l5_rate - home_l20_rate
        away_acceleration = away_l5_rate - away_l20_rate
        
        features['form_acceleration_differential'] = home_acceleration - away_acceleration
        
        # 4. VENUE-SPECIFIC MOMENTUM
        # Home team at home vs away team on road (recent)
        home_home_pct = self._get_home_win_pct(home_team, season_context, game_date, 10)
        away_away_pct = self._get_away_win_pct(away_team, season_context, game_date, 10)
        
        features['venue_momentum_gap'] = home_home_pct - away_away_pct
        
        # 5. SCORING TREND ACCELERATION
        # Goals for/against momentum
        home_gf_l5 = self._get_goals_for(home_team, season_context, game_date, 5)
        home_gf_l20 = self._get_goals_for(home_team, season_context, game_date, 20)
        away_gf_l5 = self._get_goals_for(away_team, season_context, game_date, 5)
        away_gf_l20 = self._get_goals_for(away_team, season_context, game_date, 20)
        
        home_scoring_accel = (home_gf_l5 - home_gf_l20) / 3  # Normalize
        away_scoring_accel = (away_gf_l5 - away_gf_l20) / 3
        
        features['scoring_acceleration_differential'] = home_scoring_accel - away_scoring_accel
        
        # 6. DEFENSIVE TREND
        home_ga_l5 = self._get_goals_against(home_team, season_context, game_date, 5)
        home_ga_l20 = self._get_goals_against(home_team, season_context, game_date, 20)
        away_ga_l5 = self._get_goals_against(away_team, season_context, game_date, 5)
        away_ga_l20 = self._get_goals_against(away_team, season_context, game_date, 20)
        
        home_defense_improvement = (home_ga_l20 - home_ga_l5) / 3  # Lower is better
        away_defense_improvement = (away_ga_l20 - away_ga_l5) / 3
        
        features['defensive_trend_differential'] = home_defense_improvement - away_defense_improvement
        
        # 7. FORM TRAJECTORY
        # First 15 games vs last 15 games (season-long improvement)
        home_trajectory = self._calculate_trajectory(home_team, season_context, game_date)
        away_trajectory = self._calculate_trajectory(away_team, season_context, game_date)
        
        features['trajectory_differential'] = home_trajectory - away_trajectory
        
        # 8. DESPERATION INDEX
        # Combines playoff position + games remaining + recent losses
        home_desp = self._calculate_desperation(home_team, season_context, game_date, home_position, games_remaining_home)
        away_desp = self._calculate_desperation(away_team, season_context, game_date, away_position, games_remaining_away)
        
        features['desperation_differential'] = home_desp - away_desp
        
        # 9. REST ADVANTAGE INTERACTION
        # Rest advantage matters more for older/injured teams
        home_rest = game.get('temporal_context', {}).get('home_rest_days', 1)
        away_rest = game.get('temporal_context', {}).get('away_rest_days', 1)
        rest_gap = home_rest - away_rest
        
        # Weight rest by recent games played (fatigue accumulation)
        home_recent_games = min(home_games, 10)
        away_recent_games = min(away_games, 10)
        fatigue_factor = (away_recent_games - home_recent_games) / 10
        
        features['rest_fatigue_interaction'] = rest_gap * (1 + fatigue_factor)
        
        # 10. SITUATIONAL MOTIVATION
        # Division games, playoff implications, rivalry
        is_division_game = self._is_division_game(home_team, away_team)
        playoff_implications = (home_push > 0.5 or away_push > 0.5)
        
        features['situational_motivation'] = float(is_division_game and playoff_implications)
        
        return features
    
    def _get_zero_template(self) -> Dict[str, float]:
        """Return zero-filled template"""
        return {
            'expectation_differential': 0.0,
            'playoff_push_differential': 0.0,
            'form_acceleration_differential': 0.0,
            'venue_momentum_gap': 0.0,
            'scoring_acceleration_differential': 0.0,
            'defensive_trend_differential': 0.0,
            'trajectory_differential': 0.0,
            'desperation_differential': 0.0,
            'rest_fatigue_interaction': 0.0,
            'situational_motivation': 0.0,
        }
    
    def _load_preseason_expectations(self) -> Dict[str, float]:
        """Load preseason win total expectations"""
        # Placeholder: use historical averages
        # In production, scrape from betting markets
        return {
            'TBL': 50, 'BOS': 48, 'COL': 48, 'CAR': 47, 'EDM': 47,
            'VGK': 46, 'DAL': 46, 'TOR': 45, 'NYR': 45, 'FLA': 44,
            'MIN': 43, 'WPG': 43, 'LAK': 42, 'NJD': 42, 'SEA': 41,
            'CGY': 40, 'NSH': 40, 'PIT': 40, 'VAN': 39, 'STL': 38,
            'DET': 37, 'OTT': 37, 'BUF': 36, 'NYI': 36, 'PHI': 35,
            'WSH': 35, 'ARI': 33, 'MTL': 32, 'CBJ': 31, 'SJS': 28,
            'ANA': 27, 'CHI': 26,
        }
    
    def _count_wins(self, team: str, season_data: List[Dict], before_date: str) -> int:
        """Count wins before date"""
        wins = 0
        for g in season_data:
            if g.get('date', '') >= before_date:
                continue
            if g.get('home_team') == team and g.get('home_won'):
                wins += 1
            elif g.get('away_team') == team and not g.get('home_won'):
                wins += 1
        return wins
    
    def _count_games(self, team: str, season_data: List[Dict], before_date: str) -> int:
        """Count games before date"""
        count = 0
        for g in season_data:
            if g.get('date', '') >= before_date:
                continue
            if g.get('home_team') == team or g.get('away_team') == team:
                count += 1
        return count
    
    def _count_wins_window(self, team: str, season_data: List[Dict], before_date: str, window: int) -> Optional[int]:
        """Count wins in last N games"""
        games = [g for g in season_data if g.get('date', '') < before_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not games:
            return None
        
        wins = sum(1 for g in games if 
                   (g.get('home_team') == team and g.get('home_won')) or
                   (g.get('away_team') == team and not g.get('home_won')))
        return wins
    
    def _get_playoff_position(self, team: str, season_data: List[Dict], game_date: str) -> int:
        """Get playoff position (1-16 = in, 17-32 = out)"""
        standings = {}
        for g in season_data:
            if g.get('date', '') >= game_date:
                continue
            for t in [g.get('home_team'), g.get('away_team')]:
                if t not in standings:
                    standings[t] = {'wins': 0, 'games': 0}
                standings[t]['games'] += 1
                if (t == g.get('home_team') and g.get('home_won')) or \
                   (t == g.get('away_team') and not g.get('home_won')):
                    standings[t]['wins'] += 1
        
        ranked = sorted(standings.items(), key=lambda x: x[1]['wins'] / max(x[1]['games'], 1), reverse=True)
        for i, (t, _) in enumerate(ranked):
            if t == team:
                return i + 1
        return 16
    
    def _calculate_playoff_push(self, position: int, games_remaining: int) -> float:
        """Calculate playoff push intensity"""
        if position <= 8:
            return 0.2  # Safely in
        elif position >= 20:
            return 0.1  # Out of contention
        else:
            bubble_factor = (20 - position) / 12
            urgency_factor = 1 - (games_remaining / 82)
            return 0.5 + 0.5 * bubble_factor * (1 + urgency_factor)
    
    def _get_home_win_pct(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get home win % in last N home games"""
        home_games = [g for g in season_data if g.get('date', '') < game_date and g.get('home_team') == team]
        home_games = sorted(home_games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not home_games:
            return 0.5
        
        wins = sum(1 for g in home_games if g.get('home_won'))
        return wins / len(home_games)
    
    def _get_away_win_pct(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get away win % in last N away games"""
        away_games = [g for g in season_data if g.get('date', '') < game_date and g.get('away_team') == team]
        away_games = sorted(away_games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not away_games:
            return 0.5
        
        wins = sum(1 for g in away_games if not g.get('home_won'))
        return wins / len(away_games)
    
    def _get_goals_for(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get average goals scored in last N games"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not games:
            return 3.0
        
        goals = []
        for g in games:
            if g.get('home_team') == team:
                goals.append(g.get('home_score', 0))
            else:
                goals.append(g.get('away_score', 0))
        
        return np.mean(goals)
    
    def _get_goals_against(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get average goals allowed in last N games"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not games:
            return 3.0
        
        goals = []
        for g in games:
            if g.get('home_team') == team:
                goals.append(g.get('away_score', 0))
            else:
                goals.append(g.get('home_score', 0))
        
        return np.mean(goals)
    
    def _calculate_trajectory(self, team: str, season_data: List[Dict], game_date: str) -> float:
        """Calculate trajectory (first 15 vs last 15 games)"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''))
        
        if len(games) < 30:
            return 0.0
        
        first_15 = games[:15]
        last_15 = games[-15:]
        
        first_wins = sum(1 for g in first_15 if 
                        (g.get('home_team') == team and g.get('home_won')) or
                        (g.get('away_team') == team and not g.get('home_won')))
        last_wins = sum(1 for g in last_15 if 
                       (g.get('home_team') == team and g.get('home_won')) or
                       (g.get('away_team') == team and not g.get('home_won')))
        
        return (last_wins - first_wins) / 15
    
    def _calculate_desperation(self, team: str, season_data: List[Dict], game_date: str,
                               playoff_position: int, games_remaining: int) -> float:
        """Calculate desperation index"""
        if 8 < playoff_position <= 12:
            playoff_urgency = 1.0
        elif playoff_position <= 8:
            playoff_urgency = 0.3
        else:
            playoff_urgency = 0.0
        
        time_urgency = 1 - (games_remaining / 82)
        
        recent_wins = self._count_wins_window(team, season_data, game_date, 5)
        recent_losses = 5 - recent_wins if recent_wins is not None else 2.5
        loss_urgency = recent_losses / 5
        
        return (playoff_urgency + time_urgency + loss_urgency) / 3
    
    def _is_division_game(self, home_team: str, away_team: str) -> bool:
        """Check if division game (would need division mappings)"""
        # Placeholder
        return False


def extract_focused_temporal_batch(games: List[Dict], season_context: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract focused temporal features for batch of games.
    
    Parameters
    ----------
    games : list of dict
        Games to extract features for
    season_context : list of dict, optional
        Season data for context
    
    Returns
    -------
    features : list of dict
        10 focused temporal features per game
    """
    extractor = FocusedTemporalExtractor()
    return [extractor.extract_features(g, season_context) for g in games]

