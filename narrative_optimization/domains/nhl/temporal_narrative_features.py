"""
NHL Temporal Narrative Features - Three-Scale Framework

Extracts temporal dynamics at three critical scales:

1. MACRO-TEMPORAL (Season-Long Narratives)
   - Playoff push momentum (teams fighting for position)
   - Underdog season arcs (worst-to-first trajectories)
   - Post-trade deadline effects
   - Coach change impacts
   - Injury comeback narratives

2. MESO-TEMPORAL (Recent Form Patterns)
   - Hot/cold streaks (5, 10, 20 game windows)
   - Home/away splits over recent stretch
   - Divisional rivalry momentum
   - Goalie rotation patterns
   - Power play/penalty kill trends

3. MICRO-TEMPORAL (In-Game Dynamics)
   - Period-by-period momentum
   - Comeback patterns (trailing after 1st/2nd)
   - Lead protection (prevent defense)
   - Empty net tendencies
   - Overtime/shootout history

This framework serves as the TEMPLATE for all sports domains.

Author: Temporal Narrative System
Date: November 19, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class NHLTemporalExtractor:
    """Extract multi-scale temporal narrative features for NHL"""
    
    def __init__(self):
        """Initialize temporal extractor"""
        self.season_cache = {}  # Cache season-long stats
        self.streak_cache = {}  # Cache streak data
        
    def extract_all_temporal_features(self, game: Dict, season_data: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Extract all temporal features at three scales.
        
        Parameters
        ----------
        game : dict
            Current game data
        season_data : list of dict, optional
            All games in season up to this point (for macro-temporal features)
        
        Returns
        -------
        features : dict
            Complete temporal feature set (~50 features)
        """
        features = {}
        
        # Extract at each scale
        features.update(self.extract_macro_temporal(game, season_data))
        features.update(self.extract_meso_temporal(game, season_data))
        features.update(self.extract_micro_temporal(game))
        
        return features
    
    # ========================================================================
    # MACRO-TEMPORAL: Season-Long Narratives
    # ========================================================================
    
    def extract_macro_temporal(self, game: Dict, season_data: Optional[List[Dict]]) -> Dict[str, float]:
        """
        Extract season-long narrative features.
        
        These capture the "story of the season" - playoff pushes, underdog runs,
        post-trade momentum, etc. Critical for understanding long-term betting value.
        """
        features = {}
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        game_date = game.get('date', '')
        
        if not season_data:
            # Return zeros if no season context available
            return self._get_macro_template()
        
        # Calculate games remaining in season (82-game season)
        games_played_home = self._count_team_games(home_team, season_data, before_date=game_date)
        games_played_away = self._count_team_games(away_team, season_data, before_date=game_date)
        
        games_remaining_home = 82 - games_played_home
        games_remaining_away = 82 - games_played_away
        
        # 1. PLAYOFF PUSH INTENSITY
        # Teams fighting for playoff spots have elevated motivation
        home_playoff_position = self._get_playoff_position(home_team, season_data, game_date)
        away_playoff_position = self._get_playoff_position(away_team, season_data, game_date)
        
        features['home_playoff_push'] = self._calculate_playoff_push_intensity(
            home_playoff_position, games_remaining_home
        )
        features['away_playoff_push'] = self._calculate_playoff_push_intensity(
            away_playoff_position, games_remaining_away
        )
        features['playoff_push_differential'] = features['home_playoff_push'] - features['away_playoff_push']
        
        # 2. UNDERDOG SEASON ARC
        # Teams exceeding expectations have narrative momentum
        home_expected_wins = self._get_preseason_expectation(home_team)
        away_expected_wins = self._get_preseason_expectation(away_team)
        
        home_actual_wins = self._count_wins(home_team, season_data, before_date=game_date)
        away_actual_wins = self._count_wins(away_team, season_data, before_date=game_date)
        
        features['home_vs_expectation'] = (home_actual_wins - home_expected_wins * (games_played_home / 82)) / 10
        features['away_vs_expectation'] = (away_actual_wins - away_expected_wins * (games_played_away / 82)) / 10
        features['expectation_differential'] = features['home_vs_expectation'] - features['away_vs_expectation']
        
        # 3. POST-TRADE DEADLINE EFFECT
        # Teams that made moves have changed dynamics
        trade_deadline = self._get_trade_deadline_date(game_date)
        days_since_deadline = (self._parse_date(game_date) - trade_deadline).days
        
        features['post_trade_deadline'] = float(days_since_deadline > 0 and days_since_deadline < 30)
        features['days_since_trade_deadline'] = max(0, min(days_since_deadline, 30)) / 30
        
        # 4. COACH CHANGE MOMENTUM
        # New coach "bounce" effect
        features['home_new_coach_games'] = self._games_since_coach_change(home_team, season_data, game_date)
        features['away_new_coach_games'] = self._games_since_coach_change(away_team, season_data, game_date)
        features['home_new_coach_bounce'] = float(0 < features['home_new_coach_games'] <= 10)
        features['away_new_coach_bounce'] = float(0 < features['away_new_coach_games'] <= 10)
        
        # 5. SEASON TRAJECTORY
        # Improving vs declining teams (30-game rolling win %)
        features['home_trajectory'] = self._calculate_trajectory(home_team, season_data, game_date)
        features['away_trajectory'] = self._calculate_trajectory(away_team, season_data, game_date)
        features['trajectory_differential'] = features['home_trajectory'] - features['away_trajectory']
        
        # 6. DESPERATION INDEX
        # Combines playoff push + games remaining + recent losses
        features['home_desperation'] = self._calculate_desperation(
            home_team, season_data, game_date, home_playoff_position, games_remaining_home
        )
        features['away_desperation'] = self._calculate_desperation(
            away_team, season_data, game_date, away_playoff_position, games_remaining_away
        )
        features['desperation_differential'] = features['home_desperation'] - features['away_desperation']
        
        return features
    
    # ========================================================================
    # MESO-TEMPORAL: Recent Form Patterns
    # ========================================================================
    
    def extract_meso_temporal(self, game: Dict, season_data: Optional[List[Dict]]) -> Dict[str, float]:
        """
        Extract recent form features (last 5, 10, 20 games).
        
        These capture momentum, streaks, and pattern shifts that markets
        are slow to incorporate.
        """
        features = {}
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        game_date = game.get('date', '')
        
        if not season_data:
            return self._get_meso_template()
        
        # 1. MULTI-WINDOW STREAKS
        # Different time windows reveal different patterns
        for window in [5, 10, 20]:
            home_wins = self._count_wins_in_window(home_team, season_data, game_date, window)
            away_wins = self._count_wins_in_window(away_team, season_data, game_date, window)
            
            features[f'home_l{window}_wins'] = home_wins
            features[f'away_l{window}_wins'] = away_wins
            features[f'l{window}_win_differential'] = home_wins - away_wins
        
        # 2. HOME/AWAY SPLITS (Recent)
        # Some teams are dramatically better at home
        home_home_record = self._get_home_record_recent(home_team, season_data, game_date, window=10)
        away_away_record = self._get_away_record_recent(away_team, season_data, game_date, window=10)
        
        features['home_home_win_pct_l10'] = home_home_record
        features['away_away_win_pct_l10'] = away_away_record
        features['venue_advantage_recent'] = home_home_record - away_away_record
        
        # 3. DIVISIONAL PERFORMANCE
        # Rivalry games have different dynamics
        features['home_divisional_record'] = self._get_divisional_record(home_team, season_data, game_date)
        features['away_divisional_record'] = self._get_divisional_record(away_team, season_data, game_date)
        
        # 4. SCORING TRENDS
        # Recent offensive/defensive performance
        features['home_goals_per_game_l10'] = self._get_avg_goals_for(home_team, season_data, game_date, 10)
        features['away_goals_per_game_l10'] = self._get_avg_goals_for(away_team, season_data, game_date, 10)
        features['home_goals_against_l10'] = self._get_avg_goals_against(home_team, season_data, game_date, 10)
        features['away_goals_against_l10'] = self._get_avg_goals_against(away_team, season_data, game_date, 10)
        
        # 5. SPECIAL TEAMS MOMENTUM
        # Power play/penalty kill trends
        features['home_pp_success_l10'] = self._get_powerplay_pct(home_team, season_data, game_date, 10)
        features['away_pp_success_l10'] = self._get_powerplay_pct(away_team, season_data, game_date, 10)
        
        # 6. GOALIE ROTATION PATTERN
        # Starter vs backup usage
        features['home_goalie_games_l5'] = self._get_goalie_recent_starts(
            game.get('home_goalie', ''), home_team, season_data, game_date, 5
        )
        features['away_goalie_games_l5'] = self._get_goalie_recent_starts(
            game.get('away_goalie', ''), away_team, season_data, game_date, 5
        )
        
        return features
    
    # ========================================================================
    # MICRO-TEMPORAL: In-Game Dynamics (for Live Betting)
    # ========================================================================
    
    def extract_micro_temporal(self, game: Dict) -> Dict[str, float]:
        """
        Extract in-game temporal features.
        
        These are for LIVE BETTING - period-by-period momentum,
        comeback patterns, etc. Requires checkpoint data.
        """
        features = {}
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        # Check if we have checkpoint (in-game) data
        checkpoints = game.get('checkpoints', [])
        
        if not checkpoints:
            # Pre-game: Use historical tendencies
            features.update(self._get_historical_ingame_tendencies(home_team, away_team))
        else:
            # Live game: Calculate real-time momentum
            features.update(self._calculate_live_momentum(checkpoints, home_team, away_team))
        
        return features
    
    def _get_historical_ingame_tendencies(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Get historical in-game performance patterns"""
        # These would come from historical database
        # For now, return template
        return {
            'home_comeback_rate': 0.5,  # % of games where team came back from deficit
            'away_comeback_rate': 0.5,
            'home_lead_protection_rate': 0.7,  # % of leads held
            'away_lead_protection_rate': 0.7,
            'home_ot_win_rate': 0.5,
            'away_ot_win_rate': 0.5,
            'home_avg_1st_period_goals': 1.0,
            'away_avg_1st_period_goals': 1.0,
            'home_3rd_period_strength': 0.5,  # Relative performance in 3rd
            'away_3rd_period_strength': 0.5,
        }
    
    def _calculate_live_momentum(self, checkpoints: List[Dict], home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate real-time momentum from checkpoint data"""
        features = {}
        
        if not checkpoints:
            return self._get_historical_ingame_tendencies(home_team, away_team)
        
        latest = checkpoints[-1]
        
        # Current score situation
        home_score = latest.get('home_score', 0)
        away_score = latest.get('away_score', 0)
        period = latest.get('period', 1)
        
        features['current_score_diff'] = home_score - away_score
        features['current_period'] = period / 3  # Normalize
        features['home_leading'] = float(home_score > away_score)
        features['away_leading'] = float(away_score > home_score)
        features['tied'] = float(home_score == away_score)
        
        # Momentum calculation (scoring rate in recent periods)
        if len(checkpoints) >= 2:
            prev = checkpoints[-2]
            home_goals_recent = latest.get('home_score', 0) - prev.get('home_score', 0)
            away_goals_recent = latest.get('away_score', 0) - prev.get('away_score', 0)
            
            features['home_recent_momentum'] = home_goals_recent / 3  # Normalize
            features['away_recent_momentum'] = away_goals_recent / 3
            features['momentum_shift'] = features['home_recent_momentum'] - features['away_recent_momentum']
        
        # Comeback narrative
        if len(checkpoints) >= 2:
            was_trailing = checkpoints[0].get('home_score', 0) < checkpoints[0].get('away_score', 0)
            now_leading = home_score > away_score
            features['home_comeback_active'] = float(was_trailing and now_leading)
        
        return features
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    
    def _get_macro_template(self) -> Dict[str, float]:
        """Return zero-filled macro-temporal template"""
        return {
            'home_playoff_push': 0.0, 'away_playoff_push': 0.0, 'playoff_push_differential': 0.0,
            'home_vs_expectation': 0.0, 'away_vs_expectation': 0.0, 'expectation_differential': 0.0,
            'post_trade_deadline': 0.0, 'days_since_trade_deadline': 0.0,
            'home_new_coach_games': 0.0, 'away_new_coach_games': 0.0,
            'home_new_coach_bounce': 0.0, 'away_new_coach_bounce': 0.0,
            'home_trajectory': 0.0, 'away_trajectory': 0.0, 'trajectory_differential': 0.0,
            'home_desperation': 0.0, 'away_desperation': 0.0, 'desperation_differential': 0.0,
        }
    
    def _get_meso_template(self) -> Dict[str, float]:
        """Return zero-filled meso-temporal template"""
        features = {}
        for window in [5, 10, 20]:
            features[f'home_l{window}_wins'] = 0.0
            features[f'away_l{window}_wins'] = 0.0
            features[f'l{window}_win_differential'] = 0.0
        
        features.update({
            'home_home_win_pct_l10': 0.5, 'away_away_win_pct_l10': 0.5, 'venue_advantage_recent': 0.0,
            'home_divisional_record': 0.5, 'away_divisional_record': 0.5,
            'home_goals_per_game_l10': 3.0, 'away_goals_per_game_l10': 3.0,
            'home_goals_against_l10': 3.0, 'away_goals_against_l10': 3.0,
            'home_pp_success_l10': 0.2, 'away_pp_success_l10': 0.2,
            'home_goalie_games_l5': 3.0, 'away_goalie_games_l5': 3.0,
        })
        return features
    
    def _count_team_games(self, team: str, season_data: List[Dict], before_date: str) -> int:
        """Count games played by team before given date"""
        count = 0
        for game in season_data:
            if game.get('date', '') >= before_date:
                continue
            if game.get('home_team') == team or game.get('away_team') == team:
                count += 1
        return count
    
    def _count_wins(self, team: str, season_data: List[Dict], before_date: str) -> int:
        """Count wins by team before given date"""
        wins = 0
        for game in season_data:
            if game.get('date', '') >= before_date:
                continue
            if game.get('home_team') == team and game.get('home_won'):
                wins += 1
            elif game.get('away_team') == team and not game.get('home_won'):
                wins += 1
        return wins
    
    def _count_wins_in_window(self, team: str, season_data: List[Dict], before_date: str, window: int) -> int:
        """Count wins in last N games"""
        games = []
        for game in season_data:
            if game.get('date', '') >= before_date:
                continue
            if game.get('home_team') == team or game.get('away_team') == team:
                games.append(game)
        
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        wins = 0
        for game in games:
            if game.get('home_team') == team and game.get('home_won'):
                wins += 1
            elif game.get('away_team') == team and not game.get('home_won'):
                wins += 1
        return wins
    
    def _get_playoff_position(self, team: str, season_data: List[Dict], game_date: str) -> int:
        """
        Get team's playoff position (1-16 = in, 17-32 = out).
        Simplified: just use win % ranking for now.
        """
        # Calculate standings
        standings = {}
        for game in season_data:
            if game.get('date', '') >= game_date:
                continue
            home = game.get('home_team')
            away = game.get('away_team')
            
            if home not in standings:
                standings[home] = {'wins': 0, 'games': 0}
            if away not in standings:
                standings[away] = {'wins': 0, 'games': 0}
            
            standings[home]['games'] += 1
            standings[away]['games'] += 1
            
            if game.get('home_won'):
                standings[home]['wins'] += 1
            else:
                standings[away]['wins'] += 1
        
        # Rank by win %
        ranked = sorted(
            standings.items(),
            key=lambda x: x[1]['wins'] / max(x[1]['games'], 1),
            reverse=True
        )
        
        for i, (t, _) in enumerate(ranked):
            if t == team:
                return i + 1
        return 16  # Default to bubble
    
    def _calculate_playoff_push_intensity(self, position: int, games_remaining: int) -> float:
        """
        Calculate playoff push intensity (0-1).
        Highest for teams on playoff bubble with few games left.
        """
        if position <= 8:
            # Safely in playoffs
            return 0.2
        elif position >= 20:
            # Out of contention
            return 0.1
        else:
            # On the bubble - intensity increases as games run out
            bubble_factor = (20 - position) / 12  # 0-1 scale
            urgency_factor = 1 - (games_remaining / 82)  # Higher as season ends
            return 0.5 + 0.5 * bubble_factor * (1 + urgency_factor)
    
    def _get_preseason_expectation(self, team: str) -> float:
        """Get preseason expected wins (would come from odds/projections)"""
        # Placeholder: return league average
        return 41.0
    
    def _get_trade_deadline_date(self, game_date: str) -> datetime:
        """Get trade deadline for season"""
        # NHL trade deadline is typically early March
        year = int(game_date[:4])
        return datetime(year, 3, 3)
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    
    def _games_since_coach_change(self, team: str, season_data: List[Dict], game_date: str) -> float:
        """Games since coach change (would need coach change database)"""
        # Placeholder
        return 100.0  # No recent change
    
    def _calculate_trajectory(self, team: str, season_data: List[Dict], game_date: str) -> float:
        """
        Calculate team trajectory (improving vs declining).
        Compare first 15 games to last 15 games win %.
        """
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
        
        return (last_wins - first_wins) / 15  # -1 to +1 scale
    
    def _calculate_desperation(self, team: str, season_data: List[Dict], game_date: str,
                               playoff_position: int, games_remaining: int) -> float:
        """
        Calculate desperation index combining multiple factors.
        High desperation = playoff bubble + few games left + recent losses.
        """
        # Playoff urgency
        if 8 < playoff_position <= 12:
            playoff_urgency = 1.0
        elif playoff_position <= 8:
            playoff_urgency = 0.3
        else:
            playoff_urgency = 0.0
        
        # Time urgency
        time_urgency = 1 - (games_remaining / 82)
        
        # Recent performance (losses in last 5)
        recent_losses = 5 - self._count_wins_in_window(team, season_data, game_date, 5)
        loss_urgency = recent_losses / 5
        
        return (playoff_urgency + time_urgency + loss_urgency) / 3
    
    def _get_home_record_recent(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get home win % in last N home games"""
        home_games = [g for g in season_data if g.get('date', '') < game_date and g.get('home_team') == team]
        home_games = sorted(home_games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not home_games:
            return 0.5
        
        wins = sum(1 for g in home_games if g.get('home_won'))
        return wins / len(home_games)
    
    def _get_away_record_recent(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get away win % in last N away games"""
        away_games = [g for g in season_data if g.get('date', '') < game_date and g.get('away_team') == team]
        away_games = sorted(away_games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not away_games:
            return 0.5
        
        wins = sum(1 for g in away_games if not g.get('home_won'))
        return wins / len(away_games)
    
    def _get_divisional_record(self, team: str, season_data: List[Dict], game_date: str) -> float:
        """Get divisional win % (would need division data)"""
        # Placeholder
        return 0.5
    
    def _get_avg_goals_for(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get average goals scored in last N games"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not games:
            return 3.0
        
        total_goals = 0
        for g in games:
            if g.get('home_team') == team:
                total_goals += g.get('home_score', 0)
            else:
                total_goals += g.get('away_score', 0)
        
        return total_goals / len(games)
    
    def _get_avg_goals_against(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get average goals allowed in last N games"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        if not games:
            return 3.0
        
        total_goals = 0
        for g in games:
            if g.get('home_team') == team:
                total_goals += g.get('away_score', 0)
            else:
                total_goals += g.get('home_score', 0)
        
        return total_goals / len(games)
    
    def _get_powerplay_pct(self, team: str, season_data: List[Dict], game_date: str, window: int) -> float:
        """Get power play success % in last N games (would need PP data)"""
        # Placeholder
        return 0.2
    
    def _get_goalie_recent_starts(self, goalie: str, team: str, season_data: List[Dict], 
                                  game_date: str, window: int) -> float:
        """Count goalie starts in last N team games"""
        games = [g for g in season_data if g.get('date', '') < game_date and 
                 (g.get('home_team') == team or g.get('away_team') == team)]
        games = sorted(games, key=lambda g: g.get('date', ''), reverse=True)[:window]
        
        starts = 0
        for g in games:
            if g.get('home_team') == team and g.get('home_goalie') == goalie:
                starts += 1
            elif g.get('away_team') == team and g.get('away_goalie') == goalie:
                starts += 1
        
        return starts


def extract_temporal_features_batch(games: List[Dict], season_data: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract temporal features for a batch of games.
    
    Parameters
    ----------
    games : list of dict
        Games to extract features for
    season_data : list of dict, optional
        Full season data for context
    
    Returns
    -------
    features : list of dict
        Temporal features for each game
    """
    extractor = NHLTemporalExtractor()
    return [extractor.extract_all_temporal_features(g, season_data) for g in games]

