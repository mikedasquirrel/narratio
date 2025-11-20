"""
NHL Player Performance Transformer

Extracts narrative features from NHL player statistics for prop betting.

Critical insight: Player narratives drive prop outcomes. A player on a hot streak
isn't just statistically likely to score - they carry narrative momentum that 
influences how the game unfolds. Star players in rivalry games, milestone 
chasers, and redemption stories all create exploitable prop betting edges.

Features extracted (35 total):
- Star power narrative (5): Name recognition, draft position, salary tier
- Performance momentum (8): Hot/cold streaks, recent form, scoring bursts  
- Matchup narrative (6): vs opponent history, revenge games, dominance
- Contextual amplifiers (6): Home ice, nationally televised, playoffs
- Position-specific (5): Role on team, PP usage, ice time trends
- Milestone narratives (5): Career milestones, season goals, streaks

Author: Prop Betting Narrative System
Date: November 20, 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta


class NHLPlayerPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    Extract narrative features from NHL player performance for props.
    
    Philosophy:
    - Player names carry narrative weight (McDavid vs 4th liner)
    - Hot streaks create self-fulfilling prophecies
    - Matchup histories generate storylines
    - Context amplifies or dampens performance
    - Milestones drive player motivation
    
    Accepts:
    - Dict with player stats and game logs
    - DataFrame with player columns
    - List of player game dictionaries
    
    Returns:
    - ndarray of shape (n_players, 35) narrative features
    """
    
    def __init__(
        self,
        include_goalie_features: bool = True,
        normalize: bool = True,
        milestone_awareness: bool = True
    ):
        """
        Parameters
        ----------
        include_goalie_features : bool
            Include goalie-specific prop features
        normalize : bool
            Normalize features to 0-1 range
        milestone_awareness : bool
            Track milestone narratives (99 goals, 500 points, etc)
        """
        self.include_goalie_features = include_goalie_features
        self.normalize = normalize
        self.milestone_awareness = milestone_awareness
        
        # Star player names (narrative weight)
        self.star_players = {
            # Generational talents
            'Connor McDavid': 1.0,
            'Auston Matthews': 0.95,
            'Nathan MacKinnon': 0.93,
            'Nikita Kucherov': 0.91,
            'David Pastrnak': 0.90,
            'Leon Draisaitl': 0.89,
            'Mikko Rantanen': 0.88,
            'Mitch Marner': 0.87,
            'Sidney Crosby': 0.86,  # Aging but still elite narrative
            'Alex Ovechkin': 0.85,  # Goal record chase
            
            # Elite scorers
            'Jack Hughes': 0.84,
            'Kirill Kaprizov': 0.83,
            'Matthew Tkachuk': 0.82,
            'Jason Robertson': 0.81,
            'Cale Makar': 0.80,
            'Adam Fox': 0.79,
            'Mika Zibanejad': 0.78,
            'Artemi Panarin': 0.77,
            'Patrick Kane': 0.76,
            'Steven Stamkos': 0.75,
            
            # Rising stars
            'Tim Stützle': 0.74,
            'Trevor Zegras': 0.73,
            'Cole Caufield': 0.72,
            'Jack Eichel': 0.71,
            'Elias Pettersson': 0.70,
        }
        
        # Position importance for props
        self.position_weights = {
            'C': 1.0,   # Centers involved in all plays
            'RW': 0.9,  # Wingers score goals
            'LW': 0.9,
            'D': 0.7,   # Defensemen less likely for goal props
            'G': 0.5,   # Goalies only for save props
        }
        
    def fit(self, X, y=None):
        """Fit transformer (learns normalization parameters)"""
        return self
        
    def transform(self, X) -> np.ndarray:
        """
        Transform player data to narrative features.
        
        Parameters
        ----------
        X : list of dicts or DataFrame
            Player statistics and context
            
        Returns
        -------
        features : ndarray of shape (n_samples, 35)
            Player narrative features for props
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_dict('records')
        elif not isinstance(X, list):
            X = [X]
            
        features = []
        for player_data in X:
            feat = self._extract_player_features(player_data)
            features.append(feat)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_player_features(self, data: Dict) -> List[float]:
        """Extract all narrative features for a player"""
        features = []
        
        # === STAR POWER NARRATIVE (5 features) ===
        
        # 1. Name recognition value
        player_name = data.get('player_name', '')
        star_value = self.star_players.get(player_name, 0.3)  # Default for non-stars
        features.append(star_value)
        
        # 2. Position value for props
        position = data.get('position', 'F')
        pos_weight = self.position_weights.get(position, 0.8)
        features.append(pos_weight)
        
        # 3. Season point production (star indicator)
        season_ppg = data.get('season_stats', {}).get('points', 0) / max(
            data.get('season_stats', {}).get('games_played', 1), 1
        )
        # Normalize: 0.5 PPG = average, 1.2+ = elite
        star_production = min(season_ppg / 1.2, 1.0)
        features.append(star_production)
        
        # 4. Ice time indicator (usage = opportunity)
        toi_str = data.get('season_stats', {}).get('toi_per_game', '0:00')
        toi_minutes = self._parse_toi(toi_str)
        # Normalize: 20+ minutes = high usage
        usage_rate = min(toi_minutes / 20.0, 1.0)
        features.append(usage_rate)
        
        # 5. Power play involvement
        pp_points = data.get('season_stats', {}).get('powerplay_points', 0)
        pp_involvement = min(pp_points / 25.0, 1.0)  # 25+ PP points = elite
        features.append(pp_involvement)
        
        # === PERFORMANCE MOMENTUM (8 features) ===
        
        form = data.get('recent_form', {})
        
        # 6. Recent scoring surge (last 5 vs last 10)
        l5_goals = form.get('last_5_avg_goals', 0)
        season_gpg = data.get('season_stats', {}).get('goals_per_game', 0)
        hot_scoring = l5_goals / max(season_gpg, 0.2) if season_gpg > 0 else 1.0
        features.append(min(hot_scoring, 2.0) / 2.0)  # Cap at 2x and normalize
        
        # 7. Point streak indicator
        point_streak = form.get('point_streak_games', 0)
        streak_momentum = min(point_streak / 5.0, 1.0)  # 5+ game streak = max
        features.append(streak_momentum)
        
        # 8. Multi-point game frequency
        multi_point_l10 = form.get('multi_point_games_l10', 0)
        explosion_rate = multi_point_l10 / 10.0
        features.append(explosion_rate)
        
        # 9. Shot volume trend
        l5_shots = form.get('last_5_avg_shots', 0)
        season_spg = data.get('season_stats', {}).get('shots_per_game', 0)
        shot_surge = l5_shots / max(season_spg, 2.0) if season_spg > 0 else 1.0
        features.append(min(shot_surge, 1.5) / 1.5)
        
        # 10. Goals last 5 games (raw production)
        goals_l5 = form.get('goals_last_5', 0)
        recent_scoring = min(goals_l5 / 5.0, 1.0)  # 5 goals in 5 games = max
        features.append(recent_scoring)
        
        # 11. Trend direction
        trend_map = {'hot': 1.0, 'neutral': 0.5, 'cold': 0.0}
        trend = form.get('trend', 'neutral')
        features.append(trend_map.get(trend, 0.5))
        
        # 12. Consistency (games with points / games played)
        last_5_games = data.get('last_5_games', [])
        if last_5_games:
            games_with_points = sum(1 for g in last_5_games if g.get('points', 0) > 0)
            consistency = games_with_points / len(last_5_games)
        else:
            consistency = 0.3  # Default
        features.append(consistency)
        
        # 13. Shooting percentage trend (hot/cold finishing)
        recent_games = data.get('last_5_games', [])
        if recent_games:
            recent_goals = sum(g.get('goals', 0) for g in recent_games)
            recent_shots = sum(g.get('shots', 0) for g in recent_games)
            recent_sh_pct = recent_goals / max(recent_shots, 1)
            season_sh_pct = data.get('season_stats', {}).get('shooting_pct', 0.10)
            
            # Hot shooting = 1.5x season average
            shooting_heat = recent_sh_pct / max(season_sh_pct, 0.05)
            features.append(min(shooting_heat, 2.0) / 2.0)
        else:
            features.append(0.5)
            
        # === MATCHUP NARRATIVE (6 features) ===
        
        vs_stats = data.get('vs_opponent', {})
        
        # 14. Historical dominance vs team
        vs_gpg = vs_stats.get('avg_goals', 0)
        season_gpg = data.get('season_stats', {}).get('goals_per_game', 0)
        matchup_edge = vs_gpg / max(season_gpg, 0.2) if season_gpg > 0 else 1.0
        features.append(min(matchup_edge, 2.0) / 2.0)
        
        # 15. Recent success vs team (last 5 meetings)
        l5_goals_vs = vs_stats.get('last_5_goals', 0)
        l5_points_vs = vs_stats.get('last_5_points', 0)
        recent_dominance = min(l5_points_vs / 10.0, 1.0)  # 10 points in 5 games = max
        features.append(recent_dominance)
        
        # 16. Games played vs team (familiarity)
        games_vs = vs_stats.get('games_played', 0)
        familiarity = min(games_vs / 10.0, 1.0)  # 10+ games = max familiarity
        features.append(familiarity)
        
        # 17. Revenge game indicator
        last_vs_result = self._check_revenge_game(data)
        features.append(last_vs_result)
        
        # 18. Division rival multiplier
        is_division_rival = data.get('is_division_rival', False)
        features.append(1.0 if is_division_rival else 0.0)
        
        # 19. Playoff history
        playoff_history = data.get('playoff_history_vs_opponent', False)
        features.append(1.0 if playoff_history else 0.0)
        
        # === CONTEXTUAL AMPLIFIERS (6 features) ===
        
        # 20. Home/away performance split
        is_home = data.get('is_home_game', True)
        home_ppg = self._get_home_away_split(data, 'home')
        away_ppg = self._get_home_away_split(data, 'away')
        
        if is_home:
            location_boost = home_ppg / max(away_ppg, 0.1)
        else:
            location_boost = away_ppg / max(home_ppg, 0.1)
            
        features.append(min(location_boost, 1.5) / 1.5)
        
        # 21. Rest advantage
        days_rest = data.get('days_rest', 1)
        rest_boost = min(days_rest / 3.0, 1.0)  # 3+ days = max rest
        features.append(rest_boost)
        
        # 22. Back-to-back penalty
        is_b2b = data.get('is_back_to_back', False)
        features.append(0.0 if is_b2b else 1.0)
        
        # 23. National TV game
        is_national_tv = data.get('is_national_tv', False)
        features.append(1.0 if is_national_tv else 0.0)
        
        # 24. Playoff implications
        playoff_implications = data.get('playoff_implications', False)
        features.append(1.0 if playoff_implications else 0.0)
        
        # 25. Time zone travel
        timezone_change = abs(data.get('timezone_change', 0))
        travel_penalty = 1.0 - min(timezone_change / 3.0, 1.0)
        features.append(travel_penalty)
        
        # === POSITION-SPECIFIC FEATURES (5) ===
        
        # 26. Line placement (1st line = 1.0)
        line_number = data.get('line_number', 3)
        line_quality = 1.0 - (line_number - 1) * 0.25
        features.append(max(line_quality, 0.25))
        
        # 27. Linemate quality
        linemate_quality = data.get('linemate_star_power', 0.5)
        features.append(linemate_quality)
        
        # 28. PP unit (1st unit = 1.0)
        pp_unit = data.get('powerplay_unit', 2)
        pp_opportunity = 1.0 if pp_unit == 1 else 0.3
        features.append(pp_opportunity)
        
        # 29. Recent ice time trend
        ice_time_trend = self._calculate_ice_time_trend(data)
        features.append(ice_time_trend)
        
        # 30. Defensive responsibility (affects offense)
        defensive_role = data.get('defensive_assignments', 0.5)
        offensive_freedom = 1.0 - defensive_role
        features.append(offensive_freedom)
        
        # === MILESTONE NARRATIVES (5 features) ===
        
        if self.milestone_awareness:
            # 31. Goals to milestone (99, 199, 299, etc.)
            season_goals = data.get('season_stats', {}).get('goals', 0)
            goals_to_milestone = self._distance_to_milestone(season_goals, [25, 30, 40, 50])
            milestone_proximity = 1.0 - min(goals_to_milestone / 5.0, 1.0)
            features.append(milestone_proximity)
            
            # 32. Points to milestone
            season_points = data.get('season_stats', {}).get('points', 0)
            points_to_milestone = self._distance_to_milestone(season_points, [50, 75, 100])
            points_proximity = 1.0 - min(points_to_milestone / 10.0, 1.0)
            features.append(points_proximity)
            
            # 33. Career milestone proximity
            career_goals = data.get('career_goals', 0)
            career_milestone = self._distance_to_milestone(career_goals, [100, 200, 300, 400, 500])
            career_proximity = 1.0 - min(career_milestone / 10.0, 1.0)
            features.append(career_proximity)
            
            # 34. Hat trick watch (2 goals already)
            current_game_goals = data.get('current_game_goals', 0)
            hat_trick_watch = 1.0 if current_game_goals >= 2 else 0.0
            features.append(hat_trick_watch)
            
            # 35. Contract year boost
            is_contract_year = data.get('is_contract_year', False)
            features.append(1.0 if is_contract_year else 0.0)
        else:
            # Fill with defaults
            features.extend([0.5] * 5)
            
        return features
        
    def _parse_toi(self, toi_str: str) -> float:
        """Parse time on ice string to minutes"""
        if ':' in str(toi_str):
            parts = str(toi_str).split(':')
            return int(parts[0]) + int(parts[1]) / 60.0
        return 0.0
        
    def _check_revenge_game(self, data: Dict) -> float:
        """Check if this is a revenge game scenario"""
        # Would need last game vs opponent result
        # For now return 0.5 (neutral)
        return 0.5
        
    def _get_home_away_split(self, data: Dict, location: str) -> float:
        """Get home/away scoring average"""
        games = data.get('last_5_games', [])
        location_games = [g for g in games if g.get('home_away', '') == location]
        
        if location_games:
            return np.mean([g.get('points', 0) for g in location_games])
        
        # Default to season average
        return data.get('season_stats', {}).get('points_per_game', 0.5)
        
    def _calculate_ice_time_trend(self, data: Dict) -> float:
        """Calculate if player's ice time is trending up"""
        games = data.get('last_5_games', [])
        
        if len(games) >= 3:
            recent_toi = [self._parse_toi(g.get('toi', '0:00')) for g in games[:3]]
            older_toi = [self._parse_toi(g.get('toi', '0:00')) for g in games[3:]]
            
            if older_toi:
                trend = np.mean(recent_toi) / max(np.mean(older_toi), 1.0)
                return min(trend, 1.5) / 1.5
                
        return 0.5  # Neutral
        
    def _distance_to_milestone(self, current: int, milestones: List[int]) -> int:
        """Calculate distance to nearest milestone"""
        distances = [m - current for m in milestones if m > current]
        return min(distances) if distances else 999
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return [
            # Star Power (5)
            'star_name_value', 'position_prop_value', 'star_production_rate',
            'ice_time_usage', 'powerplay_involvement',
            
            # Performance Momentum (8)
            'scoring_surge', 'point_streak', 'multi_point_frequency',
            'shot_volume_trend', 'recent_goal_production', 'form_trend',
            'scoring_consistency', 'shooting_percentage_heat',
            
            # Matchup Narrative (6)
            'historical_dominance', 'recent_success_vs', 'matchup_familiarity',
            'revenge_game', 'division_rival', 'playoff_history',
            
            # Contextual Amplifiers (6)
            'home_away_edge', 'rest_advantage', 'back_to_back_fresh',
            'national_tv_game', 'playoff_implications', 'travel_freshness',
            
            # Position-Specific (5)
            'line_quality', 'linemate_quality', 'powerplay_unit',
            'ice_time_trend', 'offensive_freedom',
            
            # Milestone Narratives (5)
            'goals_milestone_proximity', 'points_milestone_proximity',
            'career_milestone_proximity', 'hat_trick_watch', 'contract_year'
        ]


# Convenience function
def extract_player_features(players_data: List[Dict]) -> np.ndarray:
    """
    Extract narrative features from player data.
    
    Parameters
    ----------
    players_data : list of dict
        Player statistics and context
        
    Returns
    -------
    features : ndarray
        Player narrative features (n_players, 35)
    """
    transformer = NHLPlayerPerformanceTransformer()
    return transformer.transform(players_data)
    

def test_transformer():
    """Test the transformer with sample data"""
    # Sample player data
    sample_player = {
        'player_name': 'Auston Matthews',
        'position': 'C',
        'is_home_game': True,
        'season_stats': {
            'games_played': 20,
            'goals': 15,
            'assists': 10,
            'points': 25,
            'shots': 80,
            'shooting_pct': 0.188,
            'toi_per_game': '21:32',
            'powerplay_points': 8,
            'goals_per_game': 0.75,
            'points_per_game': 1.25,
            'shots_per_game': 4.0,
        },
        'recent_form': {
            'last_5_avg_goals': 1.0,
            'last_5_avg_shots': 4.5,
            'goals_last_5': 5,
            'trend': 'hot',
            'point_streak_games': 7,
            'multi_point_games_l10': 4,
        },
        'vs_opponent': {
            'games_played': 12,
            'avg_goals': 0.83,
            'avg_points': 1.42,
            'last_5_goals': 4,
            'last_5_points': 7,
        },
        'last_5_games': [
            {'date': '2024-11-18', 'goals': 2, 'assists': 1, 'shots': 5, 
             'points': 3, 'toi': '22:15', 'home_away': 'home'},
            {'date': '2024-11-16', 'goals': 1, 'assists': 0, 'shots': 4,
             'points': 1, 'toi': '21:45', 'home_away': 'away'},
            {'date': '2024-11-14', 'goals': 0, 'assists': 2, 'shots': 3,
             'points': 2, 'toi': '20:30', 'home_away': 'away'},
            {'date': '2024-11-12', 'goals': 1, 'assists': 1, 'shots': 5,
             'points': 2, 'toi': '21:00', 'home_away': 'home'},
            {'date': '2024-11-10', 'goals': 1, 'assists': 0, 'shots': 6,
             'points': 1, 'toi': '22:00', 'home_away': 'home'},
        ],
        'line_number': 1,
        'powerplay_unit': 1,
        'days_rest': 2,
        'is_national_tv': True,
    }
    
    # Create transformer
    transformer = NHLPlayerPerformanceTransformer()
    
    # Extract features
    features = transformer.transform([sample_player])
    
    print("NHL Player Performance Transformer Test")
    print("=" * 80)
    print(f"\nPlayer: {sample_player['player_name']}")
    print(f"Position: {sample_player['position']}")
    print(f"Recent form: {sample_player['recent_form']['trend']}")
    print(f"\nExtracted {len(features[0])} narrative features")
    
    # Show feature values
    feature_names = transformer.get_feature_names()
    print("\nTop narrative signals:")
    
    # Get top 10 features
    top_indices = np.argsort(features[0])[-10:][::-1]
    
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {features[0][idx]:.3f}")
        

if __name__ == "__main__":
    test_transformer()
