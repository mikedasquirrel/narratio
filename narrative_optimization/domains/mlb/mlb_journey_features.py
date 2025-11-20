"""
MLB Journey Features - Transformer-Guided Optimization
Extracts journey completion, quest structure, and comedy mythos features

Key Transformer Finding: MLB = 13.5% journey completion (HIGHEST in sports)
- 2x NFL (6.9%)
- 8x NBA (~1.7%)
- Quest dominant (26.2% purity)
- Comedy mythos pattern

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
from typing import Dict
from datetime import datetime


class MLBJourneyFeatures:
    """Extract journey-based features following transformer insights"""
    
    def __init__(self):
        # Quest stages mapped to season timeline
        self.quest_stages = {
            'call_to_adventure': (1, 40),      # April-early May
            'tests_and_trials': (41, 100),     # May-July
            'approach_ordeal': (101, 130),     # August
            'supreme_ordeal': (131, 162),      # September
            'return_elixir': (163, 180)        # Playoffs
        }
        
        # Comedy mythos indicators
        self.comedy_patterns = [
            'underdog_wins', 'comeback', 'happy_ending',
            'unexpected_hero', 'reversal_fortune'
        ]
    
    def extract_season_arc_features(self, game: Dict, team_stats: Dict) -> Dict[str, float]:
        """
        Season arc features - Map to journey stages
        
        Captures the 13.5% journey completion signal
        """
        features = {}
        
        # Game number in season (1-162)
        game_number = game.get('game_number', 81)
        
        # Season progress (0 to 1)
        features['season_progress'] = game_number / 162.0
        
        # Quest stage encoding
        for stage_name, (start, end) in self.quest_stages.items():
            features[f'quest_stage_{stage_name}'] = 1.0 if start <= game_number <= end else 0.0
        
        # Playoff proximity (games back from wild card)
        wins = team_stats.get('wins', 81)
        losses = team_stats.get('losses', 81)
        games_played = wins + losses
        
        # Estimate wild card cutoff (typically ~87 wins)
        games_remaining = 162 - games_played
        wins_needed_for_wildcard = max(0, 87 - wins)
        
        features['playoff_proximity'] = 1.0 - min(1.0, wins_needed_for_wildcard / max(games_remaining, 1))
        features['in_playoff_race'] = 1.0 if wins_needed_for_wildcard <= games_remaining else 0.0
        
        # Streak features (momentum in journey)
        # Would need recent game data - using win% as proxy
        win_pct = wins / max(games_played, 1)
        features['momentum_positive'] = 1.0 if win_pct > 0.550 else 0.0
        features['momentum_negative'] = 1.0 if win_pct < 0.450 else 0.0
        
        # Season turning points
        features['midseason_inflection'] = 1.0 if 75 <= game_number <= 87 else 0.0
        features['stretch_run'] = 1.0 if game_number > 130 else 0.0
        
        # Comeback narrative (recovering from bad start)
        if game_number > 40:
            # If winning now but started poorly
            early_win_pace = 0.400  # Assume poor start if below this
            features['comeback_narrative'] = 1.0 if (win_pct > 0.500) and (wins / min(40, games_played) < early_win_pace) else 0.0
        else:
            features['comeback_narrative'] = 0.0
        
        return features
    
    def extract_quest_structure_features(self, game: Dict, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """
        Quest structure features - 26.2% purity finding
        
        Maps Campbell's Hero's Journey to baseball season
        """
        features = {}
        
        game_number = game.get('game_number', 81)
        month = game.get('month', 6)
        
        # Quest stage weights (stronger in September)
        if month <= 5:
            features['quest_intensity'] = 0.3  # Early season
        elif month <= 7:
            features['quest_intensity'] = 0.5  # Mid season
        elif month == 8:
            features['quest_intensity'] = 0.7  # August
        else:
            features['quest_intensity'] = 1.0  # September climax
        
        # Quest objectives
        home_wins = home_stats.get('wins', 81)
        away_wins = away_stats.get('wins', 81)
        
        # Division quest (both teams competing)
        features['division_quest'] = 1.0 if (home_wins > 75 and away_wins > 75) else 0.0
        
        # Wild card quest (85-92 win range)
        features['wildcard_quest_home'] = 1.0 if 80 <= home_wins <= 95 else 0.0
        features['wildcard_quest_away'] = 1.0 if 80 <= away_wins <= 95 else 0.0
        
        # Hundred win quest (approaching milestone)
        features['hundred_win_quest_home'] = 1.0 if home_wins >= 95 else 0.0
        features['hundred_win_quest_away'] = 1.0 if away_wins >= 95 else 0.0
        
        # Rivalry quest (from game context)
        features['rivalry_quest'] = 1.0 if game.get('is_rivalry', False) else 0.0
        
        # High-stakes quest (both teams good)
        features['high_stakes_quest'] = features['division_quest'] * features['quest_intensity']
        
        return features
    
    def extract_comedy_mythos_features(self, game: Dict, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """
        Comedy mythos features - Dominant pattern in MLB
        
        Comedy = happy resolutions, not tragedy
        """
        features = {}
        
        home_wins = home_stats.get('wins', 81)
        away_wins = away_stats.get('wins', 81)
        home_total = home_wins + home_stats.get('losses', 81)
        away_total = away_wins + away_stats.get('losses', 81)
        
        home_win_pct = home_wins / max(home_total, 1)
        away_win_pct = away_wins / max(away_total, 1)
        
        # Underdog success potential (weaker team can win - comedy)
        win_pct_gap = abs(home_win_pct - away_win_pct)
        features['underdog_potential'] = 1.0 - min(win_pct_gap / 0.3, 1.0)  # High when teams close
        
        # Unexpected hero (both teams have chance)
        features['unexpected_outcome_possible'] = 1.0 if win_pct_gap < 0.15 else 0.0
        
        # Reversal of fortune potential (bad team vs good team)
        features['reversal_opportunity'] = 1.0 if win_pct_gap > 0.20 else 0.0
        
        # Happy resolution context (home team winning)
        features['home_comedy_setup'] = 1.0 if home_win_pct > 0.500 else 0.0
        
        # Avoid tragedy (teams not in dire straits)
        features['avoid_tragic_spiral'] = 1.0 if (home_win_pct > 0.350 and away_win_pct > 0.350) else 0.0
        
        # Competitive balance (comedy thrives on uncertainty)
        features['competitive_balance'] = 1.0 - win_pct_gap
        
        # Late season redemption (September comebacks)
        month = game.get('month', 6)
        if month >= 8:
            # Redemption arc for struggling teams
            features['late_redemption_home'] = 1.0 if 0.450 <= home_win_pct <= 0.520 else 0.0
            features['late_redemption_away'] = 1.0 if 0.450 <= away_win_pct <= 0.520 else 0.0
        else:
            features['late_redemption_home'] = 0.0
            features['late_redemption_away'] = 0.0
        
        return features
    
    def compute_journey_completion_score(self, all_features: Dict[str, float]) -> float:
        """
        Compute overall journey completion score
        
        Target: 13.5% mean (transformer finding)
        High-journey games: 20%+
        """
        # Weighted combination
        score = (
            all_features.get('season_progress', 0) * 0.3 +
            all_features.get('playoff_proximity', 0) * 0.4 +
            all_features.get('quest_intensity', 0) * 0.2 +
            all_features.get('rivalry_quest', 0) * 0.1
        )
        
        # Boost for high-stakes contexts
        if all_features.get('stretch_run', 0) > 0 and all_features.get('in_playoff_race', 0) > 0:
            score *= 1.5
        
        # Normalize to 0-1 range
        return min(score, 1.0)
    
    def extract_all_journey_features(self, game: Dict, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """
        Extract complete journey feature set
        
        Returns ~30 journey-based features
        """
        features = {}
        
        # Season arc features
        features.update(self.extract_season_arc_features(game, home_stats))
        features.update(self.extract_season_arc_features(game, away_stats))  # For away team
        
        # Quest structure features
        features.update(self.extract_quest_structure_features(game, home_stats, away_stats))
        
        # Comedy mythos features
        features.update(self.extract_comedy_mythos_features(game, home_stats, away_stats))
        
        # Overall journey completion score
        features['journey_completion_score'] = self.compute_journey_completion_score(features)
        
        # High-journey flag (above MLB mean of 13.5%)
        features['high_journey_game'] = 1.0 if features['journey_completion_score'] > 0.15 else 0.0
        
        # Extreme journey flag (top quartile - 20%+)
        features['extreme_journey_game'] = 1.0 if features['journey_completion_score'] > 0.20 else 0.0
        
        return features
    
    def create_interaction_features(self, journey_features: Dict, nominative_features: Dict) -> Dict[str, float]:
        """
        Create nominative × journey interaction features
        
        Combines transformer insights with nominative approach
        """
        interactions = {}
        
        # Journey × Nominative richness
        journey_score = journey_features.get('journey_completion_score', 0)
        total_players = nominative_features.get('total_players', 0)
        
        interactions['journey_x_nominative'] = journey_score * (total_players / 50.0)
        
        # Quest stage × International names
        quest_intensity = journey_features.get('quest_intensity', 0)
        intl_names = nominative_features.get('home_international_names', 0) + nominative_features.get('away_international_names', 0)
        
        interactions['quest_x_international'] = quest_intensity * intl_names
        
        # Comedy × Name complexity
        competitive_balance = journey_features.get('competitive_balance', 0)
        name_complexity = nominative_features.get('home_name_complexity', 0) + nominative_features.get('away_name_complexity', 0)
        
        interactions['comedy_x_complexity'] = competitive_balance * (name_complexity / 100.0)
        
        # Playoff proximity × Rivalry
        playoff_prox = journey_features.get('playoff_proximity', 0)
        is_rivalry = nominative_features.get('is_rivalry', 0)
        
        interactions['playoff_x_rivalry'] = playoff_prox * is_rivalry
        
        # Stretch run × Historic stadium
        stretch_run = journey_features.get('stretch_run', 0)
        historic_stadium = nominative_features.get('is_historic_stadium', 0)
        
        interactions['stretch_x_stadium'] = stretch_run * historic_stadium
        
        return interactions


if __name__ == '__main__':
    # Example usage
    extractor = MLBJourneyFeatures()
    
    # Example game in September playoff race
    example_game = {
        'game_id': 12345,
        'game_number': 145,  # Late September
        'month': 9,
        'is_rivalry': True
    }
    
    example_home_stats = {
        'wins': 86,
        'losses': 58
    }
    
    example_away_stats = {
        'wins': 88,
        'losses': 56
    }
    
    # Extract features
    features = extractor.extract_all_journey_features(example_game, example_home_stats, example_away_stats)
    
    print("MLB Journey Features - Transformer Guided")
    print("=" * 80)
    print(f"Total features: {len(features)}")
    print(f"\nJourney Completion Score: {features['journey_completion_score']:.3f}")
    print(f"  (MLB mean: 0.135, this game: {features['journey_completion_score']:.3f})")
    print(f"\nQuest Intensity: {features['quest_intensity']:.2f}")
    print(f"Playoff Proximity: {features['playoff_proximity']:.3f}")
    print(f"High Journey Game: {'YES' if features['high_journey_game'] > 0 else 'NO'}")
    print(f"Extreme Journey Game: {'YES' if features['extreme_journey_game'] > 0 else 'NO'}")
    print("\nQuest Features:")
    print(f"  Division Quest: {features['division_quest']}")
    print(f"  Wild Card Quest (Home): {features['wildcard_quest_home']}")
    print(f"  Rivalry Quest: {features['rivalry_quest']}")
    print("\nComedy Mythos:")
    print(f"  Competitive Balance: {features['competitive_balance']:.3f}")
    print(f"  Underdog Potential: {features['underdog_potential']:.3f}")
    print("=" * 80)

