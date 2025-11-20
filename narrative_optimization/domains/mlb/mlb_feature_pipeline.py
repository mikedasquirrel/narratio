"""
MLB Feature Pipeline - Extract Nominative + Statistical Features
The "narrative" is the composition of named entities (players, teams) + stats + context

Author: Narrative Optimization Framework  
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Add transformers to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from transformers.nominative import extract_name_features
except:
    print("Warning: Nominative transformers not available")

# Import journey features
from mlb_journey_features import MLBJourneyFeatures


class MLBFeaturePipeline:
    """Extract all features from MLB game data - nominative + statistical"""
    
    def __init__(self):
        self.feature_names = []
        self.journey_extractor = MLBJourneyFeatures()
        
    def extract_nominative_features(self, game: Dict, home_roster: List[Dict], 
                                   away_roster: List[Dict]) -> Dict[str, float]:
        """
        Extract features from PLAYER NAMES (the core nominative features)
        
        32 players per game = rich nominative context
        """
        features = {}
        
        # Home team player names
        home_names = [p['full_name'] for p in home_roster if p.get('full_name')]
        away_names = [p['full_name'] for p in away_roster if p.get('full_name')]
        
        # Name-based features
        features['home_roster_size'] = len(home_names)
        features['away_roster_size'] = len(away_names)
        features['total_players'] = len(home_names) + len(away_names)
        
        # Name length features (nominative complexity)
        if home_names:
            features['home_avg_name_length'] = np.mean([len(name) for name in home_names])
            features['home_total_name_chars'] = sum(len(name) for name in home_names)
        else:
            features['home_avg_name_length'] = 0
            features['home_total_name_chars'] = 0
            
        if away_names:
            features['away_avg_name_length'] = np.mean([len(name) for name in away_names])
            features['away_total_name_chars'] = sum(len(name) for name in away_names)
        else:
            features['away_avg_name_length'] = 0
            features['away_total_name_chars'] = 0
        
        # Name syllable counts (approximation: vowels)
        features['home_name_complexity'] = sum(
            sum(c.lower() in 'aeiouy' for c in name) for name in home_names
        )
        features['away_name_complexity'] = sum(
            sum(c.lower() in 'aeiouy' for c in name) for name in away_names
        )
        
        # International names (contains non-ASCII or specific patterns)
        features['home_international_names'] = sum(
            1 for name in home_names if any(ord(c) > 127 for c in name) or 
            any(pattern in name.lower() for pattern in ['rodriguez', 'martinez', 'hernandez', 'garcia'])
        )
        features['away_international_names'] = sum(
            1 for name in away_names if any(ord(c) > 127 for c in name) or
            any(pattern in name.lower() for pattern in ['rodriguez', 'martinez', 'hernandez', 'garcia'])
        )
        
        # Star player indicators (position + name patterns)
        home_pitchers = [p for p in home_roster if p.get('position_code') == 'P']
        away_pitchers = [p for p in away_roster if p.get('position_code') == 'P']
        
        features['home_pitcher_count'] = len(home_pitchers)
        features['away_pitcher_count'] = len(away_pitchers)
        
        # Position diversity (more positions = more nominative richness)
        home_positions = set(p.get('position_code', 'UNK') for p in home_roster)
        away_positions = set(p.get('position_code', 'UNK') for p in away_roster)
        
        features['home_position_diversity'] = len(home_positions)
        features['away_position_diversity'] = len(away_positions)
        
        return features
    
    def extract_team_features(self, game: Dict, home_stats: Dict, away_stats: Dict) -> Dict[str, float]:
        """Extract team-level nominative and statistical features"""
        features = {}
        
        # Team name features
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        features['home_team_name_length'] = len(home_team)
        features['away_team_name_length'] = len(away_team)
        
        # Team statistics
        features['home_wins'] = home_stats.get('wins', 0)
        features['home_losses'] = home_stats.get('losses', 0)
        features['away_wins'] = away_stats.get('wins', 0)
        features['away_losses'] = away_stats.get('losses', 0)
        
        # Win percentages
        home_total = features['home_wins'] + features['home_losses']
        away_total = features['away_wins'] + features['away_losses']
        
        features['home_win_pct'] = features['home_wins'] / max(home_total, 1)
        features['away_win_pct'] = features['away_wins'] / max(away_total, 1)
        
        # Record differential
        features['win_diff'] = features['home_wins'] - features['away_wins']
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
        
        # Quality indicators
        features['home_winning_record'] = 1.0 if features['home_win_pct'] > 0.500 else 0.0
        features['away_winning_record'] = 1.0 if features['away_win_pct'] > 0.500 else 0.0
        features['both_winning'] = features['home_winning_record'] * features['away_winning_record']
        
        return features
    
    def extract_context_features(self, game: Dict) -> Dict[str, float]:
        """Extract contextual features (rivalry, stadium, timing)"""
        features = {}
        
        # Rivalry (strong nominative effect)
        features['is_rivalry'] = 1.0 if game.get('is_rivalry') else 0.0
        
        # Historic stadium (nominative gravity)
        features['is_historic_stadium'] = 1.0 if game.get('is_historic_stadium') else 0.0
        
        # Temporal features
        month = game.get('month', 6)
        features['month'] = month
        features['early_season'] = 1.0 if month <= 5 else 0.0
        features['mid_season'] = 1.0 if 6 <= month <= 7 else 0.0
        features['late_season'] = 1.0 if month >= 8 else 0.0
        
        # Specific stadium names (nominative)
        venue = game.get('venue', '')
        features['venue_wrigley'] = 1.0 if 'Wrigley' in venue else 0.0
        features['venue_fenway'] = 1.0 if 'Fenway' in venue else 0.0
        features['venue_yankee'] = 1.0 if 'Yankee' in venue else 0.0
        features['venue_dodger'] = 1.0 if 'Dodger' in venue else 0.0
        
        # Specific rivalries (nominative pairs)
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        
        features['rivalry_yanks_sox'] = 1.0 if {home, away} == {'NYY', 'BOS'} else 0.0
        features['rivalry_dodgers_giants'] = 1.0 if {home, away} == {'LAD', 'SF'} else 0.0
        features['rivalry_cubs_cards'] = 1.0 if {home, away} == {'CHC', 'STL'} else 0.0
        features['rivalry_astros_rangers'] = 1.0 if {home, away} == {'HOU', 'TEX'} else 0.0
        
        return features
    
    def extract_pitcher_features(self, game: Dict) -> Dict[str, float]:
        """Extract pitcher nominative features"""
        features = {}
        
        # Pitcher names (key nominative elements)
        home_pitcher = game.get('home_pitcher', '')
        away_pitcher = game.get('away_pitcher', '')
        
        features['has_home_pitcher'] = 1.0 if home_pitcher else 0.0
        features['has_away_pitcher'] = 1.0 if away_pitcher else 0.0
        
        if home_pitcher:
            features['home_pitcher_name_length'] = len(home_pitcher)
            features['home_pitcher_name_complexity'] = sum(c.lower() in 'aeiouy' for c in home_pitcher)
        else:
            features['home_pitcher_name_length'] = 0
            features['home_pitcher_name_complexity'] = 0
            
        if away_pitcher:
            features['away_pitcher_name_length'] = len(away_pitcher)
            features['away_pitcher_name_complexity'] = sum(c.lower() in 'aeiouy' for c in away_pitcher)
        else:
            features['away_pitcher_name_length'] = 0
            features['away_pitcher_name_complexity'] = 0
        
        # Pitcher name differential
        features['pitcher_name_diff'] = abs(
            features['home_pitcher_name_length'] - features['away_pitcher_name_length']
        )
        
        return features
    
    def extract_interaction_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features between nominative and statistical"""
        features = {}
        
        # Nominative richness × Team quality
        nom_richness = base_features.get('total_players', 0)
        features['nom_richness_x_home_qual'] = nom_richness * base_features.get('home_win_pct', 0.5)
        features['nom_richness_x_away_qual'] = nom_richness * base_features.get('away_win_pct', 0.5)
        
        # Rivalry × Record differential
        features['rivalry_x_record_diff'] = (
            base_features.get('is_rivalry', 0) * abs(base_features.get('win_pct_diff', 0))
        )
        
        # Historic stadium × Late season
        features['stadium_x_late_season'] = (
            base_features.get('is_historic_stadium', 0) * base_features.get('late_season', 0)
        )
        
        # International names × Team quality
        home_intl = base_features.get('home_international_names', 0)
        away_intl = base_features.get('away_international_names', 0)
        features['intl_names_x_quality'] = (
            (home_intl + away_intl) * (base_features.get('home_win_pct', 0.5) + base_features.get('away_win_pct', 0.5)) / 2
        )
        
        return features
    
    def extract_all_features(self, game: Dict, home_stats: Dict, away_stats: Dict,
                            home_roster: List[Dict], away_roster: List[Dict]) -> Dict[str, float]:
        """
        Extract complete feature set for a game
        
        Returns:
            Dictionary of all features (nominative + journey)
        """
        all_features = {}
        
        # 1. Nominative features (player names) - THE CORE
        nominative_features = self.extract_nominative_features(game, home_roster, away_roster)
        all_features.update(nominative_features)
        
        # 2. Team features (names + stats)
        all_features.update(self.extract_team_features(game, home_stats, away_stats))
        
        # 3. Context features (rivalry, stadium, timing)
        all_features.update(self.extract_context_features(game))
        
        # 4. Pitcher features (names)
        all_features.update(self.extract_pitcher_features(game))
        
        # 5. Journey features (TRANSFORMER-GUIDED) - 13.5% journey completion
        journey_features = self.journey_extractor.extract_all_journey_features(game, home_stats, away_stats)
        all_features.update(journey_features)
        
        # 6. Nominative × Journey interactions
        journey_interactions = self.journey_extractor.create_interaction_features(journey_features, nominative_features)
        all_features.update(journey_interactions)
        
        # 7. Original interaction features
        all_features.update(self.extract_interaction_features(all_features))
        
        return all_features
    
    def extract_batch(self, games: List[Dict], stats_dict: Dict[str, Dict], 
                     roster_dict: Dict[str, List[Dict]]) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract features for multiple games
        
        Args:
            games: List of game dictionaries
            stats_dict: Team stats by team code
            roster_dict: Team rosters by team code
            
        Returns:
            (feature_matrix, game_ids, feature_names)
        """
        game_features = []
        game_ids = []
        
        for game in games:
            game_id = game.get('game_id')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            # Get required data
            home_stats = stats_dict.get(home_team, {})
            away_stats = stats_dict.get(away_team, {})
            home_roster = roster_dict.get(home_team, [])
            away_roster = roster_dict.get(away_team, [])
            
            # Extract features
            features = self.extract_all_features(game, home_stats, away_stats, home_roster, away_roster)
            
            game_features.append(features)
            game_ids.append(str(game_id))
        
        # Convert to matrix
        if not game_features:
            return np.array([]), [], []
        
        # Get consistent feature names
        feature_names = sorted(game_features[0].keys())
        self.feature_names = feature_names
        
        # Build matrix
        matrix = []
        for features in game_features:
            row = [features.get(fname, 0.0) for fname in feature_names]
            matrix.append(row)
        
        return np.array(matrix), game_ids, feature_names


if __name__ == '__main__':
    # Example usage
    pipeline = MLBFeaturePipeline()
    
    # Example game
    example_game = {
        'game_id': 123456,
        'home_team': 'BOS',
        'away_team': 'NYY',
        'venue': 'Fenway Park',
        'is_rivalry': True,
        'is_historic_stadium': True,
        'month': 9,
        'home_pitcher': 'Chris Sale',
        'away_pitcher': 'Gerrit Cole'
    }
    
    example_home_stats = {'wins': 85, 'losses': 65, 'win_pct': 0.567}
    example_away_stats = {'wins': 90, 'losses': 60, 'win_pct': 0.600}
    
    example_home_roster = [
        {'full_name': 'Rafael Devers', 'position_code': '3B'},
        {'full_name': 'Xander Bogaerts', 'position_code': 'SS'},
        {'full_name': 'Chris Sale', 'position_code': 'P'}
    ]
    
    example_away_roster = [
        {'full_name': 'Aaron Judge', 'position_code': 'RF'},
        {'full_name': 'Juan Soto', 'position_code': 'LF'},
        {'full_name': 'Gerrit Cole', 'position_code': 'P'}
    ]
    
    features = pipeline.extract_all_features(
        example_game, example_home_stats, example_away_stats,
        example_home_roster, example_away_roster
    )
    
    print("MLB Feature Extraction (Nominative + Statistical)")
    print("=" * 80)
    print(f"Total features extracted: {len(features)}")
    print("\nKey Nominative Features:")
    print(f"  Total players: {features['total_players']}")
    print(f"  Home international names: {features['home_international_names']}")
    print(f"  Away international names: {features['away_international_names']}")
    print(f"  Is rivalry: {features['is_rivalry']}")
    print(f"  Is historic stadium: {features['is_historic_stadium']}")
    print("\nKey Statistical Features:")
    print(f"  Home win %: {features['home_win_pct']:.3f}")
    print(f"  Away win %: {features['away_win_pct']:.3f}")
    print(f"  Win % differential: {features['win_pct_diff']:.3f}")
    print("=" * 80)

