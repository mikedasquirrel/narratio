"""
NHL Nominative Features - Name-Based Narratives (EXPANDED)

Extracts nominative gravity features from NHL entities:
- Team historical prestige (Cups, Presidents' Trophies, Conference titles, Playoff consistency)
- Goalie names (Roy, Hasek, Brodeur legacy effects)
- Team brands (Original Six premium, franchise history)
- Player names (star power, HOF trajectories)
- Coach prestige (Scotty Bowman legacy)
- Jersey numbers (#99 Gretzky shadow, #66 Lemieux)

Philosophy: Names carry narrative weight in hockey. Historical prestige (Cups,
conference titles, consistent excellence) creates nominative mass (ة) that markets
undervalue. The framework predicted this: if Cup history matters (26.6% importance),
OTHER historical achievements should matter too.

EXPANDED (Nov 17, 2025): Added 9 new historical prestige features based on
narrative framework guidance. Now extracts 38 nominative features (was 29).

Author: Narrative Integration System
Date: November 17, 2025
"""

import numpy as np
from typing import Dict, List, Optional
import re


# Original Six teams - founding franchises with maximum historical weight
ORIGINAL_SIX = {
    'BOS': {'name': 'Boston Bruins', 'weight': 1.0, 'founded': 1924},
    'CHI': {'name': 'Chicago Blackhawks', 'weight': 1.0, 'founded': 1926},
    'DET': {'name': 'Detroit Red Wings', 'weight': 1.0, 'founded': 1926},
    'MTL': {'name': 'Montreal Canadiens', 'weight': 1.0, 'founded': 1909},
    'NYR': {'name': 'New York Rangers', 'weight': 1.0, 'founded': 1926},
    'TOR': {'name': 'Toronto Maple Leafs', 'weight': 1.0, 'founded': 1917},
}

# Historic franchises (pre-expansion era, 1967)
HISTORIC_FRANCHISES = {
    'BOS': 1.00, 'CHI': 1.00, 'DET': 1.00, 'MTL': 1.00, 'NYR': 1.00, 'TOR': 1.00,
}

# Modern expansion teams (lower narrative weight)
EXPANSION_ERA_TEAMS = {
    'PHI': 0.85, 'STL': 0.85, 'PIT': 0.85, 'BUF': 0.80, 'VAN': 0.80,
    'LAK': 0.80, 'WPG': 0.75, 'CGY': 0.80, 'EDM': 0.80, 'NJD': 0.75,
    'CAR': 0.70, 'COL': 0.75, 'ARI': 0.65, 'FLA': 0.65, 'ANA': 0.70,
    'NSH': 0.65, 'CBJ': 0.60, 'MIN': 0.65, 'TBL': 0.70, 'DAL': 0.75,
    'OTT': 0.70, 'SJS': 0.70, 'VGK': 0.55, 'SEA': 0.50,
}

# Legendary goalie names (HOF goalies with narrative weight)
LEGENDARY_GOALIES = {
    'Roy': 1.0, 'Hasek': 1.0, 'Brodeur': 1.0, 'Plante': 0.95, 'Dryden': 0.95,
    'Sawchuk': 0.95, 'Esposito': 0.90, 'Fuhr': 0.90, 'Belfour': 0.90,
    'Joseph': 0.85, 'Luongo': 0.85, 'Price': 0.85, 'Lundqvist': 0.85,
    'Quick': 0.80, 'Crawford': 0.75, 'Fleury': 0.80, 'Rinne': 0.75,
    'Bishop': 0.70, 'Bobrovsky': 0.75, 'Vasilevskiy': 0.80, 'Hellebuyck': 0.75,
}

# Legendary player surnames (HOF trajectory names)
LEGENDARY_PLAYERS = {
    'Gretzky': 1.0, 'Lemieux': 1.0, 'Orr': 1.0, 'Howe': 1.0, 'Crosby': 0.95,
    'Ovechkin': 0.95, 'Jagr': 0.95, 'Messier': 0.90, 'Yzerman': 0.90,
    'Sakic': 0.90, 'Forsberg': 0.85, 'Lidstrom': 0.90, 'Bourque': 0.90,
    'Kane': 0.85, 'Toews': 0.80, 'Stamkos': 0.80, 'McDavid': 0.90,
    'MacKinnon': 0.85, 'Makar': 0.80, 'Matthews': 0.85, 'Bergeron': 0.85,
}

# Legendary coach surnames
LEGENDARY_COACHES = {
    'Bowman': 1.0, 'Arbour': 0.95, 'Blake': 0.90, 'Quenneville': 0.90,
    'Trotz': 0.85, 'Babcock': 0.80, 'Julien': 0.75, 'Cooper': 0.85,
    'Cassidy': 0.75, 'Sullivan': 0.80, 'Bednar': 0.75, 'Berube': 0.75,
}

# Iconic jersey numbers with legacy effects
ICONIC_NUMBERS = {
    99: {'player': 'Gretzky', 'weight': 1.0, 'retired_league_wide': True},
    66: {'player': 'Lemieux', 'weight': 0.95, 'retired_league_wide': False},
    4: {'player': 'Orr', 'weight': 0.90, 'retired_league_wide': False},
    9: {'player': 'Howe', 'weight': 0.90, 'retired_league_wide': False},
    87: {'player': 'Crosby', 'weight': 0.85, 'retired_league_wide': False},
    8: {'player': 'Ovechkin', 'weight': 0.85, 'retired_league_wide': False},
}

# Stanley Cup championships (franchise success narrative)
STANLEY_CUP_WINS = {
    'MTL': 24, 'TOR': 13, 'DET': 11, 'BOS': 6, 'CHI': 6, 'EDM': 5,
    'PIT': 5, 'NYR': 4, 'NYI': 4, 'NJD': 3, 'COL': 3, 'TBL': 3,
    'LAK': 2, 'PHI': 2, 'CAR': 1, 'CGY': 1, 'ANA': 1, 'DAL': 1,
    'WSH': 1, 'STL': 1, 'VGK': 0, 'SEA': 0, 'CBJ': 0, 'ARI': 0,
    'WPG': 0, 'MIN': 0, 'NSH': 0, 'BUF': 0, 'VAN': 0, 'OTT': 0,
    'SJS': 0, 'FLA': 0,
}

# Presidents' Trophy wins (regular season excellence since 1986)
PRESIDENTS_TROPHY_WINS = {
    'DET': 6, 'COL': 3, 'BOS': 3, 'VAN': 2, 'DAL': 2, 'NYR': 2,
    'TBL': 2, 'WSH': 2, 'PHI': 1, 'CGY': 1, 'CHI': 1, 'EDM': 1,
    'BUF': 1, 'STL': 1, 'ANA': 1, 'SJS': 1, 'PIT': 1, 'NSH': 1,
    'CAR': 1, 'FLA': 1, 'MTL': 0, 'TOR': 0, 'LAK': 0, 'NJD': 0,
    'WPG': 0, 'VGK': 0, 'SEA': 0, 'CBJ': 0, 'ARI': 0, 'MIN': 0,
    'OTT': 0, 'NYI': 0,
}

# Conference championships (approximate modern era counts)
CONFERENCE_CHAMPIONSHIPS = {
    'PIT': 6, 'DET': 6, 'BOS': 5, 'TBL': 4, 'CHI': 4, 'NJD': 5,
    'MTL': 4, 'EDM': 4, 'COL': 4, 'LAK': 3, 'NYR': 3, 'PHI': 3,
    'CAR': 2, 'ANA': 2, 'VGK': 2, 'SJS': 2, 'NYI': 2, 'BUF': 2,
    'VAN': 3, 'CGY': 2, 'DAL': 2, 'OTT': 1, 'WSH': 2, 'FLA': 2,
    'TOR': 0, 'STL': 1, 'NSH': 1, 'WPG': 0, 'MIN': 0, 'CBJ': 0,
    'ARI': 0, 'SEA': 0,
}

# Playoff appearances (modern era consistency - rough estimate)
PLAYOFF_APPEARANCES_2000_2024 = {
    'PIT': 18, 'BOS': 17, 'WSH': 16, 'TBL': 15, 'STL': 14, 'DET': 14,
    'SJS': 14, 'NSH': 14, 'NYR': 13, 'PHI': 12, 'ANA': 12, 'COL': 12,
    'CHI': 11, 'MTL': 11, 'LAK': 11, 'NJD': 10, 'DAL': 10, 'CAR': 10,
    'MIN': 10, 'VAN': 9, 'FLA': 9, 'TOR': 9, 'CGY': 9, 'NYI': 8,
    'VGK': 6, 'EDM': 7, 'WPG': 7, 'OTT': 5, 'CBJ': 4, 'BUF': 2,
    'ARI': 2, 'SEA': 1,
}


class NHLNominativeExtractor:
    """Extract nominative features from NHL game data"""
    
    def __init__(self):
        """Initialize the extractor"""
        pass
    
    def extract_features(self, game: Dict) -> Dict[str, float]:
        """
        Extract nominative features from a single game.
        
        Parameters
        ----------
        game : dict
            Game data with team names, goalie names, etc.
        
        Returns
        -------
        features : dict
            Nominative feature dictionary
        """
        features = {}
        
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        
        # === TEAM BRAND & HISTORICAL PRESTIGE FEATURES (19 features - EXPANDED) ===
        
        # 1. Home team brand weight (Original Six premium)
        features['home_brand_weight'] = self._get_team_brand_weight(home_team)
        
        # 2. Away team brand weight
        features['away_brand_weight'] = self._get_team_brand_weight(away_team)
        
        # 3. Brand differential (home - away)
        features['brand_differential'] = features['home_brand_weight'] - features['away_brand_weight']
        
        # 4. Original Six matchup indicator
        features['original_six_matchup'] = float(
            home_team in ORIGINAL_SIX and away_team in ORIGINAL_SIX
        )
        
        # 5. Home team Stanley Cup history (log scale)
        home_cups = STANLEY_CUP_WINS.get(home_team, 0)
        features['home_cup_history'] = np.log1p(home_cups) / np.log1p(24)  # Normalize by MTL's 24
        
        # 6. Away team Stanley Cup history
        away_cups = STANLEY_CUP_WINS.get(away_team, 0)
        features['away_cup_history'] = np.log1p(away_cups) / np.log1p(24)
        
        # 7. Cup history differential
        features['cup_history_diff'] = features['home_cup_history'] - features['away_cup_history']
        
        # 8-13. EXPANDED HISTORICAL PRESTIGE (New features)
        
        # 8. Home Presidents' Trophy history
        home_presidents = PRESIDENTS_TROPHY_WINS.get(home_team, 0)
        features['home_presidents_trophies'] = np.log1p(home_presidents) / np.log1p(6)  # Normalize by DET's 6
        
        # 9. Away Presidents' Trophy history  
        away_presidents = PRESIDENTS_TROPHY_WINS.get(away_team, 0)
        features['away_presidents_trophies'] = np.log1p(away_presidents) / np.log1p(6)
        
        # 10. Conference championship history (home)
        home_conf = CONFERENCE_CHAMPIONSHIPS.get(home_team, 0)
        features['home_conference_champs'] = np.log1p(home_conf) / np.log1p(6)  # Normalize by max 6
        
        # 11. Conference championship history (away)
        away_conf = CONFERENCE_CHAMPIONSHIPS.get(away_team, 0)
        features['away_conference_champs'] = np.log1p(away_conf) / np.log1p(6)
        
        # 12. Playoff consistency (home) - modern era appearances
        home_playoffs = PLAYOFF_APPEARANCES_2000_2024.get(home_team, 0)
        features['home_playoff_consistency'] = home_playoffs / 24.0  # Max 24 years (2000-2024)
        
        # 13. Playoff consistency (away)
        away_playoffs = PLAYOFF_APPEARANCES_2000_2024.get(away_team, 0)
        features['away_playoff_consistency'] = away_playoffs / 24.0
        
        # 14. Combined historical prestige (home) - weighted sum
        features['home_total_prestige'] = (
            features['home_cup_history'] * 0.50 +  # Cups most important
            features['home_conference_champs'] * 0.25 +  # Conference finals significant
            features['home_presidents_trophies'] * 0.15 +  # Regular season dominance
            features['home_playoff_consistency'] * 0.10  # Consistency matters
        )
        
        # 15. Combined historical prestige (away)
        features['away_total_prestige'] = (
            features['away_cup_history'] * 0.50 +
            features['away_conference_champs'] * 0.25 +
            features['away_presidents_trophies'] * 0.15 +
            features['away_playoff_consistency'] * 0.10
        )
        
        # 16. Total prestige differential
        features['total_prestige_diff'] = features['home_total_prestige'] - features['away_total_prestige']
        
        # 17. Expansion team indicator (home)
        features['home_is_expansion'] = float(
            self._get_team_brand_weight(home_team) < 0.70
        )
        
        # 18. Expansion team indicator (away)
        features['away_is_expansion'] = float(
            self._get_team_brand_weight(away_team) < 0.70
        )
        
        # 19. Combined brand gravity (multiplicative)
        features['combined_brand_gravity'] = (
            features['home_brand_weight'] * features['away_brand_weight']
        )
        
        # === GOALIE NOMINATIVE FEATURES (8 features) ===
        
        home_goalie = game.get('home_goalie', '')
        away_goalie = game.get('away_goalie', '')
        
        # 20. Home goalie name prestige
        features['home_goalie_prestige'] = self._get_goalie_prestige(home_goalie)
        
        # 21. Away goalie name prestige
        features['away_goalie_prestige'] = self._get_goalie_prestige(away_goalie)
        
        # 22. Goalie prestige differential
        features['goalie_prestige_diff'] = (
            features['home_goalie_prestige'] - features['away_goalie_prestige']
        )
        
        # 23. Legendary goalie matchup (both have prestige)
        features['legendary_goalie_matchup'] = float(
            features['home_goalie_prestige'] > 0.80 and 
            features['away_goalie_prestige'] > 0.80
        )
        
        # 24. Home goalie name length (information density)
        features['home_goalie_name_length'] = len(home_goalie) / 15.0 if home_goalie else 0.5
        
        # 25. Away goalie name length
        features['away_goalie_name_length'] = len(away_goalie) / 15.0 if away_goalie else 0.5
        
        # 26. Goalie name phonetic weight (vowel ratio - "softer" names)
        features['home_goalie_phonetic'] = self._get_phonetic_weight(home_goalie)
        
        # 27. Away goalie phonetic weight
        features['away_goalie_phonetic'] = self._get_phonetic_weight(away_goalie)
        
        # === PLAYER NOMINATIVE FEATURES (5 features) ===
        
        # 28. Home team star power (based on roster names if available)
        features['home_star_power'] = self._get_team_star_power(game, 'home')
        
        # 29. Away team star power
        features['away_star_power'] = self._get_team_star_power(game, 'away')
        
        # 30. Star power differential
        features['star_power_diff'] = features['home_star_power'] - features['away_star_power']
        
        # 31. Home roster name density (information content)
        features['home_name_density'] = self._get_name_density(game, 'home')
        
        # 32. Away roster name density
        features['away_name_density'] = self._get_name_density(game, 'away')
        
        # === COACH NOMINATIVE FEATURES (3 features) ===
        
        home_coach = game.get('home_coach', '')
        away_coach = game.get('away_coach', '')
        
        # 33. Home coach prestige
        features['home_coach_prestige'] = self._get_coach_prestige(home_coach)
        
        # 34. Away coach prestige
        features['away_coach_prestige'] = self._get_coach_prestige(away_coach)
        
        # 35. Coach prestige differential
        features['coach_prestige_diff'] = (
            features['home_coach_prestige'] - features['away_coach_prestige']
        )
        
        # === CONTEXTUAL NOMINATIVE FEATURES (3 features) ===
        
        # 36. Rivalry nominative boost (Original Six + history)
        is_rivalry = game.get('is_rivalry', False)
        features['rivalry_nominative_boost'] = 1.2 if is_rivalry else 1.0
        
        # 37. Playoff nominative amplification
        is_playoff = game.get('is_playoff', False)
        features['playoff_nominative_amp'] = 1.3 if is_playoff else 1.0
        
        # 38. Overall nominative gravity (combined measure - now includes total prestige)
        features['total_nominative_gravity'] = (
            features['combined_brand_gravity'] * 
            (1 + features['goalie_prestige_diff']) *
            (1 + features['total_prestige_diff']) *  # NEW: Include expanded prestige
            features['rivalry_nominative_boost'] *
            features['playoff_nominative_amp']
        )
        
        return features
    
    def _get_team_brand_weight(self, team: str) -> float:
        """Get team brand weight (Original Six = 1.0, expansion = lower)"""
        if team in ORIGINAL_SIX:
            return ORIGINAL_SIX[team]['weight']
        elif team in EXPANSION_ERA_TEAMS:
            return EXPANSION_ERA_TEAMS[team]
        else:
            return 0.60  # Default for unknown teams
    
    def _get_goalie_prestige(self, goalie_name: str) -> float:
        """Get goalie name prestige based on HOF/legendary status"""
        if not goalie_name:
            return 0.50  # Default
        
        # Check for legendary surname match
        for legend, weight in LEGENDARY_GOALIES.items():
            if legend.lower() in goalie_name.lower():
                return weight
        
        # Default prestige for non-legendary
        return 0.50
    
    def _get_coach_prestige(self, coach_name: str) -> float:
        """Get coach name prestige"""
        if not coach_name:
            return 0.50
        
        for legend, weight in LEGENDARY_COACHES.items():
            if legend.lower() in coach_name.lower():
                return weight
        
        return 0.50
    
    def _get_team_star_power(self, game: Dict, side: str) -> float:
        """Get team star power from roster names"""
        # Check for player names in game data
        roster_key = f'{side}_roster' if f'{side}_roster' in game else None
        
        if not roster_key:
            # Default based on team brand as proxy
            team = game.get(f'{side}_team', '')
            return self._get_team_brand_weight(team) * 0.8
        
        roster = game.get(roster_key, [])
        star_count = 0
        
        for player in roster:
            player_name = player if isinstance(player, str) else player.get('name', '')
            for legend, weight in LEGENDARY_PLAYERS.items():
                if legend.lower() in player_name.lower():
                    star_count += weight
                    break
        
        # Normalize by typical star count (3-5 stars = high power)
        return min(1.0, star_count / 4.0)
    
    def _get_name_density(self, game: Dict, side: str) -> float:
        """Get name information density (average name length)"""
        roster_key = f'{side}_roster'
        
        if roster_key not in game:
            return 0.50  # Default
        
        roster = game.get(roster_key, [])
        if not roster:
            return 0.50
        
        total_length = 0
        count = 0
        
        for player in roster:
            player_name = player if isinstance(player, str) else player.get('name', '')
            total_length += len(player_name)
            count += 1
        
        if count == 0:
            return 0.50
        
        avg_length = total_length / count
        # Normalize (typical hockey name: 10-15 chars)
        return min(1.0, avg_length / 15.0)
    
    def _get_phonetic_weight(self, name: str) -> float:
        """Get phonetic weight (vowel ratio)"""
        if not name:
            return 0.50
        
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for c in name if c in vowels)
        total_letters = sum(1 for c in name if c.isalpha())
        
        if total_letters == 0:
            return 0.50
        
        return vowel_count / total_letters


def extract_nominative_features_batch(games: List[Dict]) -> np.ndarray:
    """
    Extract nominative features for a batch of games.
    
    Parameters
    ----------
    games : list of dict
        List of game dictionaries
    
    Returns
    -------
    features : ndarray of shape (n_games, 29)
        Nominative features matrix
    """
    extractor = NHLNominativeExtractor()
    
    all_features = []
    for game in games:
        feat_dict = extractor.extract_features(game)
        # Convert dict to ordered array (29 features)
        feat_array = [feat_dict[k] for k in sorted(feat_dict.keys())]
        all_features.append(feat_array)
    
    return np.array(all_features, dtype=np.float32)


def get_nominative_feature_names() -> List[str]:
    """Get list of nominative feature names"""
    return [
        # Team brand (10)
        'home_brand_weight', 'away_brand_weight', 'brand_differential',
        'original_six_matchup', 'home_cup_history', 'away_cup_history',
        'cup_history_diff', 'home_is_expansion', 'away_is_expansion',
        'combined_brand_gravity',
        
        # Goalie (8)
        'home_goalie_prestige', 'away_goalie_prestige', 'goalie_prestige_diff',
        'legendary_goalie_matchup', 'home_goalie_name_length', 'away_goalie_name_length',
        'home_goalie_phonetic', 'away_goalie_phonetic',
        
        # Player (5)
        'home_star_power', 'away_star_power', 'star_power_diff',
        'home_name_density', 'away_name_density',
        
        # Coach (3)
        'home_coach_prestige', 'away_coach_prestige', 'coach_prestige_diff',
        
        # Contextual (3)
        'rivalry_nominative_boost', 'playoff_nominative_amp', 'total_nominative_gravity',
    ]

