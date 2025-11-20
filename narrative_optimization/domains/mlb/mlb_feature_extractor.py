"""
MLB Feature Extractor - Domain-Specific Features

Extracts MLB-specific features that complement transformer features:
- Rivalry indicators
- Stadium effects
- Team name characteristics
- Historical context
- Playoff race intensity
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class MLBFeatureExtractor:
    """
    Extracts domain-specific features for MLB games.
    
    Features include:
    - Rivalry strength (Yankees-Red Sox, Dodgers-Giants, etc.)
    - Stadium prestige/historical significance
    - Team name memorability and phonetic features
    - Playoff race intensity
    - Historical matchup context
    """
    
    # Major MLB rivalries with strength scores
    RIVALRIES = {
        ('NYY', 'BOS'): 1.0,  # Yankees-Red Sox (strongest)
        ('LAD', 'SF'): 0.95,  # Dodgers-Giants
        ('CHC', 'STL'): 0.90, # Cubs-Cardinals
        ('NYY', 'NYM'): 0.85, # Subway Series
        ('LAA', 'LAD'): 0.80, # Freeway Series
        ('CHC', 'CWS'): 0.75, # Crosstown Classic
        ('BAL', 'WSN'): 0.70, # Beltway Series
        ('TEX', 'HOU'): 0.70, # Lone Star Series
    }
    
    # Historic stadiums (higher prestige)
    HISTORIC_STADIUMS = {
        'Fenway Park': 1.0,
        'Wrigley Field': 1.0,
        'Yankee Stadium': 0.95,
        'Dodger Stadium': 0.90,
        'Busch Stadium': 0.85,
        'Oracle Park': 0.85,
        'Coors Field': 0.80,
        'Camden Yards': 0.80,
    }
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract_features(self, games: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract domain-specific features for all games.
        
        Parameters
        ----------
        games : list of dict
            Game data with team, venue, context information
        
        Returns
        -------
        features : np.ndarray
            Feature matrix of shape (n_games, n_features)
        """
        features = []
        
        for game in games:
            game_features = self._extract_game_features(game)
            features.append(game_features)
        
        return np.array(features)
    
    def _extract_game_features(self, game: Dict[str, Any]) -> List[float]:
        """Extract features for a single game."""
        features = []
        
        home_team = game['home_team']
        away_team = game['away_team']
        venue = game['venue']
        context = game['context']
        
        # 1. Rivalry strength (0-1)
        rivalry_strength = self._get_rivalry_strength(
            home_team['abbreviation'],
            away_team['abbreviation']
        )
        features.append(rivalry_strength)
        
        # 2. Is rivalry game (binary)
        is_rivalry = 1.0 if rivalry_strength > 0 else 0.0
        features.append(is_rivalry)
        
        # 3. Stadium prestige (0-1)
        stadium_prestige = self._get_stadium_prestige(venue['name'])
        features.append(stadium_prestige)
        
        # 4. Is historic stadium (binary)
        is_historic = 1.0 if stadium_prestige > 0.8 else 0.0
        features.append(is_historic)
        
        # 5. Playoff race intensity (0-1)
        playoff_intensity = self._get_playoff_intensity(
            home_team['record'],
            away_team['record'],
            game['season']
        )
        features.append(playoff_intensity)
        
        # 6. Is playoff race game (binary)
        is_playoff_race = 1.0 if context.get('playoff_race', False) else 0.0
        features.append(is_playoff_race)
        
        # 7. Home team win percentage
        home_record = home_team['record']
        home_win_pct = home_record['wins'] / (home_record['wins'] + home_record['losses']) if (home_record['wins'] + home_record['losses']) > 0 else 0.5
        features.append(home_win_pct)
        
        # 8. Away team win percentage
        away_record = away_team['record']
        away_win_pct = away_record['wins'] / (away_record['wins'] + away_record['losses']) if (away_record['wins'] + away_record['losses']) > 0 else 0.5
        features.append(away_win_pct)
        
        # 9. Win percentage differential
        win_pct_diff = home_win_pct - away_win_pct
        features.append(win_pct_diff)
        
        # 10. Team name length (home)
        home_name_len = len(home_team['name']) / 50.0  # Normalize
        features.append(home_name_len)
        
        # 11. Team name length (away)
        away_name_len = len(away_team['name']) / 50.0  # Normalize
        features.append(away_name_len)
        
        # 12. City name memorability (home) - simple heuristic
        home_city_mem = self._city_memorability(home_team['city'])
        features.append(home_city_mem)
        
        # 13. City name memorability (away)
        away_city_mem = self._city_memorability(away_team['city'])
        features.append(away_city_mem)
        
        # 14. Nickname memorability (home)
        home_nickname_mem = self._nickname_memorability(home_team['nickname'])
        features.append(home_nickname_mem)
        
        # 15. Nickname memorability (away)
        away_nickname_mem = self._nickname_memorability(away_team['nickname'])
        features.append(away_nickname_mem)
        
        # 16. Season (normalized)
        season_norm = (game['season'] - 2015) / 10.0  # 2015-2024 range
        features.append(season_norm)
        
        # 17. Month (from date, normalized)
        month = self._extract_month(game.get('date', ''))
        month_norm = month / 12.0
        features.append(month_norm)
        
        # 18. Is late season (August-September, playoff push)
        is_late_season = 1.0 if 8 <= month <= 9 else 0.0
        features.append(is_late_season)
        
        # 19. Combined team prestige (sum of city/nickname memorability)
        combined_prestige = (home_city_mem + home_nickname_mem + away_city_mem + away_nickname_mem) / 4.0
        features.append(combined_prestige)
        
        # 20. Rivalry × Playoff intensity interaction
        rivalry_playoff_interaction = rivalry_strength * playoff_intensity
        features.append(rivalry_playoff_interaction)
        
        return features
    
    def _get_rivalry_strength(self, team1_abbr: str, team2_abbr: str) -> float:
        """Get rivalry strength between two teams."""
        key1 = (team1_abbr, team2_abbr)
        key2 = (team2_abbr, team1_abbr)
        
        if key1 in self.RIVALRIES:
            return self.RIVALRIES[key1]
        elif key2 in self.RIVALRIES:
            return self.RIVALRIES[key2]
        else:
            return 0.0
    
    def _get_stadium_prestige(self, stadium_name: str) -> float:
        """Get stadium prestige score."""
        for historic_name, prestige in self.HISTORIC_STADIUMS.items():
            if historic_name.lower() in stadium_name.lower():
                return prestige
        return 0.5  # Default moderate prestige
    
    def _get_playoff_intensity(self, home_record: Dict, away_record: Dict, season: int) -> float:
        """Calculate playoff race intensity."""
        home_win_pct = home_record['wins'] / (home_record['wins'] + home_record['losses']) if (home_record['wins'] + home_record['losses']) > 0 else 0.5
        away_win_pct = away_record['wins'] / (away_record['wins'] + away_record['losses']) if (away_record['wins'] + away_record['losses']) > 0 else 0.5
        
        # Both teams above .500 = high intensity
        if home_win_pct > 0.5 and away_win_pct > 0.5:
            # Closer to .500, more intense (tight race)
            avg_win_pct = (home_win_pct + away_win_pct) / 2.0
            intensity = 1.0 - abs(avg_win_pct - 0.5) * 2.0  # Higher when closer to .500
            return max(0.0, min(1.0, intensity))
        else:
            return 0.0
    
    def _city_memorability(self, city: str) -> float:
        """Simple heuristic for city name memorability."""
        city_lower = city.lower()
        
        # Major cities get higher scores
        major_cities = ['new york', 'los angeles', 'chicago', 'boston', 'philadelphia']
        if any(major in city_lower for major in major_cities):
            return 1.0
        
        # Shorter names more memorable
        length_score = 1.0 - (len(city) / 30.0)
        
        # Common letters/patterns
        common_patterns = ['ton', 'ville', 'city', 'burg']
        pattern_score = 0.5 if any(p in city_lower for p in common_patterns) else 1.0
        
        return (length_score + pattern_score) / 2.0
    
    def _nickname_memorability(self, nickname: str) -> float:
        """Simple heuristic for team nickname memorability."""
        nickname_lower = nickname.lower()
        
        # Classic/iconic nicknames
        iconic = ['yankees', 'red sox', 'dodgers', 'giants', 'cubs', 'cardinals']
        if any(iconic_name in nickname_lower for iconic_name in iconic):
            return 1.0
        
        # Animal names (memorable)
        animals = ['tigers', 'lions', 'bears', 'eagles', 'hawks', 'cardinals']
        if any(animal in nickname_lower for animal in animals):
            return 0.8
        
        # Length-based (shorter = more memorable)
        length_score = 1.0 - (len(nickname) / 20.0)
        
        return max(0.3, min(1.0, length_score))
    
    def _extract_month(self, date_str: str) -> int:
        """Extract month from date string."""
        try:
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts) >= 2:
                    return int(parts[1])
        except:
            pass
        return 6  # Default to June
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        return [
            'rivalry_strength',
            'is_rivalry',
            'stadium_prestige',
            'is_historic_stadium',
            'playoff_intensity',
            'is_playoff_race',
            'home_win_pct',
            'away_win_pct',
            'win_pct_diff',
            'home_name_len',
            'away_name_len',
            'home_city_mem',
            'away_city_mem',
            'home_nickname_mem',
            'away_nickname_mem',
            'season_norm',
            'month_norm',
            'is_late_season',
            'combined_prestige',
            'rivalry_playoff_interaction'
        ]


def main():
    """Test feature extractor."""
    import json
    
    # Load sample data
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'mlb_complete_dataset.json'
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run collect_mlb_data.py first")
        return
    
    with open(dataset_path) as f:
        games = json.load(f)
    
    print(f"Loaded {len(games)} games")
    
    # Extract features
    extractor = MLBFeatureExtractor()
    features = extractor.extract_features(games[:100])  # Test on first 100
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature names: {extractor.get_feature_names()}")
    print(f"\nSample features (first game):")
    for name, value in zip(extractor.get_feature_names(), features[0]):
        print(f"  {name}: {value:.3f}")


if __name__ == '__main__':
    main()

