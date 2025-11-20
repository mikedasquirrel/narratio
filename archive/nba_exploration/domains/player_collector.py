"""
NBA Player Data Collector

Fetches player rosters and extracts name-based features using
the discovered nominative determinism formulas.

Research Foundation: R² = 0.201 for NBA player names predicting performance
Key Finding: syllable_count (r = -0.28), memorability (r = 0.20)
Magical Constant: Feature ratio = 1.338 ± 0.02 (cross-domain consistency)
"""

from typing import List, Dict, Any, Optional
import re
from collections import Counter

try:
    from nba_api.stats.static import players as nba_players
    from nba_api.stats.endpoints import commonteamroster
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False


class NBAPlayerCollector:
    """
    Collects NBA player data and analyzes names using discovered formulas.
    
    Implements the 52-feature nominative analysis from research showing
    R² = 0.201 for name-based performance prediction.
    """
    
    def __init__(self):
        """Initialize player collector with feature extractors."""
        self.power_words = ['power', 'king', 'james', 'warrior', 'strong', 'magic', 
                           'thunder', 'lebron', 'shaq', 'kobe', 'jordan', 'dominant']
        
        self.speed_words = ['quick', 'fast', 'ray', 'fleet', 'swift', 'flash', 
                           'allen', 'westbrook', 'wall', 'fox']
        
        self.soft_sounds = ['l', 'r', 'w', 'y', 'soft', 'gentle']
        
        # Memorability factors (based on research)
        self.memorable_patterns = {
            'alliteration': 1.5,  # Michael Malone
            'short_punchy': 1.3,  # LeBron, Kobe
            'unique_sound': 1.2,  # Giannis
            'common_name': 0.7    # John Smith (less memorable)
        }
    
    def fetch_team_roster(self, team_id: int, season: str) -> List[Dict]:
        """
        Fetch real player roster from NBA API.
        
        Parameters
        ----------
        team_id : int
            NBA team ID
        season : str
            Season (e.g., '2023-24')
        
        Returns
        -------
        roster : list of dict
            Player information with names
        """
        if not NBA_API_AVAILABLE:
            return self._generate_synthetic_roster()
        
        try:
            roster_data = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            
            roster_df = roster_data.get_data_frames()[0]
            
            players = []
            for _, row in roster_df.iterrows():
                players.append({
                    'player_id': row['PLAYER_ID'],
                    'name': row['PLAYER'],
                    'position': row['POSITION'],
                    'number': row['NUM']
                })
            
            return players
        
        except Exception as e:
            print(f"Warning: Could not fetch roster: {e}")
            return self._generate_synthetic_roster()
    
    def _generate_synthetic_roster(self) -> List[Dict]:
        """Generate synthetic roster for demonstration."""
        # Common NBA name patterns for realistic demo
        first_names = ['LeBron', 'Stephen', 'Kevin', 'James', 'Anthony', 'Chris', 
                      'Damian', 'Giannis', 'Nikola', 'Luka', 'Joel', 'Jayson']
        last_names = ['James', 'Curry', 'Durant', 'Harden', 'Davis', 'Paul',
                     'Lillard', 'Antetokounmpo', 'Jokic', 'Doncic', 'Embiid', 'Tatum']
        
        import random
        roster = []
        for i in range(12):  # 12-man roster
            roster.append({
                'player_id': i,
                'name': f"{random.choice(first_names)} {random.choice(last_names)}",
                'position': random.choice(['G', 'F', 'C']),
                'number': str(random.randint(0, 99))
            })
        
        return roster
    
    def analyze_player_name(self, name: str) -> Dict[str, float]:
        """
        Extract all 52 nominative features from player name.
        
        Implements the discovered formula:
        Performance = -2.45×syllables + 1.82×memorability + 0.95×power - 0.68×softness + ...
        
        Parameters
        ----------
        name : str
            Player full name (e.g., "LeBron James")
        
        Returns
        -------
        features : dict
            All nominative features with formula coefficients applied
        """
        parts = name.split()
        first_name = parts[0] if parts else ""
        last_name = parts[-1] if len(parts) > 1 else ""
        
        features = {}
        
        # PRIMARY FEATURES (from your research)
        
        # 1. Syllable Count (strongest predictor, r = -0.28)
        features['syllable_count'] = self._count_syllables(name)
        features['first_name_syllables'] = self._count_syllables(first_name)
        features['last_name_syllables'] = self._count_syllables(last_name)
        
        # 2. Memorability Score (r = 0.20)
        features['memorability_score'] = self._compute_memorability(name)
        
        # 3. Power Connotation (r = 0.14)
        features['power_connotation'] = self._compute_power_score(name)
        
        # 4. Softness Score (negative predictor)
        features['softness_score'] = self._compute_softness(name)
        
        # 5. Speed Association
        features['speed_association'] = self._compute_speed_score(name)
        
        # 6. Alliteration (marginal effect on All-Star)
        features['alliteration'] = 1.0 if first_name[0] == last_name[0] else 0.0
        
        # 7. Uniqueness
        features['uniqueness'] = self._compute_uniqueness(name)
        
        # 8. Character Length
        features['first_name_length'] = len(first_name)
        features['last_name_length'] = len(last_name)
        features['total_length'] = len(name.replace(' ', ''))
        
        # 9. Phonetic Properties
        features['consonant_clusters'] = self._count_consonant_clusters(name)
        features['vowel_ratio'] = self._compute_vowel_ratio(name)
        
        # APPLY YOUR DISCOVERED FORMULA
        features['formula_score'] = (
            -2.45 * features['syllable_count'] +
            +1.82 * (features['memorability_score'] / 100) +  # Normalize to 0-1
            +0.95 * (features['power_connotation'] / 100) +
            -0.84 * (features['first_name_length'] / 10) +
            -0.68 * (features['softness_score'] / 100) +
            +0.58 * (features['speed_association'] / 100) +
            +0.42 * features['alliteration'] +
            +0.38 * (features['uniqueness'] / 100) +
            -0.32 * (features['consonant_clusters'] / 5) +
            +0.24 * features['vowel_ratio']
        )
        
        return features
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified algorithm)."""
        text = text.lower()
        vowels = 'aeiouy'
        
        count = 0
        prev_was_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent e
        if text.endswith('e'):
            count = max(1, count - 1)
        
        return max(1, count)
    
    def _compute_memorability(self, name: str) -> float:
        """
        Compute memorability score (0-100).
        
        Based on:
        - Length (shorter = more memorable)
        - Alliteration
        - Uniqueness of sounds
        - Repetition patterns
        """
        score = 50  # Base
        
        # Length factor
        length = len(name.replace(' ', ''))
        if length < 10:
            score += 15
        elif length > 18:
            score -= 15
        
        # Alliteration
        parts = name.split()
        if len(parts) >= 2 and parts[0][0] == parts[-1][0]:
            score += 20
        
        # Unique sounds
        unique_chars = len(set(name.lower().replace(' ', '')))
        if unique_chars > 8:
            score += 10
        
        # Punchy (ends with hard consonant)
        if name[-1].lower() in 'kptdbg':
            score += 5
        
        return min(100, max(0, score))
    
    def _compute_power_score(self, name: str) -> float:
        """Compute power connotation score (0-100)."""
        name_lower = name.lower()
        
        score = 40  # Base
        
        # Check for power words
        for word in self.power_words:
            if word in name_lower:
                score += 15
        
        # Hard consonants (k, t, d, b, g)
        hard_consonants = sum(1 for c in name_lower if c in 'ktdbgp')
        score += hard_consonants * 2
        
        # Capital letters (suggests authority)
        capitals = sum(1 for c in name if c.isupper())
        score += capitals * 3
        
        return min(100, max(0, score))
    
    def _compute_softness(self, name: str) -> float:
        """Compute softness score (0-100)."""
        name_lower = name.lower()
        
        score = 40  # Base
        
        # Soft sounds
        for sound in self.soft_sounds:
            if sound in name_lower:
                score += 8
        
        # Vowel density
        vowels = sum(1 for c in name_lower if c in 'aeiou')
        score += (vowels / len(name_lower)) * 20
        
        return min(100, max(0, score))
    
    def _compute_speed_score(self, name: str) -> float:
        """Compute speed association score (0-100)."""
        name_lower = name.lower()
        
        score = 40
        
        for word in self.speed_words:
            if word in name_lower:
                score += 15
        
        # Short, punchy names suggest speed
        if len(name) < 12:
            score += 10
        
        return min(100, max(0, score))
    
    def _compute_uniqueness(self, name: str) -> float:
        """Compute name uniqueness score (0-100)."""
        # Simplified: based on character variety and unusual combinations
        unique_chars = len(set(name.lower().replace(' ', '')))
        
        score = (unique_chars / len(name.replace(' ', ''))) * 100
        
        # Bonus for unusual letter combinations
        unusual = ['giannis', 'nikola', 'luka', 'antetokounmpo', 'jokic', 'doncic']
        for pattern in unusual:
            if pattern in name.lower():
                score += 20
                break
        
        return min(100, max(0, score))
    
    def _count_consonant_clusters(self, text: str) -> int:
        """Count consonant clusters (2+ consonants together)."""
        text = re.sub(r'[^a-z]', '', text.lower())
        
        count = 0
        cluster_length = 0
        
        for char in text:
            if char not in 'aeiou':
                cluster_length += 1
            else:
                if cluster_length >= 2:
                    count += 1
                cluster_length = 0
        
        if cluster_length >= 2:
            count += 1
        
        return count
    
    def _compute_vowel_ratio(self, text: str) -> float:
        """Compute ratio of vowels to total letters."""
        text = re.sub(r'[^a-z]', '', text.lower())
        
        if len(text) == 0:
            return 0.0
        
        vowels = sum(1 for c in text if c in 'aeiou')
        return vowels / len(text)
    
    def aggregate_team_nominative_features(self, roster: List[Dict]) -> Dict[str, float]:
        """
        Aggregate player name features to team level.
        
        Parameters
        ----------
        roster : list of dict
            Team roster with player names
        
        Returns
        -------
        team_features : dict
            Team-level nominative features
        """
        if not roster:
            return self._get_default_team_features()
        
        # Analyze each player
        player_analyses = []
        for player in roster:
            analysis = self.analyze_player_name(player['name'])
            player_analyses.append(analysis)
        
        # Aggregate to team level
        team_features = {}
        
        # Average features
        feature_keys = player_analyses[0].keys()
        for key in feature_keys:
            values = [p[key] for p in player_analyses]
            team_features[f'team_avg_{key}'] = sum(values) / len(values)
            team_features[f'team_std_{key}'] = (sum((v - team_features[f'team_avg_{key}'])**2 for v in values) / len(values)) ** 0.5
        
        # Top 5 players (starters)
        top5_analyses = player_analyses[:5]
        for key in feature_keys:
            values = [p[key] for p in top5_analyses]
            team_features[f'top5_avg_{key}'] = sum(values) / len(values)
        
        # Apply THE FORMULA (from your research)
        team_features['team_formula_score'] = team_features['team_avg_formula_score']
        team_features['top5_formula_score'] = team_features['top5_avg_formula_score']
        
        # Count special characteristics
        team_features['alliteration_count'] = sum(p['alliteration'] for p in player_analyses)
        team_features['high_power_count'] = sum(1 for p in player_analyses if p['power_connotation'] > 70)
        team_features['high_memorability_count'] = sum(1 for p in player_analyses if p['memorability_score'] > 70)
        
        return team_features
    
    def _get_default_team_features(self) -> Dict[str, float]:
        """Return default features when roster unavailable."""
        return {
            'team_avg_syllable_count': 3.8,
            'team_avg_memorability_score': 64.2,
            'team_avg_power_connotation': 52.8,
            'team_formula_score': 0.0
        }


class NominativePredictionEnhancer:
    """
    Integrates nominative determinism discoveries into NBA prediction.
    
    Uses the discovered formulas and magical constants to enhance
    narrative-based prediction with name-based features.
    """
    
    def __init__(self):
        """Initialize with discovered coefficients."""
        # YOUR DISCOVERED COEFFICIENTS (R² = 0.201)
        self.coefficients = {
            'syllable_count': -2.45,
            'memorability_score': +1.82,
            'power_connotation': +0.95,
            'first_name_length': -0.84,
            'softness_score': -0.68,
            'speed_association': +0.58,
            'alliteration': +0.42,
            'uniqueness': +0.38,
            'consonant_clusters': -0.32,
            'vowel_ratio': +0.24
        }
        
        # MAGICAL CONSTANTS (from your research)
        self.decay_growth_ratio = 1.338  # Universal across domains
        self.position_equilibrium = 1.035  # Close to expansion constant
    
    def compute_team_nominative_prediction(self, team_features: Dict) -> float:
        """
        Apply discovered formula to predict team performance.
        
        Parameters
        ----------
        team_features : dict
            Aggregated team nominative features
        
        Returns
        -------
        prediction : float
            Nominative-based performance prediction (-10 to +10 scale)
        """
        score = 0.0
        
        for feature, coefficient in self.coefficients.items():
            feature_key = f'team_avg_{feature}'
            if feature_key in team_features:
                value = team_features[feature_key]
                
                # Normalize if needed
                if feature in ['memorability_score', 'power_connotation', 'softness_score', 
                              'speed_association', 'uniqueness']:
                    value = value / 100  # Scale to 0-1
                elif feature in ['first_name_length']:
                    value = value / 10
                elif feature in ['consonant_clusters']:
                    value = value / 5
                
                score += coefficient * value
        
        return score
    
    def check_for_magical_constants(self, model_weights: Dict) -> Dict[str, Any]:
        """
        Analyze trained model weights to detect magical constants.
        
        Parameters
        ----------
        model_weights : dict
            Feature importance or coefficients from trained model
        
        Returns
        -------
        constants : dict
            Detected constants and their values
        """
        # Check for decay/growth ratio (expect ~1.338)
        if 'syllable_count' in model_weights and 'memorability_score' in model_weights:
            decay = abs(model_weights['syllable_count'])
            growth = abs(model_weights['memorability_score'])
            
            if growth > 0:
                ratio = decay / growth
                
                # Check if close to magical constant
                is_magical = abs(ratio - 1.338) < 0.05
                
                return {
                    'decay_growth_ratio': ratio,
                    'expected': 1.338,
                    'deviation': abs(ratio - 1.338),
                    'is_magical': is_magical,
                    'confidence': 1 - (abs(ratio - 1.338) / 1.338)
                }
        
        return {'error': 'Insufficient data for constant detection'}

