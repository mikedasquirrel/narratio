"""
Player Name Analyzer - Validated Sport Linguistics

Extracts nominative features from player names using VALIDATED correlations:
- Syllables: r = -0.191*** (basketball validated)
- Harshness: r = 0.196*** (basketball validated)
- Memorability: r = 0.182*** (basketball validated)

From sports meta-analysis project.
"""

import re
from typing import Dict, List
import numpy as np


class PlayerNameAnalyzer:
    """
    Analyzes player names for nominative power using validated linguistics.
    
    Based on validated findings:
    - Basketball harshness r=0.196 (p<0.001)
    - Basketball syllables r=-0.191 (p<0.001)
    - Basketball memorability r=0.182 (p<0.001)
    """
    
    def __init__(self):
        # Harsh consonants (from phonetic symbolism research)
        self.harsh_consonants = set('KGTDPB')
        
        # Soft consonants
        self.soft_consonants = set('LMNSR')
        
        # Vowels for syllable counting
        self.vowels = set('AEIOUY')
        
        # Power sounds (aggressive phonetics)
        self.power_sounds = ['ACK', 'OCK', 'UCK', 'ASH', 'AST', 'ANG']
    
    def analyze_name(self, full_name: str) -> Dict[str, float]:
        """
        Analyze single name for all nominative features.
        
        Returns dict with validated metrics plus extensions.
        """
        name_upper = full_name.upper()
        
        # 1. Syllable count (validated r=-0.191)
        syllables = self._count_syllables(full_name)
        
        # 2. Harshness score (validated r=0.196)
        harshness = self._calculate_harshness(name_upper)
        
        # 3. Memorability (validated r=0.182)
        memorability = self._calculate_memorability(full_name)
        
        # 4. Length
        length = len(full_name.replace(' ', ''))
        
        # 5. Hard consonant ratio
        consonants = [c for c in name_upper if c.isalpha() and c not in self.vowels]
        hard_ratio = sum(1 for c in consonants if c in self.harsh_consonants) / max(len(consonants), 1)
        
        # 6. Soft consonant ratio
        soft_ratio = sum(1 for c in consonants if c in self.soft_consonants) / max(len(consonants), 1)
        
        # 7. Power sounds presence
        has_power_sound = any(sound in name_upper for sound in self.power_sounds)
        
        # 8. Vowel/consonant ratio
        vowel_count = sum(1 for c in name_upper if c in self.vowels)
        consonant_count = len(consonants)
        vowel_ratio = vowel_count / max(vowel_count + consonant_count, 1)
        
        # 9. Name power score (composite)
        name_power = (
            -syllables * 0.191 +  # Validated coefficient
            harshness * 0.196 +   # Validated coefficient
            memorability * 0.182  # Validated coefficient
        )
        
        return {
            'syllables': syllables,
            'harshness': harshness,
            'memorability': memorability,
            'length': length,
            'hard_consonant_ratio': hard_ratio,
            'soft_consonant_ratio': soft_ratio,
            'has_power_sound': 1.0 if has_power_sound else 0.0,
            'vowel_ratio': vowel_ratio,
            'name_power_score': name_power
        }
    
    def _count_syllables(self, name: str) -> int:
        """Count syllables in name."""
        name = name.lower()
        
        # Simple syllable counting
        count = 0
        previous_was_vowel = False
        
        for char in name:
            is_vowel = char in 'aeiouy'
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        return max(1, count)
    
    def _calculate_harshness(self, name_upper: str) -> float:
        """
        Calculate phonetic harshness score.
        
        Based on: Proportion of hard consonants (K, G, T, D, P, B)
        """
        consonants = [c for c in name_upper if c.isalpha() and c not in self.vowels]
        if not consonants:
            return 0.0
        
        harsh_count = sum(1 for c in consonants if c in self.harsh_consonants)
        harshness = harsh_count / len(consonants)
        
        return harshness
    
    def _calculate_memorability(self, name: str) -> float:
        """
        Calculate name memorability.
        
        Factors:
        - Uniqueness (unusual letter combinations)
        - Phonetic distinctiveness
        - Rhythmic quality
        - Not too long, not too short
        """
        # Optimal length (5-8 chars for last name)
        last_name = name.split()[-1] if ' ' in name else name
        length_score = 1.0 - abs(len(last_name) - 6.5) / 10.0
        length_score = max(0, length_score)
        
        # Syllable rhythm (2-3 syllables optimal)
        syllables = self._count_syllables(name)
        syllable_score = 1.0 - abs(syllables - 2.5) / 5.0
        syllable_score = max(0, syllable_score)
        
        # Distinctiveness (uncommon letters = more memorable)
        rare_letters = set('QXZJK')
        has_rare = any(c in name.upper() for c in rare_letters)
        rare_score = 0.3 if has_rare else 0.0
        
        # Combine
        memorability = (length_score * 0.4 + syllable_score * 0.4 + rare_score * 0.2)
        
        return memorability
    
    def analyze_roster(self, player_names: List[str], importance_weights: List[float] = None) -> Dict[str, float]:
        """
        Analyze full roster, weighted by player importance.
        
        Parameters
        ----------
        player_names : list
            List of player names
        importance_weights : list, optional
            Importance weight per player (minutes, usage, etc.)
        
        Returns
        -------
        roster_metrics : dict
            Aggregated roster name features
        """
        if importance_weights is None:
            importance_weights = [1.0] * len(player_names)
        
        # Normalize weights
        total_weight = sum(importance_weights)
        weights_norm = [w / total_weight for w in importance_weights]
        
        # Analyze each player
        player_analyses = [self.analyze_name(name) for name in player_names]
        
        # Aggregate weighted
        roster_metrics = {}
        
        for feature in player_analyses[0].keys():
            weighted_avg = sum(
                analysis[feature] * weight
                for analysis, weight in zip(player_analyses, weights_norm)
            )
            roster_metrics[f'roster_{feature}'] = weighted_avg
        
        # Add roster-level features
        roster_metrics['roster_size'] = len(player_names)
        roster_metrics['name_diversity'] = len(set(player_names)) / len(player_names)
        
        # Star player dominance (top weighted player)
        if importance_weights:
            max_weight = max(importance_weights)
            roster_metrics['star_dominance'] = max_weight / total_weight
        
        return roster_metrics


def load_basketball_athletes(sports_meta_path: str) -> Dict[str, Dict]:
    """Load player name data from sports meta-analysis."""
    import json
    
    with open(f'{sports_meta_path}/basketball_athletes.json', 'r') as f:
        athletes = json.load(f)
    
    # Index by name for quick lookup
    athlete_dict = {}
    for athlete in athletes:
        name = athlete['full_name']
        athlete_dict[name] = athlete
    
    return athlete_dict


def create_player_analyzer():
    """Factory function."""
    return PlayerNameAnalyzer()

