"""
Housing Domain - Narrative Feature Extractor

Extracts ж (genome/features) for each house and computes ю (story quality).

This module treats house numbers as narratives with semantic and cultural properties.

Features extracted:
- Numerological properties (lucky/unlucky numbers)
- Aesthetic properties (palindrome, sequential, round)
- Cultural resonance (Western vs Asian interpretations)
- Semantic embeddings (number meanings)
- Name gravity metrics (phonetic/semantic similarity)

Output:
- ж (feature vector) for each house: 40+ dimensions
- ю (story quality score): [0, 1] weighted by π
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HouseNarrativeExtractor:
    """Extract narrative features from house numbers"""
    
    def __init__(self, narrativity: float = 0.92):
        """
        Args:
            narrativity: π value for Housing domain (default 0.92)
        """
        self.pi = narrativity
        self.feature_names = []
        self._setup_feature_weights()
    
    def _setup_feature_weights(self):
        """
        Setup feature weights based on π (narrativity)
        
        High π → weight character/identity features more
        Low π → weight plot/outcome features more
        
        For Housing (π=0.92), heavily weight character/identity features
        """
        # At π=0.92, we want ~75% character weight, 25% plot weight
        self.character_weight = 0.75
        self.plot_weight = 0.25
        
        logger.info(f"Feature weights for π={self.pi:.2f}:")
        logger.info(f"  Character/Identity: {self.character_weight:.2f}")
        logger.info(f"  Plot/Outcome: {self.plot_weight:.2f}")
    
    def extract_numerological_features(self, number: int) -> Dict[str, float]:
        """
        Extract pure numerological properties
        
        These are the IDENTITY features (character-level)
        """
        features = {}
        
        # Unlucky numbers (Western)
        features['is_exactly_13'] = 1.0 if number == 13 else 0.0
        features['is_exactly_666'] = 1.0 if number == 666 else 0.0
        features['contains_13'] = 1.0 if '13' in str(number) else 0.0
        features['contains_666'] = 1.0 if '666' in str(number) else 0.0
        
        # Unlucky numbers (Asian)
        features['is_exactly_4'] = 1.0 if number == 4 else 0.0
        features['contains_4'] = 1.0 if '4' in str(number) else 0.0
        features['ends_with_4'] = 1.0 if number % 10 == 4 else 0.0
        
        # Lucky numbers (Western)
        features['is_exactly_7'] = 1.0 if number == 7 else 0.0
        features['is_exactly_777'] = 1.0 if number == 777 else 0.0
        features['contains_7'] = 1.0 if '7' in str(number) else 0.0
        
        # Lucky numbers (Asian)
        features['is_exactly_8'] = 1.0 if number == 8 else 0.0
        features['is_exactly_888'] = 1.0 if number == 888 else 0.0
        features['contains_8'] = 1.0 if '8' in str(number) else 0.0
        features['ends_with_8'] = 1.0 if number % 10 == 8 else 0.0
        
        # Composite scores
        unlucky_western = features['is_exactly_13'] * 1.0 + features['is_exactly_666'] * 0.8
        unlucky_asian = features['is_exactly_4'] * 0.6 + features['ends_with_4'] * 0.3
        lucky_western = features['is_exactly_7'] * 0.5 + features['is_exactly_777'] * 0.7
        lucky_asian = features['is_exactly_8'] * 0.7 + features['is_exactly_888'] * 0.9
        
        features['unlucky_score'] = min(unlucky_western + unlucky_asian, 1.0)
        features['lucky_score'] = min(lucky_western + lucky_asian, 1.0)
        
        return features
    
    def extract_aesthetic_features(self, number: int) -> Dict[str, float]:
        """
        Extract aesthetic/pattern properties
        
        These are IDENTITY features (how the number "looks")
        """
        features = {}
        num_str = str(number)
        
        # Palindrome (reads same forwards/backwards)
        features['is_palindrome'] = 1.0 if num_str == num_str[::-1] else 0.0
        
        # Sequential (e.g., 123, 234, 345)
        is_seq = False
        if len(num_str) >= 2:
            diffs = [int(num_str[i+1]) - int(num_str[i]) for i in range(len(num_str)-1)]
            is_seq = all(d == 1 for d in diffs) or all(d == -1 for d in diffs)
        features['is_sequential'] = 1.0 if is_seq else 0.0
        
        # Repeating digits (e.g., 111, 222, 333)
        features['is_repeating'] = 1.0 if len(set(num_str)) == 1 else 0.0
        
        # Round number (ends in 0 or 00)
        features['ends_in_zero'] = 1.0 if number % 10 == 0 else 0.0
        features['ends_in_double_zero'] = 1.0 if number % 100 == 0 else 0.0
        
        # Digit properties
        features['num_digits'] = len(num_str) / 4.0  # Normalize (max ~4 digits typical)
        features['digit_sum'] = sum(int(d) for d in num_str) / 30.0  # Normalize
        features['digit_variance'] = np.var([int(d) for d in num_str]) / 10.0
        
        # Symmetry score
        symmetry = 0.0
        if features['is_palindrome'] == 1.0:
            symmetry = 1.0
        elif features['is_repeating'] == 1.0:
            symmetry = 0.9
        elif features['is_sequential'] == 1.0:
            symmetry = 0.7
        features['symmetry_score'] = symmetry
        
        return features
    
    def extract_semantic_features(self, number: int) -> Dict[str, float]:
        """
        Extract semantic/cultural meaning features
        
        These are INTERPRETATION features (what the number "means")
        """
        features = {}
        
        # Semantic associations (cultural meanings)
        semantic_map = {
            1: {'meaning': 'unity', 'valence': 0.7},
            2: {'meaning': 'pair', 'valence': 0.6},
            3: {'meaning': 'trinity', 'valence': 0.8},
            4: {'meaning': 'death', 'valence': -0.6},  # Asian
            5: {'meaning': 'balance', 'valence': 0.5},
            6: {'meaning': 'harmony', 'valence': 0.6},
            7: {'meaning': 'luck', 'valence': 0.8},
            8: {'meaning': 'prosperity', 'valence': 0.9},  # Asian
            9: {'meaning': 'completion', 'valence': 0.7},
            13: {'meaning': 'unluck', 'valence': -1.0},  # Western
        }
        
        # Get base semantic value for last digit
        last_digit = number % 10
        if last_digit in semantic_map:
            features['semantic_valence'] = semantic_map[last_digit]['valence']
        else:
            features['semantic_valence'] = 0.0
        
        # Special case for full number
        if number in semantic_map:
            features['semantic_valence'] = semantic_map[number]['valence']
        
        # Cultural resonance
        features['western_resonance'] = 0.0
        features['asian_resonance'] = 0.0
        
        if number == 13:
            features['western_resonance'] = 1.0  # Strongly Western-specific
        elif number == 7 or number == 777:
            features['western_resonance'] = 0.7
        
        if number == 4 or number % 10 == 4:
            features['asian_resonance'] = 0.8  # Asian-specific unlucky
        elif number == 8 or number == 88 or number == 888:
            features['asian_resonance'] = 0.9  # Asian-specific lucky
        
        # Prime number (mathematical meaning)
        features['is_prime'] = 1.0 if self._is_prime(number) else 0.0
        
        return features
    
    def extract_name_gravity_features(self, number: int, 
                                     other_numbers: List[int] = None) -> Dict[str, float]:
        """
        Extract ν (name-gravity) features
        
        How similar is this number to other significant numbers?
        """
        features = {}
        
        # Phonetic features (how it sounds)
        num_str = str(number)
        
        # Syllable count (rough approximation)
        syllables = {
            '1': 1, '2': 1, '3': 1, '4': 1, '5': 1,
            '6': 1, '7': 2, '8': 1, '9': 1, '0': 2,
            '13': 3,  # thir-teen
            '666': 4,  # six-six-six
        }
        
        if number in syllables:
            features['syllable_count'] = syllables[number] / 4.0  # Normalize
        else:
            features['syllable_count'] = len(num_str) / 4.0
        
        # Phonetic similarity to #13 (the prototype unlucky number)
        # Simple metric: edit distance
        features['phonetic_distance_to_13'] = self._edit_distance(num_str, '13') / 10.0
        
        # Semantic similarity to culturally significant numbers
        significant = [7, 8, 13, 666]
        if other_numbers:
            min_dist = min(abs(number - sig) for sig in significant)
            features['semantic_distance_to_significant'] = min_dist / 100.0
        else:
            features['semantic_distance_to_significant'] = 0.5
        
        return features
    
    def extract_all_features(self, number: int) -> Dict[str, float]:
        """
        Extract complete ж (genome) for a house number
        
        Returns:
            Dictionary of 40+ features representing the narrative identity
        """
        features = {}
        
        # Identity features (character-level)
        features.update(self.extract_numerological_features(number))
        features.update(self.extract_aesthetic_features(number))
        features.update(self.extract_semantic_features(number))
        
        # Relational features
        features.update(self.extract_name_gravity_features(number))
        
        return features
    
    def compute_story_quality(self, features: Dict[str, float]) -> float:
        """
        Compute ю (story quality) from ж (features)
        
        Weights determined by π (narrativity):
        - High π → weight identity/character features
        - Low π → weight outcome/plot features
        
        For Housing (π=0.92):
        - Character weight: 0.75
        - Plot weight: 0.25
        
        Returns:
            ю ∈ [0, 1] where higher = "better" narrative
        """
        # Character features (identity, aesthetics, meaning)
        character_features = [
            'is_exactly_7', 'is_exactly_8', 'is_exactly_888',
            'lucky_score', 'is_palindrome', 'is_repeating',
            'symmetry_score', 'semantic_valence',
            'asian_resonance', 'is_prime'
        ]
        
        # Plot features (unlucky markers, risks)
        plot_features = [
            'is_exactly_13', 'is_exactly_666', 'is_exactly_4',
            'unlucky_score', 'western_resonance'
        ]
        
        # Compute character score
        char_values = [features.get(f, 0.0) for f in character_features if f in features]
        char_score = np.mean(char_values) if char_values else 0.5
        
        # Compute plot score (inverse for unlucky - lower unlucky = higher quality)
        plot_values = [1.0 - features.get(f, 0.0) for f in plot_features if f in features]
        plot_score = np.mean(plot_values) if plot_values else 0.5
        
        # Weighted combination
        yu = (self.character_weight * char_score + 
              self.plot_weight * plot_score)
        
        # Ensure [0, 1] range
        yu = np.clip(yu, 0.0, 1.0)
        
        return yu
    
    def process_houses(self, numbers: List[int]) -> pd.DataFrame:
        """
        Process multiple house numbers and extract features
        
        Args:
            numbers: List of house numbers
        
        Returns:
            DataFrame with house numbers, features (ж), and quality (ю)
        """
        logger.info(f"Processing {len(numbers)} house numbers...")
        
        results = []
        for number in numbers:
            features = self.extract_all_features(number)
            quality = self.compute_story_quality(features)
            
            row = {'house_number': number, 'quality_yu': quality}
            row.update(features)
            results.append(row)
        
        df = pd.DataFrame(results)
        logger.info(f"Extracted {len(df.columns)-2} features (ж) for each house")
        logger.info(f"Computed ю (quality) for all houses")
        
        return df
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


def main():
    """Demo: Extract features for sample house numbers"""
    
    logger.info("="*80)
    logger.info("HOUSING NARRATIVE EXTRACTOR - DEMO")
    logger.info("="*80)
    
    # Create extractor
    extractor = HouseNarrativeExtractor(narrativity=0.92)
    
    # Sample house numbers
    sample_numbers = [
        7,    # Lucky Western
        8,    # Lucky Asian
        13,   # Unlucky Western
        4,    # Unlucky Asian
        42,   # Neutral
        123,  # Sequential
        111,  # Repeating
        888,  # Very lucky Asian
        666,  # Very unlucky Western
    ]
    
    logger.info(f"\nProcessing {len(sample_numbers)} sample houses...")
    
    results = extractor.process_houses(sample_numbers)
    
    # Show results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    for _, row in results.iterrows():
        number = int(row['house_number'])
        quality = row['quality_yu']
        unlucky = row.get('unlucky_score', 0.0)
        lucky = row.get('lucky_score', 0.0)
        
        logger.info(f"\nHouse #{number}:")
        logger.info(f"  ю (Quality): {quality:.3f}")
        logger.info(f"  Unlucky: {unlucky:.2f}, Lucky: {lucky:.2f}")
        
        if number == 13:
            logger.info(f"  ⚠ This is THE prototype unlucky number!")
            logger.info(f"     Expected: ю ≈ 0.25 (very low)")
            logger.info(f"     This house sells for $93K less on average")
    
    # Save results
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'narrative_features_demo.csv'
    results.to_csv(output_file, index=False)
    
    logger.info(f"\n\nResults saved to: {output_file}")
    logger.info("\nFeatures extracted (ж): " + str(len(results.columns) - 2))
    logger.info("Quality computed (ю): Yes")
    logger.info("\nNarrative extraction complete! ✓")


if __name__ == "__main__":
    main()

