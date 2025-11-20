"""
Compatibility Analyzer for Marriage Prediction

Implements four competing theories of name compatibility:
1. Similarity Theory: Similar names → compatibility
2. Complementarity Theory: Opposite names → balance
3. Golden Ratio Theory: φ relationship → harmony
4. Resonance Theory: Harmonic ratios → success

Tests which theory best predicts relationship outcomes.

Author: Narrative Optimization Research
Date: November 2025
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats


class CompatibilityAnalyzer:
    """
    Analyze name compatibility using multiple theoretical frameworks.
    """
    
    PHI = 1.618033988749  # Golden ratio
    
    def __init__(self):
        """Initialize compatibility analyzer."""
        pass
    
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Theory 1: Similarity predicts compatibility.
        
        Parameters
        ----------
        name1, name2 : str
            Partner names
        
        Returns
        -------
        float
            Similarity score (0-1, higher = more similar)
        """
        # Edit distance similarity
        edit_dist = self._levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        edit_similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0
        
        # Syllable similarity
        syll1 = self._count_syllables(name1)
        syll2 = self._count_syllables(name2)
        syll_similarity = 1.0 - abs(syll1 - syll2) / max(syll1, syll2)
        
        # Length similarity
        len_similarity = 1.0 - abs(len(name1) - len(name2)) / max(len(name1), len(name2))
        
        # Weighted average
        similarity = (0.4 * edit_similarity + 
                     0.3 * syll_similarity + 
                     0.3 * len_similarity)
        
        return similarity
    
    def calculate_complementarity(self, name1: str, name2: str) -> float:
        """
        Theory 2: Complementarity predicts compatibility.
        
        Opposite characteristics create balance.
        
        Parameters
        ----------
        name1, name2 : str
            Partner names
        
        Returns
        -------
        float
            Complementarity score (0-1)
        """
        # Complementarity is inverse of similarity
        similarity = self.calculate_similarity(name1, name2)
        complementarity = 1.0 - similarity
        
        # But extreme dissimilarity is bad (cultural/class mismatch)
        # Optimal complementarity is moderate difference
        if complementarity > 0.7:  # Too different
            complementarity *= 0.7
        
        return complementarity
    
    def calculate_golden_ratio_fit(self, name1: str, name2: str) -> float:
        """
        Theory 3: Golden ratio (φ ≈ 1.618) predicts harmony.
        
        Names with syllable or length ratios near φ → optimal compatibility.
        
        Parameters
        ----------
        name1, name2 : str
            Partner names
        
        Returns
        -------
        float
            Golden ratio fit score (0-1, higher = closer to φ)
        """
        syll1 = self._count_syllables(name1)
        syll2 = self._count_syllables(name2)
        
        # Calculate ratio (always larger / smaller)
        syll_ratio = max(syll1, syll2) / min(syll1, syll2) if min(syll1, syll2) > 0 else 1.0
        
        # Distance from φ
        phi_distance = abs(syll_ratio - self.PHI)
        
        # Convert to score (closer = better)
        phi_score = max(0, 1.0 - phi_distance / self.PHI)
        
        return phi_score
    
    def calculate_resonance(self, name1: str, name2: str) -> float:
        """
        Theory 4: Harmonic ratios predict success.
        
        Musical harmony ratios (2:1, 3:2, 4:3) in name features.
        
        Parameters
        ----------
        name1, name2 : str
            Partner names
        
        Returns
        -------
        float
            Resonance score (0-1)
        """
        syll1 = self._count_syllables(name1)
        syll2 = self._count_syllables(name2)
        
        # Calculate ratio
        ratio = max(syll1, syll2) / min(syll1, syll2) if min(syll1, syll2) > 0 else 1.0
        
        # Check against harmonic ratios
        harmonic_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]  # 1:1, 3:2, 2:1, 5:2, 3:1
        
        # Find closest harmonic
        distances = [abs(ratio - hr) for hr in harmonic_ratios]
        min_distance = min(distances)
        
        # Convert to score
        resonance = max(0, 1.0 - min_distance)
        
        return resonance
    
    def calculate_all_theories(self, name1: str, name2: str) -> Dict:
        """
        Calculate scores for all four theories.
        
        Parameters
        ----------
        name1, name2 : str
            Partner names
        
        Returns
        -------
        dict
            Scores for all theories
        """
        return {
            'similarity': self.calculate_similarity(name1, name2),
            'complementarity': self.calculate_complementarity(name1, name2),
            'golden_ratio': self.calculate_golden_ratio_fit(name1, name2),
            'resonance': self.calculate_resonance(name1, name2)
        }
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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
    
    def _count_syllables(self, name: str) -> int:
        """Count syllables in name."""
        import re
        name = name.lower()
        name = re.sub(r'e$', '', name)
        vowels = 'aeiouy'
        syllables = 0
        previous_was_vowel = False
        
        for char in name:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        return max(1, syllables)
    
    def test_theories(self, couples: List[Dict]) -> Dict:
        """
        Test which theory best predicts outcomes.
        
        Parameters
        ----------
        couples : list of dict
            Couple records with names and outcomes
        
        Returns
        -------
        dict
            Theory comparison results
        """
        if not couples:
            return {'error': 'No couples data'}
        
        # Calculate theory scores for all couples
        similarity_scores = []
        complementarity_scores = []
        phi_scores = []
        resonance_scores = []
        outcomes = []
        
        for couple in couples:
            name1 = couple.get('name1', couple.get('partner1_name', ''))
            name2 = couple.get('name2', couple.get('partner2_name', ''))
            outcome = couple.get('relative_success', couple.get('duration_years'))
            
            if name1 and name2 and outcome is not None:
                theories = self.calculate_all_theories(name1, name2)
                
                similarity_scores.append(theories['similarity'])
                complementarity_scores.append(theories['complementarity'])
                phi_scores.append(theories['golden_ratio'])
                resonance_scores.append(theories['resonance'])
                outcomes.append(float(outcome))
        
        if len(outcomes) < 3:
            return {'error': 'Insufficient data for theory testing'}
        
        outcomes = np.array(outcomes)
        
        # Test each theory
        results = {}
        
        for theory_name, scores in [
            ('similarity', similarity_scores),
            ('complementarity', complementarity_scores),
            ('golden_ratio', phi_scores),
            ('resonance', resonance_scores)
        ]:
            r, p = stats.pearsonr(scores, outcomes)
            results[theory_name] = {
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'r_squared': r ** 2
            }
        
        # Determine winner
        winner = max(results.items(), key=lambda x: abs(x[1]['correlation']))
        
        return {
            'n_couples': len(outcomes),
            'theories': results,
            'winner': {
                'theory': winner[0],
                'correlation': winner[1]['correlation'],
                'p_value': winner[1]['p_value']
            }
        }


if __name__ == '__main__':
    # Demo
    analyzer = CompatibilityAnalyzer()
    
    theories = analyzer.calculate_all_theories('James', 'Jennifer')
    print("Compatibility Theories:")
    for theory, score in theories.items():
        print(f"  {theory}: {score:.3f}")

