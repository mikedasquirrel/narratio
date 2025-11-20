"""
Narrative Mass Transformer (μ Measurement)

Measures importance and stakes (μ = importance × stakes).
Needed for gravitational force calculations (φ and ة).

μ represents the "gravitational mass" of a narrative instance.
Higher mass instances exert stronger gravitational pull.

Author: Narrative Integration System
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Set
from collections import Counter

from .base import NarrativeTransformer


class NarrativeMassTransformer(NarrativeTransformer):
    """
    Extracts features measuring narrative mass (μ).
    
    Theory: μ = importance × stakes
    - Used in gravitational calculations: φ = (μ₁ × μ₂ × similarity) / distance²
    - Higher mass instances have greater influence on narrative space
    
    Mass Components:
    1. Stakes: How much is on the line?
    2. Importance: How significant is this instance?
    3. Gravitas: How serious/weighty is the language?
    
    Typical ranges:
    - Routine instance: μ ≈ 0.5-1.0
    - Important instance: μ ≈ 1.5-2.0
    - Critical/historic instance: μ ≈ 2.5-3.0
    
    Features Extracted (10 total):
    1. Stakes indicators (championship, final, decisive, career-defining)
    2. Importance markers (historic, legendary, pivotal, milestone)
    3. Temporal urgency (now-or-never, last chance, crucial moment)
    4. Consequence magnitude (life-changing vs routine)
    5. Gravitas score (serious, weighty language)
    6. Significance markers (major, significant, critical)
    7. Legacy language (legacy, immortal, eternal, remembered)
    8. Finality markers (final, ultimate, climactic)
    9. Magnitude language (massive, enormous, monumental)
    10. Mass estimate μ ∈ [0.3, 3.0]
    
    Parameters
    ----------
    mass_range : tuple, default=(0.3, 3.0)
        Min and max mass values
    
    Examples
    --------
    >>> transformer = NarrativeMassTransformer()
    >>> features = transformer.fit_transform(narratives)
    >>> 
    >>> # Check mass distribution
    >>> mass_values = features[:, 9]  # Column 10 is μ
    >>> print(f"Average μ: {mass_values.mean():.2f}")
    >>> print(f"Max μ: {mass_values.max():.2f}")
    >>> 
    >>> # High-mass instances (μ > 2.0) have strongest gravitational pull
    >>> high_mass = mass_values > 2.0
    >>> print(f"High-mass instances: {high_mass.sum()}")
    """
    
    def __init__(self, mass_range: tuple = (0.3, 3.0)):
        super().__init__(
            narrative_id="narrative_mass",
            description="Measures μ (importance × stakes) for gravitational calculations"
        )
        
        self.min_mass, self.max_mass = mass_range
        
        # Stakes indicators
        self.stakes_patterns = [
            # High stakes
            r'\bchampionship\b', r'\bfinal\b', r'\bfinals\b',
            r'\bdecisive\b', r'\bcareer-defining\b', r'\bdo or die\b',
            r'\bmake or break\b', r'\bwin or lose\b', r'\ball or nothing\b',
            r'\bcrucial\b', r'\bpivotal\b', r'\bcritical\b',
            r'\bdefining\b', r'\bdefining moment\b',
            # Tournament/competition context
            r'\bplayoffs\b', r'\bsuper bowl\b', r'\bworld series\b',
            r'\bworld cup\b', r'\bolympics\b', r'\bgrand finale\b',
            r'\belimination\b', r'\bknockout\b'
        ]
        
        # Importance markers
        self.importance_patterns = [
            r'\bhistoric\b', r'\bhistorical\b', r'\blegendary\b',
            r'\bmilestone\b', r'\blandmark\b', r'\bbreakthrough\b',
            r'\bgroundbreaking\b', r'\bepic\b', r'\bmonumental\b',
            r'\bextraordinary\b', r'\bremarkable\b', r'\bunprecedented\b',
            r'\brare\b', r'\bunique\b', r'\bonce-in-a-lifetime\b',
            r'\brecord-breaking\b', r'\bworld-class\b'
        ]
        
        # Temporal urgency
        self.urgency_patterns = [
            r'\bnow or never\b', r'\blast chance\b', r'\bfinal opportunity\b',
            r'\bcrucial moment\b', r'\bdefining moment\b', r'\bmoment of truth\b',
            r'\bpoint of no return\b', r'\bturning point\b',
            r'\bimmediately\b', r'\burgent\b', r'\bpressing\b',
            r'\btime-sensitive\b', r'\bdeadline\b'
        ]
        
        # Consequence magnitude
        self.consequence_patterns = [
            # High consequences
            r'\blife-changing\b', r'\blife-altering\b', r'\btransformative\b',
            r'\bprofound impact\b', r'\bfar-reaching\b', r'\blong-lasting\b',
            r'\bpermanent\b', r'\birreversible\b', r'\bdefining\b',
            # Career/legacy impact
            r'\bcareer\b', r'\blegacy\b', r'\breputation\b',
            r'\bdefine\w* career\b', r'\blegacy on the line\b'
        ]
        
        # Gravitas (serious, weighty language)
        self.gravitas_patterns = [
            r'\bserious\b', r'\bsolemn\b', r'\bgrave\b', r'\bweighty\b',
            r'\bprofound\b', r'\bdeep\b', r'\bsubstantive\b',
            r'\bsignificant\b', r'\bmajor\b', r'\bsubstantial\b',
            r'\bconsequential\b', r'\bmomentous\b'
        ]
        
        # Significance markers
        self.significance_patterns = [
            r'\bmajor\b', r'\bsignificant\b', r'\bcritical\b',
            r'\bvital\b', r'\bessential\b', r'\bkey\b',
            r'\bfundamental\b', r'\bcentral\b', r'\bcore\b',
            r'\bimportant\b', r'\bvaluable\b', r'\bprecious\b'
        ]
        
        # Legacy language
        self.legacy_patterns = [
            r'\blegacy\b', r'\bimmortal\b', r'\beternal\b',
            r'\bremembered\b', r'\bforgotten\b', r'\benduring\b',
            r'\btimeless\b', r'\bforever\b', r'\bhistory books\b',
            r'\bannals\b', r'\bposterity\b'
        ]
        
        # Finality markers
        self.finality_patterns = [
            r'\bfinal\b', r'\bultimate\b', r'\bclimactic\b',
            r'\bconcluding\b', r'\bend\b', r'\blast\b',
            r'\bfinale\b', r'\bculmination\b', r'\bzenith\b'
        ]
        
        # Magnitude language
        self.magnitude_patterns = [
            r'\bmassive\b', r'\benormous\b', r'\bmonumental\b',
            r'\bcolossal\b', r'\bgigantic\b', r'\bimmense\b',
            r'\btremendous\b', r'\bastounding\b', r'\bstaggering\b',
            r'\boverwhelming\b', r'\btowering\b'
        ]
        
        # Routine/low-stakes markers (negative indicators)
        self.routine_patterns = [
            r'\broutine\b', r'\bregular\b', r'\bnormal\b',
            r'\bordinary\b', r'\btypical\b', r'\busual\b',
            r'\baverage\b', r'\bmundane\b', r'\bunremarkable\b'
        ]
        
        # Domain statistics
        self.domain_mean_mass_ = None
        self.domain_std_mass_ = None
    
    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count matches for regex patterns."""
        text_lower = text.lower()
        count = 0
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        
        return count
    
    def _compute_mass(
        self,
        stakes: float,
        importance: float,
        urgency: float,
        consequences: float,
        gravitas: float,
        significance: float,
        legacy: float,
        finality: float,
        magnitude: float,
        routine: float,
        n_words: int
    ) -> float:
        """
        Compute narrative mass (μ).
        
        Combines multiple indicators with weights.
        
        Parameters
        ----------
        Various density measurements
        n_words : int
            Text length (for normalization)
        
        Returns
        -------
        mass : float
            Mass estimate μ ∈ [min_mass, max_mass]
        """
        # Normalize by text length (convert to per 100 words)
        if n_words == 0:
            return 1.0  # Default neutral mass
        
        per_100 = 100.0 / n_words
        
        # Weight different components
        mass_score = (
            0.20 * stakes * per_100 +          # High stakes = high mass
            0.18 * importance * per_100 +      # Important events = high mass
            0.12 * urgency * per_100 +         # Time pressure adds mass
            0.15 * consequences * per_100 +    # High consequences = high mass
            0.10 * gravitas * per_100 +        # Serious tone = higher mass
            0.10 * significance * per_100 +    # Significance markers
            0.08 * legacy * per_100 +          # Legacy implications
            0.05 * finality * per_100 +        # Finality adds mass
            0.07 * magnitude * per_100 -       # Magnitude language
            0.05 * routine * per_100           # Routine reduces mass (negative)
        )
        
        # Normalize to [0, 1] range first
        # Typical high-mass text might have 10-15 total matches per 100 words
        mass_normalized = mass_score / 15.0
        mass_normalized = max(0.0, min(1.0, mass_normalized))
        
        # Scale to [min_mass, max_mass] range
        mass = self.min_mass + mass_normalized * (self.max_mass - self.min_mass)
        
        # Boost for very high indicators (outliers)
        if mass_score > 20:
            mass = min(self.max_mass, mass * 1.2)
        
        return mass
    
    def fit(self, X, y=None):
        """
        Learn domain mass statistics.
        
        Parameters
        ----------
        X : list of str
            Training texts
        y : ignored
        
        Returns
        -------
        self
        """
        # Compute mass for all texts
        mass_values = []
        
        for text in X:
            words = text.split()
            n_words = len(words)
            
            if n_words == 0:
                mass_values.append(1.0)
                continue
            
            # Count pattern matches
            stakes = self._count_pattern_matches(text, self.stakes_patterns)
            importance = self._count_pattern_matches(text, self.importance_patterns)
            urgency = self._count_pattern_matches(text, self.urgency_patterns)
            consequences = self._count_pattern_matches(text, self.consequence_patterns)
            gravitas = self._count_pattern_matches(text, self.gravitas_patterns)
            significance = self._count_pattern_matches(text, self.significance_patterns)
            legacy = self._count_pattern_matches(text, self.legacy_patterns)
            finality = self._count_pattern_matches(text, self.finality_patterns)
            magnitude = self._count_pattern_matches(text, self.magnitude_patterns)
            routine = self._count_pattern_matches(text, self.routine_patterns)
            
            mass = self._compute_mass(
                stakes, importance, urgency, consequences, gravitas,
                significance, legacy, finality, magnitude, routine, n_words
            )
            
            mass_values.append(mass)
        
        self.domain_mean_mass_ = np.mean(mass_values)
        self.domain_std_mass_ = np.std(mass_values)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Extract narrative mass features.
        
        Parameters
        ----------
        X : list of str
            Texts to transform
        
        Returns
        -------
        features : np.ndarray, shape (n_samples, 10)
            Narrative mass features
        """
        self._validate_fitted()
        
        features = []
        
        for text in X:
            words = text.split()
            n_words = len(words)
            
            if n_words == 0:
                # Empty text - return neutral values
                features.append([0.0] * 9 + [1.0])
                continue
            
            # Count pattern matches (raw counts)
            stakes = self._count_pattern_matches(text, self.stakes_patterns)
            importance = self._count_pattern_matches(text, self.importance_patterns)
            urgency = self._count_pattern_matches(text, self.urgency_patterns)
            consequences = self._count_pattern_matches(text, self.consequence_patterns)
            gravitas = self._count_pattern_matches(text, self.gravitas_patterns)
            significance = self._count_pattern_matches(text, self.significance_patterns)
            legacy = self._count_pattern_matches(text, self.legacy_patterns)
            finality = self._count_pattern_matches(text, self.finality_patterns)
            magnitude = self._count_pattern_matches(text, self.magnitude_patterns)
            
            # Compute mass
            routine = self._count_pattern_matches(text, self.routine_patterns)
            mass = self._compute_mass(
                stakes, importance, urgency, consequences, gravitas,
                significance, legacy, finality, magnitude, routine, n_words
            )
            
            # Normalize counts to densities (per 100 words) for features
            per_100 = 100.0 / n_words
            
            # Assemble feature vector
            feature_vector = [
                stakes * per_100,        # 1. Stakes indicators
                importance * per_100,    # 2. Importance markers
                urgency * per_100,       # 3. Temporal urgency
                consequences * per_100,  # 4. Consequence magnitude
                gravitas * per_100,      # 5. Gravitas score
                significance * per_100,  # 6. Significance markers
                legacy * per_100,        # 7. Legacy language
                finality * per_100,      # 8. Finality markers
                magnitude * per_100,     # 9. Magnitude language
                mass                     # 10. Mass estimate μ
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names."""
        return [
            'stakes_indicators_density',
            'importance_markers_density',
            'temporal_urgency_density',
            'consequence_magnitude_density',
            'gravitas_score',
            'significance_markers_density',
            'legacy_language_density',
            'finality_markers_density',
            'magnitude_language_density',
            'narrative_mass_mu'  # This is μ!
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of learned patterns."""
        return (
            f"Narrative Mass Analysis (μ Measurement)\n"
            f"========================================\n\n"
            f"Domain Statistics:\n"
            f"  • Mean mass (μ): {self.domain_mean_mass_:.2f}\n"
            f"  • Std deviation: {self.domain_std_mass_:.2f}\n"
            f"  • Range: [{self.min_mass:.1f}, {self.max_mass:.1f}]\n\n"
            f"Interpretation:\n"
            f"  • μ < 1.0: Routine, low-stakes instances\n"
            f"  • μ ≈ 1.0-1.5: Normal importance\n"
            f"  • μ > 2.0: High-stakes, critical instances\n"
            f"  • μ > 2.5: Historic, career-defining moments\n\n"
            f"This domain: μ = {self.domain_mean_mass_:.2f}\n"
            f"  → {'High-stakes' if self.domain_mean_mass_ > 1.5 else 'Routine' if self.domain_mean_mass_ < 0.8 else 'Normal'}\n\n"
            f"Usage in Theory:\n"
            f"  • Gravitational forces: φ = (μ₁ × μ₂ × similarity) / distance²\n"
            f"  • High-mass instances exert stronger gravitational pull\n"
            f"  • Create narrative clusters around important events\n"
        )

