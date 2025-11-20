"""
Coupling Strength Transformer (κ Measurement)

Measures narrator-narrated coupling strength.
κ is central to Δ = π × r × κ formula.

κ = 1.0: Perfect self-coupling (narrator judges themselves)
κ < 1.0: External evaluation (others judge)

Author: Narrative Integration System
Date: November 2025
"""

import re
import numpy as np
from typing import List, Dict, Set
from collections import Counter

from .base import NarrativeTransformer


class CouplingStrengthTransformer(NarrativeTransformer):
    """
    Extracts features measuring narrator-narrated coupling (κ).
    
    Theory: Δ = π × r × κ
    - κ = 1: Self-rated (narrator = judge)
    - κ < 1: External evaluation
    
    The coupling strength determines how directly narrative quality
    translates to outcomes. Perfect coupling (self-perception) means
    the narrator directly controls the outcome through their narrative.
    
    Features Extracted (12 total):
    1. Self-referential density (I/me/my per 100 words)
    2. First-person percentage (vs third-person)
    3. Internal validation markers ("I feel", "I believe", "I think")
    4. External validation markers ("they said", "rated as", "judged")
    5. Self-evaluation language density
    6. External evaluation language density
    7. Judge-narrator distance score
    8. Self-agency markers ("I did", "I achieved", "I chose")
    9. External agency markers ("was given", "was selected", "was rated")
    10. Coupling score (0-1, where 1 = perfect self-coupling)
    11. Narrative stance (internal vs external perspective)
    12. Control locus (internal vs external)
    
    Parameters
    ----------
    detect_third_person : bool, default=True
        Whether to detect third-person narrative
    
    Examples
    --------
    >>> transformer = CouplingStrengthTransformer()
    >>> features = transformer.fit_transform(narratives)
    >>> 
    >>> # Check coupling strength
    >>> coupling_scores = features[:, 9]  # Column 10 is coupling score
    >>> print(f"Average κ: {coupling_scores.mean():.2f}")
    >>> 
    >>> # High κ (~1.0) = self-rated domain
    >>> # Low κ (~0.3) = external evaluation domain
    """
    
    def __init__(self, detect_third_person: bool = True):
        super().__init__(
            narrative_id="coupling_strength",
            description="Measures κ (narrator-narrated coupling) for Δ = π × r × κ"
        )
        
        self.detect_third_person = detect_third_person
        
        # Pronoun patterns
        self.first_person_pronouns = {
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        }
        
        self.third_person_pronouns = {
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'they', 'them', 'their', 'theirs', 'themselves', 'it', 'its', 'itself'
        }
        
        # Validation markers
        self.internal_validation = [
            r'\bi feel\b', r'\bi believe\b', r'\bi think\b', r'\bi know\b',
            r'\bmy opinion\b', r'\bmy view\b', r'\bto me\b', r'\bi see\b',
            r'\bi understand\b', r'\bi perceive\b', r'\bin my experience\b'
        ]
        
        self.external_validation = [
            r'\bthey said\b', r'\brated as\b', r'\bjudged\b', r'\bscored\b',
            r'\bwas rated\b', r'\bwas judged\b', r'\bwas scored\b',
            r'\breviewed as\b', r'\bevaluated as\b', r'\bconsidered\b',
            r'\bperceived as\b', r'\bseen as\b', r'\bregarded as\b'
        ]
        
        # Self-evaluation language
        self.self_evaluation = [
            r'\bi am\b', r'\bi\'m\b', r'\bi was\b', r'\bi have\b',
            r'\bmy strength\b', r'\bmy weakness\b', r'\bi excel\b',
            r'\bi struggle\b', r'\bmy ability\b', r'\bmy skill\b'
        ]
        
        # External evaluation language
        self.external_evaluation = [
            r'\bwas described as\b', r'\bwas called\b', r'\bwas named\b',
            r'\breceived\b', r'\bawarded\b', r'\brecognized as\b',
            r'\backnowledged as\b', r'\blabeled as\b'
        ]
        
        # Agency markers
        self.self_agency = [
            r'\bi did\b', r'\bi made\b', r'\bi created\b', r'\bi achieved\b',
            r'\bi chose\b', r'\bi decided\b', r'\bi selected\b', r'\bi built\b',
            r'\bi accomplished\b', r'\bi completed\b', r'\bi performed\b'
        ]
        
        self.external_agency = [
            r'\bwas given\b', r'\bwas assigned\b', r'\bwas selected\b',
            r'\bwas chosen\b', r'\bwas rated\b', r'\bwas deemed\b',
            r'\bwas considered\b', r'\bwas awarded\b', r'\breceived\b'
        ]
        
        # Domain statistics
        self.domain_mean_coupling_ = None
        self.domain_std_coupling_ = None
    
    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """
        Count matches for regex patterns.
        
        Parameters
        ----------
        text : str
            Input text
        patterns : list of str
            Regex patterns
        
        Returns
        -------
        count : int
            Total matches
        """
        text_lower = text.lower()
        count = 0
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        
        return count
    
    def _compute_pronoun_ratio(self, text: str) -> float:
        """
        Compute first-person / (first + third person) ratio.
        
        Parameters
        ----------
        text : str
            Input text
        
        Returns
        -------
        ratio : float
            First-person ratio (0-1)
        """
        words = text.lower().split()
        
        first_count = sum(1 for w in words if w.strip('.,!?;:') in self.first_person_pronouns)
        third_count = sum(1 for w in words if w.strip('.,!?;:') in self.third_person_pronouns)
        
        total = first_count + third_count
        
        if total == 0:
            return 0.5  # Neutral
        
        return first_count / total
    
    def _compute_coupling_score(
        self,
        self_ref_density: float,
        first_person_ratio: float,
        internal_val: float,
        external_val: float,
        self_eval: float,
        external_eval: float,
        self_agency: float,
        external_agency: float
    ) -> float:
        """
        Compute overall coupling score (κ).
        
        Combines multiple indicators into single score.
        
        Parameters
        ----------
        Various density/ratio measurements
        
        Returns
        -------
        coupling : float
            Coupling strength κ ∈ [0, 1]
        """
        # Normalize densities (cap at reasonable values)
        max_density = 10.0
        self_ref_norm = min(self_ref_density / max_density, 1.0)
        
        # Internal vs external validation
        total_val = internal_val + external_val + 0.001  # Avoid division by zero
        internal_val_ratio = internal_val / total_val
        
        # Self vs external evaluation
        total_eval = self_eval + external_eval + 0.001
        self_eval_ratio = self_eval / total_eval
        
        # Self vs external agency
        total_agency = self_agency + external_agency + 0.001
        self_agency_ratio = self_agency / total_agency
        
        # Weighted combination
        # Higher weight on validation and agency (direct indicators)
        coupling = (
            0.15 * self_ref_norm +           # Self-referential density
            0.20 * first_person_ratio +      # First-person perspective
            0.25 * internal_val_ratio +      # Internal validation
            0.20 * self_eval_ratio +         # Self-evaluation
            0.20 * self_agency_ratio         # Self-agency
        )
        
        # Ensure in [0, 1]
        coupling = max(0.0, min(1.0, coupling))
        
        return coupling
    
    def _compute_narrative_stance(self, first_person_ratio: float, coupling: float) -> float:
        """
        Compute narrative stance score.
        
        1.0 = fully internal perspective
        0.0 = fully external perspective
        
        Parameters
        ----------
        first_person_ratio : float
            First-person pronoun ratio
        coupling : float
            Coupling score
        
        Returns
        -------
        stance : float
            Narrative stance (0-1)
        """
        return (first_person_ratio + coupling) / 2
    
    def _compute_locus_of_control(self, self_agency: float, external_agency: float) -> float:
        """
        Compute locus of control.
        
        1.0 = internal locus (I control outcomes)
        0.0 = external locus (others/circumstances control)
        
        Parameters
        ----------
        self_agency : float
            Self-agency markers
        external_agency : float
            External agency markers
        
        Returns
        -------
        locus : float
            Locus of control (0-1)
        """
        total = self_agency + external_agency + 0.001
        return self_agency / total
    
    def fit(self, X, y=None):
        """
        Learn domain coupling statistics.
        
        Parameters
        ----------
        X : list of str
            Training texts
        y : ignored
        
        Returns
        -------
        self
        """
        # Compute coupling scores for all texts
        coupling_scores = []
        
        for text in X:
            words = text.split()
            n_words = len(words)
            
            if n_words == 0:
                coupling_scores.append(0.5)
                continue
            
            # Quick coupling estimate
            self_ref_density = self._count_pattern_matches(text, [r'\b(i|me|my|mine)\b']) / n_words * 100
            first_person_ratio = self._compute_pronoun_ratio(text)
            internal_val = self._count_pattern_matches(text, self.internal_validation)
            external_val = self._count_pattern_matches(text, self.external_validation)
            self_eval = self._count_pattern_matches(text, self.self_evaluation)
            external_eval = self._count_pattern_matches(text, self.external_evaluation)
            self_agency = self._count_pattern_matches(text, self.self_agency)
            external_agency = self._count_pattern_matches(text, self.external_agency)
            
            coupling = self._compute_coupling_score(
                self_ref_density, first_person_ratio,
                internal_val, external_val,
                self_eval, external_eval,
                self_agency, external_agency
            )
            
            coupling_scores.append(coupling)
        
        self.domain_mean_coupling_ = np.mean(coupling_scores)
        self.domain_std_coupling_ = np.std(coupling_scores)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Extract coupling strength features.
        
        Parameters
        ----------
        X : list of str
            Texts to transform
        
        Returns
        -------
        features : np.ndarray, shape (n_samples, 12)
            Coupling strength features
        """
        self._validate_fitted()
        
        features = []
        
        for text in X:
            words = text.split()
            n_words = len(words)
            
            if n_words == 0:
                # Empty text - return zeros
                features.append([0.0] * 12)
                continue
            
            # 1. Self-referential density (I/me/my per 100 words)
            self_ref_density = self._count_pattern_matches(text, [r'\b(i|me|my|mine)\b']) / n_words * 100
            
            # 2. First-person percentage
            first_person_ratio = self._compute_pronoun_ratio(text)
            
            # 3. Internal validation markers
            internal_val = self._count_pattern_matches(text, self.internal_validation) / n_words * 100
            
            # 4. External validation markers
            external_val = self._count_pattern_matches(text, self.external_validation) / n_words * 100
            
            # 5. Self-evaluation language density
            self_eval = self._count_pattern_matches(text, self.self_evaluation) / n_words * 100
            
            # 6. External evaluation language density
            external_eval = self._count_pattern_matches(text, self.external_evaluation) / n_words * 100
            
            # 7. Judge-narrator distance (inverse of coupling)
            # Will compute after coupling score
            
            # 8. Self-agency markers
            self_agency = self._count_pattern_matches(text, self.self_agency) / n_words * 100
            
            # 9. External agency markers
            external_agency = self._count_pattern_matches(text, self.external_agency) / n_words * 100
            
            # 10. Coupling score (κ)
            coupling = self._compute_coupling_score(
                self_ref_density, first_person_ratio,
                internal_val, external_val,
                self_eval, external_eval,
                self_agency, external_agency
            )
            
            # 7. Judge-narrator distance (now we can compute)
            judge_distance = 1.0 - coupling
            
            # 11. Narrative stance
            narrative_stance = self._compute_narrative_stance(first_person_ratio, coupling)
            
            # 12. Locus of control
            locus_control = self._compute_locus_of_control(self_agency, external_agency)
            
            # Assemble feature vector
            feature_vector = [
                self_ref_density,
                first_person_ratio,
                internal_val,
                external_val,
                self_eval,
                external_eval,
                judge_distance,
                self_agency,
                external_agency,
                coupling,  # This is κ!
                narrative_stance,
                locus_control
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names."""
        return [
            'self_referential_density',
            'first_person_ratio',
            'internal_validation_density',
            'external_validation_density',
            'self_evaluation_density',
            'external_evaluation_density',
            'judge_narrator_distance',
            'self_agency_density',
            'external_agency_density',
            'coupling_score_kappa',  # This is κ!
            'narrative_stance',
            'locus_of_control'
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of learned patterns."""
        return (
            f"Coupling Strength Analysis (κ Measurement)\n"
            f"==========================================\n\n"
            f"Domain Statistics:\n"
            f"  • Mean coupling (κ): {self.domain_mean_coupling_:.3f}\n"
            f"  • Std deviation: {self.domain_std_coupling_:.3f}\n\n"
            f"Interpretation:\n"
            f"  • κ ≈ 1.0: Self-rated domain (narrator judges self)\n"
            f"  • κ ≈ 0.5: Mixed evaluation\n"
            f"  • κ ≈ 0.3: External evaluation (others judge)\n\n"
            f"This domain: κ = {self.domain_mean_coupling_:.3f}\n"
            f"  → {'Self-rated' if self.domain_mean_coupling_ > 0.7 else 'External evaluation' if self.domain_mean_coupling_ < 0.4 else 'Mixed'}\n\n"
            f"Theory: Δ = π × r × κ\n"
            f"  κ determines how directly narrative quality affects outcomes.\n"
            f"  Higher κ = stronger narrative-outcome coupling.\n"
        )

