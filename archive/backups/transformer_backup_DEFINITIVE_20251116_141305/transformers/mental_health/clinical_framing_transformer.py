"""
Clinical Framing Transformer

Analyzes how disorder names are framed (medical vs colloquial terminology)
and how framing affects stigma and treatment seeking.

Medical framing: "Major Depressive Disorder" (clinical, distant)
Colloquial framing: "Depression" (familiar, accessible)

Hypothesis: Overly medical framing → clinical distance → higher stigma

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict
import re

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.base_transformer import TextNarrativeTransformer


class ClinicalFramingTransformer(TextNarrativeTransformer):
    """
    Analyzes clinical vs colloquial framing of disorder names.
    
    Tests whether overly medical terminology creates barriers to help-seeking.
    """
    
    def __init__(self):
        """Initialize clinical framing transformer."""
        super().__init__(
            narrative_id="clinical_framing",
            description="Analyzes medical vs colloquial framing of disorder names"
        )
        
        # Medical terminology markers
        self.clinical_terms = {
            'disorder', 'syndrome', 'disease', 'condition', 'dysfunction',
            'pathology', 'impairment', 'deficit', 'abnormality'
        }
        
        self.diagnostic_modifiers = {
            'major', 'minor', 'acute', 'chronic', 'severe', 'moderate', 'mild',
            'persistent', 'recurrent', 'episodic', 'generalized', 'specific'
        }
        
        self.acronyms = {
            'ocd', 'ptsd', 'adhd', 'mdd', 'gad', 'sad', 'bpd', 'npd', 'aspd'
        }
    
    def fit(self, X, y=None):
        """
        Learn framing patterns from disorder names.
        
        Parameters
        ----------
        X : list of str
            Disorder names
        y : array-like, optional
            Stigma or treatment seeking scores
        
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Analyze framing distribution
        clinical_scores = []
        accessibility_scores = []
        
        for name in X:
            clinical = self._calculate_clinical_score(name)
            accessible = self._calculate_accessibility_score(name)
            
            clinical_scores.append(clinical)
            accessibility_scores.append(accessible)
        
        self.metadata['corpus_stats'] = {
            'n_disorders': len(X),
            'clinical_framing': {
                'mean': np.mean(clinical_scores),
                'std': np.std(clinical_scores),
                'range': (min(clinical_scores), max(clinical_scores))
            },
            'accessibility': {
                'mean': np.mean(accessibility_scores),
                'std': np.std(accessibility_scores)
            }
        }
        
        if y is not None:
            y_array = np.array(y)
            self.metadata['correlations'] = {
                'clinical_vs_stigma': np.corrcoef(clinical_scores, y_array)[0, 1],
                'accessibility_vs_seeking': np.corrcoef(accessibility_scores, y_array)[0, 1]
            }
        
        self.metadata['n_features'] = 12
        self.metadata['feature_names'] = self._generate_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform disorder names to framing features."""
        self._validate_fitted()
        self._validate_input(X)
        
        features = []
        for name in X:
            feature_vec = self._extract_framing_features(name)
            features.append(feature_vec)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_clinical_score(self, name: str) -> float:
        """Calculate how medical/clinical the name sounds (0-100)."""
        name_lower = name.lower()
        score = 0
        
        # Clinical terminology
        for term in self.clinical_terms:
            if term in name_lower:
                score += 30
        
        # Diagnostic modifiers
        for modifier in self.diagnostic_modifiers:
            if modifier in name_lower:
                score += 15
        
        # Full diagnostic format (e.g., "Major Depressive Disorder")
        if any(term in name_lower for term in self.clinical_terms) and len(name.split()) >= 2:
            score += 20
        
        # Acronyms
        if name_lower in self.acronyms or name.upper() == name:
            score += 25
        
        return min(100, score)
    
    def _calculate_accessibility_score(self, name: str) -> float:
        """Calculate how accessible/familiar the name is (0-100)."""
        name_lower = name.lower()
        
        # Start at 100, reduce for barriers
        score = 100
        
        # Medical terminology reduces accessibility
        clinical = self._calculate_clinical_score(name)
        score -= clinical * 0.5
        
        # Long names harder to discuss
        if len(name) > 20:
            score -= 20
        elif len(name) > 30:
            score -= 30
        
        # Multiple words add complexity
        words = len(name.split())
        if words > 2:
            score -= (words - 2) * 10
        
        # Common colloquial terms increase accessibility
        colloquial_terms = {'anxiety', 'depression', 'stress', 'worry', 'fear', 'panic'}
        for term in colloquial_terms:
            if term in name_lower:
                score += 20
        
        return max(0, min(100, score))
    
    def _extract_framing_features(self, name: str) -> List[float]:
        """Extract complete framing feature vector."""
        features = []
        name_lower = name.lower()
        
        # 1. Clinical score
        clinical_score = self._calculate_clinical_score(name)
        features.append(clinical_score)
        
        # 2. Accessibility score
        accessibility_score = self._calculate_accessibility_score(name)
        features.append(accessibility_score)
        
        # 3. Has clinical terminology
        has_clinical = any(term in name_lower for term in self.clinical_terms)
        features.append(1.0 if has_clinical else 0.0)
        
        # 4. Has diagnostic modifier
        has_modifier = any(mod in name_lower for mod in self.diagnostic_modifiers)
        features.append(1.0 if has_modifier else 0.0)
        
        # 5. Is acronym
        is_acronym = (name_lower in self.acronyms or 
                     (len(name) <= 5 and name.upper() == name))
        features.append(1.0 if is_acronym else 0.0)
        
        # 6. Word count (complexity indicator)
        word_count = len(name.split())
        features.append(float(word_count))
        
        # 7. Has colloquial term
        colloquial = {'anxiety', 'depression', 'stress', 'worry', 'fear', 'panic', 'eating'}
        has_colloquial = any(term in name_lower for term in colloquial)
        features.append(1.0 if has_colloquial else 0.0)
        
        # 8. Full diagnostic format
        full_diagnostic = (has_clinical and word_count >= 2 and has_modifier)
        features.append(1.0 if full_diagnostic else 0.0)
        
        # 9. Name length
        features.append(float(len(name)))
        
        # 10. Clinical distance score (clinical - accessibility)
        distance = (clinical_score - accessibility_score) / 2
        features.append(distance)
        
        # 11. Formality ratio
        formality = clinical_score / (accessibility_score + 1)
        features.append(formality)
        
        # 12. Stigma amplification (high clinical × low accessibility)
        stigma_amp = (clinical_score / 100) * (1 - accessibility_score / 100)
        features.append(stigma_amp)
        
        return features
    
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names."""
        return [
            'clinical_score',
            'accessibility_score',
            'has_clinical_term',
            'has_diagnostic_modifier',
            'is_acronym',
            'word_count',
            'has_colloquial_term',
            'full_diagnostic_format',
            'name_length',
            'clinical_distance',
            'formality_ratio',
            'stigma_amplification'
        ]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of framing patterns."""
        stats = self.metadata['corpus_stats']
        
        interpretation = f"""
Clinical Framing Analysis ({stats['n_disorders']} disorders)

FRAMING DISTRIBUTION
--------------------
Clinical Score: {stats['clinical_framing']['mean']:.1f} ± {stats['clinical_framing']['std']:.1f}
Accessibility Score: {stats['accessibility']['mean']:.1f} ± {stats['accessibility']['std']:.1f}
"""
        
        if 'correlations' in self.metadata:
            corr = self.metadata['correlations']
            interpretation += f"""
IMPACT ON HELP-SEEKING
----------------------
Clinical framing → Stigma: r = {corr['clinical_vs_stigma']:.3f}
Accessibility → Treatment seeking: r = {corr['accessibility_vs_seeking']:.3f}

INTERPRETATION
--------------
{'High' if corr['clinical_vs_stigma'] > 0.20 else 'Moderate'} correlation between clinical framing and stigma.
Overly medical terminology creates distance from everyday experience.
"""
        
        interpretation += """
MECHANISM
---------
Medical terminology → Clinical distance ↑
Clinical distance → "Not for me" perception
Accessibility → Help-seeking ↑ → Better outcomes
"""
        
        return interpretation


if __name__ == '__main__':
    # Demo
    transformer = ClinicalFramingTransformer()
    
    disorders = [
        'Major Depressive Disorder',
        'Depression',
        'PTSD',
        'Anxiety'
    ]
    
    transformer.fit(disorders)
    features = transformer.transform(disorders)
    
    print(transformer.get_interpretation())

