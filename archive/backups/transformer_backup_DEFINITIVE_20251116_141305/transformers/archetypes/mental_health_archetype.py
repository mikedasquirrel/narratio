"""
Mental Health-Specific Archetype Transformer

Measures distance from Mental Health's domain-specific Ξ (golden narratio).
Mental Health Ξ: Clinical framing + phonetic severity + treatment seeking + stigma association

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class MentalHealthArchetypeTransformer(DomainArchetypeTransformer):
    """
    Mental Health-specific Ξ measurement.
    
    Enhances base archetype extraction with mental health-specific context:
    - Clinical terminology recognition
    - Phonetic pattern detection (harsh vs soft sounds)
    - Treatment context
    - Stigma indicators
    
    Mental Health's Ξ emphasizes:
    - Clinical framing (30%)
    - Phonetic severity (25%)
    - Treatment seeking (20%)
    - Stigma association (15%)
    - Severity indicator (10%)
    
    Examples
    --------
    >>> transformer = MentalHealthArchetypeTransformer()
    >>> features = transformer.fit_transform(mh_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('mental_health')
        super().__init__(config)
        
        # Mental health-specific context recognition
        self.clinical_terms = [
            'diagnosis', 'symptom', 'syndrome', 'disorder', 'condition',
            'clinical', 'pathology', 'etiology'
        ]
        self.harsh_sounds = [
            'schizophrenia', 'psychosis', 'bipolar', 'disorder',
            'severe', 'chronic', 'acute'
        ]
        self.treatment_context = [
            'therapy', 'treatment', 'medication', 'intervention',
            'counseling', 'support', 'care'
        ]
        self.stigma_markers = [
            'stigma', 'stereotyped', 'prejudice', 'discrimination',
            'misunderstood', 'judged'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Mental health-specific archetype extraction with clinical context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Clinical terminology (30% boost - professional framing)
            clinical_boost = 1.3 if any(term in text_lower for term in self.clinical_terms) else 1.0
            
            # Harsh phonetic patterns (25% boost - severity indicator)
            phonetic_boost = 1.25 if any(sound in text_lower for sound in self.harsh_sounds) else 1.0
            
            # Treatment context (20% boost)
            treatment_boost = 1.2 if any(context in text_lower for context in self.treatment_context) else 1.0
            
            # Stigma markers (15% boost - negative association)
            stigma_boost = 1.15 if any(marker in text_lower for marker in self.stigma_markers) else 1.0
            
            enhanced = base_features[i] * clinical_boost * phonetic_boost * treatment_boost * stigma_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get mental health-specific feature names."""
        base_names = super().get_feature_names()
        return [f"mental_health_{name}" for name in base_names]

