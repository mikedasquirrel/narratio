"""
Housing-Specific Archetype Transformer

Measures distance from Housing's domain-specific Ξ (golden narratio).
Housing Ξ: Numerology + cultural superstition + street valence + address prestige

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List
import re

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class HousingArchetypeTransformer(DomainArchetypeTransformer):
    """
    Housing-specific Ξ measurement.
    
    PURE NOMINATIVE DOMAIN - Cleanest test of name-gravity
    
    Enhances base archetype extraction with housing-specific context:
    - #13 detection (unlucky number)
    - Street name valence
    - Address prestige indicators
    - Cultural superstition markers
    
    Housing's Ξ emphasizes:
    - Numerology (40% - highest weight)
    - Cultural superstition (25%)
    - Street valence (15%)
    - Address prestige (12%)
    - Location association (8%)
    
    Note: Pure nominative domain (π=0.92) with $93K effect from #13.
    
    Examples
    --------
    >>> transformer = HousingArchetypeTransformer()
    >>> features = transformer.fit_transform(housing_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('housing')
        super().__init__(config)
        
        # Housing-specific context recognition
        self.unlucky_numbers = ['13', 'thirteen', '#13']
        self.lucky_numbers = ['7', 'seven', '8', 'eight', '777', '888']
        self.prestige_streets = [
            'park avenue', 'madison avenue', 'fifth avenue', 'avenue',
            'boulevard', 'drive', 'court'
        ]
        self.superstition_markers = [
            'unlucky', 'lucky', 'superstition', 'taboo', 'avoid',
            'skip', 'unfortunate', 'fortunate'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Housing-specific archetype extraction with numerology context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # #13 detection (50% boost - key finding)
            has_unlucky = any(num in text_lower for num in self.unlucky_numbers)
            # Also check for "number 13" or "house 13"
            if re.search(r'\b(house|number|#|no\.?)\s*13\b', text_lower):
                has_unlucky = True
            
            unlucky_boost = 1.5 if has_unlucky else 1.0
            
            # Lucky numbers (20% boost - positive association)
            has_lucky = any(num in text_lower for num in self.lucky_numbers)
            lucky_boost = 1.2 if has_lucky else 1.0
            
            # Prestige street (25% boost)
            prestige_boost = 1.25 if any(street in text_lower for street in self.prestige_streets) else 1.0
            
            # Superstition markers (20% boost)
            superstition_boost = 1.2 if any(marker in text_lower for marker in self.superstition_markers) else 1.0
            
            enhanced = base_features[i] * unlucky_boost * lucky_boost * prestige_boost * superstition_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get housing-specific feature names."""
        base_names = super().get_feature_names()
        return [f"housing_{name}" for name in base_names]

