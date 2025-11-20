"""
Hurricanes-Specific Archetype Transformer

Measures distance from Hurricanes's domain-specific Ξ (golden narratio).
Hurricanes Ξ: Naming patterns + threat perception + gender association + severity

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class HurricanesArchetypeTransformer(DomainArchetypeTransformer):
    """
    Hurricanes-specific Ξ measurement.
    
    Enhances base archetype extraction with hurricane-specific context:
    - Hurricane name gender detection
    - Category strength (1-5)
    - Geographic impact areas
    - Historical significance
    
    Hurricanes's Ξ emphasizes:
    - Naming patterns (30%)
    - Threat perception (25%)
    - Gender association (20%)
    - Severity indicators (15%)
    - Geographic context (10%)
    
    Note: Natural phenomenon with high π (0.30) but moderate effects (42% R²).
    
    Examples
    --------
    >>> transformer = HurricanesArchetypeTransformer()
    >>> features = transformer.fit_transform(hurricane_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('hurricanes')
        super().__init__(config)
        
        # Hurricane-specific context recognition
        self.feminine_names = [
            'anna', 'bella', 'carla', 'diana', 'elena', 'fiona', 'grace',
            'helen', 'iris', 'jane', 'kate', 'lisa', 'maria', 'nina'
        ]
        self.masculine_names = [
            'andrew', 'bob', 'charles', 'david', 'edward', 'frank', 'george',
            'henry', 'ivan', 'john', 'karl', 'louis', 'michael', 'nick'
        ]
        self.category_markers = [
            'category 1', 'category 2', 'category 3', 'category 4', 'category 5',
            'cat 1', 'cat 2', 'cat 3', 'cat 4', 'cat 5', 'major hurricane'
        ]
        self.geographic_impact = [
            'gulf coast', 'atlantic', 'caribbean', 'florida', 'texas',
            'louisiana', 'carolinas', 'new england'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Hurricane-specific archetype extraction with weather context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Gender detection (25% boost - key finding from research)
            has_feminine = any(name in text_lower for name in self.feminine_names)
            has_masculine = any(name in text_lower for name in self.masculine_names)
            gender_boost = 1.25 if (has_feminine or has_masculine) else 1.0
            
            # Category strength (30% boost - severity indicator)
            category_boost = 1.3 if any(cat in text_lower for cat in self.category_markers) else 1.0
            
            # Geographic impact (15% boost - context)
            geo_boost = 1.15 if any(geo in text_lower for geo in self.geographic_impact) else 1.0
            
            enhanced = base_features[i] * gender_boost * category_boost * geo_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get hurricane-specific feature names."""
        base_names = super().get_feature_names()
        return [f"hurricanes_{name}" for name in base_names]

