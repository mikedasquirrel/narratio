"""
Boxing-Specific Archetype Transformer

Measures distance from Boxing's domain-specific Ξ (golden narratio).
Boxing Ξ: Physical dominance + fighting style + warrior spirit + ring generalship

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class BoxingArchetypeTransformer(DomainArchetypeTransformer):
    """
    Boxing-specific Ξ measurement.
    
    Enhances base archetype extraction with boxing-specific context:
    - Championship bout recognition (title fights)
    - Legendary venue recognition (Madison Square Garden, etc.)
    - Weight class significance
    
    Boxing's Ξ emphasizes:
    - Physical dominance (35%)
    - Fighting style (25%)
    - Warrior spirit (20%)
    - Ring generalship (12%)
    - Underdog story (8%)
    
    Note: Boxing has high θ (0.80-0.95) which historically suppresses
    narrative effects. This transformer tests if proper Ξ measurement helps.
    
    Examples
    --------
    >>> transformer = BoxingArchetypeTransformer()
    >>> features = transformer.fit_transform(boxing_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('boxing')
        super().__init__(config)
        
        # Boxing-specific context recognition
        self.title_keywords = [
            'championship', 'title fight', 'belt', 'champion', 'undisputed', 'unified'
        ]
        self.legendary_venues = [
            'madison square garden', 'msg', 'caesars palace', 'mgm grand',
            'yankee stadium', 'wembley', 'tokyo dome'
        ]
        self.weight_classes = [
            'heavyweight', 'light heavyweight', 'middleweight', 'welterweight',
            'lightweight', 'featherweight', 'bantamweight'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        Boxing-specific archetype extraction with bout context.
        
        Parameters
        ----------
        X : list of str
            Texts to extract from
        
        Returns
        -------
        ndarray
            Enhanced archetype features with boxing-specific boosts
        """
        base_features = super()._extract_archetype_features(X)
        
        # Add boxing-specific enhancements
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Check for title fight context (25% boost)
            title_boost = 1.25 if any(keyword in text_lower for keyword in self.title_keywords) else 1.0
            
            # Check for legendary venue (15% boost)
            venue_boost = 1.15 if any(venue in text_lower for venue in self.legendary_venues) else 1.0
            
            # Check for weight class mention (10% boost - adds context)
            weight_boost = 1.1 if any(wc in text_lower for wc in self.weight_classes) else 1.0
            
            # Apply domain-specific boosts
            enhanced = base_features[i] * title_boost * venue_boost * weight_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get boxing-specific feature names."""
        base_names = super().get_feature_names()
        return [f"boxing_{name}" for name in base_names]

