"""
Golf-Specific Archetype Transformer

Measures distance from Golf's domain-specific Ξ (golden narratio).
Golf Ξ: Mental game mastery + elite skill + course knowledge + pressure performance

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class GolfArchetypeTransformer(DomainArchetypeTransformer):
    """
    Golf-specific Ξ measurement.
    
    Enhances base archetype extraction with golf-specific context:
    - Major championship boosts (Masters, US Open, etc.)
    - Iconic course recognition (Augusta, Pebble Beach, etc.)
    - Tournament prestige modulation
    
    Golf's Ξ emphasizes:
    - Mental game mastery (30%)
    - Elite skill requirements (25%)
    - Course mastery (20%)
    - Pressure performance (15%)
    - Veteran wisdom (10%)
    
    Examples
    --------
    >>> transformer = GolfArchetypeTransformer()
    >>> features = transformer.fit_transform(golf_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('golf')
        super().__init__(config)
        
        # Golf-specific context recognition
        self.major_championships = [
            'masters', 'us open', 'open championship', 'british open', 'pga championship'
        ]
        self.iconic_courses = [
            'augusta', 'pebble beach', 'st andrews', 'oakmont', 'pinehurst', 
            'winged foot', 'shinnecock', 'royal troon', 'carnoustie'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        Golf-specific archetype extraction with course/tournament context.
        
        Parameters
        ----------
        X : list of str
            Texts to extract from
        
        Returns
        -------
        ndarray
            Enhanced archetype features with golf-specific boosts
        """
        base_features = super()._extract_archetype_features(X)
        
        # Add golf-specific enhancements
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Check for major championship context (20% boost)
            major_boost = 1.2 if any(major in text_lower for major in self.major_championships) else 1.0
            
            # Check for iconic course mentions (10% boost)
            course_boost = 1.1 if any(course in text_lower for course in self.iconic_courses) else 1.0
            
            # Apply domain-specific boosts
            enhanced = base_features[i] * major_boost * course_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get golf-specific feature names."""
        base_names = super().get_feature_names()
        # Add prefix for clarity
        return [f"golf_{name}" for name in base_names]

