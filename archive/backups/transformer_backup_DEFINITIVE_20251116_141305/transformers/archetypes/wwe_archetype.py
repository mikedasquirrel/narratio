"""
WWE-Specific Archetype Transformer

Measures distance from WWE's domain-specific Ξ (golden narratio).
WWE Ξ: Character arc + long-term payoff + betrayal/redemption + meta-awareness

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class WWEArchetypeTransformer(DomainArchetypeTransformer):
    """
    WWE-specific Ξ measurement.
    
    Enhances base archetype extraction with WWE-specific context:
    - WrestleMania and major PPV recognition
    - Legendary performer mentions
    - Multi-year storyline detection
    
    WWE's Ξ emphasizes:
    - Character arc (30%)
    - Long-term payoff (25%)
    - Betrayal/redemption (20%)
    - Meta-awareness (15%)
    - Crowd reaction (10%)
    
    Note: WWE is a PRESTIGE domain with highest π (0.974). Everyone
    knows it's fake, yet narrative quality predicts engagement.
    Uses inverted prestige equation: Д = ة + θ - λ
    
    Examples
    --------
    >>> transformer = WWEArchetypeTransformer()
    >>> features = transformer.fit_transform(wwe_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('wwe')
        super().__init__(config)
        
        # WWE-specific context recognition
        self.major_events = [
            'wrestlemania', 'summerslam', 'royal rumble', 'survivor series',
            'money in the bank', 'hell in a cell'
        ]
        self.legendary_performers = [
            'undertaker', 'hulk hogan', 'stone cold', 'the rock', 'john cena',
            'shawn michaels', 'bret hart', 'ric flair', 'triple h'
        ]
        self.storyline_duration_markers = [
            'years in the making', 'long-term', 'epic', 'saga', 'culmination',
            'historic', 'legacy', 'iconic', 'legendary'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        WWE-specific archetype extraction with storyline context.
        
        Parameters
        ----------
        X : list of str
            Texts to extract from
        
        Returns
        -------
        ndarray
            Enhanced archetype features with WWE-specific boosts
        """
        base_features = super()._extract_archetype_features(X)
        
        # Add WWE-specific enhancements
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Check for major event context (40% boost - these are peak moments)
            event_boost = 1.4 if any(event in text_lower for event in self.major_events) else 1.0
            
            # Check for legendary performer (25% boost - adds gravitas)
            legend_boost = 1.25 if any(legend in text_lower for legend in self.legendary_performers) else 1.0
            
            # Check for long-term storyline markers (30% boost - payoff moments)
            duration_boost = 1.3 if any(marker in text_lower for marker in self.storyline_duration_markers) else 1.0
            
            # Apply domain-specific boosts
            enhanced = base_features[i] * event_boost * legend_boost * duration_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get WWE-specific feature names."""
        base_names = super().get_feature_names()
        return [f"wwe_{name}" for name in base_names]

