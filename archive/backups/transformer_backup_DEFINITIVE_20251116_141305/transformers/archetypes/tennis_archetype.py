"""
Tennis-Specific Archetype Transformer

Measures distance from Tennis's domain-specific Ξ (golden narratio).
Tennis Ξ: Mental toughness + grand slam pressure + surface mastery + rivalry

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class TennisArchetypeTransformer(DomainArchetypeTransformer):
    """
    Tennis-specific Ξ measurement.
    
    Enhances base archetype extraction with tennis-specific context:
    - Grand Slam tournament recognition
    - Surface type significance (clay, grass, hard court)
    - Historic rivalry detection
    
    Tennis's Ξ emphasizes:
    - Mental toughness (30%)
    - Grand slam pressure (25%)
    - Surface mastery (20%)
    - Rivalry (15%)
    - Physical conditioning (10%)
    
    Note: Tennis is individual sport achieving 93% R², similar to Golf.
    Should maintain high performance with proper Ξ measurement.
    
    Examples
    --------
    >>> transformer = TennisArchetypeTransformer()
    >>> features = transformer.fit_transform(tennis_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('tennis')
        super().__init__(config)
        
        # Tennis-specific context recognition
        self.grand_slams = [
            'australian open', 'french open', 'roland garros', 'wimbledon', 'us open'
        ]
        self.surfaces = [
            'clay', 'grass', 'hard court', 'hardcourt'
        ]
        self.rivalries = [
            'federer vs nadal', 'federer-nadal', 'djokovic vs nadal', 'djokovic-nadal',
            'federer vs djokovic', 'federer-djokovic', 'big three'
        ]
        self.legendary_players = [
            'federer', 'nadal', 'djokovic', 'serena', 'murray', 'sampras', 'agassi'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        Tennis-specific archetype extraction with match context.
        
        Parameters
        ----------
        X : list of str
            Texts to extract from
        
        Returns
        -------
        ndarray
            Enhanced archetype features with tennis-specific boosts
        """
        base_features = super()._extract_archetype_features(X)
        
        # Add tennis-specific enhancements
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Check for Grand Slam context (35% boost - highest prestige)
            slam_boost = 1.35 if any(slam in text_lower for slam in self.grand_slams) else 1.0
            
            # Check for surface mention (15% boost - surface mastery matters)
            surface_boost = 1.15 if any(surface in text_lower for surface in self.surfaces) else 1.0
            
            # Check for historic rivalry (25% boost - narrative richness)
            rivalry_boost = 1.25 if any(rivalry in text_lower for rivalry in self.rivalries) else 1.0
            
            # Check for legendary player (10% boost)
            legend_boost = 1.1 if any(player in text_lower for player in self.legendary_players) else 1.0
            
            # Apply domain-specific boosts
            enhanced = base_features[i] * slam_boost * surface_boost * rivalry_boost * legend_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get tennis-specific feature names."""
        base_names = super().get_feature_names()
        return [f"tennis_{name}" for name in base_names]

