"""
NBA-Specific Archetype Transformer

Measures distance from NBA's domain-specific Ξ (golden narratio).
NBA Ξ: Star narrative + team chemistry + momentum + coaching + matchup advantage

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class NBAArchetypeTransformer(DomainArchetypeTransformer):
    """
    NBA-specific Ξ measurement.
    
    Enhances base archetype extraction with NBA-specific context:
    - Playoff game recognition (higher stakes)
    - Rivalry game detection (historic matchups)
    - Star player mentions (Lebron, Curry, etc.)
    
    NBA's Ξ emphasizes:
    - Star narrative (30%)
    - Team chemistry (25%)
    - Momentum (20%)
    - Coaching (15%)
    - Matchup advantage (10%)
    
    Note: NBA is a team sport with distributed agency, historically
    achieving only ~15% R². Tests if proper Ξ measurement improves this.
    
    Examples
    --------
    >>> transformer = NBAArchetypeTransformer()
    >>> features = transformer.fit_transform(nba_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('nba')
        super().__init__(config)
        
        # NBA-specific context recognition
        self.playoff_keywords = [
            'playoffs', 'playoff', 'finals', 'championship', 'elimination', 'game 7'
        ]
        self.star_players = [
            'lebron', 'curry', 'durant', 'giannis', 'jokic', 'luka', 'embiid',
            'tatum', 'kawhi', 'davis', 'harden', 'lillard'
        ]
        self.rivalry_teams = [
            'lakers vs celtics', 'lakers-celtics', 'warriors vs cavs',
            'heat vs spurs', 'bulls vs pistons'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        NBA-specific archetype extraction with game context.
        
        Parameters
        ----------
        X : list of str
            Texts to extract from
        
        Returns
        -------
        ndarray
            Enhanced archetype features with NBA-specific boosts
        """
        base_features = super()._extract_archetype_features(X)
        
        # Add NBA-specific enhancements
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Check for playoff context (30% boost)
            playoff_boost = 1.3 if any(keyword in text_lower for keyword in self.playoff_keywords) else 1.0
            
            # Check for star player mentions (15% boost)
            star_boost = 1.15 if any(player in text_lower for player in self.star_players) else 1.0
            
            # Check for rivalry game (20% boost)
            rivalry_boost = 1.2 if any(rivalry in text_lower for rivalry in self.rivalry_teams) else 1.0
            
            # Apply domain-specific boosts
            enhanced = base_features[i] * playoff_boost * star_boost * rivalry_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get NBA-specific feature names."""
        base_names = super().get_feature_names()
        return [f"nba_{name}" for name in base_names]

