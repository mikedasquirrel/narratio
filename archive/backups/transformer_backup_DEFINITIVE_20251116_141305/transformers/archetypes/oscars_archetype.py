"""
Oscars-Specific Archetype Transformer

Measures distance from Oscars's domain-specific Ξ (golden narratio).
Oscars Ξ: Campaign narrative + cultural moment + emotional resonance + technical excellence

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class OscarsArchetypeTransformer(DomainArchetypeTransformer):
    """
    Oscars-specific Ξ measurement.
    
    PRESTIGE DOMAIN - Uses inverted equation (Д = ة + θ - λ)
    
    Enhances base archetype extraction with Oscars-specific context:
    - Best Picture category recognition
    - Awards season timing
    - Campaign momentum markers
    - Cultural relevance indicators
    
    Oscars's Ξ emphasizes:
    - Campaign narrative (30%)
    - Cultural moment (25%)
    - Emotional resonance (20%)
    - Technical excellence (15%)
    - Prestige factors (10%)
    
    Examples
    --------
    >>> transformer = OscarsArchetypeTransformer()
    >>> features = transformer.fit_transform(oscar_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('oscars')
        super().__init__(config)
        
        # Oscars-specific context recognition
        self.best_picture = [
            'best picture', 'best film', 'picture', 'film', 'movie'
        ]
        self.awards_season = [
            'awards season', 'oscar season', 'for your consideration', 'campaign',
            'golden globes', 'sag', 'bafta', 'critics choice'
        ]
        self.cultural_moments = [
            'cultural', 'zeitgeist', 'moment', 'relevant', 'timely', 'resonance'
        ]
        self.prestige_directors = [
            'spielberg', 'scorsese', 'nolan', 'coen', 'anderson', 'villeneuve'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Oscars-specific archetype extraction with campaign context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Best Picture category (40% boost - highest prestige)
            bp_boost = 1.4 if any(bp in text_lower for bp in self.best_picture) else 1.0
            
            # Awards season momentum (30% boost)
            season_boost = 1.3 if any(season in text_lower for season in self.awards_season) else 1.0
            
            # Cultural moment indicators (25% boost)
            cultural_boost = 1.25 if any(moment in text_lower for moment in self.cultural_moments) else 1.0
            
            # Prestige director (15% boost)
            director_boost = 1.15 if any(director in text_lower for director in self.prestige_directors) else 1.0
            
            enhanced = base_features[i] * bp_boost * season_boost * cultural_boost * director_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get Oscars-specific feature names."""
        base_names = super().get_feature_names()
        return [f"oscars_{name}" for name in base_names]

