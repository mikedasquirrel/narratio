"""
Startups-Specific Archetype Transformer

Measures distance from Startups's domain-specific Ξ (golden narratio).
Startups Ξ: Market fit + innovation + execution + team quality + scalability

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class StartupsArchetypeTransformer(DomainArchetypeTransformer):
    """
    Startups-specific Ξ measurement.
    
    Enhances base archetype extraction with startup-specific context:
    - Y Combinator / accelerator recognition
    - Funding stage indicators
    - Traction metrics
    - Market opportunity language
    
    Startups's Ξ emphasizes:
    - Market fit (30%)
    - Innovation (25%)
    - Execution (20%)
    - Team quality (15%)
    - Scalability (10%)
    
    Note: Startups achieved 98% R² - tests if proper Ξ measurement maintains this.
    
    Examples
    --------
    >>> transformer = StartupsArchetypeTransformer()
    >>> features = transformer.fit_transform(startup_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('startups')
        super().__init__(config)
        
        # Startup-specific context recognition
        self.accelerators = [
            'y combinator', 'yc', 'techstars', '500 startups', 'accelerator',
            'incubator', 'seed fund'
        ]
        self.funding_stages = [
            'seed', 'series a', 'series b', 'series c', 'funding', 'raised',
            'investment', 'venture capital', 'vc'
        ]
        self.traction_markers = [
            'traction', 'revenue', 'users', 'customers', 'growth',
            'mrr', 'arr', 'cac', 'ltv'
        ]
        self.market_opportunity = [
            'market', 'opportunity', 'billion', 'trillion', 'tam', 'sam',
            'addressable', 'total addressable market'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Startup-specific archetype extraction with business context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Accelerator recognition (30% boost - validation signal)
            accelerator_boost = 1.3 if any(acc in text_lower for acc in self.accelerators) else 1.0
            
            # Funding stage (25% boost - progress indicator)
            funding_boost = 1.25 if any(stage in text_lower for stage in self.funding_stages) else 1.0
            
            # Traction markers (20% boost - execution proof)
            traction_boost = 1.2 if any(marker in text_lower for marker in self.traction_markers) else 1.0
            
            # Market opportunity (15% boost - scale potential)
            market_boost = 1.15 if any(opp in text_lower for opp in self.market_opportunity) else 1.0
            
            enhanced = base_features[i] * accelerator_boost * funding_boost * traction_boost * market_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get startup-specific feature names."""
        base_names = super().get_feature_names()
        return [f"startups_{name}" for name in base_names]

