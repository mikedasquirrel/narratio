"""
Crypto-Specific Archetype Transformer

Measures distance from Crypto's domain-specific Ξ (golden narratio).
Crypto Ξ: Innovation + decentralization + community + technical legitimacy

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class CryptoArchetypeTransformer(DomainArchetypeTransformer):
    """
    Crypto-specific Ξ measurement.
    
    Enhances base archetype extraction with crypto-specific context:
    - Major crypto mentions (Bitcoin, Ethereum)
    - Category positioning (DeFi, NFT, Layer-1, etc.)
    - Technical innovation markers
    - Community/ecosystem language
    
    Crypto's Ξ emphasizes:
    - Innovation (30%)
    - Decentralization (25%)
    - Community (20%)
    - Technical legitimacy (15%)
    - Use case clarity (10%)
    
    Note: Crypto has low π (0.32) but tests if proper Ξ measurement improves from 17% R².
    
    Examples
    --------
    >>> transformer = CryptoArchetypeTransformer()
    >>> features = transformer.fit_transform(crypto_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('crypto')
        super().__init__(config)
        
        # Crypto-specific context recognition
        self.major_cryptos = [
            'bitcoin', 'ethereum', 'btc', 'eth', 'blockchain'
        ]
        self.categories = [
            'defi', 'nft', 'layer-1', 'layer 1', 'layer-2', 'layer 2',
            'stablecoin', 'exchange', 'wallet', 'protocol'
        ]
        self.innovation_markers = [
            'revolutionary', 'breakthrough', 'next-gen', 'cutting-edge',
            'innovative', 'novel', 'pioneering'
        ]
        self.community_markers = [
            'community', 'ecosystem', 'adoption', 'users', 'network',
            'developer', 'builders'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Crypto-specific archetype extraction with ecosystem context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # Major crypto connection (25% boost - ecosystem positioning)
            major_boost = 1.25 if any(crypto in text_lower for crypto in self.major_cryptos) else 1.0
            
            # Category positioning (20% boost)
            category_boost = 1.2 if any(cat in text_lower for cat in self.categories) else 1.0
            
            # Innovation markers (15% boost)
            innovation_boost = 1.15 if any(marker in text_lower for marker in self.innovation_markers) else 1.0
            
            # Community markers (10% boost)
            community_boost = 1.1 if any(marker in text_lower for marker in self.community_markers) else 1.0
            
            enhanced = base_features[i] * major_boost * category_boost * innovation_boost * community_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get crypto-specific feature names."""
        base_names = super().get_feature_names()
        return [f"crypto_{name}" for name in base_names]

