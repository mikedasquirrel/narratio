"""
Chess-Specific Archetype Transformer

Measures distance from Chess's domain-specific Ξ (golden narratio).
Chess Ξ: Strategic depth + opening theory + endgame mastery + time pressure

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List

from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig


class ChessArchetypeTransformer(DomainArchetypeTransformer):
    """
    Chess-specific Ξ measurement.
    
    Enhances base archetype extraction with chess-specific context:
    - World Championship recognition
    - Opening system mentions (Sicilian, Ruy Lopez, etc.)
    - Time control significance (classical vs blitz)
    - Historical game references
    
    Chess's Ξ emphasizes:
    - Strategic depth (30%)
    - Opening theory (25%)
    - Endgame mastery (20%)
    - Time pressure (15%)
    - Psychological (10%)
    
    Examples
    --------
    >>> transformer = ChessArchetypeTransformer()
    >>> features = transformer.fit_transform(chess_narratives, outcomes)
    >>> story_quality = features[:, -1]
    """
    
    def __init__(self):
        config = DomainConfig('chess')
        super().__init__(config)
        
        # Chess-specific context recognition
        self.world_championships = [
            'world championship', 'world title', 'wc match', 'championship match'
        ]
        self.major_openings = [
            'sicilian', 'ruy lopez', 'queen\'s gambit', 'king\'s gambit',
            'french defense', 'caro-kann', 'english opening', 'nimzo-indian'
        ]
        self.time_controls = [
            'classical', 'rapid', 'blitz', 'bullet', 'time control'
        ]
        self.legendary_players = [
            'carlsen', 'kasparov', 'fischer', 'karpov', 'anand', 'caruana', 'nepomniachtchi'
        ]
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """Chess-specific archetype extraction with game context."""
        base_features = super()._extract_archetype_features(X)
        
        enhanced_features = []
        for i, text in enumerate(X):
            text_lower = text.lower()
            
            # World Championship boost (35%)
            wc_boost = 1.35 if any(wc in text_lower for wc in self.world_championships) else 1.0
            
            # Opening theory mention (15% boost)
            opening_boost = 1.15 if any(opening in text_lower for opening in self.major_openings) else 1.0
            
            # Time control significance (10% boost)
            time_boost = 1.1 if any(tc in text_lower for tc in self.time_controls) else 1.0
            
            # Legendary player mention (10% boost)
            legend_boost = 1.1 if any(player in text_lower for player in self.legendary_players) else 1.0
            
            enhanced = base_features[i] * wc_boost * opening_boost * time_boost * legend_boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
    
    def get_feature_names(self) -> List[str]:
        """Get chess-specific feature names."""
        base_names = super().get_feature_names()
        return [f"chess_{name}" for name in base_names]

