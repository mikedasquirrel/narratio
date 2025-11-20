"""
Domain-Adaptive Transformer Base

Base class for making transformers domain-adaptive.
Transformers can learn patterns from domain config and discovered archetypes.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from ..config import DomainConfig


class DomainAdaptiveTransformer(ABC):
    """
    Base for domain-adaptive transformers.
    
    Features:
    - Accept domain_config to customize patterns
    - Use discovered archetypes
    - Adapt to domain characteristics
    - Learn from feedback
    """
    
    def __init__(
        self,
        transformer_id: str,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict[str, Dict]] = None
    ):
        self.transformer_id = transformer_id
        self.domain_config = domain_config
        self.learned_patterns = learned_patterns or {}
        
        # Base patterns (generic)
        self.base_patterns = self._init_base_patterns()
        
        # Domain-specific patterns
        self.domain_patterns = {}
        if domain_config:
            self.domain_patterns = self._load_domain_patterns()
        
        # Learned pattern weights
        self.pattern_weights = {}
        
    @abstractmethod
    def _init_base_patterns(self) -> Dict[str, List[str]]:
        """Initialize base (generic) patterns."""
        pass
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific patterns from config."""
        if not self.domain_config:
            return {}
        
        # Try to get domain-specific patterns for this transformer
        domain_patterns = self.domain_config.get_domain_specific_patterns(
            self.transformer_id
        )
        
        return domain_patterns or {}
    
    def update_learned_patterns(self, patterns: Dict[str, Dict]):
        """
        Update with learned patterns from discovery.
        
        Parameters
        ----------
        patterns : dict
            Learned patterns
        """
        self.learned_patterns = patterns
        
        # Extract weights
        for pattern_name, pattern_data in patterns.items():
            self.pattern_weights[pattern_name] = pattern_data.get('correlation', 0.5)
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Get combined patterns (base + domain + learned)."""
        combined = {}
        
        # Add base
        combined.update(self.base_patterns)
        
        # Add domain-specific
        combined.update(self.domain_patterns)
        
        # Add learned
        for pattern_name, pattern_data in self.learned_patterns.items():
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            combined[pattern_name] = keywords
        
        return combined
    
    def get_pattern_weight(self, pattern_name: str) -> float:
        """Get weight for a pattern."""
        return self.pattern_weights.get(pattern_name, 1.0)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract features using adaptive patterns.
        
        Parameters
        ----------
        text : str
            Text to extract from
        
        Returns
        -------
        dict
            Feature name -> value
        """
        features = {}
        all_patterns = self.get_all_patterns()
        
        for pattern_name, keywords in all_patterns.items():
            # Check presence
            matches = sum(1 for kw in keywords if kw.lower() in text.lower())
            
            # Weight by learned importance
            weight = self.get_pattern_weight(pattern_name)
            
            features[f"{self.transformer_id}_{pattern_name}"] = matches * weight
        
        return features
    
    def adapt_to_feedback(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        features: np.ndarray
    ):
        """
        Adapt transformer based on feedback.
        
        Parameters
        ----------
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        features : ndarray
            Extracted features
        """
        # Calculate feature-outcome correlations
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        for i, pattern_name in enumerate(self.get_all_patterns().keys()):
            if i >= features.shape[1]:
                break
            
            feature_col = features[:, i]
            
            if len(np.unique(feature_col)) > 1 and len(np.unique(outcomes)) > 1:
                corr = abs(np.corrcoef(feature_col, outcomes)[0, 1])
                self.pattern_weights[pattern_name] = corr


class AdaptiveNarrativePotentialTransformer(DomainAdaptiveTransformer):
    """
    Domain-adaptive version of NarrativePotentialTransformer.
    """
    
    def __init__(
        self,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict] = None
    ):
        super().__init__(
            transformer_id="narrative_potential",
            domain_config=domain_config,
            learned_patterns=learned_patterns
        )
    
    def _init_base_patterns(self) -> Dict[str, List[str]]:
        """Initialize base patterns for narrative potential."""
        return {
            'high_stakes': ['crucial', 'critical', 'important', 'decisive', 'key'],
            'urgency': ['urgent', 'immediate', 'pressing', 'time-sensitive'],
            'opportunity': ['opportunity', 'chance', 'possibility', 'potential'],
            'flexibility': ['flexible', 'adaptive', 'versatile', 'adjustable'],
            'openness': ['open', 'possible', 'uncertain', 'undecided']
        }


class AdaptiveConflictTensionTransformer(DomainAdaptiveTransformer):
    """
    Domain-adaptive version of ConflictTensionTransformer.
    """
    
    def __init__(
        self,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict] = None
    ):
        super().__init__(
            transformer_id="conflict_tension",
            domain_config=domain_config,
            learned_patterns=learned_patterns
        )
    
    def _init_base_patterns(self) -> Dict[str, List[str]]:
        """Initialize base patterns for conflict/tension."""
        return {
            'rivalry': ['rivalry', 'rival', 'competition', 'versus', 'against'],
            'confrontation': ['confrontation', 'clash', 'battle', 'fight', 'showdown'],
            'tension': ['tension', 'pressure', 'stress', 'strain'],
            'stakes': ['stakes', 'on the line', 'at stake', 'risk'],
            'intensity': ['intense', 'fierce', 'heated', 'aggressive']
        }


class AdaptiveEmotionalResonanceTransformer(DomainAdaptiveTransformer):
    """
    Domain-adaptive version of EmotionalResonanceTransformer.
    """
    
    def __init__(
        self,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict] = None
    ):
        super().__init__(
            transformer_id="emotional_resonance",
            domain_config=domain_config,
            learned_patterns=learned_patterns
        )
    
    def _init_base_patterns(self) -> Dict[str, List[str]]:
        """Initialize base patterns for emotional resonance."""
        return {
            'triumph': ['triumph', 'victory', 'success', 'achievement', 'glory'],
            'struggle': ['struggle', 'difficulty', 'challenge', 'obstacle'],
            'redemption': ['redemption', 'comeback', 'recovery', 'return'],
            'inspiration': ['inspiring', 'motivating', 'uplifting', 'encouraging'],
            'drama': ['dramatic', 'thrilling', 'exciting', 'captivating']
        }


class TransformerAdapter:
    """
    Adapts existing transformers to be domain-aware.
    """
    
    def __init__(self):
        self.adapted_transformers = {}
        
    def adapt_transformer(
        self,
        transformer,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict] = None
    ):
        """
        Wrap existing transformer to make it domain-adaptive.
        
        Parameters
        ----------
        transformer : object
            Existing transformer
        domain_config : DomainConfig, optional
            Domain configuration
        learned_patterns : dict, optional
            Learned patterns
        
        Returns
        -------
        object
            Adapted transformer
        """
        # Store original
        transformer._original_transform = transformer.transform
        
        # Add domain awareness
        transformer.domain_config = domain_config
        transformer.learned_patterns = learned_patterns or {}
        
        # Override transform method
        def domain_aware_transform(X):
            # Original features
            original_features = transformer._original_transform(X)
            
            # Add domain-specific features if available
            if domain_config and hasattr(transformer, 'narrative_id'):
                domain_patterns = domain_config.get_domain_specific_patterns(
                    transformer.narrative_id
                )
                
                if domain_patterns:
                    # Extract additional features
                    additional_features = []
                    for text in X:
                        score = 0.0
                        for pattern_list in domain_patterns.values():
                            for pattern in pattern_list:
                                if pattern.lower() in text.lower():
                                    score += 1.0
                        additional_features.append(score)
                    
                    # Combine with original
                    additional_features = np.array(additional_features).reshape(-1, 1)
                    if len(original_features.shape) == 1:
                        original_features = original_features.reshape(-1, 1)
                    
                    combined = np.hstack([original_features, additional_features])
                    return combined
            
            return original_features
        
        transformer.transform = domain_aware_transform
        
        return transformer
    
    def adapt_all_transformers(
        self,
        transformers: List,
        domain_config: Optional[DomainConfig] = None,
        learned_patterns: Optional[Dict] = None
    ) -> List:
        """Adapt a list of transformers."""
        adapted = []
        
        for transformer in transformers:
            adapted_transformer = self.adapt_transformer(
                transformer,
                domain_config,
                learned_patterns
            )
            adapted.append(adapted_transformer)
        
        return adapted

