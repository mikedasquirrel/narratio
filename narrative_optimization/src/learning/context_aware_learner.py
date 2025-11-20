"""
Context-Aware Learning

Learns situational patterns that depend on context.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class ContextAwareLearner:
    """
    Learns patterns that activate conditionally based on context.
    
    Context types:
    - Domain context (which domain?)
    - Temporal context (what time period?)
    - Entity context (what entities involved?)
    - Situational context (what conditions?)
    """
    
    def __init__(self):
        self.contextual_patterns = defaultdict(dict)
        self.context_embeddings = {}
        
    def extract_context(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Extract context from text and metadata.
        
        Parameters
        ----------
        text : str
            Text
        metadata : dict, optional
            Additional metadata
        
        Returns
        -------
        dict
            Context features
        """
        context = {}
        
        # Domain context (if provided)
        if metadata and 'domain' in metadata:
            context['domain'] = metadata['domain']
        
        # Temporal context (if provided)
        if metadata and 'timestamp' in metadata:
            context['timestamp'] = metadata['timestamp']
            
            # Time of day
            if 'hour' in metadata:
                hour = metadata['hour']
                if 6 <= hour < 12:
                    context['time_of_day'] = 'morning'
                elif 12 <= hour < 18:
                    context['time_of_day'] = 'afternoon'
                elif 18 <= hour < 22:
                    context['time_of_day'] = 'evening'
                else:
                    context['time_of_day'] = 'night'
        
        # Entity context (count mentions)
        entities = self._extract_entities(text)
        context['n_entities'] = len(entities)
        context['entities'] = entities
        
        # Situational context (keywords)
        if 'championship' in text.lower() or 'final' in text.lower():
            context['situation'] = 'high_stakes'
        elif 'regular' in text.lower() or 'season' in text.lower():
            context['situation'] = 'regular'
        else:
            context['situation'] = 'unknown'
        
        return context
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity names (simplified)."""
        # Simple: find capitalized words
        words = text.split()
        entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return entities
    
    def discover_contextual_patterns(
        self,
        texts: List[str],
        contexts: List[Dict],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Discover patterns specific to contexts.
        
        Parameters
        ----------
        texts : list
            Texts
        contexts : list
            Context for each text
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            context_id -> patterns
        """
        # Group by context
        context_groups = defaultdict(list)
        
        for i, context in enumerate(contexts):
            # Create context signature
            context_sig = self._context_signature(context)
            context_groups[context_sig].append(i)
        
        # Discover patterns in each context
        from .universal_learner import UniversalArchetypeLearner
        
        for context_sig, indices in context_groups.items():
            if len(indices) < 10:  # Need enough samples
                continue
            
            # Get texts/outcomes for this context
            context_texts = [texts[i] for i in indices]
            context_outcomes = outcomes[indices]
            
            # Discover patterns
            learner = UniversalArchetypeLearner()
            patterns = learner.discover_patterns(context_texts, context_outcomes)
            
            # Store
            self.contextual_patterns[context_sig] = patterns
        
        return dict(self.contextual_patterns)
    
    def _context_signature(self, context: Dict) -> str:
        """Create string signature for context."""
        parts = []
        
        for key in sorted(context.keys()):
            if key not in ['entities', 'timestamp']:  # Skip variable fields
                parts.append(f"{key}={context[key]}")
        
        return "::".join(parts)
    
    def predict_contextual(
        self,
        text: str,
        context: Dict
    ) -> float:
        """
        Make context-aware prediction.
        
        Parameters
        ----------
        text : str
            Text
        context : Dict
            Context
        
        Returns
        -------
        float
            Prediction
        """
        context_sig = self._context_signature(context)
        
        # Get patterns for this context
        if context_sig not in self.contextual_patterns:
            # No patterns for this context - use default
            return 0.5
        
        patterns = self.contextual_patterns[context_sig]
        
        # Score text on patterns
        scores = []
        for pattern_name, pattern_data in patterns.items():
            keywords = pattern_data.get('keywords', [])
            
            if any(kw.lower() in text.lower() for kw in keywords):
                scores.append(pattern_data.get('win_rate', 0.5))
        
        return np.mean(scores) if scores else 0.5
    
    def get_active_patterns_for_context(
        self,
        context: Dict
    ) -> Dict[str, Dict]:
        """
        Get patterns active in given context.
        
        Parameters
        ----------
        context : Dict
            Context
        
        Returns
        -------
        dict
            Active patterns
        """
        context_sig = self._context_signature(context)
        return self.contextual_patterns.get(context_sig, {})
    
    def compare_contexts(
        self,
        context1: Dict,
        context2: Dict
    ) -> float:
        """
        Calculate similarity between contexts.
        
        Parameters
        ----------
        context1, context2 : dict
            Contexts to compare
        
        Returns
        -------
        float
            Similarity (0-1)
        """
        # Jaccard similarity on context features
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        shared_keys = keys1 & keys2
        
        if len(shared_keys) == 0:
            return 0.0
        
        # Compare values for shared keys
        matches = sum(
            1 for key in shared_keys
            if context1.get(key) == context2.get(key)
        )
        
        return matches / len(shared_keys)
    
    def find_similar_contexts(
        self,
        target_context: Dict,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Find contexts similar to target.
        
        Parameters
        ----------
        target_context : dict
            Target context
        threshold : float
            Minimum similarity
        
        Returns
        -------
        list
            Similar context signatures
        """
        similar = []
        
        target_sig = self._context_signature(target_context)
        
        for context_sig in self.contextual_patterns.keys():
            if context_sig == target_sig:
                continue
            
            # Reconstruct context from signature (simplified)
            # In practice, would store full contexts
            similarity = 0.5  # Placeholder
            
            if similarity >= threshold:
                similar.append(context_sig)
        
        return similar

