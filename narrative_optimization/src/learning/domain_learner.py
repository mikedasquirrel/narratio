"""
Domain-Specific Archetype Learner

Learns patterns unique to a specific domain.
Example: Golf has "course mastery", Tennis has "surface expertise"

These patterns don't transfer across domains.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Set, Optional
from collections import Counter
import re

from ..config import ArchetypeDiscovery, DomainConfig


class DomainSpecificLearner:
    """
    Learns domain-specific archetype patterns.
    
    Domain-specific patterns are unique to one domain and don't
    generalize elsewhere.
    
    Examples:
    - Golf: "course knowledge", "major championship pressure"
    - Tennis: "surface mastery", "grand slam experience"
    - Chess: "opening preparation", "endgame technique"
    
    Parameters
    ----------
    domain_name : str
        Domain name
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.domain_config = DomainConfig(domain_name)
        self.discovery = ArchetypeDiscovery(min_pattern_frequency=0.03)
        
        self.patterns = {}  # pattern_name -> pattern_data
        self.validated_patterns = {}
        self.pattern_quality = {}  # pattern_name -> quality_score
        
    def discover_patterns(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        n_patterns: int = 5
    ) -> Dict[str, Dict]:
        """
        Discover domain-specific patterns.
        
        Parameters
        ----------
        texts : list of str
            Domain texts
        outcomes : ndarray
            Outcomes
        n_patterns : int
            Number of patterns to discover
        
        Returns
        -------
        dict
            Discovered patterns
        """
        if len(texts) < 10:
            print(f"  ⚠ Too few texts for {self.domain_name} ({len(texts)})")
            return {}
        
        # Use ArchetypeDiscovery system
        discovered = self.discovery.discover_archetypes(texts, n_archetypes=n_patterns)
        
        # Convert to standard format
        formatted = {}
        for pattern_name, pattern_data in discovered.items():
            formatted[f"{self.domain_name}_{pattern_name}"] = {
                'type': 'domain_specific',
                'domain': self.domain_name,
                'patterns': pattern_data['patterns'],
                'frequency': pattern_data.get('sample_count', 0) / len(texts),
                'coherence': pattern_data.get('coherence', 0.0),
                'sample_size': pattern_data.get('sample_count', 0)
            }
        
        self.patterns.update(formatted)
        return formatted
    
    def get_patterns(self) -> Dict[str, Dict]:
        """Get all patterns."""
        return self.patterns
    
    def get_validated_patterns(self) -> Dict[str, Dict]:
        """Get validated patterns."""
        return self.validated_patterns
    
    def set_validated_patterns(self, patterns: Dict[str, Dict]):
        """Set validated patterns."""
        self.validated_patterns = patterns
        
        # Calculate quality scores
        for pattern_name, pattern_data in patterns.items():
            self.pattern_quality[pattern_name] = pattern_data.get('correlation', 0.0)
    
    def get_average_pattern_quality(self) -> float:
        """Get average quality of patterns."""
        if len(self.pattern_quality) == 0:
            return 0.0
        return np.mean(list(self.pattern_quality.values()))
    
    def prune_weak_patterns(
        self,
        min_correlation: float = 0.05,
        min_frequency: float = 0.02
    ) -> int:
        """
        Remove weak patterns.
        
        Parameters
        ----------
        min_correlation : float
            Minimum correlation with outcomes
        min_frequency : float
            Minimum frequency in data
        
        Returns
        -------
        int
            Number of patterns pruned
        """
        to_remove = []
        
        for pattern_name, pattern_data in self.patterns.items():
            # Check correlation
            quality = self.pattern_quality.get(pattern_name, 0.0)
            if quality < min_correlation:
                to_remove.append(pattern_name)
                continue
            
            # Check frequency
            frequency = pattern_data.get('frequency', 0.0)
            if frequency < min_frequency:
                to_remove.append(pattern_name)
                continue
        
        # Remove
        for pattern_name in to_remove:
            del self.patterns[pattern_name]
            if pattern_name in self.validated_patterns:
                del self.validated_patterns[pattern_name]
            if pattern_name in self.pattern_quality:
                del self.pattern_quality[pattern_name]
        
        return len(to_remove)
    
    def merge_similar_patterns(
        self,
        similarity_threshold: float = 0.8
    ) -> int:
        """
        Merge similar patterns to reduce redundancy.
        
        Parameters
        ----------
        similarity_threshold : float
            Minimum similarity to merge
        
        Returns
        -------
        int
            Number of patterns merged
        """
        # Find similar patterns
        pattern_list = list(self.patterns.items())
        merged_count = 0
        
        for i in range(len(pattern_list)):
            for j in range(i + 1, len(pattern_list)):
                name_i, data_i = pattern_list[i]
                name_j, data_j = pattern_list[j]
                
                # Skip if already removed
                if name_i not in self.patterns or name_j not in self.patterns:
                    continue
                
                # Calculate similarity
                patterns_i = set(data_i.get('patterns', []))
                patterns_j = set(data_j.get('patterns', []))
                
                if len(patterns_i) == 0 or len(patterns_j) == 0:
                    continue
                
                overlap = len(patterns_i & patterns_j)
                union = len(patterns_i | patterns_j)
                similarity = overlap / union if union > 0 else 0.0
                
                if similarity >= similarity_threshold:
                    # Merge j into i
                    self.patterns[name_i]['patterns'] = list(patterns_i | patterns_j)
                    self.patterns[name_i]['frequency'] = max(
                        data_i.get('frequency', 0.0),
                        data_j.get('frequency', 0.0)
                    )
                    
                    # Remove j
                    del self.patterns[name_j]
                    if name_j in self.validated_patterns:
                        del self.validated_patterns[name_j]
                    if name_j in self.pattern_quality:
                        # Transfer quality to merged pattern
                        self.pattern_quality[name_i] = max(
                            self.pattern_quality.get(name_i, 0.0),
                            self.pattern_quality.get(name_j, 0.0)
                        )
                        del self.pattern_quality[name_j]
                    
                    merged_count += 1
        
        return merged_count

