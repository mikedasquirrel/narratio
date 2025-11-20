"""
Archetype Discovery and Learning System

Discovers archetypal patterns from data and makes it easy to:
1. Learn domain-specific archetypes from winners
2. Discover sub-archetypes within main patterns
3. Identify contextual boosters
4. Test and validate new archetypal patterns

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class ArchetypeDiscovery:
    """
    Discovers archetypal patterns from data.
    
    Uses unsupervised learning to find:
    - Dominant patterns in winner narratives
    - Sub-patterns within archetypes
    - Contextual features that boost archetype strength
    
    Examples
    --------
    >>> discovery = ArchetypeDiscovery()
    >>> archetypes = discovery.discover_archetypes(winner_texts, n_archetypes=5)
    >>> print(archetypes['archetype_1']['patterns'])
    ['mental game', 'pressure', 'composure', ...]
    """
    
    def __init__(self, min_pattern_frequency: float = 0.1):
        """
        Parameters
        ----------
        min_pattern_frequency : float
            Minimum frequency (0-1) for a pattern to be considered
        """
        self.min_pattern_frequency = min_pattern_frequency
        
    def discover_archetypes(
        self,
        winner_texts: List[str],
        n_archetypes: int = 5,
        ngram_range: Tuple[int, int] = (1, 3)
    ) -> Dict[str, Dict]:
        """
        Discover archetypal patterns from winner narratives.
        
        Parameters
        ----------
        winner_texts : list of str
            Texts from successful entities
        n_archetypes : int
            Number of archetypes to discover
        ngram_range : tuple
            Range of n-grams to consider (1,3) = unigrams, bigrams, trigrams
        
        Returns
        -------
        dict
            Dictionary of discovered archetypes with patterns and weights
        """
        # Extract frequent patterns using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=self.min_pattern_frequency
        )
        
        tfidf_matrix = vectorizer.fit_transform(winner_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Cluster patterns into archetype groups
        if n_archetypes > 1:
            kmeans = KMeans(n_clusters=n_archetypes, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
        else:
            clusters = np.zeros(len(winner_texts), dtype=int)
        
        # Extract top patterns for each archetype
        archetypes = {}
        
        for archetype_id in range(n_archetypes):
            # Get texts in this cluster
            cluster_indices = np.where(clusters == archetype_id)[0]
            
            if len(cluster_indices) == 0:
                # Skip empty clusters
                continue
            
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            
            # Get top patterns for this archetype
            top_indices = cluster_tfidf.argsort()[-20:][::-1]
            top_patterns = [feature_names[idx] for idx in top_indices]
            top_weights = [cluster_tfidf[idx] for idx in top_indices]
            
            # Clean and filter patterns
            patterns = [p for p in top_patterns if len(p) > 2][:15]
            
            archetypes[f'archetype_{archetype_id + 1}'] = {
                'patterns': patterns,
                'weights': dict(zip(patterns, top_weights[:len(patterns)])),
                'sample_count': len(cluster_indices),
                'coherence': self._calculate_coherence(patterns)
            }
        
        return archetypes
    
    def discover_sub_archetypes(
        self,
        texts: List[str],
        parent_archetype_patterns: List[str],
        n_sub_archetypes: int = 3
    ) -> Dict[str, Dict]:
        """
        Discover sub-archetypes within a main archetype.
        
        For example, within "mental_game" archetype, discover:
        - "pressure_performance" sub-archetype
        - "focus_concentration" sub-archetype
        - "emotional_control" sub-archetype
        
        Parameters
        ----------
        texts : list of str
            Texts containing the parent archetype
        parent_archetype_patterns : list of str
            Patterns defining the parent archetype
        n_sub_archetypes : int
            Number of sub-archetypes to discover
        
        Returns
        -------
        dict
            Sub-archetypes with their patterns
        """
        # Filter texts that contain parent archetype
        parent_texts = [
            text for text in texts
            if any(pattern.lower() in text.lower() for pattern in parent_archetype_patterns)
        ]
        
        if len(parent_texts) < 10:
            return {}
        
        # Discover patterns within parent-containing texts
        sub_archetypes = self.discover_archetypes(
            parent_texts,
            n_archetypes=n_sub_archetypes,
            ngram_range=(2, 4)  # Longer phrases for sub-patterns
        )
        
        return sub_archetypes
    
    def discover_contextual_boosters(
        self,
        texts: List[str],
        outcomes: np.ndarray,
        archetype_patterns: List[str]
    ) -> Dict[str, float]:
        """
        Discover contextual features that boost archetype effectiveness.
        
        For example:
        - "championship" boosts "mental_game" archetype
        - "major" boosts "pressure_performance"
        - "historic" boosts "legacy" narratives
        
        Parameters
        ----------
        texts : list of str
            All texts
        outcomes : ndarray
            Outcomes (higher = better)
        archetype_patterns : list of str
            Patterns defining the archetype
        
        Returns
        -------
        dict
            Contextual features and their boost multipliers
        """
        # Find texts containing the archetype
        archetype_mask = np.array([
            any(pattern.lower() in text.lower() for pattern in archetype_patterns)
            for text in texts
        ])
        
        archetype_texts = [text for i, text in enumerate(texts) if archetype_mask[i]]
        archetype_outcomes = outcomes[archetype_mask]
        
        if len(archetype_texts) < 10:
            return {}
        
        # Extract potential context words
        all_words = []
        for text in archetype_texts:
            words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(50)]
        
        # Test each context word as potential booster
        boosters = {}
        
        for word in common_words:
            # Compare outcomes with vs without this context
            with_context = np.array([
                archetype_outcomes[i] for i, text in enumerate(archetype_texts)
                if word.lower() in text.lower()
            ])
            
            without_context = np.array([
                archetype_outcomes[i] for i, text in enumerate(archetype_texts)
                if word.lower() not in text.lower()
            ])
            
            if len(with_context) > 5 and len(without_context) > 5:
                # Calculate boost effect
                mean_with = np.mean(with_context)
                mean_without = np.mean(without_context)
                
                if mean_without > 0:
                    boost = mean_with / mean_without
                    
                    # Only keep significant boosters (>10% effect)
                    if boost > 1.1:
                        boosters[word] = round(boost, 2)
        
        # Sort by boost strength
        sorted_boosters = dict(sorted(boosters.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return sorted_boosters
    
    def _calculate_coherence(self, patterns: List[str]) -> float:
        """
        Calculate coherence score for a set of patterns.
        
        Measures how semantically related the patterns are.
        Higher = more coherent archetype.
        """
        if len(patterns) < 2:
            return 1.0
        
        # Simple coherence: check for shared words
        all_words = set()
        for pattern in patterns:
            words = set(pattern.lower().split())
            all_words.update(words)
        
        # Average words per pattern
        avg_words = np.mean([len(p.split()) for p in patterns])
        
        # Shared word ratio
        shared_ratio = 1.0 - (len(all_words) / (len(patterns) * avg_words + 1))
        
        return max(0.0, min(1.0, shared_ratio))


class ArchetypeRegistry:
    """
    Registry for managing discovered and manually defined archetypes.
    
    Makes it easy to:
    - Add new archetypes (domain-specific or agnostic)
    - Organize archetypes hierarchically (archetype -> sub-archetype)
    - Manage contextual boosters
    - Export/import archetype definitions
    
    Examples
    --------
    >>> registry = ArchetypeRegistry()
    >>> 
    >>> # Add discovered archetype
    >>> registry.register_archetype(
    ...     name='mental_game',
    ...     patterns=['pressure', 'composure', 'focus'],
    ...     domain='golf',
    ...     weight=0.30
    ... )
    >>> 
    >>> # Add sub-archetype
    >>> registry.register_sub_archetype(
    ...     parent='mental_game',
    ...     name='pressure_performance',
    ...     patterns=['clutch', 'big moment', 'crunch time'],
    ...     domain='golf'
    ... )
    >>> 
    >>> # Add contextual booster
    >>> registry.add_contextual_booster(
    ...     archetype='mental_game',
    ...     context='championship',
    ...     boost_multiplier=1.3,
    ...     domain='golf'
    ... )
    """
    
    def __init__(self):
        self.archetypes = {}
        self.sub_archetypes = {}
        self.contextual_boosters = {}
        
    def register_archetype(
        self,
        name: str,
        patterns: List[str],
        domain: Optional[str] = None,
        weight: float = 1.0,
        description: str = ""
    ):
        """
        Register a new archetype.
        
        Parameters
        ----------
        name : str
            Archetype name (e.g., 'mental_game', 'elite_skill')
        patterns : list of str
            Patterns defining this archetype
        domain : str, optional
            Domain (e.g., 'golf') or None for domain-agnostic
        weight : float
            Importance weight for this archetype
        description : str
            Human-readable description
        """
        key = f"{domain}::{name}" if domain else f"agnostic::{name}"
        
        self.archetypes[key] = {
            'name': name,
            'patterns': patterns,
            'domain': domain,
            'weight': weight,
            'description': description,
            'sub_archetypes': []
        }
        
    def register_sub_archetype(
        self,
        parent: str,
        name: str,
        patterns: List[str],
        domain: Optional[str] = None,
        weight: float = 1.0
    ):
        """
        Register a sub-archetype under a parent.
        
        Parameters
        ----------
        parent : str
            Parent archetype name
        name : str
            Sub-archetype name
        patterns : list of str
            Patterns for sub-archetype
        domain : str, optional
            Domain or None
        weight : float
            Weight within parent
        """
        parent_key = f"{domain}::{parent}" if domain else f"agnostic::{parent}"
        sub_key = f"{parent_key}::{name}"
        
        self.sub_archetypes[sub_key] = {
            'parent': parent,
            'name': name,
            'patterns': patterns,
            'domain': domain,
            'weight': weight
        }
        
        # Link to parent
        if parent_key in self.archetypes:
            self.archetypes[parent_key]['sub_archetypes'].append(name)
    
    def add_contextual_booster(
        self,
        archetype: str,
        context: str,
        boost_multiplier: float,
        domain: Optional[str] = None
    ):
        """
        Add a contextual booster for an archetype.
        
        Parameters
        ----------
        archetype : str
            Archetype name
        context : str
            Context keyword (e.g., 'championship', 'playoff')
        boost_multiplier : float
            Boost multiplier (e.g., 1.3 = 30% boost)
        domain : str, optional
            Domain or None
        """
        key = f"{domain}::{archetype}" if domain else f"agnostic::{archetype}"
        
        if key not in self.contextual_boosters:
            self.contextual_boosters[key] = {}
        
        self.contextual_boosters[key][context] = boost_multiplier
    
    def get_archetype(self, name: str, domain: Optional[str] = None) -> Optional[Dict]:
        """Get archetype definition."""
        key = f"{domain}::{name}" if domain else f"agnostic::{name}"
        return self.archetypes.get(key)
    
    def get_sub_archetypes(self, parent: str, domain: Optional[str] = None) -> List[Dict]:
        """Get all sub-archetypes for a parent."""
        parent_key = f"{domain}::{parent}" if domain else f"agnostic::{parent}"
        
        sub_archs = []
        for key, sub in self.sub_archetypes.items():
            if key.startswith(parent_key):
                sub_archs.append(sub)
        
        return sub_archs
    
    def get_contextual_boosters(self, archetype: str, domain: Optional[str] = None) -> Dict[str, float]:
        """Get contextual boosters for an archetype."""
        key = f"{domain}::{archetype}" if domain else f"agnostic::{archetype}"
        return self.contextual_boosters.get(key, {})
    
    def list_archetypes(self, domain: Optional[str] = None) -> List[str]:
        """List all archetype names for a domain."""
        if domain:
            return [
                arch['name'] for key, arch in self.archetypes.items()
                if arch['domain'] == domain
            ]
        else:
            return [arch['name'] for arch in self.archetypes.values()]
    
    def export_to_dict(self) -> Dict:
        """Export entire registry to dictionary (for saving)."""
        return {
            'archetypes': self.archetypes,
            'sub_archetypes': self.sub_archetypes,
            'contextual_boosters': self.contextual_boosters
        }
    
    def import_from_dict(self, data: Dict):
        """Import registry from dictionary (for loading)."""
        self.archetypes = data.get('archetypes', {})
        self.sub_archetypes = data.get('sub_archetypes', {})
        self.contextual_boosters = data.get('contextual_boosters', {})


class ArchetypeValidator:
    """
    Validates and tests discovered archetypes.
    
    Tests if a discovered archetype:
    - Improves prediction accuracy
    - Has statistical significance
    - Generalizes across data splits
    
    Examples
    --------
    >>> validator = ArchetypeValidator()
    >>> is_valid = validator.validate_archetype(
    ...     archetype_patterns=['pressure', 'clutch'],
    ...     texts=all_texts,
    ...     outcomes=all_outcomes
    ... )
    >>> print(f"Valid archetype: {is_valid}")
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        
    def validate_archetype(
        self,
        archetype_patterns: List[str],
        texts: List[str],
        outcomes: np.ndarray,
        baseline_score: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Validate an archetype's predictive power.
        
        Parameters
        ----------
        archetype_patterns : list of str
            Patterns defining the archetype
        texts : list of str
            All texts
        outcomes : ndarray
            Outcomes
        baseline_score : float, optional
            Baseline correlation without archetype
        
        Returns
        -------
        dict
            Validation results including significance, improvement, etc.
        """
        # Score each text on this archetype
        archetype_scores = np.array([
            sum(1 for pattern in archetype_patterns if pattern.lower() in text.lower())
            for text in texts
        ])
        
        # Normalize
        if archetype_scores.max() > 0:
            archetype_scores = archetype_scores / archetype_scores.max()
        
        # Calculate correlation with outcomes
        if len(archetype_scores) > 0 and archetype_scores.std() > 0:
            correlation = np.corrcoef(archetype_scores, outcomes)[0, 1]
        else:
            correlation = 0.0
        
        # Statistical significance (t-test approximation)
        n = len(archetype_scores)
        if n > 3:
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
            # Rough p-value approximation
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        # Improvement over baseline
        improvement = None
        if baseline_score is not None:
            improvement = correlation - baseline_score
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': p_value < self.significance_threshold,
            'improvement': improvement,
            'sample_size': n,
            'patterns': archetype_patterns
        }
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        from scipy.special import betainc
        x = df / (df + t**2)
        return 1 - 0.5 * betainc(df/2, 0.5, x) if df > 0 else 0.5

