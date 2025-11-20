"""
Imperative Gravity Calculator

Calculates cross-domain gravitational forces (ф_imperative).

Instances are pulled toward structurally similar domains for comparison
and learning. This is the force that enables cross-domain intelligence.

Formula:
ф_imperative(instance→domain) = (μ × structural_similarity) / domain_distance²

Where:
- structural_similarity = f(π similarity, θ overlap, archetype resonance)
- domain_distance = sqrt(Σ(feature_importance_differences²))

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance
from config.domain_config import DomainConfig


class ImperativeGravityCalculator:
    """
    Calculate cross-domain gravitational forces.
    
    Enables:
    - Finding structurally similar domains
    - Cross-domain pattern transfer
    - Imperative neighbor identification
    - Domain similarity matrix construction
    """
    
    def __init__(self, all_domain_configs: Optional[Dict[str, DomainConfig]] = None):
        """
        Initialize calculator.
        
        Parameters
        ----------
        all_domain_configs : dict, optional
            {domain_name: DomainConfig} for all domains
        """
        self.domain_configs = all_domain_configs or {}
        
        # Cache for structural similarity calculations
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.distance_cache: Dict[Tuple[str, str], float] = {}
    
    def calculate_cross_domain_forces(
        self,
        instance: StoryInstance,
        target_domains: List[str],
        exclude_same_domain: bool = True
    ) -> Dict[str, float]:
        """
        Calculate imperative gravity forces from instance to all target domains.
        
        Parameters
        ----------
        instance : StoryInstance
            Source instance
        target_domains : list of str
            Domains to calculate forces toward
        exclude_same_domain : bool
            Whether to exclude instance's own domain
        
        Returns
        -------
        dict
            {domain_name: force_magnitude}
        """
        forces = {}
        
        for target_domain in target_domains:
            # Skip same domain if requested
            if exclude_same_domain and target_domain == instance.domain:
                continue
            
            # Calculate force
            force = self._calculate_force_to_domain(instance, target_domain)
            forces[target_domain] = force
        
        return forces
    
    def find_gravitational_neighbors(
        self,
        instance: StoryInstance,
        all_domains: List[str],
        n_neighbors: int = 5,
        exclude_same_domain: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find the N domains with strongest imperative pull.
        
        These are the domains to learn from.
        
        Parameters
        ----------
        instance : StoryInstance
            Source instance
        all_domains : list of str
            All available domains
        n_neighbors : int
            Number of neighbors to return
        exclude_same_domain : bool
            Whether to exclude instance's own domain
        
        Returns
        -------
        list of tuple
            [(domain_name, force_magnitude), ...] sorted by force (descending)
        """
        forces = self.calculate_cross_domain_forces(
            instance,
            all_domains,
            exclude_same_domain
        )
        
        # Sort by force magnitude (descending)
        sorted_forces = sorted(forces.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_forces[:n_neighbors]
    
    def calculate_domain_similarity_matrix(
        self,
        domains: List[str]
    ) -> np.ndarray:
        """
        Calculate structural similarity matrix for all domains.
        
        Parameters
        ----------
        domains : list of str
            Domain names
        
        Returns
        -------
        ndarray
            Similarity matrix (n_domains, n_domains)
        """
        n = len(domains)
        similarity_matrix = np.zeros((n, n))
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self._calculate_domain_similarity(domain1, domain2)
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _calculate_force_to_domain(
        self,
        instance: StoryInstance,
        target_domain: str
    ) -> float:
        """
        Calculate force from instance to target domain.
        
        ф_imperative = (μ × similarity) / distance²
        """
        # Get mass
        mass = instance.mass if instance.mass else 1.0
        
        # Calculate structural similarity
        similarity = self._calculate_domain_similarity(
            instance.domain,
            target_domain
        )
        
        # Calculate domain distance
        distance = self._calculate_domain_distance(
            instance.domain,
            target_domain
        )
        
        # Prevent division by zero
        distance_sq = max(distance ** 2, 0.01)
        
        # Calculate force
        force = (mass * similarity) / distance_sq
        
        return force
    
    def _calculate_domain_similarity(
        self,
        domain1: str,
        domain2: str
    ) -> float:
        """
        Calculate structural similarity between two domains.
        
        Considers:
        - π similarity
        - θ range overlap
        - Archetype pattern overlap
        
        Returns
        -------
        float
            Similarity score (0-1)
        """
        # Check cache
        cache_key = tuple(sorted([domain1, domain2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get domain configs
        config1 = self.domain_configs.get(domain1)
        config2 = self.domain_configs.get(domain2)
        
        if config1 is None or config2 is None:
            # Default similarity if configs not available
            similarity = 0.5
        else:
            similarities = []
            
            # π similarity
            pi1 = config1.get_pi()
            pi2 = config2.get_pi()
            pi_sim = 1.0 - abs(pi1 - pi2)
            similarities.append(pi_sim)
            
            # θ range overlap
            theta_range1 = config1.get_theta_range()
            theta_range2 = config2.get_theta_range()
            theta_overlap = self._calculate_range_overlap(theta_range1, theta_range2)
            similarities.append(theta_overlap)
            
            # λ range overlap
            lambda_range1 = config1.get_lambda_range()
            lambda_range2 = config2.get_lambda_range()
            lambda_overlap = self._calculate_range_overlap(lambda_range1, lambda_range2)
            similarities.append(lambda_overlap)
            
            # Prestige domain similarity
            prestige1 = config1.is_prestige_domain()
            prestige2 = config2.is_prestige_domain()
            prestige_sim = 1.0 if prestige1 == prestige2 else 0.0
            similarities.append(prestige_sim)
            
            # Weight and combine
            similarity = (
                0.40 * similarities[0] +  # π similarity (most important)
                0.25 * similarities[1] +  # θ overlap
                0.20 * similarities[2] +  # λ overlap
                0.15 * similarities[3]    # prestige similarity
            )
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _calculate_domain_distance(
        self,
        domain1: str,
        domain2: str
    ) -> float:
        """
        Calculate distance between domains in feature space.
        
        Returns
        -------
        float
            Distance (0-infinity, lower = more similar)
        """
        # Check cache
        cache_key = tuple(sorted([domain1, domain2]))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Get configs
        config1 = self.domain_configs.get(domain1)
        config2 = self.domain_configs.get(domain2)
        
        if config1 is None or config2 is None:
            distance = 1.0
        else:
            # Feature vector: [π, θ_min, θ_max, λ_min, λ_max, prestige]
            vec1 = np.array([
                config1.get_pi(),
                config1.get_theta_range()[0],
                config1.get_theta_range()[1],
                config1.get_lambda_range()[0],
                config1.get_lambda_range()[1],
                1.0 if config1.is_prestige_domain() else 0.0
            ])
            
            vec2 = np.array([
                config2.get_pi(),
                config2.get_theta_range()[0],
                config2.get_theta_range()[1],
                config2.get_lambda_range()[0],
                config2.get_lambda_range()[1],
                1.0 if config2.is_prestige_domain() else 0.0
            ])
            
            # Euclidean distance
            distance = np.linalg.norm(vec1 - vec2)
        
        # Cache result
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def _calculate_range_overlap(
        self,
        range1: Tuple[float, float],
        range2: Tuple[float, float]
    ) -> float:
        """
        Calculate overlap between two ranges.
        
        Returns
        -------
        float
            Overlap score (0-1)
        """
        min1, max1 = range1
        min2, max2 = range2
        
        # Calculate overlap
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_end <= overlap_start:
            # No overlap
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        
        # Calculate union
        union_start = min(min1, min2)
        union_end = max(max1, max2)
        union_length = union_end - union_start
        
        if union_length == 0:
            return 1.0
        
        # Overlap ratio (Jaccard-like)
        overlap_ratio = overlap_length / union_length
        
        return overlap_ratio
    
    def get_domain_clusters(
        self,
        domains: List[str],
        similarity_threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Identify clusters of structurally similar domains.
        
        Parameters
        ----------
        domains : list of str
            All domains
        similarity_threshold : float
            Minimum similarity to be in same cluster
        
        Returns
        -------
        list of list
            Clusters of domain names
        """
        # Build similarity matrix
        similarity_matrix = self.calculate_domain_similarity_matrix(domains)
        
        # Simple clustering: greedy approach
        clusters = []
        assigned = set()
        
        for i, domain in enumerate(domains):
            if domain in assigned:
                continue
            
            # Start new cluster
            cluster = [domain]
            assigned.add(domain)
            
            # Find similar domains
            for j, other_domain in enumerate(domains):
                if other_domain in assigned:
                    continue
                
                if similarity_matrix[i, j] >= similarity_threshold:
                    cluster.append(other_domain)
                    assigned.add(other_domain)
            
            clusters.append(cluster)
        
        return clusters
    
    def explain_gravitational_pull(
        self,
        instance: StoryInstance,
        target_domain: str
    ) -> Dict[str, Any]:
        """
        Explain why instance is pulled toward target domain.
        
        Parameters
        ----------
        instance : StoryInstance
            Source instance
        target_domain : str
            Target domain
        
        Returns
        -------
        dict
            Explanation with component breakdowns
        """
        force = self._calculate_force_to_domain(instance, target_domain)
        similarity = self._calculate_domain_similarity(instance.domain, target_domain)
        distance = self._calculate_domain_distance(instance.domain, target_domain)
        
        explanation = {
            'source_domain': instance.domain,
            'target_domain': target_domain,
            'force_magnitude': float(force),
            'structural_similarity': float(similarity),
            'domain_distance': float(distance),
            'interpretation': self._interpret_force(force),
            'learning_potential': self._assess_learning_potential(similarity, distance)
        }
        
        return explanation
    
    def _interpret_force(self, force: float) -> str:
        """Generate human-readable interpretation of force magnitude."""
        if force > 10.0:
            return "Very strong gravitational pull - highly similar structures"
        elif force > 5.0:
            return "Strong pull - significant structural similarity"
        elif force > 2.0:
            return "Moderate pull - some structural overlap"
        elif force > 0.5:
            return "Weak pull - limited structural similarity"
        else:
            return "Negligible pull - very different structures"
    
    def _assess_learning_potential(self, similarity: float, distance: float) -> str:
        """Assess potential for cross-domain learning."""
        if similarity > 0.8:
            return "High - domains are highly analogous, direct pattern transfer likely effective"
        elif similarity > 0.6:
            return "Good - substantial overlap, many transferable patterns"
        elif similarity > 0.4:
            return "Moderate - some transferable concepts but requires adaptation"
        elif similarity > 0.2:
            return "Low - limited transferability, mainly conceptual insights"
        else:
            return "Minimal - domains too dissimilar for effective pattern transfer"

