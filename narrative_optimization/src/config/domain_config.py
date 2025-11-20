"""
Domain Configuration Loader

Loads and manages domain-specific configuration for archetypal analysis.

Author: Narrative Integration System
Date: November 2025
"""

from typing import Dict, List, Any, Optional, Tuple
from .domain_archetypes import DOMAIN_ARCHETYPES, get_generic_archetype


class DomainConfig:
    """
    Loads and manages domain-specific configuration.
    
    Provides access to domain-specific archetypal patterns, weights,
    and parameters discovered from domain analyses.
    
    Parameters
    ----------
    domain_name : str
        Name of the domain (e.g., 'golf', 'boxing', 'nba', 'wwe', 'tennis')
    
    Attributes
    ----------
    domain_name : str
        The domain name
    archetype : dict
        Domain-specific archetype configuration
    
    Examples
    --------
    >>> config = DomainConfig('golf')
    >>> patterns = config.get_archetype_patterns()
    >>> print(patterns['mental_game'])
    ['all mental', 'between the ears', 'choking', 'clutch', ...]
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.archetype = DOMAIN_ARCHETYPES.get(domain_name, get_generic_archetype())
        
    def get_archetype_patterns(self) -> Dict[str, List[str]]:
        """
        Get domain-specific archetype patterns.
        
        Returns
        -------
        dict
            Dictionary mapping archetype names to pattern lists
        """
        return self.archetype['archetype_patterns']
    
    def get_archetype_weights(self) -> Dict[str, float]:
        """
        Get weights for each archetype dimension.
        
        Returns
        -------
        dict
            Dictionary mapping archetype names to weights (sum to 1.0)
        """
        return self.archetype['archetype_weights']
    
    def is_prestige_domain(self) -> bool:
        """
        Check if this is a prestige domain (uses inverted equation).
        
        Returns
        -------
        bool
            True if prestige domain (WWE, Oscars), False otherwise
        """
        return self.archetype.get('prestige_domain', False)
    
    def get_pi(self) -> float:
        """
        Get domain narrativity (π).
        
        Returns
        -------
        float
            Narrativity value (0-1)
        """
        return self.archetype['pi']
    
    def get_theta_range(self) -> Tuple[float, float]:
        """
        Get expected theta (awareness resistance) range.
        
        Returns
        -------
        tuple
            (min_theta, max_theta)
        """
        return self.archetype['theta_range']
    
    def get_lambda_range(self) -> Tuple[float, float]:
        """
        Get expected lambda (fundamental constraints) range.
        
        Returns
        -------
        tuple
            (min_lambda, max_lambda)
        """
        return self.archetype['lambda_range']
    
    def get_nominative_richness_requirement(self) -> int:
        """
        Get required number of proper nouns for optimal narrative.
        
        Returns
        -------
        int
            Number of proper nouns needed
        """
        return self.archetype['nominative_richness_requirement']
    
    def get_blind_narratio(self) -> Optional[float]:
        """
        Get discovered Blind Narratio (Β) for this domain.
        
        Β = equilibrium ratio between deterministic and free will forces
        
        Returns
        -------
        float or None
            Blind Narratio value if discovered, None otherwise
        """
        return self.archetype.get('blind_narratio', None)
    
    def get_pi_sensitivity(self) -> float:
        """
        Get π sensitivity parameter for instance-level variation.
        
        Determines how much π can vary within domain based on
        instance complexity:
        
        π_effective = π_base + pi_sensitivity × complexity
        
        Returns
        -------
        float
            Sensitivity parameter (typically 0.0 - 0.3)
        """
        return self.archetype.get('pi_sensitivity', 0.2)
    
    def calculate_effective_pi(self, instance_complexity: float) -> float:
        """
        Calculate instance-specific π based on complexity.
        
        Revolutionary finding from Supreme Court domain:
        π is NOT domain-constant, it varies by instance complexity.
        
        Parameters
        ----------
        instance_complexity : float
            Complexity score for specific instance (0-1)
            - 0 = simple, clear, low complexity
            - 1 = highly complex, ambiguous, contested
        
        Returns
        -------
        float
            Effective π for this instance (clipped to [0, 1])
        """
        base_pi = self.get_pi()
        sensitivity = self.get_pi_sensitivity()
        
        # π increases with complexity (more complex = more narrative matters)
        pi_effective = base_pi + sensitivity * instance_complexity
        
        # Clip to valid range
        return max(0.0, min(1.0, pi_effective))
    
    def get_awareness_amplification_range(self) -> Tuple[float, float]:
        """
        Get expected awareness amplification (θ_amp) range for domain.
        
        θ_amp is distinct from θ_resistance:
        - θ_resistance: awareness suppressing narrative effects
        - θ_amp: awareness amplifying potential energy
        
        Returns
        -------
        tuple
            (min_theta_amp, max_theta_amp)
        """
        return self.archetype.get('theta_amplification_range', (0.0, 0.5))
    
    def get_imperative_gravity_neighbors(self) -> List[str]:
        """
        Get typical cross-domain gravitational neighbors.
        
        Domains with similar structural properties that instances
        in this domain are typically pulled toward.
        
        Returns
        -------
        list of str
            Domain names with high structural similarity
        """
        return self.archetype.get('imperative_gravity_neighbors', [])
    
    def get_domain_specific_patterns(self, pattern_type: str) -> Optional[List[str]]:
        """
        Get domain-specific patterns for a given type.
        
        Parameters
        ----------
        pattern_type : str
            Type of pattern (e.g., 'potential', 'awareness', 'constraints')
        
        Returns
        -------
        list or None
            List of patterns if available, None otherwise
        """
        # This can be extended to support more pattern types
        # For now, return archetype patterns if they match
        patterns = self.archetype['archetype_patterns']
        return patterns.get(pattern_type, None)
    
    def __repr__(self) -> str:
        return f"DomainConfig(domain='{self.domain_name}', pi={self.get_pi():.3f}, prestige={self.is_prestige_domain()})"

