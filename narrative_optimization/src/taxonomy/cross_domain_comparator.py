"""
Cross-Domain Comparator - Compare domains across dimensions
"""

from typing import List, Dict
from .domain_registry import DomainMetadata
import numpy as np


class CrossDomainComparator:
    """Compare and analyze relationships between domains."""
    
    @staticmethod
    def compare_two_domains(domain1: DomainMetadata, 
                           domain2: DomainMetadata) -> Dict:
        """Compare two domains across all dimensions."""
        return {
            'domains': (domain1.display_name, domain2.display_name),
            'visibility_diff': domain2.visibility - domain1.visibility,
            'effect_diff': (domain2.effect_size_observed - domain1.effect_size_observed
                          if domain1.effect_size_observed and domain2.effect_size_observed
                          else None),
            'same_type': domain1.domain_type == domain2.domain_type,
            'both_implemented': (domain1.status == 'completed' and 
                               domain2.status == 'completed')
        }
    
    @staticmethod
    def find_similar_domains(target: DomainMetadata, 
                            all_domains: List[DomainMetadata],
                            n: int = 3) -> List[DomainMetadata]:
        """Find domains most similar to target."""
        similarities = []
        for domain in all_domains:
            if domain.id == target.id:
                continue
            
            # Calculate similarity score
            vis_sim = 100 - abs(target.visibility - domain.visibility)
            type_sim = 100 if target.domain_type == domain.type else 0
            importance_sim = 100 if target.narrative_importance == domain.narrative_importance else 50
            
            total_sim = (0.5 * vis_sim + 0.3 * type_sim + 0.2 * importance_sim)
            similarities.append((domain, total_sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in similarities[:n]]
    
    @staticmethod
    def calculate_correlation(domains: List[DomainMetadata]) -> Dict:
        """Calculate correlation between visibility and effect size."""
        # Filter domains with observed effects
        valid_domains = [d for d in domains if d.effect_size_observed is not None]
        
        if len(valid_domains) < 3:
            return {'error': 'Insufficient data for correlation'}
        
        visibilities = np.array([d.visibility for d in valid_domains])
        effects = np.array([d.effect_size_observed for d in valid_domains])
        
        correlation = np.corrcoef(visibilities, effects)[0, 1]
        
        return {
            'n_domains': len(valid_domains),
            'correlation': correlation,
            'interpretation': 'Negative correlation confirms visibility moderation' 
                            if correlation < -0.5 else 'Check model assumptions'
        }

