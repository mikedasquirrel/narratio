"""
Taxonomy Builder - Automatic categorization and organization

Builds hierarchical taxonomy from domain metadata.
"""

from typing import Dict, List
from .domain_registry import DomainRegistry, DomainMetadata


class TaxonomyBuilder:
    """Build and navigate domain taxonomy."""
    
    def __init__(self, registry: DomainRegistry):
        """Initialize with domain registry."""
        self.registry = registry
    
    def build_visibility_taxonomy(self) -> Dict:
        """Organize domains by visibility levels."""
        return {
            'ultra_high': self.registry.get_domains_by_visibility_range(80, 100),
            'high': self.registry.get_domains_by_visibility_range(60, 79),
            'medium': self.registry.get_domains_by_visibility_range(40, 59),
            'low': self.registry.get_domains_by_visibility_range(20, 39),
            'very_low': self.registry.get_domains_by_visibility_range(0, 19)
        }
    
    def build_type_taxonomy(self) -> Dict[str, List[DomainMetadata]]:
        """Organize domains by type."""
        types = set(d.domain_type for d in self.registry.get_all_domains())
        return {t: self.registry.get_domains_by_type(t) for t in types}
    
    def build_narrative_importance_taxonomy(self) -> Dict:
        """Organize by narrative importance."""
        all_domains = self.registry.get_all_domains()
        return {
            'high': [d for d in all_domains if 'high' in d.narrative_importance],
            'medium': [d for d in all_domains if 'medium' in d.narrative_importance 
                      and 'high' not in d.narrative_importance],
            'low': [d for d in all_domains if d.narrative_importance == 'low']
        }
    
    def build_quadrant_taxonomy(self) -> Dict:
        """Organize by visibility × narrative importance quadrants."""
        all_domains = self.registry.get_all_domains()
        
        return {
            'high_vis_high_narrative': [
                d for d in all_domains 
                if d.visibility >= 60 and 'high' in d.narrative_importance
            ],
            'high_vis_low_narrative': [
                d for d in all_domains 
                if d.visibility >= 60 and d.narrative_importance == 'low'
            ],
            'low_vis_high_narrative': [
                d for d in all_domains 
                if d.visibility < 60 and 'high' in d.narrative_importance
            ],
            'low_vis_low_narrative': [
                d for d in all_domains 
                if d.visibility < 60 and 'medium' not in d.narrative_importance 
                and 'high' not in d.narrative_importance
            ]
        }

