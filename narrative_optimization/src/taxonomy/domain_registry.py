"""
Domain Registry - Central registry of all narrative analysis domains

Maintains metadata for all domains including:
- Visibility scores
- Narrative importance
- Effect sizes (observed/predicted)
- Implementation status
- Data sources

Author: Narrative Optimization Research
Date: November 2025
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DomainMetadata:
    """Metadata for a single domain."""
    id: str
    name: str
    display_name: str
    visibility: float  # 0-100%
    narrative_importance: str  # low, medium, high
    domain_type: str  # sports, natural, social, medical, historical
    effect_size_observed: Optional[float] = None
    effect_size_predicted: Optional[float] = None
    status: str = 'proposed'  # proposed, in_progress, completed
    n_samples: Optional[int] = None
    data_source: Optional[str] = None
    implementation_date: Optional[str] = None
    route_prefix: Optional[str] = None
    description: str = ''
    key_findings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'visibility': self.visibility,
            'narrative_importance': self.narrative_importance,
            'domain_type': self.domain_type,
            'effect_size_observed': self.effect_size_observed,
            'effect_size_predicted': self.effect_size_predicted,
            'status': self.status,
            'n_samples': self.n_samples,
            'data_source': self.data_source,
            'implementation_date': self.implementation_date,
            'route_prefix': self.route_prefix,
            'description': self.description,
            'key_findings': self.key_findings
        }


class DomainRegistry:
    """
    Central registry of all narrative analysis domains.
    
    Provides:
    - Domain registration and lookup
    - Cross-domain queries
    - Status tracking
    - Metadata management
    """
    
    def __init__(self):
        """Initialize registry with existing domains."""
        self.domains: Dict[str, DomainMetadata] = {}
        self._initialize_existing_domains()
    
    def _initialize_existing_domains(self):
        """Load existing implemented domains."""
        
        # NBA
        self.register_domain(DomainMetadata(
            id='nba',
            name='nba',
            display_name='NBA Basketball',
            visibility=75.0,
            narrative_importance='medium-high',
            domain_type='sports',
            effect_size_observed=0.24,
            effect_size_predicted=0.23,
            status='completed',
            n_samples=500,
            data_source='NBA API',
            implementation_date='2025-11-01',
            route_prefix='/nba',
            description='NBA player names and performance analysis',
            key_findings=[
                'Name memorability correlates with career earnings',
                'Position-appropriate names show selection effects',
                'Brand value adds to pure statistics'
            ]
        ))
        
        # Hurricanes
        self.register_domain(DomainMetadata(
            id='hurricanes',
            name='hurricanes',
            display_name='Hurricane Names',
            visibility=25.0,
            narrative_importance='high',
            domain_type='natural',
            effect_size_observed=0.32,
            effect_size_predicted=0.30,
            status='completed',
            n_samples=100,
            data_source='NOAA + Historical Records',
            implementation_date='2025-11-10',
            route_prefix='/hurricanes',
            description='Hurricane name gender effects on evacuation behavior',
            key_findings=[
                'Feminine names → 18.6% lower evacuation rate',
                "Cohen's d = 0.947 (large effect)",
                'Name perception affects life-or-death decisions'
            ]
        ))
        
        # MLB Baseball
        self.register_domain(DomainMetadata(
            id='mlb',
            name='mlb',
            display_name='MLB Baseball',
            visibility=80.0,
            narrative_importance='medium-high',
            domain_type='sports',
            effect_size_observed=None,
            effect_size_predicted=0.25,
            status='completed',
            n_samples=24000,
            data_source='MLB Stats API',
            implementation_date='2025-11-11',
            route_prefix='/mlb',
            description='MLB game analysis with team names, rivalries, stadiums, and narratives',
            key_findings=[
                'Medium-high narrativity (π ≈ 0.25-0.30)',
                'Rivalry games show stronger narrative effects',
                'Historic stadiums create narrative context',
                'Team sport with individual performance elements'
            ]
        ))
        
        # Reference domains from literature
        self._add_reference_domains()
    
    def _add_reference_domains(self):
        """Add reference domains from literature (not yet implemented)."""
        
        reference = [
            ('adult_film', 'Adult Film', 95.0, 'low', 'social', 0.00),
            ('baseball_mlb', 'MLB Baseball', 80.0, 'medium', 'sports', 0.19),
            ('football_nfl', 'NFL Football', 70.0, 'medium', 'sports', 0.21),
            ('ships', 'Naval Ships', 50.0, 'medium', 'historical', 0.18),
            ('board_games', 'Board Games', 40.0, 'medium', 'social', 0.14),
            ('elections', 'Elections', 40.0, 'medium-high', 'social', 0.22),
            ('bands_music', 'Bands/Music', 35.0, 'high', 'social', 0.19),
            ('immigration', 'Immigration', 30.0, 'medium', 'social', 0.20),
            ('mtg_cards', 'MTG Cards', 30.0, 'medium', 'social', 0.15),
            ('mental_health', 'Mental Health', 25.0, 'high', 'medical', 0.29),
            ('cryptocurrencies', 'Cryptocurrencies', 15.0, 'high', 'social', 0.28),
        ]
        
        for id, name, vis, importance, type, effect in reference:
            if id not in self.domains:  # Don't override implemented ones
                self.register_domain(DomainMetadata(
                    id=id,
                    name=id,
                    display_name=name,
                    visibility=vis,
                    narrative_importance=importance,
                    domain_type=type,
                    effect_size_observed=effect,
                    status='proposed',
                    description=f'{name} domain from literature'
                ))
    
    def register_domain(self, domain: DomainMetadata):
        """Register a new domain."""
        self.domains[domain.id] = domain
    
    def get_domain(self, domain_id: str) -> Optional[DomainMetadata]:
        """Get domain by ID."""
        return self.domains.get(domain_id)
    
    def get_all_domains(self) -> List[DomainMetadata]:
        """Get all registered domains."""
        return list(self.domains.values())
    
    def get_domains_by_status(self, status: str) -> List[DomainMetadata]:
        """Get domains with specific status."""
        return [d for d in self.domains.values() if d.status == status]
    
    def get_domains_by_type(self, domain_type: str) -> List[DomainMetadata]:
        """Get domains of specific type."""
        return [d for d in self.domains.values() if d.domain_type == domain_type]
    
    def get_domains_by_visibility_range(self, min_vis: float, 
                                       max_vis: float) -> List[DomainMetadata]:
        """Get domains within visibility range."""
        return [d for d in self.domains.values() 
                if min_vis <= d.visibility <= max_vis]
    
    def get_implemented_domains(self) -> List[DomainMetadata]:
        """Get domains with completed implementation."""
        return self.get_domains_by_status('completed')
    
    def get_proposed_domains(self) -> List[DomainMetadata]:
        """Get proposed but not yet implemented domains."""
        return self.get_domains_by_status('proposed')
    
    def update_domain_status(self, domain_id: str, status: str):
        """Update domain implementation status."""
        if domain_id in self.domains:
            self.domains[domain_id].status = status
            if status == 'completed' and not self.domains[domain_id].implementation_date:
                self.domains[domain_id].implementation_date = datetime.now().strftime('%Y-%m-%d')
    
    def update_observed_effect(self, domain_id: str, effect_size: float):
        """Update observed effect size for domain."""
        if domain_id in self.domains:
            self.domains[domain_id].effect_size_observed = effect_size
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all domains."""
        all_domains = self.get_all_domains()
        implemented = self.get_implemented_domains()
        
        # Visibility statistics
        visibilities = [d.visibility for d in all_domains]
        effects = [d.effect_size_observed for d in all_domains 
                  if d.effect_size_observed is not None]
        
        return {
            'total_domains': len(all_domains),
            'implemented': len(implemented),
            'proposed': len(self.get_proposed_domains()),
            'in_progress': len(self.get_domains_by_status('in_progress')),
            'visibility_range': (min(visibilities), max(visibilities)),
            'visibility_mean': sum(visibilities) / len(visibilities),
            'effect_range': (min(effects), max(effects)) if effects else (None, None),
            'effect_mean': sum(effects) / len(effects) if effects else None,
            'domain_types': list(set(d.domain_type for d in all_domains))
        }
    
    def export_to_json(self, filepath: str):
        """Export registry to JSON file."""
        data = {
            'domains': [d.to_dict() for d in self.get_all_domains()],
            'summary': self.get_summary_statistics(),
            'export_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_taxonomy_report(self) -> str:
        """Generate comprehensive taxonomy report."""
        stats = self.get_summary_statistics()
        
        report = f"""
{'='*70}
NARRATIVE DOMAIN TAXONOMY
{'='*70}

OVERVIEW
--------
Total Domains: {stats['total_domains']}
Implemented: {stats['implemented']}
In Progress: {stats['in_progress']}
Proposed: {stats['proposed']}

VISIBILITY SPECTRUM
-------------------
Range: {stats['visibility_range'][0]:.0f}% - {stats['visibility_range'][1]:.0f}%
Mean: {stats['visibility_mean']:.1f}%

EFFECT SIZES
------------
Range: {stats['effect_range'][0]:.3f} - {stats['effect_range'][1]:.3f}
Mean: {stats['effect_mean']:.3f}

DOMAIN TYPES
------------
"""
        
        for dtype in stats['domain_types']:
            domains_of_type = self.get_domains_by_type(dtype)
            report += f"  {dtype}: {len(domains_of_type)} domains\n"
        
        report += "\n" + "="*70 + "\n\nIMPLEMENTED DOMAINS\n" + "="*70 + "\n\n"
        
        for domain in sorted(self.get_implemented_domains(), 
                           key=lambda x: x.visibility):
            report += f"{domain.display_name}\n"
            report += f"  Visibility: {domain.visibility:.0f}%\n"
            report += f"  Effect: r = {domain.effect_size_observed:.3f}\n"
            report += f"  Type: {domain.domain_type}\n"
            report += f"  Status: {domain.status}\n"
            report += f"  Route: {domain.route_prefix}\n"
            if domain.key_findings:
                report += "  Key Findings:\n"
                for finding in domain.key_findings:
                    report += f"    • {finding}\n"
            report += "\n"
        
        report += "="*70 + "\n"
        
        return report


if __name__ == '__main__':
    # Demo
    registry = DomainRegistry()
    print(registry.generate_taxonomy_report())

