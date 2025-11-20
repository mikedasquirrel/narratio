"""
Domain Registry

Central registry of all analyzed domains with metadata and relationships.

Author: Narrative Integration System
Date: November 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DomainEntry:
    """Entry for a single domain."""
    name: str
    pi: float
    domain_type: str
    status: str  # 'active', 'experimental', 'archived'
    r_squared: Optional[float] = None
    delta: Optional[float] = None
    n_samples: Optional[int] = None
    similar_domains: Optional[List[str]] = None
    patterns_count: Optional[int] = None
    last_updated: Optional[str] = None
    data_path: Optional[str] = None


class DomainRegistry:
    """
    Central registry of all domains.
    
    Features:
    - Track all analyzed domains
    - Store domain relationships
    - Query by characteristics
    - Export registry data
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent.parent.parent / 'domain_registry.json'
        
        self.registry_path = registry_path
        self.domains: Dict[str, DomainEntry] = {}
        
        # Load existing if available
        if self.registry_path.exists():
            self.load()
    
    def register_domain(
        self,
        name: str,
        pi: float,
        domain_type: str,
        **kwargs
    ):
        """
        Register or update a domain.
        
        Parameters
        ----------
        name : str
            Domain name
        pi : float
            Narrativity
        domain_type : str
            Domain type
        **kwargs
            Additional fields
        """
        entry = DomainEntry(
            name=name,
            pi=pi,
            domain_type=domain_type,
            status=kwargs.get('status', 'active'),
            r_squared=kwargs.get('r_squared'),
            delta=kwargs.get('delta'),
            n_samples=kwargs.get('n_samples'),
            similar_domains=kwargs.get('similar_domains'),
            patterns_count=kwargs.get('patterns_count'),
            last_updated=datetime.now().isoformat(),
            data_path=kwargs.get('data_path')
        )
        
        self.domains[name] = entry
    
    def get_domain(self, name: str) -> Optional[DomainEntry]:
        """Get domain entry."""
        return self.domains.get(name)
    
    def get_all_domains(self) -> List[DomainEntry]:
        """Get all domains."""
        return list(self.domains.values())
    
    def get_domains_by_type(self, domain_type: str) -> List[DomainEntry]:
        """Get domains of specific type."""
        return [d for d in self.domains.values() if d.domain_type == domain_type]
    
    def get_domains_by_pi_range(
        self,
        min_pi: float,
        max_pi: float
    ) -> List[DomainEntry]:
        """Get domains in π range."""
        return [
            d for d in self.domains.values()
            if min_pi <= d.pi <= max_pi
        ]
    
    def get_high_performers(self, min_r_squared: float = 0.7) -> List[DomainEntry]:
        """Get high-performing domains."""
        return [
            d for d in self.domains.values()
            if d.r_squared and d.r_squared >= min_r_squared
        ]
    
    def find_similar_domains(
        self,
        domain_name: str,
        n: int = 5
    ) -> List[tuple]:
        """
        Find domains similar to target.
        
        Returns
        -------
        list of (domain_name, similarity)
        """
        if domain_name not in self.domains:
            return []
        
        target = self.domains[domain_name]
        
        similarities = []
        
        for other_name, other in self.domains.items():
            if other_name == domain_name:
                continue
            
            # Calculate similarity
            sim = self._calculate_similarity(target, other)
            similarities.append((other_name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def _calculate_similarity(self, domain1: DomainEntry, domain2: DomainEntry) -> float:
        """Calculate similarity between domains."""
        sim_scores = []
        
        # Type similarity
        if domain1.domain_type == domain2.domain_type:
            sim_scores.append(1.0)
        else:
            sim_scores.append(0.0)
        
        # π similarity
        pi_diff = abs(domain1.pi - domain2.pi)
        pi_sim = 1.0 - min(1.0, pi_diff)
        sim_scores.append(pi_sim)
        
        # Performance similarity (if both have R²)
        if domain1.r_squared and domain2.r_squared:
            r2_diff = abs(domain1.r_squared - domain2.r_squared)
            r2_sim = 1.0 - min(1.0, r2_diff)
            sim_scores.append(r2_sim)
        
        return np.mean(sim_scores)
    
    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        domains_list = list(self.domains.values())
        
        if len(domains_list) == 0:
            return {}
        
        return {
            'total_domains': len(domains_list),
            'by_type': self._count_by_field('domain_type'),
            'by_status': self._count_by_field('status'),
            'avg_pi': np.mean([d.pi for d in domains_list]),
            'pi_range': (min(d.pi for d in domains_list), max(d.pi for d in domains_list)),
            'high_performers': len(self.get_high_performers()),
            'total_samples': sum(d.n_samples for d in domains_list if d.n_samples)
        }
    
    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count domains by field value."""
        counts = {}
        for domain in self.domains.values():
            value = getattr(domain, field)
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def export_summary(self) -> str:
        """Export human-readable summary."""
        stats = self.get_statistics()
        
        summary = "# Domain Registry Summary\n\n"
        summary += f"**Total Domains**: {stats['total_domains']}\n"
        summary += f"**Average π**: {stats['avg_pi']:.3f}\n"
        summary += f"**π Range**: [{stats['pi_range'][0]:.2f}, {stats['pi_range'][1]:.2f}]\n"
        summary += f"**High Performers** (R² ≥ 70%): {stats['high_performers']}\n\n"
        
        summary += "## By Type\n\n"
        for domain_type, count in stats['by_type'].items():
            summary += f"- {domain_type}: {count} domains\n"
        summary += "\n"
        
        summary += "## All Domains\n\n"
        for domain in sorted(self.domains.values(), key=lambda d: d.pi, reverse=True):
            r2_str = f"{domain.r_squared:.1%}" if domain.r_squared else "N/A"
            summary += f"- **{domain.name}** (π={domain.pi:.2f}, R²={r2_str})\n"
        
        return summary
    
    def save(self):
        """Save registry to disk."""
        data = {
            'domains': {
                name: asdict(entry)
                for name, entry in self.domains.items()
            },
            'saved_at': datetime.now().isoformat()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load registry from disk."""
        if not self.registry_path.exists():
            return
        
        with open(self.registry_path) as f:
            data = json.load(f)
        
        self.domains = {
            name: DomainEntry(**entry_data)
            for name, entry_data in data.get('domains', {}).items()
        }


# Global registry
_global_registry = None


def get_domain_registry() -> DomainRegistry:
    """Get global domain registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DomainRegistry()
    return _global_registry


def register_domain(name: str, **kwargs):
    """Register domain in global registry."""
    registry = get_domain_registry()
    registry.register_domain(name, **kwargs)
    registry.save()


def list_all_domains() -> List[str]:
    """List all registered domains."""
    registry = get_domain_registry()
    return list(registry.domains.keys())

