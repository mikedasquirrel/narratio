"""
Cross-Domain Comparison Matrix

Generate comparison matrix showing clustering, transformer effectiveness,
and patterns across domains.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.domain_config import DomainConfig


class DomainComparisonMatrix:
    """
    Generate cross-domain comparison matrix.
    
    Analyzes:
    - Domain clustering (which domains are similar?)
    - Transformer effectiveness (which transformers work best in each domain?)
    - Feature importance rankings
    - Validation metrics dashboard
    """
    
    def __init__(self, project_root: Path = None):
        """Initialize comparison matrix"""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent
        
        self.project_root = Path(project_root)
        self.domains_dir = self.project_root / 'narrative_optimization' / 'domains'
    
    def load_all_domains(self) -> Dict[str, DomainConfig]:
        """Load all domain configurations"""
        domains = {}
        
        for domain_dir in self.domains_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            config_path = domain_dir / 'config.yaml'
            if config_path.exists():
                try:
                    config = DomainConfig.from_yaml(config_path)
                    domains[config.domain] = config
                except Exception as e:
                    print(f"Warning: Could not load {domain_dir.name}: {e}")
        
        return domains
    
    def compute_domain_clustering(self, domains: Dict[str, DomainConfig]) -> Dict[str, Any]:
        """
        Compute domain clustering based on п and domain type.
        
        Parameters
        ----------
        domains : dict
            {domain_name: DomainConfig}
            
        Returns
        -------
        clustering : dict
            Clustering results with clusters and dendrogram data
        """
        domain_names = list(domains.keys())
        
        # Feature matrix: [п, type_encoded, ...]
        features = []
        for domain_name in domain_names:
            config = domains[domain_name]
            # Encode domain type as numeric
            type_encoding = {
                'sports': 0.0,
                'sports_individual': 0.2,
                'sports_team': 0.4,
                'entertainment': 0.6,
                'nominative': 0.8,
                'business': 1.0,
                'medical': 1.2,
                'hybrid': 0.5
            }
            type_val = type_encoding.get(config.type.value, 0.5)
            
            features.append([config.pi, type_val])
        
        features = np.array(features)
        
        # Compute distance matrix
        distances = pdist(features, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Form clusters (3 clusters)
        clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
        
        # Group domains by cluster
        cluster_groups = {}
        for i, domain_name in enumerate(domain_names):
            cluster_id = clusters[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append({
                'domain': domain_name,
                'pi': domains[domain_name].pi,
                'type': domains[domain_name].type.value
            })
        
        return {
            'clusters': cluster_groups,
            'linkage_matrix': linkage_matrix.tolist(),
            'distance_matrix': distance_matrix.tolist(),
            'domain_names': domain_names
        }
    
    def analyze_transformer_effectiveness(
        self,
        domains: Dict[str, DomainConfig]
    ) -> Dict[str, Any]:
        """
        Analyze transformer effectiveness across domains.
        
        Parameters
        ----------
        domains : dict
            Domain configurations
            
        Returns
        -------
        effectiveness : dict
            {transformer_name: {best_domains, worst_domains, avg_correlation}}
        """
        # This would require actual analysis results
        # For now, return structure
        
        effectiveness = {}
        
        # Common transformers
        transformers = [
            'nominative', 'self_perception', 'narrative_potential',
            'linguistic', 'ensemble', 'relational', 'conflict',
            'suspense', 'authenticity', 'expertise', 'statistical'
        ]
        
        for transformer in transformers:
            effectiveness[transformer] = {
                'best_domains': [],
                'worst_domains': [],
                'avg_correlation': 0.0,
                'domain_scores': {}
            }
        
        return effectiveness
    
    def generate_comparison_report(
        self,
        clustering: Dict[str, Any],
        effectiveness: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable comparison report.
        
        Parameters
        ----------
        clustering : dict
            Clustering results
        effectiveness : dict
            Transformer effectiveness
            
        Returns
        -------
        report : str
            Formatted report
        """
        lines = [
            "=" * 80,
            "CROSS-DOMAIN COMPARISON MATRIX",
            "=" * 80,
            "\nDomain Clustering:",
            "-" * 80
        ]
        
        for cluster_id, domains in sorted(clustering['clusters'].items()):
            lines.append(f"\nCluster {cluster_id}:")
            for domain_info in domains:
                lines.append(
                    f"  • {domain_info['domain']:20s} "
                    f"(п={domain_info['pi']:.3f}, type={domain_info['type']})"
                )
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def generate_comparison_matrix(self) -> Dict[str, Any]:
        """
        Generate complete comparison matrix for all domains.
        
        Returns
        -------
        matrix : dict
            Complete comparison analysis
        """
        domains = self.load_all_domains()
        
        if not domains:
            return {'error': 'No domains found'}
        
        # Compute clustering
        clustering = self.compute_domain_clustering(domains)
        
        # Analyze transformer effectiveness
        effectiveness = self.analyze_transformer_effectiveness(domains)
        
        # Generate report
        report = self.generate_comparison_report(clustering, effectiveness)
        
        return {
            'domains': {name: config.to_dict() for name, config in domains.items()},
            'clustering': clustering,
            'transformer_effectiveness': effectiveness,
            'report': report,
            'total_domains': len(domains)
        }


if __name__ == '__main__':
    comparator = DomainComparisonMatrix()
    matrix = comparator.generate_comparison_matrix()
    print(matrix['report'])

