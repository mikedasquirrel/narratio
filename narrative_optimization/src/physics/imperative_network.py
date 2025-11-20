"""
Imperative Gravity Network Builder

Builds complete cross-domain gravitational network from all instances.

Creates:
1. Full force matrix between all instance pairs across domains
2. Domain-level aggregated forces
3. Gravitational clusters
4. Network graph data structures
5. Exportable network representations

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.story_instance import StoryInstance
from physics.imperative_gravity import ImperativeGravityCalculator
from config.domain_config import DomainConfig


class ImperativeGravityNetwork:
    """
    Complete cross-domain gravitational network.
    
    Represents the full structure of imperative forces connecting
    all story instances across all domains.
    """
    
    def __init__(
        self,
        domain_configs: Dict[str, DomainConfig],
        instances_by_domain: Optional[Dict[str, List[StoryInstance]]] = None
    ):
        """
        Initialize network builder.
        
        Parameters
        ----------
        domain_configs : dict
            {domain_name: DomainConfig}
        instances_by_domain : dict, optional
            {domain_name: [instances]}
        """
        self.domain_configs = domain_configs
        self.instances_by_domain = instances_by_domain or {}
        
        self.calculator = ImperativeGravityCalculator(domain_configs)
        
        # Network structures
        self.domain_forces: Dict[Tuple[str, str], float] = {}  # Domain-to-domain
        self.instance_forces: Dict[Tuple[str, str], float] = {}  # Instance-to-instance
        self.domain_clusters: List[Set[str]] = []
        
        # Network metadata
        self.network_built = False
        self.build_timestamp = None
    
    def build_complete_network(
        self,
        force_threshold: float = 0.5,
        verbose: bool = True
    ):
        """
        Build complete cross-domain gravitational network.
        
        Parameters
        ----------
        force_threshold : float
            Minimum force magnitude to store (efficiency)
        verbose : bool
            Print progress
        """
        if verbose:
            print("\n" + "="*70)
            print("BUILDING COMPLETE IMPERATIVE GRAVITY NETWORK")
            print("="*70 + "\n")
        
        # Step 1: Domain-level forces
        if verbose:
            print("Step 1/4: Calculating domain-level forces...")
        
        self._build_domain_forces(verbose)
        
        # Step 2: Instance-level forces (if instances provided)
        if self.instances_by_domain:
            if verbose:
                print("\nStep 2/4: Calculating instance-level forces...")
            
            self._build_instance_forces(force_threshold, verbose)
        else:
            if verbose:
                print("\nStep 2/4: Skipping instance-level forces (no instances provided)")
        
        # Step 3: Identify clusters
        if verbose:
            print("\nStep 3/4: Identifying gravitational clusters...")
        
        self._identify_clusters(verbose)
        
        # Step 4: Calculate network statistics
        if verbose:
            print("\nStep 4/4: Calculating network statistics...")
        
        self.network_stats = self._calculate_network_stats()
        
        self.network_built = True
        self.build_timestamp = datetime.now()
        
        if verbose:
            print("\n" + "="*70)
            print("NETWORK BUILD COMPLETE")
            print("="*70)
            self._print_network_summary()
    
    def _build_domain_forces(self, verbose: bool):
        """Calculate forces between all domain pairs."""
        domains = list(self.domain_configs.keys())
        n_domains = len(domains)
        total_pairs = n_domains * (n_domains - 1) // 2
        
        calculated = 0
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains):
                if i < j:  # Avoid duplicates
                    # Calculate structural similarity
                    similarity = self.calculator._calculate_domain_similarity(
                        domain1, domain2
                    )
                    
                    # Calculate distance
                    distance = self.calculator._calculate_domain_distance(
                        domain1, domain2
                    )
                    
                    # Calculate force (using unit mass)
                    force = similarity / (distance ** 2 + 0.01)
                    
                    # Store
                    self.domain_forces[(domain1, domain2)] = force
                    self.domain_forces[(domain2, domain1)] = force  # Symmetric
                    
                    calculated += 1
                    
                    if verbose and calculated % 50 == 0:
                        print(f"  Calculated {calculated}/{total_pairs} domain pairs...", end="\r")
        
        if verbose:
            print(f"  ✓ Calculated {calculated} domain pairs ({n_domains} domains)       ")
    
    def _build_instance_forces(self, force_threshold: float, verbose: bool):
        """Calculate forces between instances across domains."""
        total_instances = sum(len(instances) for instances in self.instances_by_domain.values())
        
        if verbose:
            print(f"  Processing {total_instances} instances...")
        
        calculated = 0
        stored = 0
        
        # For each instance, calculate forces to instances in other domains
        for source_domain, source_instances in self.instances_by_domain.items():
            for source_instance in source_instances:
                # Get top imperative neighbors (limit for efficiency)
                all_domains = [d for d in self.domain_configs.keys() if d != source_domain]
                
                neighbors = self.calculator.find_gravitational_neighbors(
                    instance=source_instance,
                    all_domains=all_domains,
                    n_neighbors=5,  # Top 5 only
                    exclude_same_domain=True
                )
                
                # Store forces above threshold
                for target_domain, force in neighbors:
                    if force >= force_threshold:
                        key = (source_instance.instance_id, target_domain)
                        self.instance_forces[key] = force
                        stored += 1
                
                calculated += 1
                
                if verbose and calculated % 10 == 0:
                    print(f"  Processed {calculated}/{total_instances} instances (stored {stored} forces)...", end="\r")
        
        if verbose:
            print(f"  ✓ Processed {calculated} instances, stored {stored} significant forces       ")
    
    def _identify_clusters(self, verbose: bool):
        """Identify gravitational clusters of domains."""
        domains = list(self.domain_configs.keys())
        
        # Get clusters using calculator
        self.domain_clusters = self.calculator.get_domain_clusters(
            domains=domains,
            similarity_threshold=0.7
        )
        
        if verbose:
            print(f"  ✓ Identified {len(self.domain_clusters)} gravitational clusters")
            for i, cluster in enumerate(self.domain_clusters):
                if len(cluster) > 1:
                    print(f"    Cluster {i+1}: {', '.join(cluster)}")
    
    def _calculate_network_stats(self) -> Dict:
        """Calculate network statistics."""
        stats = {
            'n_domains': len(self.domain_configs),
            'n_domain_connections': len(self.domain_forces) // 2,  # Symmetric
            'n_clusters': len(self.domain_clusters),
            'avg_cluster_size': np.mean([len(c) for c in self.domain_clusters]),
            'build_timestamp': self.build_timestamp.isoformat() if self.build_timestamp else None
        }
        
        if self.instances_by_domain:
            stats['n_instances'] = sum(len(instances) for instances in self.instances_by_domain.values())
            stats['n_instance_forces'] = len(self.instance_forces)
        
        # Force statistics
        if self.domain_forces:
            forces = [f for f in self.domain_forces.values()]
            stats['force_mean'] = float(np.mean(forces))
            stats['force_std'] = float(np.std(forces))
            stats['force_max'] = float(np.max(forces))
            stats['force_min'] = float(np.min(forces))
        
        return stats
    
    def _print_network_summary(self):
        """Print network summary."""
        print("\nNetwork Summary:")
        print(f"  Domains: {self.network_stats['n_domains']}")
        print(f"  Domain connections: {self.network_stats['n_domain_connections']}")
        print(f"  Gravitational clusters: {self.network_stats['n_clusters']}")
        
        if 'n_instances' in self.network_stats:
            print(f"  Instances: {self.network_stats['n_instances']}")
            print(f"  Instance forces: {self.network_stats['n_instance_forces']}")
        
        if 'force_mean' in self.network_stats:
            print(f"\nForce Statistics:")
            print(f"  Mean: {self.network_stats['force_mean']:.3f}")
            print(f"  Std: {self.network_stats['force_std']:.3f}")
            print(f"  Range: [{self.network_stats['force_min']:.3f}, {self.network_stats['force_max']:.3f}]")
    
    def get_strongest_connections(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get N strongest domain-domain connections.
        
        Parameters
        ----------
        n : int
            Number of connections to return
        
        Returns
        -------
        list of tuple
            [(domain1, domain2, force), ...]
        """
        # Get unique pairs (not duplicates from symmetry)
        unique_pairs = {}
        for (d1, d2), force in self.domain_forces.items():
            key = tuple(sorted([d1, d2]))
            if key not in unique_pairs:
                unique_pairs[key] = force
        
        # Sort by force
        sorted_pairs = sorted(unique_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        result = [(pair[0], pair[1], force) for pair, force in sorted_pairs[:n]]
        
        return result
    
    def get_domain_neighbors(
        self,
        domain: str,
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top N gravitational neighbors for a domain.
        
        Parameters
        ----------
        domain : str
            Source domain
        n : int
            Number of neighbors
        
        Returns
        -------
        list of tuple
            [(neighbor_domain, force), ...]
        """
        # Get all forces from this domain
        domain_forces = []
        
        for (d1, d2), force in self.domain_forces.items():
            if d1 == domain and d2 != domain:
                domain_forces.append((d2, force))
            elif d2 == domain and d1 != domain:
                domain_forces.append((d1, force))
        
        # Sort by force
        domain_forces.sort(key=lambda x: x[1], reverse=True)
        
        return domain_forces[:n]
    
    def export_network(self, output_path: str):
        """
        Export complete network to JSON.
        
        Parameters
        ----------
        output_path : str
            Path to save network data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        network_data = {
            'metadata': {
                'version': '2.0',
                'build_timestamp': self.build_timestamp.isoformat() if self.build_timestamp else None,
                'n_domains': len(self.domain_configs),
                'n_clusters': len(self.domain_clusters)
            },
            'domains': list(self.domain_configs.keys()),
            'domain_forces': [
                {
                    'source': d1,
                    'target': d2,
                    'force': float(force)
                }
                for (d1, d2), force in self.domain_forces.items()
                if d1 < d2  # Only unique pairs
            ],
            'clusters': [
                {
                    'cluster_id': i,
                    'domains': list(cluster),
                    'size': len(cluster)
                }
                for i, cluster in enumerate(self.domain_clusters)
            ],
            'statistics': self.network_stats
        }
        
        # Add instance forces if available
        if self.instance_forces:
            network_data['instance_forces'] = [
                {
                    'source_instance': inst_id,
                    'target_domain': domain,
                    'force': float(force)
                }
                for (inst_id, domain), force in self.instance_forces.items()
            ]
        
        with open(output_path, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        print(f"✓ Exported network to {output_path}")
    
    def generate_network_report(self) -> str:
        """
        Generate comprehensive network analysis report.
        
        Returns
        -------
        str
            Formatted report
        """
        if not self.network_built:
            return "Network not built yet. Call build_complete_network() first."
        
        lines = []
        lines.append("=" * 70)
        lines.append("IMPERATIVE GRAVITY NETWORK ANALYSIS")
        lines.append("=" * 70)
        lines.append("")
        
        # Overview
        lines.append(f"Built: {self.build_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Domains: {self.network_stats['n_domains']}")
        lines.append(f"Connections: {self.network_stats['n_domain_connections']}")
        lines.append(f"Clusters: {self.network_stats['n_clusters']}")
        
        if 'n_instances' in self.network_stats:
            lines.append(f"Instances: {self.network_stats['n_instances']}")
            lines.append(f"Instance forces: {self.network_stats['n_instance_forces']}")
        
        # Strongest connections
        lines.append("\n" + "-" * 70)
        lines.append("TOP 10 STRONGEST CONNECTIONS")
        lines.append("-" * 70)
        
        strongest = self.get_strongest_connections(n=10)
        for i, (d1, d2, force) in enumerate(strongest, 1):
            lines.append(f"{i:2d}. {d1:15s} ←→ {d2:15s}  Force: {force:6.2f}")
        
        # Clusters
        lines.append("\n" + "-" * 70)
        lines.append("GRAVITATIONAL CLUSTERS")
        lines.append("-" * 70)
        
        for i, cluster in enumerate(self.domain_clusters, 1):
            if len(cluster) > 1:
                lines.append(f"\nCluster {i} ({len(cluster)} domains):")
                lines.append(f"  {', '.join(sorted(cluster))}")
                
                # Calculate intra-cluster connectivity
                intra_forces = []
                for d1 in cluster:
                    for d2 in cluster:
                        if d1 < d2:
                            force = self.domain_forces.get((d1, d2), 0)
                            if force > 0:
                                intra_forces.append(force)
                
                if intra_forces:
                    lines.append(f"  Avg intra-cluster force: {np.mean(intra_forces):.2f}")
        
        # Domain-specific analysis
        lines.append("\n" + "-" * 70)
        lines.append("DOMAIN-SPECIFIC NEIGHBORS")
        lines.append("-" * 70)
        
        # Sample a few interesting domains
        sample_domains = ['golf', 'supreme_court', 'oscars', 'nba', 'boxing']
        available_samples = [d for d in sample_domains if d in self.domain_configs]
        
        for domain in available_samples[:5]:
            neighbors = self.get_domain_neighbors(domain, n=3)
            lines.append(f"\n{domain.upper()}:")
            for i, (neighbor, force) in enumerate(neighbors, 1):
                lines.append(f"  {i}. {neighbor:15s}  Force: {force:6.2f}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def export_for_visualization(self, output_dir: str):
        """
        Export network data optimized for visualization tools.
        
        Parameters
        ----------
        output_dir : str
            Directory to save visualization data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Nodes file (domains)
        nodes = []
        for domain in self.domain_configs.keys():
            config = self.domain_configs[domain]
            
            # Count connections
            n_connections = sum(1 for (d1, d2) in self.domain_forces.keys() 
                              if (d1 == domain or d2 == domain) and d1 != d2)
            
            nodes.append({
                'id': domain,
                'pi': config.get_pi(),
                'theta_min': config.get_theta_range()[0],
                'theta_max': config.get_theta_range()[1],
                'lambda_min': config.get_lambda_range()[0],
                'lambda_max': config.get_lambda_range()[1],
                'prestige': config.is_prestige_domain(),
                'n_connections': n_connections // 2,  # Symmetric
                'blind_narratio': config.get_blind_narratio()
            })
        
        with open(output_dir / 'nodes.json', 'w') as f:
            json.dump(nodes, f, indent=2)
        
        # 2. Edges file (forces)
        edges = []
        seen = set()
        
        for (d1, d2), force in self.domain_forces.items():
            key = tuple(sorted([d1, d2]))
            if key not in seen:
                edges.append({
                    'source': d1,
                    'target': d2,
                    'force': float(force),
                    'similarity': float(self.calculator._calculate_domain_similarity(d1, d2))
                })
                seen.add(key)
        
        with open(output_dir / 'edges.json', 'w') as f:
            json.dump(edges, f, indent=2)
        
        # 3. Clusters file
        clusters_data = [
            {
                'id': i,
                'domains': list(cluster),
                'size': len(cluster)
            }
            for i, cluster in enumerate(self.domain_clusters)
        ]
        
        with open(output_dir / 'clusters.json', 'w') as f:
            json.dump(clusters_data, f, indent=2)
        
        print(f"✓ Exported visualization data to {output_dir}")
        print(f"  nodes.json: {len(nodes)} domains")
        print(f"  edges.json: {len(edges)} connections")
        print(f"  clusters.json: {len(clusters_data)} clusters")
    
    def get_transfer_learning_recommendations(
        self,
        source_domain: str,
        target_instance: StoryInstance,
        n_recommendations: int = 3
    ) -> List[Dict]:
        """
        Get recommendations for cross-domain transfer learning.
        
        Parameters
        ----------
        source_domain : str
            Domain of the instance
        target_instance : StoryInstance
            Instance to improve predictions for
        n_recommendations : int
            Number of recommendations
        
        Returns
        -------
        list of dict
            Recommendations with explanations
        """
        # Get imperative neighbors
        all_domains = [d for d in self.domain_configs.keys() if d != source_domain]
        neighbors = self.calculator.find_gravitational_neighbors(
            target_instance,
            all_domains,
            n_neighbors=n_recommendations
        )
        
        recommendations = []
        
        for neighbor_domain, force in neighbors:
            explanation = self.calculator.explain_gravitational_pull(
                target_instance,
                neighbor_domain
            )
            
            recommendations.append({
                'domain': neighbor_domain,
                'force': float(force),
                'similarity': explanation['structural_similarity'],
                'learning_potential': explanation['learning_potential'],
                'interpretation': explanation['interpretation'],
                'suggested_patterns': self._suggest_transferable_patterns(
                    source_domain,
                    neighbor_domain
                )
            })
        
        return recommendations
    
    def _suggest_transferable_patterns(
        self,
        source_domain: str,
        target_domain: str
    ) -> List[str]:
        """Suggest which patterns might transfer between domains."""
        suggestions = []
        
        # Get configs
        source_config = self.domain_configs.get(source_domain)
        target_config = self.domain_configs.get(target_domain)
        
        if not source_config or not target_config:
            return suggestions
        
        # Compare π
        pi_diff = abs(source_config.get_pi() - target_config.get_pi())
        if pi_diff < 0.15:
            suggestions.append("Narrative structure patterns (similar π)")
        
        # Compare θ
        source_theta = source_config.get_theta_range()
        target_theta = target_config.get_theta_range()
        theta_overlap = self.calculator._calculate_range_overlap(source_theta, target_theta)
        if theta_overlap > 0.6:
            suggestions.append("Awareness dynamics patterns")
        
        # Compare λ
        source_lambda = source_config.get_lambda_range()
        target_lambda = target_config.get_lambda_range()
        lambda_overlap = self.calculator._calculate_range_overlap(source_lambda, target_lambda)
        if lambda_overlap > 0.6:
            suggestions.append("Constraint management patterns")
        
        # Prestige
        if source_config.is_prestige_domain() == target_config.is_prestige_domain():
            if source_config.is_prestige_domain():
                suggestions.append("Prestige equation dynamics")
            else:
                suggestions.append("Standard equation patterns")
        
        return suggestions


def build_network_from_repository(
    domain_configs: Dict[str, DomainConfig],
    repository,
    output_dir: str = 'results/imperative_network'
) -> ImperativeGravityNetwork:
    """
    Convenience function to build network from repository.
    
    Parameters
    ----------
    domain_configs : dict
        All domain configurations
    repository : InstanceRepository
        Instance repository
    output_dir : str
        Output directory for exports
    
    Returns
    -------
    ImperativeGravityNetwork
        Built network
    """
    # Get instances by domain
    instances_by_domain = {}
    for domain in domain_configs.keys():
        instances = repository.get_instances_by_domain(domain)
        if instances:
            instances_by_domain[domain] = instances
    
    # Build network
    network = ImperativeGravityNetwork(domain_configs, instances_by_domain)
    network.build_complete_network(verbose=True)
    
    # Export
    network.export_network(f"{output_dir}/network.json")
    network.export_for_visualization(output_dir)
    
    # Generate report
    report = network.generate_network_report()
    with open(f"{output_dir}/network_report.txt", 'w') as f:
        f.write(report)
    
    print(f"\n✓ Network analysis complete. Results in {output_dir}")
    
    return network


if __name__ == '__main__':
    print("Imperative Gravity Network Builder")
    print("Use build_network_from_repository() to build from existing data")

