"""
Phylogenetic Tree Builder

Constructs phylogenetic trees from comparison "DNA" (feature vectors)
to show evolutionary relationships between narratives.

Implements the biological metaphor from theory:
- Each comparison is an organism
- Feature vectors are DNA
- Similar DNA = close evolutionary relationship
- Tree structure shows narrative lineages
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings


@dataclass
class PhylogeneticNode:
    """Node in phylogenetic tree."""
    node_id: str
    parent: Optional[str]
    children: List[str]
    dna: np.ndarray  # Feature vector
    domain: str
    weight: float  # Context weight (gravitational mass)
    distance_to_parent: float
    is_leaf: bool
    metadata: Dict[str, Any]
    
    def __repr__(self):
        type_str = "LEAF" if self.is_leaf else "ANCESTOR"
        return f"{type_str} | {self.domain} (weight={self.weight:.2f}, distance={self.distance_to_parent:.3f})"


@dataclass
class SpeciationEvent:
    """Detected speciation event (domain divergence)."""
    event_id: str
    ancestor_domain: str
    descendant_domains: List[str]
    divergence_time: float  # Genetic distance
    characteristics: Dict[str, Any]
    significance: float  # How major was the split?


class PhylogeneticTreeBuilder:
    """
    Builds phylogenetic trees from narrative DNA.
    
    Uses hierarchical clustering on feature vectors to construct
    trees showing evolutionary relationships between comparisons.
    """
    
    def __init__(self, method: str = 'ward'):
        """
        Parameters
        ----------
        method : str
            Linkage method: 'ward', 'average', 'complete', 'single'
        """
        self.method = method
        self.nodes: Dict[str, PhylogeneticNode] = {}
        self.speciation_events: List[SpeciationEvent] = []
        self.linkage_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
    
    def build_tree(
        self,
        comparison_dnas: List[np.ndarray],
        comparison_ids: List[str],
        domains: List[str],
        weights: List[float],
        metadata: Optional[List[Dict]] = None
    ) -> Dict[str, PhylogeneticNode]:
        """
        Build phylogenetic tree from comparison DNA.
        
        Parameters
        ----------
        comparison_dnas : List[np.ndarray]
            Feature vectors (DNA) for each comparison
        comparison_ids : List[str]
            Unique IDs for each comparison
        domains : List[str]
            Domain for each comparison
        weights : List[float]
            Context weight (gravitational mass) for each
        metadata : List[Dict], optional
            Additional metadata for each comparison
            
        Returns
        -------
        Dict mapping node IDs to PhylogeneticNode objects
        """
        if metadata is None:
            metadata = [{} for _ in comparison_ids]
        
        n = len(comparison_dnas)
        
        if n < 2:
            raise ValueError("Need at least 2 comparisons to build tree")
        
        # Compute distance matrix (genetic distance)
        dna_matrix = np.array(comparison_dnas)
        self.distance_matrix = squareform(pdist(dna_matrix, metric='euclidean'))
        
        # Perform hierarchical clustering
        self.linkage_matrix = linkage(dna_matrix, method=self.method)
        
        # Build tree structure
        self.nodes = {}
        
        # Create leaf nodes
        for i, (comp_id, dna, domain, weight, meta) in enumerate(
            zip(comparison_ids, comparison_dnas, domains, weights, metadata)
        ):
            node = PhylogeneticNode(
                node_id=comp_id,
                parent=None,  # Will be set when building internal nodes
                children=[],
                dna=dna,
                domain=domain,
                weight=weight,
                distance_to_parent=0.0,  # Will be set when building internal nodes
                is_leaf=True,
                metadata=meta
            )
            self.nodes[comp_id] = node
        
        # Create internal nodes from linkage matrix
        for i, (idx1, idx2, distance, count) in enumerate(self.linkage_matrix):
            # Internal node ID
            internal_id = f"ancestor_{i}"
            
            # Get children
            child1_id = comparison_ids[int(idx1)] if idx1 < n else f"ancestor_{int(idx1 - n)}"
            child2_id = comparison_ids[int(idx2)] if idx2 < n else f"ancestor_{int(idx2 - n)}"
            
            # Average DNA of children
            child1_dna = self.nodes[child1_id].dna
            child2_dna = self.nodes[child2_id].dna
            ancestor_dna = (child1_dna + child2_dna) / 2
            
            # Determine domain (most common among descendants)
            descendants = self._get_all_descendants([child1_id, child2_id])
            descendant_domains = [self.nodes[d].domain for d in descendants if d in self.nodes]
            if descendant_domains:
                ancestor_domain = max(set(descendant_domains), key=descendant_domains.count)
            else:
                ancestor_domain = "unknown"
            
            # Average weight
            ancestor_weight = (self.nodes[child1_id].weight + self.nodes[child2_id].weight) / 2
            
            # Create internal node
            internal_node = PhylogeneticNode(
                node_id=internal_id,
                parent=None,  # Will be set for non-root
                children=[child1_id, child2_id],
                dna=ancestor_dna,
                domain=ancestor_domain,
                weight=ancestor_weight,
                distance_to_parent=0.0,
                is_leaf=False,
                metadata={'merge_distance': distance, 'descendant_count': count}
            )
            
            self.nodes[internal_id] = internal_node
            
            # Set parent for children
            self.nodes[child1_id].parent = internal_id
            self.nodes[child1_id].distance_to_parent = distance / 2
            self.nodes[child2_id].parent = internal_id
            self.nodes[child2_id].distance_to_parent = distance / 2
        
        # Detect speciation events
        self.speciation_events = self._detect_speciation_events()
        
        return self.nodes
    
    def _get_all_descendants(self, node_ids: List[str]) -> List[str]:
        """Get all leaf descendants of given nodes."""
        descendants = []
        
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.is_leaf:
                    descendants.append(node_id)
                else:
                    descendants.extend(self._get_all_descendants(node.children))
        
        return descendants
    
    def _detect_speciation_events(self) -> List[SpeciationEvent]:
        """
        Detect speciation events (where domains diverge).
        
        A speciation event occurs when an internal node's children
        belong to different domains.
        """
        events = []
        event_counter = 0
        
        for node_id, node in self.nodes.items():
            if node.is_leaf or not node.children:
                continue
            
            # Get domains of children
            child_domains = [
                self.nodes[child_id].domain
                for child_id in node.children
                if child_id in self.nodes
            ]
            
            # Check if domains differ (speciation)
            unique_domains = set(child_domains)
            
            if len(unique_domains) > 1:
                # This is a speciation event
                event = SpeciationEvent(
                    event_id=f"speciation_{event_counter}",
                    ancestor_domain=node.domain,
                    descendant_domains=list(unique_domains),
                    divergence_time=node.metadata.get('merge_distance', 0.0),
                    characteristics={
                        'ancestor_node': node_id,
                        'child_nodes': node.children,
                        'descendant_count': node.metadata.get('descendant_count', 0)
                    },
                    significance=self._calculate_speciation_significance(node, unique_domains)
                )
                
                events.append(event)
                event_counter += 1
        
        return events
    
    def _calculate_speciation_significance(
        self,
        node: PhylogeneticNode,
        descendant_domains: set
    ) -> float:
        """
        Calculate how significant a speciation event is.
        
        More significant if:
        - Large divergence distance
        - Many descendants
        - High-weight comparisons
        """
        distance = node.metadata.get('merge_distance', 0.0)
        count = node.metadata.get('descendant_count', 1)
        weight = node.weight
        
        # Normalize components
        distance_score = min(1.0, distance / 10.0)
        count_score = min(1.0, count / 20.0)
        weight_score = (weight - 0.3) / 2.7  # Scale 0.3-3.0 to 0-1
        
        # Weighted average
        significance = 0.4 * distance_score + 0.3 * count_score + 0.3 * weight_score
        
        return significance
    
    def get_evolutionary_distance(self, node_id1: str, node_id2: str) -> float:
        """
        Calculate evolutionary distance between two nodes.
        
        Distance is sum of branch lengths from node1 to common ancestor
        to node2.
        """
        # Find common ancestor
        ancestors1 = self._get_ancestors(node_id1)
        ancestors2 = self._get_ancestors(node_id2)
        
        # Find most recent common ancestor
        common = set(ancestors1) & set(ancestors2)
        
        if not common:
            return float('inf')
        
        # Get closest common ancestor (appears latest in both paths)
        mrca = None
        min_total_depth = float('inf')
        
        for ancestor in common:
            depth1 = ancestors1.index(ancestor) if ancestor in ancestors1 else float('inf')
            depth2 = ancestors2.index(ancestor) if ancestor in ancestors2 else float('inf')
            total_depth = depth1 + depth2
            
            if total_depth < min_total_depth:
                min_total_depth = total_depth
                mrca = ancestor
        
        if mrca is None:
            return float('inf')
        
        # Sum distances
        distance1 = self._distance_to_ancestor(node_id1, mrca)
        distance2 = self._distance_to_ancestor(node_id2, mrca)
        
        return distance1 + distance2
    
    def _get_ancestors(self, node_id: str) -> List[str]:
        """Get list of ancestors from node to root."""
        ancestors = []
        current = node_id
        
        while current is not None and current in self.nodes:
            ancestors.append(current)
            current = self.nodes[current].parent
        
        return ancestors
    
    def _distance_to_ancestor(self, node_id: str, ancestor_id: str) -> float:
        """Calculate total distance from node to ancestor."""
        distance = 0.0
        current = node_id
        
        while current is not None and current in self.nodes:
            if current == ancestor_id:
                return distance
            
            distance += self.nodes[current].distance_to_parent
            current = self.nodes[current].parent
        
        return float('inf')
    
    def get_species_catalog(self, distance_threshold: float = 2.0) -> Dict[str, List[str]]:
        """
        Classify comparisons into species based on genetic distance.
        
        Parameters
        ----------
        distance_threshold : float
            Max distance within a species
            
        Returns
        -------
        Dict mapping species names to member IDs
        """
        if self.linkage_matrix is None:
            raise ValueError("Must build tree first")
        
        # Get flat clusters
        leaf_nodes = [n for n in self.nodes.values() if n.is_leaf]
        n_leaves = len(leaf_nodes)
        
        # Cluster assignments
        clusters = fcluster(self.linkage_matrix, distance_threshold, criterion='distance')
        
        # Group by cluster
        species_catalog = {}
        for i, node in enumerate(leaf_nodes):
            cluster_id = clusters[i]
            species_name = f"{node.domain}_species_{cluster_id}"
            
            if species_name not in species_catalog:
                species_catalog[species_name] = []
            
            species_catalog[species_name].append(node.node_id)
        
        return species_catalog
    
    def generate_newick_tree(self) -> str:
        """
        Generate tree in Newick format for visualization.
        
        Returns
        -------
        str : Newick format tree
        """
        # Find root (node with no parent)
        root = None
        for node in self.nodes.values():
            if node.parent is None and not node.is_leaf:
                root = node
                break
        
        if root is None:
            return ""
        
        return self._newick_recursive(root.node_id) + ";"
    
    def _newick_recursive(self, node_id: str) -> str:
        """Recursively build Newick string."""
        node = self.nodes[node_id]
        
        if node.is_leaf:
            return f"{node_id}:{node.distance_to_parent:.3f}"
        
        # Internal node
        children_newick = [
            self._newick_recursive(child_id)
            for child_id in node.children
        ]
        
        children_str = ",".join(children_newick)
        
        return f"({children_str}){node_id}:{node.distance_to_parent:.3f}"
    
    def generate_report(self) -> str:
        """Generate phylogenetic analysis report."""
        report = []
        report.append("=" * 80)
        report.append("PHYLOGENETIC TREE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        if not self.nodes:
            report.append("No tree built yet.")
            return "\n".join(report)
        
        # Summary statistics
        leaf_nodes = [n for n in self.nodes.values() if n.is_leaf]
        internal_nodes = [n for n in self.nodes.values() if not n.is_leaf]
        
        report.append("TREE STRUCTURE:")
        report.append(f"  Total nodes: {len(self.nodes)}")
        report.append(f"  Leaf nodes (comparisons): {len(leaf_nodes)}")
        report.append(f"  Internal nodes (ancestors): {len(internal_nodes)}")
        report.append("")
        
        # Domain distribution
        domains = {}
        for node in leaf_nodes:
            domains[node.domain] = domains.get(node.domain, 0) + 1
        
        report.append("DOMAIN DISTRIBUTION:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {domain}: {count} comparisons")
        report.append("")
        
        # Speciation events
        report.append("SPECIATION EVENTS:")
        report.append(f"  Total events detected: {len(self.speciation_events)}")
        report.append("")
        
        for event in sorted(self.speciation_events, key=lambda e: e.significance, reverse=True)[:10]:
            report.append(f"  {event.event_id}:")
            report.append(f"    Ancestor: {event.ancestor_domain}")
            report.append(f"    Descendants: {', '.join(event.descendant_domains)}")
            report.append(f"    Divergence: {event.divergence_time:.3f}")
            report.append(f"    Significance: {event.significance:.3f}")
            report.append("")
        
        # Species catalog
        species = self.get_species_catalog()
        report.append("SPECIES CATALOG:")
        report.append(f"  Total species: {len(species)}")
        
        for species_name, members in sorted(species.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            report.append(f"  {species_name}: {len(members)} members")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

