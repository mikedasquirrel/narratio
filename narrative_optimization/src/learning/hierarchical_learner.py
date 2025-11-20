"""
Hierarchical Archetype Learning

Learns multi-level archetype hierarchies:
- Universal archetypes (level 0)
- Domain archetypes (level 1)
- Sub-archetypes (level 2+)

Weights propagate through hierarchy: children inherit from parents.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import networkx as nx

from ..config.advanced_archetype_discovery import HierarchicalArchetypeDiscovery


@dataclass
class HierarchyNode:
    """Node in archetype hierarchy."""
    name: str
    level: int
    patterns: List[str]
    parent: Optional[str]
    children: List[str]
    weight: float
    inherited_weight: float  # Weight inherited from parent


class HierarchicalArchetypeLearner:
    """
    Learns hierarchical archetype structures.
    
    Hierarchy levels:
    0. Universal (e.g., "comeback")
    1. Domain (e.g., "golf_mental_game")
    2. Sub-domain (e.g., "golf_major_pressure")
    3. Micro (e.g., "golf_sunday_final_hole")
    
    Weights propagate: child weight = parent_weight * local_weight
    
    Parameters
    ----------
    max_depth : int
        Maximum hierarchy depth
    min_samples_per_level : int
        Minimum samples to create child level
    weight_decay : float
        How much weight decays at each level (0-1)
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_per_level: int = 10,
        weight_decay: float = 0.9
    ):
        self.max_depth = max_depth
        self.min_samples_per_level = min_samples_per_level
        self.weight_decay = weight_decay
        
        # Hierarchy storage
        self.hierarchy = nx.DiGraph()  # Directed graph for hierarchy
        self.nodes = {}  # node_name -> HierarchyNode
        
        # Pattern discoverer
        self.discoverer = HierarchicalArchetypeDiscovery()
        
    def build_hierarchy(
        self,
        root_name: str,
        root_patterns: List[str],
        texts: List[str],
        outcomes: np.ndarray,
        domain: Optional[str] = None
    ) -> Dict[str, HierarchyNode]:
        """
        Build complete hierarchy from root patterns.
        
        Parameters
        ----------
        root_name : str
            Root node name
        root_patterns : list of str
            Root patterns
        texts : list of str
            All texts
        outcomes : ndarray
            Outcomes
        domain : str, optional
            Domain context
        
        Returns
        -------
        dict
            Hierarchy nodes
        """
        print(f"  Building hierarchy from root: {root_name}")
        
        # Create root node
        root = HierarchyNode(
            name=root_name,
            level=0,
            patterns=root_patterns,
            parent=None,
            children=[],
            weight=1.0,
            inherited_weight=1.0
        )
        
        self.nodes[root_name] = root
        self.hierarchy.add_node(root_name, node=root)
        
        # Recursively build levels
        self._build_level(
            parent=root,
            texts=texts,
            outcomes=outcomes,
            current_depth=1
        )
        
        print(f"  ✓ Hierarchy built: {len(self.nodes)} total nodes")
        
        return self.nodes
    
    def _build_level(
        self,
        parent: HierarchyNode,
        texts: List[str],
        outcomes: np.ndarray,
        current_depth: int
    ):
        """Recursively build hierarchy levels."""
        if current_depth > self.max_depth:
            return
        
        # Filter texts matching parent patterns
        matching_texts = []
        matching_outcomes = []
        
        for i, text in enumerate(texts):
            if any(pattern.lower() in text.lower() for pattern in parent.patterns):
                matching_texts.append(text)
                matching_outcomes.append(outcomes[i])
        
        if len(matching_texts) < self.min_samples_per_level:
            return  # Not enough data for children
        
        # Discover sub-patterns
        hierarchy_dict = self.discoverer.build_hierarchy(
            parent_patterns=parent.patterns,
            texts=matching_texts,
            max_depth=1,  # One level at a time
            min_samples_per_level=self.min_samples_per_level
        )
        
        if parent.name not in hierarchy_dict:
            return
        
        children_data = hierarchy_dict[parent.name].get('children', {})
        
        for child_name, child_data in children_data.items():
            # Create child node
            child = HierarchyNode(
                name=child_name,
                level=current_depth,
                patterns=child_data['patterns'],
                parent=parent.name,
                children=[],
                weight=child_data.get('sample_count', 0) / len(matching_texts),
                inherited_weight=parent.inherited_weight * self.weight_decay
            )
            
            self.nodes[child_name] = child
            parent.children.append(child_name)
            
            # Add to graph
            self.hierarchy.add_node(child_name, node=child)
            self.hierarchy.add_edge(parent.name, child_name)
            
            # Recursively build next level
            self._build_level(
                parent=child,
                texts=matching_texts,
                outcomes=np.array(matching_outcomes),
                current_depth=current_depth + 1
            )
    
    def get_effective_weight(self, node_name: str) -> float:
        """
        Get effective weight for a node (including inheritance).
        
        Parameters
        ----------
        node_name : str
            Node name
        
        Returns
        -------
        float
            Effective weight (parent_weight * local_weight)
        """
        if node_name not in self.nodes:
            return 0.0
        
        node = self.nodes[node_name]
        return node.weight * node.inherited_weight
    
    def get_path_to_root(self, node_name: str) -> List[str]:
        """Get path from node to root."""
        path = []
        current = node_name
        
        while current:
            path.append(current)
            node = self.nodes.get(current)
            if node is None:
                break
            current = node.parent
        
        return list(reversed(path))
    
    def get_all_descendants(self, node_name: str) -> List[str]:
        """Get all descendants of a node."""
        if node_name not in self.hierarchy:
            return []
        
        return list(nx.descendants(self.hierarchy, node_name))
    
    def propagate_weights_top_down(self):
        """
        Propagate weights from root to leaves.
        
        Each child's inherited weight = parent's inherited weight * decay
        """
        # Topological sort (root to leaves)
        for node_name in nx.topological_sort(self.hierarchy):
            node = self.nodes[node_name]
            
            if node.parent:
                parent = self.nodes[node.parent]
                node.inherited_weight = parent.inherited_weight * self.weight_decay * node.weight
    
    def propagate_updates_bottom_up(self, updates: Dict[str, float]):
        """
        Propagate updates from leaves to root.
        
        If a child pattern improves, parent should reflect that.
        
        Parameters
        ----------
        updates : dict
            Node name -> performance update
        """
        # Reverse topological sort (leaves to root)
        for node_name in reversed(list(nx.topological_sort(self.hierarchy))):
            if node_name in updates:
                node = self.nodes[node_name]
                
                # Update node weight
                node.weight = min(1.0, node.weight + updates[node_name])
                
                # Propagate to parent
                if node.parent and node.parent in self.nodes:
                    parent = self.nodes[node.parent]
                    # Parent weight = average of children weights
                    child_weights = [
                        self.nodes[child].weight
                        for child in parent.children
                        if child in self.nodes
                    ]
                    if child_weights:
                        parent.weight = np.mean(child_weights)
    
    def get_hierarchy_summary(self) -> str:
        """Get text summary of hierarchy."""
        summary = "Archetype Hierarchy:\n\n"
        
        # Find roots (nodes with no parents)
        roots = [name for name, node in self.nodes.items() if node.parent is None]
        
        for root in roots:
            summary += self._format_subtree(root, indent=0)
        
        return summary
    
    def _format_subtree(self, node_name: str, indent: int) -> str:
        """Format subtree recursively."""
        node = self.nodes[node_name]
        
        indent_str = "  " * indent
        effective_weight = self.get_effective_weight(node_name)
        
        text = f"{indent_str}{node.name} (level={node.level}, weight={effective_weight:.3f})\n"
        text += f"{indent_str}  Patterns: {', '.join(node.patterns[:3])}...\n"
        
        for child_name in node.children:
            text += self._format_subtree(child_name, indent + 1)
        
        return text
    
    def export_hierarchy_json(self) -> Dict:
        """Export hierarchy as JSON-serializable dict."""
        return {
            'nodes': {
                name: {
                    'level': node.level,
                    'patterns': node.patterns,
                    'parent': node.parent,
                    'children': node.children,
                    'weight': node.weight,
                    'inherited_weight': node.inherited_weight,
                    'effective_weight': self.get_effective_weight(name)
                }
                for name, node in self.nodes.items()
            },
            'edges': list(self.hierarchy.edges()),
            'max_depth': self.max_depth
        }
    
    def visualize_hierarchy_ascii(self) -> str:
        """Create ASCII art visualization of hierarchy."""
        roots = [name for name, node in self.nodes.items() if node.parent is None]
        
        viz = "Hierarchy Visualization:\n\n"
        
        for root in roots:
            viz += self._draw_node_ascii(root, "", is_last=True)
        
        return viz
    
    def _draw_node_ascii(self, node_name: str, prefix: str, is_last: bool) -> str:
        """Draw node and children in ASCII."""
        node = self.nodes[node_name]
        
        # Current node
        connector = "└── " if is_last else "├── "
        text = f"{prefix}{connector}{node.name} [{self.get_effective_weight(node_name):.2f}]\n"
        
        # Children
        children = node.children
        for i, child in enumerate(children):
            is_last_child = (i == len(children) - 1)
            extension = "    " if is_last else "│   "
            text += self._draw_node_ascii(child, prefix + extension, is_last_child)
        
        return text

