"""
Imperative Gravity Visualization

Creates visualizations for cross-domain gravitational networks.

Visualizations:
1. Network graph (force-directed layout)
2. Domain similarity heatmap
3. Gravitational cluster diagram
4. Interactive exploration tool

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Network visualizations disabled.")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.imperative_gravity import ImperativeGravityCalculator


class ImperativeGravityVisualizer:
    """
    Visualize cross-domain imperative gravity networks.
    """
    
    def __init__(
        self,
        imperative_calculator: ImperativeGravityCalculator,
        domain_names: List[str]
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        imperative_calculator : ImperativeGravityCalculator
            Calculator with loaded domain configs
        domain_names : list of str
            All domain names to visualize
        """
        self.calculator = imperative_calculator
        self.domain_names = domain_names
        self.similarity_matrix = None
        self.network_graph = None
    
    def build_network_graph(
        self,
        force_threshold: float = 2.0
    ) -> Optional[object]:
        """
        Build networkx graph of domain connections.
        
        Parameters
        ----------
        force_threshold : float
            Minimum force magnitude to include edge
        
        Returns
        -------
        networkx.Graph or None
            Network graph if networkx available
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX not available")
            return None
        
        # Calculate similarity matrix
        self.similarity_matrix = self.calculator.calculate_domain_similarity_matrix(
            self.domain_names
        )
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for domain in self.domain_names:
            G.add_node(domain)
        
        # Add edges (forces above threshold)
        for i, domain1 in enumerate(self.domain_names):
            for j, domain2 in enumerate(self.domain_names):
                if i < j:  # Avoid duplicates
                    similarity = self.similarity_matrix[i, j]
                    
                    # Estimate force magnitude (using unit mass)
                    distance = self.calculator._calculate_domain_distance(domain1, domain2)
                    force = similarity / (distance ** 2 + 0.01)
                    
                    if force >= force_threshold:
                        G.add_edge(domain1, domain2, weight=force, similarity=similarity)
        
        self.network_graph = G
        return G
    
    def visualize_network(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12),
        force_threshold: float = 2.0,
        layout: str = 'spring'
    ):
        """
        Create network visualization.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        force_threshold : float
            Minimum force to show edge
        layout : str
            'spring', 'circular', or 'kamada_kawai'
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX required for network visualization")
            return
        
        # Build graph if not already built
        if self.network_graph is None:
            self.build_network_graph(force_threshold)
        
        G = self.network_graph
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 200 + 300 for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.7,
            ax=ax
        )
        
        # Draw edges (thicker = stronger force)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        
        # Normalize weights for visualization
        edge_widths = [3 * w / max_weight for w in weights]
        edge_colors = weights
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color=edge_colors,
            edge_cmap=plt.cm.YlOrRd,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(
            'Imperative Gravity Network\n(Cross-Domain Structural Connections)',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        
        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                   norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Force Magnitude', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved network visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_similarity_heatmap(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 12)
    ):
        """
        Create similarity heatmap for all domains.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        # Calculate similarity matrix if needed
        if self.similarity_matrix is None:
            self.similarity_matrix = self.calculator.calculate_domain_similarity_matrix(
                self.domain_names
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            self.similarity_matrix,
            xticklabels=self.domain_names,
            yticklabels=self.domain_names,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Structural Similarity'},
            ax=ax
        )
        
        ax.set_title('Domain Similarity Matrix\n(Structural Overlap)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_domain_clusters(
        self,
        clusters: List[List[str]],
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Visualize identified domain clusters.
        
        Parameters
        ----------
        clusters : list of list
            Domain clusters
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        if not NETWORKX_AVAILABLE:
            print("NetworkX required for cluster visualization")
            return
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with cluster assignment
        node_colors = []
        color_map = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for cluster_id, cluster in enumerate(clusters):
            for domain in cluster:
                G.add_node(domain, cluster=cluster_id)
                node_colors.append(color_map[cluster_id])
        
        # Add edges within clusters (fully connected)
        for cluster in clusters:
            for i, domain1 in enumerate(cluster):
                for domain2 in cluster[i+1:]:
                    G.add_edge(domain1, domain2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw
        nx.draw(
            G, pos,
            node_color=node_colors,
            node_size=800,
            with_labels=True,
            font_size=9,
            font_weight='bold',
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title('Domain Clusters\n(Gravitational Groupings)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color_map[i], markersize=10,
                      label=f'Cluster {i+1} ({len(cluster)} domains)')
            for i, cluster in enumerate(clusters)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved cluster visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_domain_space_2d(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Project domain space to 2D using dimensionality reduction.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        """
        from sklearn.manifold import MDS
        
        # Calculate similarity matrix if needed
        if self.similarity_matrix is None:
            self.similarity_matrix = self.calculator.calculate_domain_similarity_matrix(
                self.domain_names
            )
        
        # Convert similarity to distance
        distance_matrix = 1 - self.similarity_matrix
        
        # Project to 2D
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=200,
            alpha=0.6,
            c=range(len(self.domain_names)),
            cmap='tab20'
        )
        
        # Labels
        for i, domain in enumerate(self.domain_names):
            ax.annotate(
                domain,
                (coords[i, 0], coords[i, 1]),
                fontsize=9,
                fontweight='bold',
                ha='center'
            )
        
        ax.set_title('Domain Space Projection\n(2D MDS based on Structural Similarity)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved 2D projection to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_all_visualizations(
        self,
        output_dir: str,
        force_threshold: float = 2.0
    ):
        """
        Create all visualizations and save to directory.
        
        Parameters
        ----------
        output_dir : str
            Directory to save visualizations
        force_threshold : float
            Minimum force for network edges
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating imperative gravity visualizations...")
        print(f"Output directory: {output_dir}\n")
        
        # 1. Network graph
        print("1/4 Creating network graph...")
        self.visualize_network(
            output_path=str(output_dir / 'imperative_gravity_network.png'),
            force_threshold=force_threshold
        )
        
        # 2. Similarity heatmap
        print("2/4 Creating similarity heatmap...")
        self.visualize_similarity_heatmap(
            output_path=str(output_dir / 'domain_similarity_heatmap.png')
        )
        
        # 3. Domain space projection
        print("3/4 Creating domain space projection...")
        self.visualize_domain_space_2d(
            output_path=str(output_dir / 'domain_space_2d.png')
        )
        
        # 4. Clusters
        print("4/4 Creating cluster visualization...")
        clusters = self.calculator.get_domain_clusters(
            self.domain_names,
            similarity_threshold=0.7
        )
        self.visualize_domain_clusters(
            clusters=clusters,
            output_path=str(output_dir / 'domain_clusters.png')
        )
        
        # Export data
        self._export_network_data(output_dir)
        
        print(f"\n✓ All visualizations saved to {output_dir}")
    
    def _export_network_data(self, output_dir: Path):
        """Export network data as JSON."""
        if self.similarity_matrix is None:
            return
        
        # Export similarity matrix
        similarity_data = {
            'domains': self.domain_names,
            'similarity_matrix': self.similarity_matrix.tolist()
        }
        
        with open(output_dir / 'similarity_matrix.json', 'w') as f:
            json.dump(similarity_data, f, indent=2)
        
        # Export network edges
        if NETWORKX_AVAILABLE and self.network_graph:
            edges_data = {
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'force': data['weight'],
                        'similarity': data['similarity']
                    }
                    for u, v, data in self.network_graph.edges(data=True)
                ]
            }
            
            with open(output_dir / 'network_edges.json', 'w') as f:
                json.dump(edges_data, f, indent=2)
        
        print(f"  ✓ Exported network data to JSON")
    
    def interactive_explorer(self):
        """
        Create interactive visualization (requires plotly).
        
        Opens in browser for exploration.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly required for interactive visualization")
            print("Install: pip install plotly")
            return
        
        # Calculate similarity if needed
        if self.similarity_matrix is None:
            self.similarity_matrix = self.calculator.calculate_domain_similarity_matrix(
                self.domain_names
            )
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=self.similarity_matrix,
            x=self.domain_names,
            y=self.domain_names,
            colorscale='YlOrRd',
            hovertemplate='<b>%{x}</b> ←→ <b>%{y}</b><br>Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Domain Similarity Matrix',
            xaxis_title='Domain',
            yaxis_title='Domain',
            width=1000,
            height=900
        )
        
        fig.show()
        print("✓ Opened interactive explorer in browser")


def create_gravity_network_visualization(
    all_domain_configs: Dict[str, object],
    output_dir: str = 'results/imperative_gravity',
    force_threshold: float = 2.0
):
    """
    Convenience function to create all visualizations.
    
    Parameters
    ----------
    all_domain_configs : dict
        {domain_name: DomainConfig}
    output_dir : str
        Output directory
    force_threshold : float
        Minimum force for edges
    """
    # Create calculator
    calculator = ImperativeGravityCalculator(all_domain_configs)
    
    # Create visualizer
    domain_names = list(all_domain_configs.keys())
    visualizer = ImperativeGravityVisualizer(calculator, domain_names)
    
    # Generate all visualizations
    visualizer.create_all_visualizations(output_dir, force_threshold)
    
    return visualizer


if __name__ == '__main__':
    # Example usage
    print("Imperative Gravity Visualization Module")
    print("Import and use create_gravity_network_visualization()")

