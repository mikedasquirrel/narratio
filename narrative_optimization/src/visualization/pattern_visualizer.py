"""
Pattern Visualization System

Visualizes learned patterns, hierarchies, and relationships.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class PatternVisualizer:
    """
    Visualize learned patterns and archetypes.
    
    Visualizations:
    - Pattern space (2D/3D embedding)
    - Hierarchy trees
    - Performance over time
    - Cross-domain comparisons
    - Causal graphs
    """
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.fig_size = (12, 8)
        
    def visualize_pattern_space(
        self,
        patterns: Dict[str, Dict],
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Visualize patterns in 2D space.
        
        Parameters
        ----------
        patterns : dict
            Patterns with embeddings
        method : str
            'tsne' or 'pca'
        save_path : str, optional
            Path to save figure
        """
        # Extract pattern embeddings (simplified)
        pattern_names = list(patterns.keys())
        
        # Create simple embeddings based on keywords
        embeddings = []
        for pattern_data in patterns.values():
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            # Simple: use keyword counts as features
            embedding = [
                len(keywords),
                pattern_data.get('frequency', 0.0),
                pattern_data.get('correlation', 0.0),
                pattern_data.get('coherence', 0.0)
            ]
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        if embeddings.shape[0] < 2:
            print("Not enough patterns to visualize")
            return
        
        # Reduce to 2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        
        coords = reducer.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=[p.get('correlation', 0) for p in patterns.values()],
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        
        # Add labels
        for i, name in enumerate(pattern_names):
            ax.annotate(
                name.replace('_', ' ')[:20],
                (coords[i, 0], coords[i, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_title('Pattern Space Visualization')
        plt.colorbar(scatter, label='Correlation')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def visualize_hierarchy(
        self,
        hierarchy_nodes: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize pattern hierarchy.
        
        Parameters
        ----------
        hierarchy_nodes : dict
            Hierarchy nodes
        save_path : str, optional
            Path to save
        """
        import networkx as nx
        
        # Build graph
        G = nx.DiGraph()
        
        for node_name, node_data in hierarchy_nodes.items():
            G.add_node(node_name, **node_data)
            
            if 'children' in node_data:
                for child in node_data['children']:
                    G.add_edge(node_name, child)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Draw nodes
        node_colors = [node_data.get('effective_weight', 0.5) for _, node_data in hierarchy_nodes.items()]
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=500,
            cmap='YlOrRd',
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, arrows=True)
        
        # Draw labels
        labels = {name: name.replace('_', '\n')[:15] for name in hierarchy_nodes.keys()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title('Pattern Hierarchy')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_learning_history(
        self,
        learning_history: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot learning progress over time.
        
        Parameters
        ----------
        learning_history : list
            Learning metrics history
        save_path : str, optional
            Path to save
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        
        iterations = [m['iteration'] for m in learning_history]
        
        # R² progression
        r_squared = [m['r_squared_after'] for m in learning_history]
        axes[0, 0].plot(iterations, r_squared, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_title('Performance Over Time')
        axes[0, 0].grid(alpha=0.3)
        
        # Pattern counts
        discovered = [m['patterns_discovered'] for m in learning_history]
        validated = [m['patterns_validated'] for m in learning_history]
        axes[0, 1].plot(iterations, discovered, label='Discovered', marker='o')
        axes[0, 1].plot(iterations, validated, label='Validated', marker='s')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Pattern Discovery')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Improvements
        improvements = [m['improvement'] for m in learning_history]
        axes[1, 0].bar(iterations, improvements, alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Improvement')
        axes[1, 0].set_title('Performance Gains')
        axes[1, 0].grid(alpha=0.3)
        
        # Coherence
        coherence = [m['coherence_score'] for m in learning_history]
        axes[1, 1].plot(iterations, coherence, marker='o', color='green', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Coherence')
        axes[1, 1].set_title('Pattern Coherence')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_domain_comparison(
        self,
        domains: List[str],
        metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Compare metrics across domains.
        
        Parameters
        ----------
        domains : list
            Domain names
        metrics : dict
            domain -> {metric: value}
        save_path : str, optional
            Path to save
        """
        metric_names = list(next(iter(metrics.values())).keys())
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        x = np.arange(len(domains))
        width = 0.8 / len(metric_names)
        
        for i, metric in enumerate(metric_names):
            values = [metrics[d].get(metric, 0) for d in domains]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Domain')
        ax.set_ylabel('Value')
        ax.set_title('Cross-Domain Comparison')
        ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
        ax.set_xticklabels(domains, rotation=45, ha='right')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_causal_graph(
        self,
        causal_graph,
        save_path: Optional[str] = None
    ):
        """
        Visualize causal graph.
        
        Parameters
        ----------
        causal_graph : nx.DiGraph
            Causal graph
        save_path : str, optional
            Path to save
        """
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Layout
        pos = nx.spring_layout(causal_graph, k=3, iterations=100)
        
        # Draw nodes
        node_colors = []
        for node in causal_graph.nodes():
            if node == 'outcome':
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(
            causal_graph, pos, ax=ax,
            node_color=node_colors,
            node_size=800,
            alpha=0.9
        )
        
        # Draw edges with weights
        edges = causal_graph.edges()
        weights = [causal_graph[u][v].get('weight', 1.0) for u, v in edges]
        
        nx.draw_networkx_edges(
            causal_graph, pos, ax=ax,
            width=[abs(w) * 3 for w in weights],
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            causal_graph, pos, ax=ax,
            font_size=9,
            font_weight='bold'
        )
        
        ax.set_title('Causal Graph: Patterns → Outcomes')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_pattern_performance(
        self,
        patterns: Dict[str, Dict],
        top_n: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot top patterns by performance.
        
        Parameters
        ----------
        patterns : dict
            Patterns
        top_n : int
            Number of top patterns to show
        save_path : str, optional
            Path to save
        """
        # Sort by correlation
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: abs(x[1].get('correlation', 0)),
            reverse=True
        )[:top_n]
        
        names = [name.replace('_', ' ')[:30] for name, _ in sorted_patterns]
        correlations = [data.get('correlation', 0) for _, data in sorted_patterns]
        frequencies = [data.get('frequency', 0) for _, data in sorted_patterns]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        
        # Correlation
        y_pos = np.arange(len(names))
        ax1.barh(y_pos, correlations, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Correlation')
        ax1.set_title('Top Patterns by Correlation')
        ax1.grid(alpha=0.3, axis='x')
        
        # Frequency
        ax2.barh(y_pos, frequencies, alpha=0.7, color='green')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('Frequency')
        ax2.set_title('Pattern Frequency')
        ax2.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

