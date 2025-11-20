"""
Interactive Network Visualizations for Narrative Analysis

Force-directed graphs showing relationships between narrative elements.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path


class NetworkVisualizer:
    """
    Create interactive network visualizations for narrative elements.
    
    Parameters
    ----------
    output_dir : str, optional
        Directory for saving plots
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_narrative_network(
        self,
        network: nx.Graph,
        title: str = "Narrative Element Network",
        node_size_attr: Optional[str] = None,
        edge_width_attr: str = 'weight',
        color_by: Optional[str] = None,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Create interactive force-directed network graph.
        
        Parameters
        ----------
        network : NetworkX Graph
            Network to visualize
        title : str
            Plot title
        node_size_attr : str, optional
            Node attribute for sizing (e.g., 'degree_centrality')
        edge_width_attr : str
            Edge attribute for width
        color_by : str, optional
            Node attribute for coloring
        save_name : str, optional
            Filename to save
        show : bool
            Whether to display
        
        Returns
        -------
        fig : plotly Figure
        """
        # Compute layout
        pos = nx.spring_layout(network, k=0.5, iterations=50)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        edge_widths = []
        
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge width
            if edge_width_attr and edge_width_attr in network[edge[0]][edge[1]]:
                width = network[edge[0]][edge[1]][edge_width_attr]
            else:
                width = 1
            edge_widths.append(width)
        
        # Edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates and attributes
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        
        for node in network.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node text
            node_text.append(str(node))
            
            # Node size
            if node_size_attr and node_size_attr in network.nodes[node]:
                size = network.nodes[node][node_size_attr]
            else:
                size = nx.degree_centrality(network)[node]
            node_sizes.append(size * 100 + 10)
            
            # Node color
            if color_by and color_by in network.nodes[node]:
                node_colors.append(network.nodes[node][color_by])
            else:
                node_colors.append(network.degree[node])
        
        # Node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create hover text with node info
        hover_text = []
        for node in network.nodes():
            degree = network.degree[node]
            info = f"<b>{node}</b><br>"
            info += f"Connections: {degree}<br>"
            
            # Add custom attributes
            for attr, value in network.nodes[node].items():
                if isinstance(value, (int, float)):
                    info += f"{attr}: {value:.3f}<br>"
            
            hover_text.append(info)
        
        node_trace.hovertext = hover_text
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Interactive: Hover for details, click to highlight",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10, color='gray')
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        # Save
        if save_name and self.output_dir:
            fig.write_html(self.output_dir / f"{save_name}.html")
        
        if show:
            fig.show()
        
        return fig
    
    def plot_co_occurrence_network(
        self,
        cooccurrence_matrix: np.ndarray,
        labels: List[str],
        threshold: float = 0.1,
        title: str = "Co-occurrence Network",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Visualize co-occurrence relationships as network.
        
        Parameters
        ----------
        cooccurrence_matrix : array
            Co-occurrence matrix
        labels : list
            Labels for nodes
        threshold : float
            Minimum co-occurrence to show edge
        title : str
            Plot title
        save_name : str, optional
            Filename to save
        show : bool
            Whether to display
        
        Returns
        -------
        fig : plotly Figure
        """
        # Build network from matrix
        G = nx.Graph()
        
        # Add nodes
        for label in labels:
            G.add_node(label)
        
        # Add edges
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                weight = cooccurrence_matrix[i, j]
                if weight > threshold:
                    G.add_edge(labels[i], labels[j], weight=float(weight))
        
        # Add degree centrality as node attribute
        centrality = nx.degree_centrality(G)
        nx.set_node_attributes(G, centrality, 'degree_centrality')
        
        return self.plot_narrative_network(
            G,
            title=title,
            node_size_attr='degree_centrality',
            edge_width_attr='weight',
            save_name=save_name,
            show=show
        )
    
    def plot_narrative_comparison_network(
        self,
        networks: Dict[str, nx.Graph],
        title: str = "Narrative Comparison",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Compare multiple narrative networks side by side.
        
        Parameters
        ----------
        networks : dict
            Dictionary mapping names to networks
        title : str
            Plot title
        save_name : str, optional
            Filename to save
        show : bool
            Whether to display
        
        Returns
        -------
        fig : plotly Figure
        """
        from plotly.subplots import make_subplots
        
        n_networks = len(networks)
        cols = min(3, n_networks)
        rows = (n_networks + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(networks.keys()),
            specs=[[{"type": "scatter"}] * cols] * rows
        )
        
        for idx, (name, network) in enumerate(networks.items()):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Compute layout for this network
            pos = nx.spring_layout(network, k=0.5, iterations=50)
            
            # Edges
            edge_x, edge_y = [], []
            for edge in network.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Nodes
            node_x = [pos[node][0] for node in network.nodes()]
            node_y = [pos[node][1] for node in network.nodes()]
            node_sizes = [network.degree[node] * 5 + 5 for node in network.nodes()]
            
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=node_sizes,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    hoverinfo='text',
                    hovertext=[str(node) for node in network.nodes()],
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
        
        fig.update_layout(
            title_text=title,
            height=400 * rows,
            showlegend=False
        )
        
        if save_name and self.output_dir:
            fig.write_html(self.output_dir / f"{save_name}.html")
        
        if show:
            fig.show()
        
        return fig


def create_ensemble_network(
    ensemble_transformer,
    top_n: int = 30
) -> nx.Graph:
    """
    Create network from ensemble transformer.
    
    Parameters
    ----------
    ensemble_transformer : EnsembleNarrativeTransformer
        Fitted ensemble transformer
    top_n : int
        Number of top nodes to include
    
    Returns
    -------
    network : NetworkX Graph
    """
    if not hasattr(ensemble_transformer, 'network_'):
        raise ValueError("Ensemble transformer must be fitted with network_metrics=True")
    
    G = ensemble_transformer.network_
    
    # Get top n nodes by degree centrality
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [node for node, _ in top_nodes]
    
    # Create subgraph
    subgraph = G.subgraph(top_node_names).copy()
    
    # Add centrality as node attributes
    for node in subgraph.nodes():
        subgraph.nodes[node]['degree_centrality'] = centrality[node]
        if hasattr(ensemble_transformer, 'centrality_scores_'):
            subgraph.nodes[node]['betweenness'] = ensemble_transformer.centrality_scores_['betweenness'].get(node, 0)
    
    return subgraph


if __name__ == '__main__':
    # Demo
    print("Creating demo network visualization...")
    
    # Create sample network
    G = nx.karate_club_graph()
    
    visualizer = NetworkVisualizer()
    visualizer.plot_narrative_network(
        G,
        title="Demo: Karate Club Network",
        show=True
    )

