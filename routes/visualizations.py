"""
Interactive visualization routes
"""

from flask import Blueprint, render_template, request, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.visualization.network_plots import NetworkVisualizer, create_ensemble_network
from src.utils.toy_data import quick_load_toy_data

visualizations_bp = Blueprint('visualizations', __name__)


@visualizations_bp.route('/api/generate_network', methods=['POST'])
def generate_network():
    """Generate network from text input."""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Create and fit ensemble transformer
        ensemble = EnsembleNarrativeTransformer(n_top_terms=30, network_metrics=True)
        ensemble.fit(texts)
        
        # Get network data
        network = ensemble.get_ensemble_network()
        top_pairs = ensemble.get_top_ensemble_pairs(n=15)
        
        # Convert network to JSON format
        nodes = []
        for node in network.nodes():
            nodes.append({
                'id': node,
                'label': node,
                'degree': network.degree[node],
                'centrality': network.nodes[node].get('degree_centrality', 0)
            })
        
        edges = []
        for edge in network.edges():
            weight = network[edge[0]][edge[1]].get('weight', 1)
            edges.append({
                'source': edge[0],
                'target': edge[1],
                'weight': float(weight)
            })
        
        return jsonify({
            'success': True,
            'network': {
                'nodes': nodes,
                'edges': edges
            },
            'top_pairs': [
                {'term1': p[0], 'term2': p[1], 'count': p[2]}
                for p in top_pairs
            ],
            'stats': {
                'nodes': len(nodes),
                'edges': len(edges),
                'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

