"""
REST API endpoints for all transformers
"""

from flask import Blueprint, request, jsonify
from functools import wraps
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer  
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer

api_bp = Blueprint('api', __name__)

# Simple API key authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # In production, validate against database
        if not api_key or api_key != 'demo-key-12345':
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@api_bp.route('/')
def api_docs():
    """API documentation page."""
    return jsonify({
        'version': '1.0',
        'endpoints': {
            '/api/transformers': 'List available transformers',
            '/api/analyze': 'POST - Analyze text with transformers',
            '/api/ensemble': 'POST - Ensemble analysis',
            '/api/linguistic': 'POST - Linguistic analysis',
            '/api/self-perception': 'POST - Self-perception analysis',
            '/api/potential': 'POST - Narrative potential analysis'
        },
        'authentication': 'Include X-API-Key header',
        'example_key': 'demo-key-12345'
    })

@api_bp.route('/transformers')
def list_transformers():
    """List available transformers."""
    return jsonify({
        'transformers': [
            {
                'id': 'ensemble',
                'name': 'Ensemble Narrative',
                'description': 'Co-occurrence, network centrality, diversity'
            },
            {
                'id': 'relational',
                'name': 'Relational Value',
                'description': 'Complementarity, synergy, relational density'
            },
            {
                'id': 'linguistic',
                'name': 'Linguistic Patterns',
                'description': 'Voice, agency, temporality, complexity'
            },
            {
                'id': 'nominative',
                'name': 'Nominative Analysis',
                'description': 'Naming patterns, semantic fields, identity'
            },
            {
                'id': 'self-perception',
                'name': 'Self-Perception',
                'description': 'Self-reference, attribution, growth mindset'
            },
            {
                'id': 'potential',
                'name': 'Narrative Potential',
                'description': 'Future orientation, possibility, flexibility'
            }
        ]
    })

@api_bp.route('/ensemble', methods=['POST'])
@require_api_key
def api_ensemble():
    """Ensemble narrative analysis endpoint."""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'texts array required'}), 400
    
    try:
        transformer = EnsembleNarrativeTransformer(n_top_terms=50)
        transformer.fit(texts)
        
        features = []
        for text in texts:
            feat = transformer.transform([text])[0]
            features.append(feat.tolist())
        
        return jsonify({
            'success': True,
            'features': features,
            'report': transformer.get_narrative_report()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/linguistic', methods=['POST'])
@require_api_key
def api_linguistic():
    """Linguistic patterns analysis endpoint."""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'texts array required'}), 400
    
    try:
        transformer = LinguisticPatternsTransformer()
        transformer.fit(texts)
        
        features = []
        for text in texts:
            feat = transformer.transform([text])[0]
            features.append(feat.tolist())
        
        return jsonify({
            'success': True,
            'features': features,
            'report': transformer.get_narrative_report()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

