"""
API Server

RESTful API for narrative analysis.

Author: Narrative Integration System
Date: November 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.learning import LearningPipeline, UniversalArchetypeLearner
from src.registry import get_domain_registry, list_all_domains
from src.data import DataLoader

app = Flask(__name__)
CORS(app)

# Global instances
pipeline = LearningPipeline()
loader = DataLoader()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': str(Path(__file__).stat().st_mtime)
    })


@app.route('/api/domains', methods=['GET'])
def list_domains():
    """List all registered domains."""
    domains = list_all_domains()
    
    registry = get_domain_registry()
    
    domain_data = []
    for domain_name in domains:
        domain = registry.get_domain(domain_name)
        if domain:
            domain_data.append({
                'name': domain.name,
                'pi': domain.pi,
                'type': domain.domain_type,
                'r_squared': domain.r_squared,
                'delta': domain.delta,
                'status': domain.status
            })
    
    return jsonify({
        'domains': domain_data,
        'total': len(domain_data)
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """
    Analyze a text narrative.
    
    POST body:
    {
        "domain": "golf",
        "text": "narrative text...",
        "outcome": 1
    }
    """
    data = request.get_json()
    
    if not data or 'domain' not in data or 'text' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    domain = data['domain']
    text = data['text']
    outcome = data.get('outcome')
    
    try:
        analyzer = DomainSpecificAnalyzer(domain)
        
        # Analyze single text
        if outcome is not None:
            results = analyzer.analyze_complete([text], np.array([outcome]))
        else:
            # Just extract features
            results = analyzer.analyze_complete([text], np.array([0]))
        
        return jsonify({
            'domain': domain,
            'story_quality': float(results['story_quality'][0]),
            'r_squared': float(results['r_squared']),
            'delta': float(results['delta']),
            'narrativity': float(results['narrativity'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/patterns/universal', methods=['GET'])
def get_universal_patterns():
    """Get universal patterns."""
    patterns = pipeline.universal_learner.get_patterns()
    
    # Convert to serializable format
    patterns_data = {}
    for name, data in patterns.items():
        patterns_data[name] = {
            'keywords': data.get('keywords', []),
            'frequency': data.get('frequency', 0),
            'description': data.get('description', '')
        }
    
    return jsonify({
        'patterns': patterns_data,
        'total': len(patterns_data)
    })


@app.route('/api/patterns/domain/<domain_name>', methods=['GET'])
def get_domain_patterns(domain_name):
    """Get domain-specific patterns."""
    if domain_name not in pipeline.domain_learners:
        return jsonify({'error': 'Domain not found'}), 404
    
    learner = pipeline.domain_learners[domain_name]
    patterns = learner.get_patterns()
    
    # Convert to serializable
    patterns_data = {}
    for name, data in patterns.items():
        patterns_data[name] = {
            'patterns': data.get('patterns', []),
            'frequency': data.get('frequency', 0),
            'coherence': data.get('coherence', 0)
        }
    
    return jsonify({
        'domain': domain_name,
        'patterns': patterns_data,
        'total': len(patterns_data)
    })


@app.route('/api/learn', methods=['POST'])
def trigger_learning():
    """
    Trigger a learning cycle.
    
    POST body:
    {
        "domains": ["golf", "tennis"],
        "learn_universal": true,
        "learn_domain_specific": true
    }
    """
    data = request.get_json()
    
    domains = data.get('domains', [])
    learn_universal = data.get('learn_universal', True)
    learn_domain_specific = data.get('learn_domain_specific', True)
    
    try:
        metrics = pipeline.learn_cycle(
            domains=domains if domains else None,
            learn_universal=learn_universal,
            learn_domain_specific=learn_domain_specific
        )
        
        return jsonify({
            'status': 'success',
            'iteration': metrics.iteration,
            'patterns_discovered': metrics.patterns_discovered,
            'patterns_validated': metrics.patterns_validated,
            'improvement': metrics.improvement
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare/<domain1>/<domain2>', methods=['GET'])
def compare_domains(domain1, domain2):
    """Compare two domains."""
    registry = get_domain_registry()
    
    d1 = registry.get_domain(domain1)
    d2 = registry.get_domain(domain2)
    
    if not d1 or not d2:
        return jsonify({'error': 'Domain not found'}), 404
    
    return jsonify({
        'domain1': {
            'name': d1.name,
            'pi': d1.pi,
            'r_squared': d1.r_squared,
            'delta': d1.delta
        },
        'domain2': {
            'name': d2.name,
            'pi': d2.pi,
            'r_squared': d2.r_squared,
            'delta': d2.delta
        },
        'similarity': registry._calculate_similarity(d1, d2)
    })


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run API server."""
    print("="*80)
    print("NARRATIVE OPTIMIZATION API SERVER")
    print("="*80)
    print(f"\nStarting server on {host}:{port}")
    print("\nEndpoints:")
    print("  GET  /api/health")
    print("  GET  /api/domains")
    print("  POST /api/analyze")
    print("  GET  /api/patterns/universal")
    print("  GET  /api/patterns/domain/<domain>")
    print("  POST /api/learn")
    print("  GET  /api/compare/<domain1>/<domain2>")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=5000, help='Port')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.debug)

