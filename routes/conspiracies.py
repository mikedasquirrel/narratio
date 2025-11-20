"""
Conspiracy Theories Routes

Flask routes for conspiracy theory analysis and results display.

Author: Narrative Integration System
Date: November 2025
"""

from flask import Blueprint, render_template, jsonify, request
import json
import pickle
from pathlib import Path
import numpy as np

conspiracies_bp = Blueprint('conspiracies', __name__)

# Load data
DATA_DIR = Path(__file__).parent.parent / 'data'

def load_conspiracy_data():
    """Load conspiracy theory data and results."""
    try:
        # Load theory data
        with open(DATA_DIR / 'conspiracy_theories_complete.json', 'r') as f:
            theories_data = json.load(f)
        
        # Load narrativity calculation
        with open(DATA_DIR / 'conspiracy_narrativity_calculation.json', 'r') as f:
            narrativity_data = json.load(f)
        
        # Note: External experiment results removed during cleanup
        experiment_results = None
        
        return {
            'theories': theories_data['theories'],
            'metadata': theories_data['metadata'],
            'narrativity': narrativity_data,
            'results': experiment_results
        }
    except Exception as e:
        print(f"Error loading conspiracy data: {e}")
        return None


@conspiracies_bp.route('/')
@conspiracies_bp.route('/dashboard')
def dashboard():
    """Display conspiracy theory dashboard landing page."""
    return render_template('conspiracy_dashboard.html')


@conspiracies_bp.route('/results')
def results():
    """Display conspiracy theory analysis results."""
    data = load_conspiracy_data()
    
    if not data:
        return render_template('error.html', 
                             error="Conspiracy theory data not found")
    
    # Prepare data for template (experiment results removed during cleanup)
    template_data = {
        'domain_name': 'Conspiracy Theories',
        'pi': data['narrativity']['pi'],
        'theta': data['narrativity']['theta'],
        'lambda': data['narrativity']['lambda'],
        'nominative_gravity': data['narrativity']['nominative_gravity'],
        'narrative_agency': data['narrativity']['narrative_agency'],
        'n_theories': len(data['theories']),
        'theories': data['theories'],
        'results': None,
        'best_transformer': 'N/A',
        'best_f1': 0.0,
        'theoretical_impact': data['narrativity'].get('theoretical_significance', [])
    }
    
    return render_template('conspiracy_results.html', **template_data)


@conspiracies_bp.route('/leaderboard')
def leaderboard():
    """Display conspiracy theory virality leaderboard."""
    data = load_conspiracy_data()
    
    if not data:
        return jsonify({'error': 'Data not found'}), 404
    
    # Sort theories by virality score
    theories_sorted = sorted(data['theories'], 
                            key=lambda x: x['virality_score'], 
                            reverse=True)
    
    return render_template('conspiracy_leaderboard.html',
                         theories=theories_sorted,
                         n_theories=len(theories_sorted))


@conspiracies_bp.route('/forces')
def forces():
    """Display force visualization."""
    data = load_conspiracy_data()
    
    if not data:
        return jsonify({'error': 'Data not found'}), 404
    
    forces_data = {
        'pi': data['narrativity']['pi'],
        'theta': data['narrativity']['theta'],
        'lambda': data['narrativity']['lambda'],
        'nominative_gravity': data['narrativity']['nominative_gravity'],
        'narrative_agency': data['narrativity']['narrative_agency'],
        'interpretations': data['narrativity']['interpretations']
    }
    
    return render_template('conspiracy_forces.html', **forces_data)


@conspiracies_bp.route('/api/stats')
def api_stats():
    """API endpoint for conspiracy theory statistics."""
    data = load_conspiracy_data()
    
    if not data:
        return jsonify({'error': 'Data not found'}), 404
    
    # Calculate statistics
    virality_scores = [t['virality_score'] for t in data['theories']]
    believer_counts = [t['outcomes']['believer_count_estimate'] for t in data['theories']]
    
    stats = {
        'domain': 'conspiracy_theories',
        'n_theories': len(data['theories']),
        'narrativity': {
            'pi': data['narrativity']['pi'],
            'theta': data['narrativity']['theta'],
            'lambda': data['narrativity']['lambda'],
            'nominative_gravity': data['narrativity']['nominative_gravity'],
            'narrative_agency': data['narrativity']['narrative_agency']
        },
        'virality': {
            'mean': float(np.mean(virality_scores)),
            'std': float(np.std(virality_scores)),
            'min': float(np.min(virality_scores)),
            'max': float(np.max(virality_scores))
        },
        'believers': {
            'total': int(np.sum(believer_counts)),
            'mean': float(np.mean(believer_counts)),
            'max': int(np.max(believer_counts))
        },
        'model_performance': {
            'best_f1': data['results']['summary']['best_f1_score'],
            'best_transformer': data['results']['summary']['best_transformer'],
            'avg_f1': data['results']['summary']['avg_f1_score']
        }
    }
    
    return jsonify(stats)


@conspiracies_bp.route('/api/theory/<theory_id>')
def api_theory(theory_id):
    """API endpoint for individual theory details."""
    data = load_conspiracy_data()
    
    if not data:
        return jsonify({'error': 'Data not found'}), 404
    
    # Find theory
    theory = next((t for t in data['theories'] if t['id'] == theory_id), None)
    
    if not theory:
        return jsonify({'error': 'Theory not found'}), 404
    
    return jsonify(theory)


@conspiracies_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint to predict virality of custom theory description."""
    # This would require loading the trained model and making predictions
    # For now, return placeholder
    return jsonify({
        'message': 'Prediction endpoint - requires model loading',
        'status': 'not_implemented'
    })

