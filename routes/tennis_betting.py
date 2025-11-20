"""
Tennis Betting Routes - Production Interface

Real-time betting predictions using complete model (ALL 33 transformers).
Shows learned weights, feature breakdowns, and betting recommendations.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

from flask import Blueprint, render_template, jsonify
import json
import numpy as np
from pathlib import Path
import sys

tennis_betting_bp = Blueprint('tennis_betting', __name__, url_prefix='/tennis')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))


@tennis_betting_bp.route('/bet-now')
def bet_now():
    """Production betting interface for today's/upcoming ATP matches."""
    
    # Load upcoming predictions
    predictions_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'upcoming_predictions.json'
    
    predictions_data = None
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                predictions_data = json.load(f)
        except:
            predictions_data = None
    
    # Load model performance
    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'tennis_complete_results.json'
    
    model_results = None
    if results_path.exists():
        try:
            with open(results_path) as f:
                model_results = json.load(f)
        except:
            model_results = None
    
    return render_template('tennis_bet_now.html',
                         predictions=predictions_data,
                         model_results=model_results,
                         is_ready=predictions_data is not None)


@tennis_betting_bp.route('/api/model-info')
def model_info():
    """API endpoint for model information and learned weights."""
    
    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'tennis_complete_results.json'
    
    if not results_path.exists():
        return jsonify({'error': 'Model not trained yet'}), 404
    
    with open(results_path) as f:
        results = json.load(f)
    
    return jsonify({
        'model_type': results['model_selection']['best_model'],
        'performance': {
            'test_r2': results['test_performance']['r2'],
            'test_roi': results['betting_simulation']['roi_pct'],
            'test_accuracy': results['test_performance']['accuracy'],
            'baseline_r2': results['baseline_comparison']['baseline_r2'],
            'improvement': results['baseline_comparison']['improvement_pct']
        },
        'features': {
            'total': results['features']['total_features'],
            'transformers_used': results['features']['transformers_used'],
            'archetype_features': results['features']['archetype_features']
        },
        'learned_weights': results.get('learned_weights', {}),
        'cross_domain_insights': results.get('cross_domain_insights', {})
    })


@tennis_betting_bp.route('/api/upcoming-predictions')
def upcoming_predictions_api():
    """API endpoint for upcoming match predictions."""
    
    predictions_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'upcoming_predictions.json'
    
    if not predictions_path.exists():
        return jsonify({'error': 'No predictions generated yet'}), 404
    
    with open(predictions_path) as f:
        data = json.load(f)
    
    return jsonify(data)


@tennis_betting_bp.route('/api/predict-match', methods=['POST'])
def predict_single_match():
    """API endpoint to predict a single user-specified match."""
    from flask import request
    import pickle
    
    data = request.get_json()
    player1 = data.get('player1')
    player2 = data.get('player2')
    surface = data.get('surface', 'hard')
    
    # Load model
    model_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'tennis_complete_model.pkl'
    scaler_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'tennis_complete' / 'results' / 'feature_scaler.pkl'
    
    if not model_path.exists():
        return jsonify({'error': 'Model not trained yet'}), 404
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Generate prediction (simplified - in production would apply all transformers)
    # For now, return template response
    prediction = {
        'player1': player1,
        'player2': player2,
        'surface': surface,
        'predicted_winner': player1,  # Placeholder
        'confidence': 0.73,
        'probability_player1': 0.73,
        'probability_player2': 0.27,
        'betting_recommendation': f"BET {player1} if odds > -200",
        'note': 'Full transformer pipeline execution would be applied here'
    }
    
    return jsonify(prediction)

