"""
NBA Betting Routes - Production Interface

Real-time betting predictions for NBA games using complete model (ALL 33 transformers).
Shows learned weights, performance comparisons with Tennis, market inefficiency analysis.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

from flask import Blueprint, render_template, jsonify
import json
import numpy as np
from pathlib import Path
import sys

nba_betting_bp = Blueprint('nba_betting', __name__, url_prefix='/nba')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))


@nba_betting_bp.route('/bet-now')
def bet_now():
    """Production betting interface for tonight's/upcoming NBA games."""
    
    # Load model results
    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'nba_complete' / 'results' / 'nba_complete_results.json'
    
    model_results = None
    if results_path.exists():
        try:
            with open(results_path) as f:
                model_results = json.load(f)
        except:
            model_results = None
    
    # Load predictions if available
    predictions_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'nba_complete' / 'results' / 'upcoming_predictions.json'
    
    predictions_data = None
    if predictions_path.exists():
        try:
            with open(predictions_path) as f:
                predictions_data = json.load(f)
        except:
            predictions_data = None
    
    return render_template('nba_bet_now.html',
                         predictions=predictions_data,
                         model_results=model_results,
                         is_ready=model_results is not None)


@nba_betting_bp.route('/api/model-info')
def model_info():
    """API endpoint for NBA model information."""
    
    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / 'nba_complete' / 'results' / 'nba_complete_results.json'
    
    if not results_path.exists():
        return jsonify({'error': 'Model not trained yet'}), 404
    
    with open(results_path) as f:
        results = json.load(f)
    
    return jsonify({
        'model_type': 'nba_complete',
        'performance': {
            'test_r2': results['test_performance']['r2'],
            'test_accuracy': results['test_performance']['accuracy'],
            'baseline_r2': results['baseline_comparison']['baseline_r2'],
            'improvement': results['baseline_comparison']['improvement']
        },
        'features': results['features'],
        'domain_characteristics': {
            'pi': 0.49,
            'type': 'performance_dominated',
            'agency': 0.70,
            'note': 'Team sport with distributed consciousness'
        }
    })

