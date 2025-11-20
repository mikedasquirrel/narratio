"""
Temporal Linguistics Routes

Historical word usage cycles - testing if "history rhymes" quantitatively.
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path

temporal_linguistics_bp = Blueprint('temporal_linguistics', __name__)


@temporal_linguistics_bp.route('/temporal-linguistics')
def temporal_dashboard():
    """Interactive dashboard for temporal linguistic cycles."""
    # Load analysis results
    results_path = Path(__file__).parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'analysis_results.json'
    
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    
    return render_template('temporal_linguistics/dashboard.html',
                         results=results,
                         title="Temporal Linguistics: History Rhyming")


@temporal_linguistics_bp.route('/api/temporal-linguistics/words')
def word_data_api():
    """API endpoint for word frequency data."""
    results_path = Path(__file__).parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'analysis_results.json'
    
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return jsonify(data['word_data'])
    
    return jsonify({'error': 'Data not found'}), 404


@temporal_linguistics_bp.route('/api/temporal-linguistics/cycles')
def cycles_api():
    """API endpoint for cycle analysis results."""
    results_path = Path(__file__).parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'analysis_results.json'
    
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return jsonify(data['results'])
    
    return jsonify({'error': 'Data not found'}), 404

