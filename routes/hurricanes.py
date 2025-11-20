"""
Hurricane Names Narrative Analysis Routes
Flask routes for hurricane name effects dashboard

π (Storm) = 0.425 (Moderate - low agency)
π (Response) = 0.677 (Moderate-High - human agency)
R² = 91.5% (name effects +1.1%)

Key Finding: Dual π approach - nature has low narrativity, but human response has high narrativity.
Tests nominative determinism in life/death decisions.

Date: November 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path

hurricanes_bp = Blueprint('hurricanes', __name__, url_prefix='/hurricanes')

# Global cache
_cache = {}


def load_hurricane_results():
    """Load hurricane analysis results"""
    if 'results' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'hurricanes' / 'hurricane_complete_analysis.json'
            with open(path, 'r') as f:
                _cache['results'] = json.load(f)
        except Exception as e:
            print(f"Error loading hurricane results: {e}")
            _cache['results'] = None
    return _cache['results']


def load_hurricane_pi():
    """Load hurricane π calculation"""
    if 'pi' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'data' / 'domains' / 'hurricanes' / 'hurricane_narrativity_calculation.json'
            with open(path, 'r') as f:
                _cache['pi'] = json.load(f)
        except Exception as e:
            print(f"Error loading hurricane π: {e}")
            _cache['pi'] = None
    return _cache['pi']


def load_hurricane_dataset():
    """Load full hurricane dataset"""
    if 'dataset' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'data' / 'domains' / 'hurricanes' / 'hurricane_dataset_with_name_analysis.json'
            with open(path, 'r') as f:
                _cache['dataset'] = json.load(f)
        except Exception as e:
            print(f"Error loading hurricane dataset: {e}")
            _cache['dataset'] = None
    return _cache['dataset']


@hurricanes_bp.route('/')
def hurricane_dashboard():
    """Main hurricane analysis dashboard"""
    results = load_hurricane_results()
    pi_data = load_hurricane_pi()
    
    if not results:
        return "Hurricane analysis results not found", 404
    
    return render_template('hurricane_dashboard.html',
                         results=results,
                         pi_data=pi_data)


@hurricanes_bp.route('/results')
def hurricane_results():
    """Detailed hurricane results page"""
    results = load_hurricane_results()
    pi_data = load_hurricane_pi()
    
    if not results:
        return "Hurricane analysis results not found", 404
    
    # Extract key metrics
    metrics = {
        'pi_storm': results['pi_storm'],
        'pi_response': results['pi_response'],
        'theta': results['forces']['theta']['mean'],
        'lambda': results['forces']['lambda']['mean'],
        'ta_marbuta': results['forces']['ta_marbuta']['mean'],
        'r_squared': results['regression_analysis']['r2_with_names'],
        'name_contribution': results['regression_analysis']['r2_improvement'],
        'sample_size': results['sample_size'],
        'landfall_storms': results['landfall_storms']
    }
    
    return render_template('hurricane_results.html',
                         results=results,
                         pi_data=pi_data,
                         metrics=metrics)


@hurricanes_bp.route('/api/stats')
def hurricane_stats_api():
    """JSON endpoint for hurricane statistics"""
    results = load_hurricane_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    return jsonify({
        'domain': 'hurricanes',
        'pi_storm': results['pi_storm'],
        'pi_response': results['pi_response'],
        'sample_size': results['sample_size'],
        'forces': results['forces'],
        'gender_analysis': results['gender_analysis'],
        'harshness_analysis': results['harshness_analysis'],
        'regression_r2': results['regression_analysis']['r2_with_names'],
        'key_finding': 'Dual π approach: Storm (π=0.425) vs Response (π=0.677). Name effects contribute +1.1% R² beyond physical factors.',
        'theoretical_contributions': results['theoretical_contributions']
    })


@hurricanes_bp.route('/api/comparison')
def hurricane_comparison_api():
    """API endpoint for cross-domain comparison"""
    results = load_hurricane_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    comparisons = {
        'hurricanes': {
            'pi_storm': 0.425,
            'pi_response': 0.677,
            'r_squared': 0.915,
            'name_effect': '+1.1%',
            'agency': 'Zero (storm) / High (response)'
        },
        'golf': {
            'pi': 0.70,
            'r_squared': 0.977,
            'agency': 'Perfect (1.00)',
            'key_difference': 'Human performance vs natural phenomenon'
        },
        'poker': {
            'pi': 0.835,
            'r_squared': 0.047,
            'agency': 'Perfect (1.00)',
            'key_difference': 'High π but variance-dominated'
        }
    }
    
    return jsonify({
        'hurricanes': results,
        'comparisons': comparisons,
        'insight': 'Hurricanes demonstrate dual π framework - nature itself has low narrativity (zero agency) but human response has moderate-high narrativity (individual evacuation decisions).'
    })
