"""
Dinosaur Names Analysis Routes
Flask routes for dinosaur name effects in cultural transmission

π = 0.753 (High - Educational Domain)
R² = 62.6% (Name effects dominate)
Name contribution: +62.3% beyond scientific factors

Key Finding: Names matter MORE than scientific importance for cultural transmission.
Jurassic Park effect: +0.677 coefficient (massive media boost)
Nickname advantage: +0.115 coefficient (T-Rex > Tyrannosaurus)

Date: November 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path

dinosaurs_bp = Blueprint('dinosaurs', __name__, url_prefix='/dinosaurs')

# Global cache
_cache = {}


def load_dinosaur_results():
    """Load dinosaur analysis results"""
    if 'results' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'dinosaurs' / 'dinosaur_complete_analysis.json'
            with open(path, 'r') as f:
                _cache['results'] = json.load(f)
        except Exception as e:
            print(f"Error loading dinosaur results: {e}")
            _cache['results'] = None
    return _cache['results']


def load_dinosaur_pi():
    """Load dinosaur π calculation"""
    if 'pi' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'data' / 'domains' / 'dinosaurs' / 'dinosaur_narrativity_calculation.json'
            with open(path, 'r') as f:
                _cache['pi'] = json.load(f)
        except Exception as e:
            print(f"Error loading dinosaur π: {e}")
            _cache['pi'] = None
    return _cache['pi']


def load_dinosaur_dataset():
    """Load full dinosaur dataset"""
    if 'dataset' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'data' / 'domains' / 'dinosaurs' / 'dinosaur_complete_dataset.json'
            with open(path, 'r') as f:
                _cache['dataset'] = json.load(f)
        except Exception as e:
            print(f"Error loading dinosaur dataset: {e}")
            _cache['dataset'] = None
    return _cache['dataset']


@dinosaurs_bp.route('/')
def dinosaur_dashboard():
    """Main dinosaur analysis dashboard"""
    results = load_dinosaur_results()
    pi_data = load_dinosaur_pi()
    
    if not results:
        return "Dinosaur analysis results not found", 404
    
    return render_template('dinosaur_dashboard.html',
                         results=results,
                         pi_data=pi_data)


@dinosaurs_bp.route('/results')
def dinosaur_results():
    """Detailed dinosaur results page"""
    results = load_dinosaur_results()
    pi_data = load_dinosaur_pi()
    
    if not results:
        return "Dinosaur analysis results not found", 404
    
    # Extract key metrics
    metrics = {
        'pi': results['pi'],
        'theta': results['forces']['theta']['mean'],
        'lambda': results['forces']['lambda']['mean'],
        'ta_marbuta': results['forces']['ta_marbuta']['mean'],
        'r_squared': results['regression_analysis']['r2_full'],
        'name_contribution': results['regression_analysis']['r2_improvement'],
        'sample_size': results['sample_size']
    }
    
    return render_template('dinosaur_results.html',
                         results=results,
                         pi_data=pi_data,
                         metrics=metrics)


@dinosaurs_bp.route('/api/stats')
def dinosaur_stats_api():
    """JSON endpoint for dinosaur statistics"""
    results = load_dinosaur_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    return jsonify({
        'domain': 'dinosaurs',
        'pi': results['pi'],
        'sample_size': results['sample_size'],
        'forces': results['forces'],
        'regression_r2': results['regression_analysis']['r2_full'],
        'name_contribution': results['regression_analysis']['r2_improvement'],
        'key_finding': 'Names contribute 62.3% R² to cultural dominance - MORE than scientific importance. Tests nominative effects in childhood education.',
        'theoretical_contributions': results['theoretical_contributions']
    })


@dinosaurs_bp.route('/api/comparison')
def dinosaur_comparison_api():
    """API endpoint for cross-domain comparison"""
    results = load_dinosaur_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    comparisons = {
        'dinosaurs': {
            'pi': 0.753,
            'r_squared': 0.626,
            'name_effect': '+62.3%',
            'agency': 'Perfect (1.00) - kids choose',
            'key': 'Educational transmission'
        },
        'golf': {
            'pi': 0.70,
            'r_squared': 0.977,
            'agency': 'Perfect (1.00)',
            'key_difference': 'Skill dominates'
        },
        'poker': {
            'pi': 0.835,
            'r_squared': 0.047,
            'agency': 'Perfect (1.00)',
            'key_difference': 'Variance dominates'
        },
        'hurricanes': {
            'pi': 0.677,
            'r_squared': 0.915,
            'agency': 'Zero (storm) / High (response)',
            'key_difference': 'Nature dominates'
        }
    }
    
    return jsonify({
        'dinosaurs': results,
        'comparisons': comparisons,
        'insight': 'Dinosaurs demonstrate that with perfect agency (1.00) and low constraints (λ=0.285), NAME EFFECTS dominate cultural transmission. Names matter MORE than scientific importance for what children learn.'
    })

