"""
Professional Poker Narrative Analysis Routes
Flask routes for poker narrative optimization dashboard

π = 0.835 (Very High Narrativity)
R² = 4.7% (Variance-dominated but high narrativity)
θ = 0.256, λ = 0.557, ة = 0.704

Key Finding: High narrativity + high variance = modest R²
Validates that external randomness can dominate even high-π domains

Date: November 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path

poker_bp = Blueprint('poker', __name__, url_prefix='/poker')

# Global cache
_cache = {}


def load_poker_results():
    """Load poker analysis results"""
    if 'results' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'poker' / 'poker_complete_analysis.json'
            with open(path, 'r') as f:
                _cache['results'] = json.load(f)
        except Exception as e:
            print(f"Error loading poker results: {e}")
            _cache['results'] = None
    return _cache['results']


def load_poker_pi():
    """Load poker π calculation"""
    if 'pi' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'data' / 'domains' / 'poker' / 'poker_narrativity_calculation.json'
            with open(path, 'r') as f:
                _cache['pi'] = json.load(f)
        except Exception as e:
            print(f"Error loading poker π: {e}")
            _cache['pi'] = None
    return _cache['pi']


@poker_bp.route('/')
def poker_dashboard():
    """Main poker analysis dashboard"""
    results = load_poker_results()
    pi_data = load_poker_pi()
    
    if not results:
        return "Poker analysis results not found", 404
    
    return render_template('poker_dashboard.html',
                         results=results,
                         pi_data=pi_data)


@poker_bp.route('/results')
def poker_results():
    """Detailed poker results page"""
    results = load_poker_results()
    pi_data = load_poker_pi()
    
    if not results:
        return "Poker analysis results not found", 404
    
    # Extract key metrics
    metrics = {
        'pi': results['pi'],
        'theta': results['forces']['theta']['mean'],
        'lambda': results['forces']['lambda']['mean'],
        'ta_marbuta': results['forces']['ta_marbuta']['mean'],
        'r_squared': results['performance']['r_squared'],
        'correlation': results['performance']['correlation'],
        'sample_size': results['sample_size']
    }
    
    return render_template('poker_results.html',
                         results=results,
                         pi_data=pi_data,
                         metrics=metrics)


@poker_bp.route('/api/stats')
def poker_stats_api():
    """JSON endpoint for poker statistics"""
    results = load_poker_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    return jsonify({
        'domain': 'professional_poker',
        'pi': results['pi'],
        'forces': results['forces'],
        'performance': results['performance'],
        'sample_size': results['sample_size'],
        'key_finding': 'High narrativity (π=0.835) with variance-dominated outcomes (R²=4.7%). Demonstrates that external randomness can limit narrative effects even in high-π domains.',
        'theoretical_contributions': [
            'First skill+chance hybrid domain analyzed',
            'Validates variance-dominance hypothesis',
            'Shows high π does not guarantee high R²',
            'Psychological warfare quantified (θ=0.256)',
            'Rich nominatives confirmed (ة=0.704)'
        ]
    })


@poker_bp.route('/api/comparison')
def poker_comparison_api():
    """API endpoint for cross-domain comparison"""
    results = load_poker_results()
    
    if not results:
        return jsonify({'error': 'Results not found'}), 404
    
    # Comparison to other domains
    comparisons = {
        'golf': {
            'pi': 0.70,
            'r_squared': 0.977,
            'variance': 'Low (skill-dominated)',
            'key_difference': 'Golf has no external randomness (no cards/dice)'
        },
        'tennis': {
            'pi': 0.75,
            'r_squared': 0.931,
            'variance': 'Low (skill-dominated)',
            'key_difference': 'Tennis has minimal randomness factors'
        },
        'poker': {
            'pi': 0.835,
            'r_squared': 0.047,
            'variance': 'Very High (card variance)',
            'key_difference': 'Even best players face 50%+ variance from cards'
        },
        'ufc': {
            'pi': 0.722,
            'r_squared': 0.025,
            'variance': 'Low (performance-dominated)',
            'key_difference': 'UFC low R² from performance dominance, not variance'
        }
    }
    
    return jsonify({
        'poker': results,
        'comparisons': comparisons,
        'insight': 'Poker demonstrates that high narrativity (π=0.835) does not guarantee high R² when external variance dominates. This contrasts with Golf/Tennis (skill-dominated) and validates the importance of domain characteristics beyond narrativity.'
    })

