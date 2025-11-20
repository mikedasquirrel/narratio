"""
Cross-Domain Comparison Routes

Shows all domains together for comparison.
Displays Д values, п spectrum, phylogenetic relationships.
"""

from flask import Blueprint, render_template, jsonify

cross_domain_bp = Blueprint('cross_domain', __name__)


@cross_domain_bp.route('/domains/compare')
def compare_domains():
    """Cross-domain comparison page."""
    return render_template('cross_domain_comparison.html')


@cross_domain_bp.route('/api/domains/all-metrics')
def get_all_domain_metrics():
    """Get metrics for all analyzed domains with Д calculations."""
    
    domains = [
        {
            'name': 'Coin Flips',
            'slug': 'coin_flips',
            'n_organisms': 1000,
            'narrativity': 0.12,
            'coupling': 0.0,
            'r_measured': 0.044,
            'D_agency': 0.005,
            'efficiency': 0.04,
            'passes': False,
            'interpretation': 'Floor: Pure physics, zero narrative agency'
        },
        {
            'name': 'Math Problems',
            'slug': 'math',
            'n_organisms': 500,
            'narrativity': 0.15,
            'coupling': 0.1,
            'r_measured': 0.05,
            'D_agency': 0.008,
            'efficiency': 0.05,
            'passes': False,
            'interpretation': 'Logic dominates, minimal narrative effect'
        },
        {
            'name': 'NBA',
            'slug': 'nba',
            'n_organisms': 2000,
            'narrativity': 0.48,
            'coupling': 0.7,
            'r_measured': -0.048,
            'D_agency': -0.016,
            'efficiency': -0.03,
            'passes': False,
            'interpretation': 'Performance domain: Skill matters, not narrative description'
        },
        {
            'name': 'NCAA Basketball',
            'slug': 'ncaa',
            'n_organisms': 998,
            'narrativity': 0.45,
            'coupling': 0.7,
            'r_measured': -0.162,
            'D_agency': -0.051,
            'efficiency': -0.11,
            'passes': False,
            'interpretation': 'Tournament performance: Execution matters, not description'
        },
        {
            'name': 'Movies',
            'slug': 'movies',
            'n_organisms': 1000,
            'narrativity': 0.65,
            'coupling': 0.5,
            'r_measured': 0.079,
            'D_agency': 0.026,
            'efficiency': 0.04,
            'passes': False,
            'interpretation': 'Content matters more than title/genre description'
        },
        {
            'name': 'Startups',
            'slug': 'startups',
            'n_organisms': 269,
            'narrativity': 0.76,
            'coupling': 0.3,
            'r_measured': 0.980,
            'D_agency': 0.223,
            'efficiency': 0.29,
            'passes': False,
            'interpretation': 'High prediction (r=0.980) but low agency (market constrains)'
        },
        {
            'name': 'Self-Rated',
            'slug': 'self_rated',
            'n_organisms': 500,
            'narrativity': 0.95,
            'coupling': 1.0,
            'r_measured': 0.594,
            'D_agency': 0.564,
            'efficiency': 0.59,
            'passes': True,
            'interpretation': 'Narrator=judge: High narrative agency'
        },
        {
            'name': 'Character',
            'slug': 'character',
            'n_organisms': 200,
            'narrativity': 0.85,
            'coupling': 0.9,
            'r_measured': 0.806,
            'D_agency': 0.617,
            'efficiency': 0.73,
            'passes': True,
            'interpretation': 'Identity construction: Narrative constructs reality'
        }
    ]
    
    return jsonify({'domains': domains, 'threshold': 0.5})


@cross_domain_bp.route('/api/domains/phylogenetic-tree')
def get_phylogenetic_data():
    """Get phylogenetic tree data for all domains."""
    
    tree = {
        'nodes': [
            {'id': 'root', 'parent': None, 'name': 'All Domains'},
            {'id': 'business', 'parent': 'root', 'name': 'Business'},
            {'id': 'performance', 'parent': 'root', 'name': 'Performance'},
            {'id': 'identity', 'parent': 'root', 'name': 'Identity'},
            {'id': 'startups', 'parent': 'business', 'name': 'Startups', 'п': 0.76, 'Д': 0.980},
            {'id': 'crypto', 'parent': 'business', 'name': 'Crypto', 'п': 0.65, 'Д': None},
            {'id': 'nba', 'parent': 'performance', 'name': 'NBA', 'п': 0.48, 'Д': None},
            {'id': 'mental_health', 'parent': 'identity', 'name': 'Mental Health', 'п': 0.85, 'Д': None}
        ]
    }
    
    return jsonify(tree)

