"""
NHL Domain Routes

Web interface for NHL narrative analysis and domain formula results.
Displays π, Δ, r, κ calculations and structure-aware validation.

Author: Narrative Integration System
Date: November 16, 2025
"""

from flask import Blueprint, render_template, jsonify, current_app
from pathlib import Path
import json

nhl_bp = Blueprint('nhl', __name__, url_prefix='/nhl')

# Paths
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data' / 'domains'
nhl_dir = project_root / 'narrative_optimization' / 'domains' / 'nhl'


def load_nhl_context():
    """Load NHL dashboard context"""
    # Import load function from app
    import sys
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import the actual load function from app.py
    try:
        import app
        context = app.load_nhl_dashboard_context()
        # Ensure the context has the 'nhl' key
        if 'nhl' not in context:
            context['nhl'] = {}
        return context
    except Exception as e:
        print(f"Error loading NHL context: {e}")
        # Return minimal context with nhl key
        return {'nhl': {'live_predictions': [], 'validation': {}, 'patterns': []}}

@nhl_bp.route('/')
def nhl_home():
    """NHL unified page - Analysis and Betting in one place."""
    nhl_context = load_nhl_context()
    return render_template('nhl_unified.html', **nhl_context)

@nhl_bp.route('/betting')
def nhl_betting():
    """NHL betting - redirects to unified page."""
    nhl_context = load_nhl_context()
    return render_template('nhl_unified.html', **nhl_context)

@nhl_bp.route('/results')
def nhl_results():
    """NHL analysis results page"""
    
    # Load formula results
    formula_path = nhl_dir / 'nhl_formula_results.json'
    
    if formula_path.exists():
        with open(formula_path, 'r') as f:
            formula_results = json.load(f)
    else:
        formula_results = {
            'domain': 'nhl',
            'formula': {
                'pi': 0.52,
                'r': 0.0,
                'kappa': 0.75,
                'delta': 0.0,
                'efficiency': 0.0,
                'narrative_matters': False,
            },
            'message': 'Run calculate_nhl_formula.py to generate results'
        }
    
    return render_template('nhl_results.html', results=formula_results)


@nhl_bp.route('/api/formula')
def nhl_formula_api():
    """API endpoint for NHL formula"""
    
    formula_path = nhl_dir / 'nhl_formula_results.json'
    
    if formula_path.exists():
        with open(formula_path, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({
            'error': 'Formula results not found',
            'message': 'Run calculate_nhl_formula.py first'
        }), 404


@nhl_bp.route('/data')
def nhl_data():
    """NHL data overview"""
    
    data_path = data_dir / 'nhl_games_with_odds.json'
    
    if data_path.exists():
        with open(data_path, 'r') as f:
            games = json.load(f)
        
        # Calculate statistics
        stats = {
            'total_games': len(games),
            'seasons': len(set(g.get('season', '') for g in games)),
            'playoff_games': sum(1 for g in games if g.get('is_playoff', False)),
            'rivalry_games': sum(1 for g in games if g.get('is_rivalry', False)),
            'overtime_games': sum(1 for g in games if g.get('overtime', False)),
            'shootout_games': sum(1 for g in games if g.get('shootout', False)),
            'avg_goals': sum(g.get('total_goals', 0) for g in games) / len(games) if games else 0,
        }
        
        return render_template('nhl_data.html', stats=stats, sample_games=games[:10])
    else:
        return render_template('nhl_data.html', stats=None, sample_games=[])


@nhl_bp.route('/features')
def nhl_features():
    """NHL feature extraction overview"""
    
    metadata_path = nhl_dir / 'nhl_features_metadata.json'
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return render_template('nhl_features.html', metadata=metadata)
    else:
        return render_template('nhl_features.html', metadata=None)

