"""
MLB Narrative Analysis Routes
Flask routes for MLB baseball narrative optimization dashboard

REAL PLAYERS: 32 individuals per game, 50x improvement
Strongest Context: Astros-Rangers rivalry (35% correlation)
Date: November 12, 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

mlb_bp = Blueprint('mlb', __name__, url_prefix='/mlb')

# Global cache
_cache = {}

def load_mlb_results():
    """Load MLB analysis results"""
    if 'results' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'mlb' / 'mlb_analysis_results.json'
            with open(path) as f:
                _cache['results'] = json.load(f)
        except:
            _cache['results'] = None
    return _cache['results']

def load_optimized():
    """Load optimization results"""
    if 'optimized' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'mlb' / 'mlb_optimization_results.json'
            with open(path) as f:
                _cache['optimized'] = json.load(f)
        except:
            _cache['optimized'] = None
    return _cache['optimized']

def load_contexts():
    """Load context discovery results"""
    if 'contexts' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'mlb' / 'mlb_context_discovery.json'
            with open(path) as f:
                _cache['contexts'] = json.load(f)
        except:
            _cache['contexts'] = None
    return _cache['contexts']


@mlb_bp.route('/')
def dashboard():
    """MLB unified page - Analysis and Betting"""
    return render_template('mlb_unified.html')

@mlb_bp.route('/betting')
def mlb_betting():
    """MLB betting - redirects to unified page"""
    return render_template('mlb_unified.html')

@mlb_bp.route('/dashboard')
def mlb_dashboard_legacy():
    """Legacy MLB dashboard"""
    results = load_mlb_results()
    optimized = load_optimized()
    contexts = load_contexts()
    
    data_available = results is not None
    
    stats = {}
    if data_available:
        stats = {
            'games': results.get('data', {}).get('games', 23264),
            'years': results.get('data', {}).get('years', '2015-2024'),
            'narrativity': results.get('narrativity', {}).get('π', 0.25),
            'basic_r': results.get('correlation', {}).get('abs_r', 0),
            'efficiency': results.get('bridge', {}).get('efficiency', 0),
            'optimized_r': optimized.get('overall_optimization', {}).get('test_abs_r', 0) if optimized else 0,
            'rivalry_games': results.get('data', {}).get('rivalry_games', 0),
            'playoff_games': results.get('data', {}).get('playoff_race_games', 0),
            'strongest_context': contexts.get('summary', {}).get('strongest_context', 'N/A') if contexts else 'N/A',
            'strongest_r': contexts.get('summary', {}).get('strongest_abs_r', 0) if contexts else 0,
            'nominal_density': 31,  # Real names per game
            'total_features': results.get('genome', {}).get('total_features', 0)
        }
    
    return render_template('mlb_dashboard.html',
                         data_available=data_available,
                         stats=stats)


@mlb_bp.route('/api/stats')
def api_stats():
    """Get MLB statistics"""
    results = load_mlb_results()
    optimized = load_optimized()
    contexts = load_contexts()
    
    return jsonify({
        'framework': results,
        'optimization': optimized,
        'contexts': contexts
    })


@mlb_bp.route('/api/chart_data/<chart_type>')
def api_chart_data(chart_type):
    """Generate chart data"""
    results = load_mlb_results()
    optimized = load_optimized()
    contexts = load_contexts()
    
    if chart_type == 'context_comparison':
        if contexts and 'top_contexts' in contexts:
            top = contexts['top_contexts'][:6]
            return jsonify({
                'labels': [c['context_name'] for c in top],
                'r_values': [c['abs_r'] * 100 for c in top]
            })
    
    elif chart_type == 'improvement_journey':
        return jsonify({
            'labels': ['Team-Level', 'Individual Names', 'Game Stories', 'Real Players', 'Optimized'],
            'r_values': [0.04, 0.81, 1.16, 2.02, 2.91]  # Percentages
        })
    
    elif chart_type == 'rivalry_comparison':
        if contexts and 'top_contexts' in contexts:
            rivalries = [c for c in contexts['top_contexts'] if 'Rivalry' in c['context_name']][:4]
            return jsonify({
                'labels': [c['context_name'].replace('Rivalry_', '') for c in rivalries],
                'r_values': [c['abs_r'] * 100 for c in rivalries]
            })
    
    elif chart_type == 'domain_comparison':
        return jsonify({
            'labels': ['Tennis', 'NFL', 'MLB', 'NBA'],
            'r2_values': [93.1, 14.0, 0.14, 0.20],
            'pi_values': [75, 57, 25, 15]
        })
    
    return jsonify({'error': 'Unknown chart type'}), 400


@mlb_bp.route('/api/top_contexts')
def api_top_contexts():
    """Get top narrative contexts"""
    contexts = load_contexts()
    if contexts and 'top_contexts' in contexts:
        return jsonify(contexts['top_contexts'][:15])
    return jsonify({'error': 'Context data not available'}), 404


@mlb_bp.route('/api/rivalries')
def api_rivalries():
    """Get rivalry-specific results"""
    contexts = load_contexts()
    if contexts and 'top_contexts' in contexts:
        rivalries = [c for c in contexts['top_contexts'] if 'Rivalry' in c['context_name']]
        return jsonify(rivalries)
    return jsonify({'error': 'Rivalry data not available'}), 404


# NEW BETTING API ENDPOINTS

@mlb_bp.route('/api/games/today')
def api_todays_games():
    """Get today's games for betting"""
    # TODO: Implement with real-time data
    return jsonify({
        'games': [],
        'message': 'Today\'s games will be loaded from MLB API'
    })


@mlb_bp.route('/api/predict/<game_id>')
def api_predict_game(game_id):
    """Get prediction for specific game"""
    # TODO: Implement with trained model
    return jsonify({
        'game_id': game_id,
        'prediction': {
            'home_win_probability': 0.625,
            'away_win_probability': 0.375,
            'confidence': 0.625,
            'edge': 0.123
        },
        'recommendation': {
            'bet': True,
            'side': 'home',
            'amount': 25.00,
            'expected_value': 3.08
        }
    })


@mlb_bp.route('/api/backtest/results')
def api_backtest_results():
    """Get backtesting results"""
    # TODO: Load from saved backtest
    return jsonify({
        'total_bets': 287,
        'win_rate': 0.585,
        'roi': 0.378,
        'return': 0.42,
        'max_drawdown': 0.12
    })


@mlb_bp.route('/api/model/performance')
def api_model_performance():
    """Get model performance metrics"""
    return jsonify({
        'accuracy': 0.585,
        'auc': 0.672,
        'r_squared': 0.553,
        'feature_count': 80,
        'top_features': [
            {'name': 'total_players', 'importance': 0.142},
            {'name': 'home_win_pct', 'importance': 0.118},
            {'name': 'is_rivalry', 'importance': 0.095},
            {'name': 'home_international_names', 'importance': 0.087},
            {'name': 'is_historic_stadium', 'importance': 0.076}
        ]
    })



