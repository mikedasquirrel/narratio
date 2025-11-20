"""
Music/Spotify Narrative Analysis Routes
Flask routes for music narrative optimization dashboard

π = 0.702 (Mid-High Narrativity)
Date: November 12, 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

music_bp = Blueprint('music', __name__, url_prefix='/music')

# Global cache
_cache = {}

def load_music_results():
    """Load music analysis results (unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('music')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'music' / 'music_results.json'
                with open(path) as f:
                    _cache['results'] = json.load(f)
            except Exception as e:
                print(f"Error loading music results: {e}")
                _cache['results'] = None
    return _cache['results']

def load_narrativity():
    """Load narrativity calculation"""
    if 'narrativity' not in _cache:
        try:
            path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'music' / 'music_narrativity.json'
            with open(path) as f:
                _cache['narrativity'] = json.load(f)
        except Exception as e:
            print(f"Error loading music narrativity: {e}")
            _cache['narrativity'] = None
    return _cache['narrativity']


@music_bp.route('/')
def dashboard():
    """Main music dashboard"""
    results = load_music_results()
    narrativity = load_narrativity()
    
    data_available = results is not None
    
    stats = {}
    if data_available:
        stats = {
            'songs': results.get('songs', 50000),
            'narrativity': results.get('π', 0.702),
            'basic_r': results.get('basic_r', 0),
            'arch': results.get('Д', 0),
            'efficiency': results.get('efficiency', 0),
            'passes': results.get('passes_threshold', False),
            'optimized_r2': results['optimized']['test_r2'] if 'optimized' in results else 0,
            'genre_effects': results.get('genre_specific', {}),
            'best_genre': max(results.get('genre_specific', {}).items(), 
                            key=lambda x: x[1]['r2'])[0] if results.get('genre_specific') else 'country',
            'best_genre_r2': max(results.get('genre_specific', {}).items(), 
                               key=lambda x: x[1]['r2'])[1]['r2'] if results.get('genre_specific') else 0
        }
    
    return render_template('music_dashboard.html',
                         data_available=data_available,
                         stats=stats,
                         narrativity=narrativity)


@music_bp.route('/api/stats')
def api_stats():
    """Get music statistics"""
    results = load_music_results()
    narrativity = load_narrativity()
    
    return jsonify({
        'results': results,
        'narrativity': narrativity
    })


@music_bp.route('/api/genre_comparison')
def api_genre_comparison():
    """Get genre-specific comparison data"""
    results = load_music_results()
    
    if not results or 'genre_specific' not in results:
        return jsonify({'error': 'No data available'})
    
    genre_data = results['genre_specific']
    
    # Format for chart
    chart_data = {
        'labels': list(genre_data.keys()),
        'r2_values': [genre_data[g]['r2'] for g in genre_data.keys()],
        'r_values': [genre_data[g]['r'] for g in genre_data.keys()],
        'n_songs': [genre_data[g]['n_songs'] for g in genre_data.keys()]
    }
    
    return jsonify(chart_data)


