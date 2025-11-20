"""
IMDB Routes
Flask routes for IMDB/CMU Movie Summaries narrative analysis dashboard
"""

from flask import Blueprint, render_template, jsonify, request
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

imdb_bp = Blueprint('imdb', __name__, url_prefix='/imdb')

# Global cache for data
_cache = {}

def load_dataset():
    """Load IMDB dataset (cached)"""
    if 'dataset' not in _cache:
        try:
            data_path = Path(__file__).parent.parent / 'data' / 'domains' / 'imdb_movies_complete.json'
            with open(data_path, 'r', encoding='utf-8') as f:
                _cache['dataset'] = json.load(f)
        except:
            _cache['dataset'] = None
    return _cache['dataset']

def load_results():
    """Load analysis results (cached, unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('imdb')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                # Try full pipeline results first
                results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'imdb' / 'full_pipeline_results.json'
                if not results_path.exists():
                    # Fallback
                    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'imdb' / 'imdb_results.json'
                
                with open(results_path, 'r', encoding='utf-8') as f:
                    _cache['results'] = json.load(f)
            except:
                _cache['results'] = None
    return _cache['results']


@imdb_bp.route('/')
def dashboard():
    """Main IMDB narrative dashboard"""
    dataset = load_dataset()
    results = load_results()
    
    # Check if data available
    data_available = dataset is not None
    results_available = results is not None
    
    # Generate stats
    stats = {}
    if data_available:
        df = pd.DataFrame(dataset)
        stats = {
            'total_movies': len(dataset),
            'year_range': f"{df['release_year'].min()}-{df['release_year'].max()}",
            'genres': df['primary_genre'].nunique(),
            'avg_box_office': f"${df['box_office_revenue'].mean():,.0f}",
            'with_cast': sum(1 for m in dataset if m['num_actors'] > 0)
        }
    
    if results_available:
        stats.update({
            'narrativity': results['narrativity'],
            'optimal_alpha': results['alpha_analysis']['optimal_alpha'],
            'r_test': results['bridge_results']['r_narrative_test'],
            'D_test': results['bridge_results']['D_test'],
            'R2_test': results['bridge_results']['R2_test']
        })
    
    return render_template('imdb_dashboard.html',
                         data_available=data_available,
                         results_available=results_available,
                         stats=stats)


@imdb_bp.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    dataset = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    df = pd.DataFrame(dataset)
    
    # Genre distribution
    genre_dist = df['primary_genre'].value_counts().head(15).to_dict()
    
    # Year distribution
    year_dist = df['release_year'].value_counts().sort_index().to_dict()
    
    # Decade distribution
    decade_dist = df['decade'].value_counts().sort_index().to_dict()
    
    # Success distribution
    success_hist = np.histogram(df['success_score'].dropna(), bins=20)
    
    # Box office by genre
    genre_box_office = df.groupby('primary_genre')['box_office_revenue'].mean().sort_values(ascending=False).head(10).to_dict()
    
    stats = {
        'total_movies': len(dataset),
        'year_range': [int(df['release_year'].min()), int(df['release_year'].max())],
        'unique_genres': int(df['primary_genre'].nunique()),
        'avg_box_office': float(df['box_office_revenue'].mean()),
        'avg_runtime': float(df['runtime'].mean()),
        'avg_success_score': float(df['success_score'].mean()),
        'genre_distribution': genre_dist,
        'year_distribution': {int(k): int(v) for k, v in year_dist.items()},
        'decade_distribution': {int(k): int(v) for k, v in decade_dist.items()},
        'success_histogram': {
            'counts': success_hist[0].tolist(),
            'bins': success_hist[1].tolist()
        },
        'genre_box_office': {k: float(v) for k, v in genre_box_office.items()},
        'top_movies': df.nlargest(10, 'success_score')[
            ['title', 'release_year', 'primary_genre', 'box_office_revenue', 'success_score']
        ].to_dict('records')
    }
    
    return jsonify(stats)


@imdb_bp.route('/api/formula')
def api_formula():
    """Get discovered formula details"""
    results = load_results()
    
    if results is None:
        return jsonify({'error': 'Results not available'}), 404
    
    formula_data = {
        'narrativity': results['narrativity'],
        'alpha_analysis': results['alpha_analysis'],
        'bridge_results': results['bridge_results'],
        'top_features': results['feature_importance']['top_features'][:30],
        'genre_results': results.get('genre_results', {}),
        'transformer_counts': results['transformer_counts']
    }
    
    return jsonify(formula_data)


@imdb_bp.route('/api/search')
def api_search():
    """Search for movies"""
    query = request.args.get('q', '').lower()
    limit = request.args.get('limit', 20, type=int)
    
    dataset = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Search in titles
    matches = [
        movie for movie in dataset
        if query in movie['title'].lower()
    ]
    
    # Limit results
    matches = matches[:limit]
    
    # Format results
    results = [
        {
            'wikipedia_id': m['wikipedia_id'],
            'title': m['title'],
            'release_year': m['release_year'],
            'primary_genre': m['primary_genre'],
            'box_office_revenue': m['box_office_revenue'],
            'success_score': m['success_score'],
            'actors': m['actors'][:5]
        }
        for m in matches
    ]
    
    return jsonify({'results': results, 'count': len(results)})


@imdb_bp.route('/api/movie/<int:movie_id>')
def api_movie_details(movie_id):
    """Get details for a specific movie"""
    dataset = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Find movie by wikipedia_id
    movie = next((m for m in dataset if m['wikipedia_id'] == movie_id), None)
    
    if movie is None:
        return jsonify({'error': 'Movie not found'}), 404
    
    return jsonify(movie)


@imdb_bp.route('/api/genre/<genre_name>')
def api_genre_analysis(genre_name):
    """Get analysis for a specific genre"""
    dataset = load_dataset()
    results = load_results()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Filter by genre
    df = pd.DataFrame(dataset)
    genre_movies = df[df['primary_genre'] == genre_name]
    
    if len(genre_movies) == 0:
        return jsonify({'error': f'Genre {genre_name} not found'}), 404
    
    # Genre stats
    genre_stats = {
        'name': genre_name,
        'count': len(genre_movies),
        'avg_box_office': float(genre_movies['box_office_revenue'].mean()),
        'avg_success_score': float(genre_movies['success_score'].mean()),
        'avg_runtime': float(genre_movies['runtime'].mean()),
        'year_range': [int(genre_movies['release_year'].min()), int(genre_movies['release_year'].max())],
        'top_movies': genre_movies.nlargest(10, 'success_score')[
            ['title', 'release_year', 'box_office_revenue', 'success_score']
        ].to_dict('records')
    }
    
    # Add genre-specific results if available
    if results and 'genre_results' in results and genre_name in results['genre_results']:
        genre_stats['narrative_correlation'] = results['genre_results'][genre_name]
    
    return jsonify(genre_stats)


@imdb_bp.route('/genre/<genre_name>')
def genre_page(genre_name):
    """Genre-specific analysis page"""
    dataset = load_dataset()
    
    if dataset is None:
        return "Data not available", 404
    
    # Check if genre exists
    df = pd.DataFrame(dataset)
    if genre_name not in df['primary_genre'].values:
        return f"Genre {genre_name} not found", 404
    
    return render_template('imdb_genre.html', genre_name=genre_name)


@imdb_bp.route('/movie/<int:movie_id>')
def movie_page(movie_id):
    """Individual movie detail page"""
    dataset = load_dataset()
    
    if dataset is None:
        return "Data not available", 404
    
    # Find movie
    movie = next((m for m in dataset if m['wikipedia_id'] == movie_id), None)
    
    if movie is None:
        return "Movie not found", 404
    
    return render_template('imdb_movie.html', movie=movie)


@imdb_bp.route('/explore')
def explore():
    """Interactive movie explorer"""
    return render_template('imdb_explore.html')


@imdb_bp.route('/formula')
def formula():
    """Detailed formula page"""
    results = load_results()
    
    if results is None:
        return "Results not available", 404
    
    return render_template('imdb_formula.html', results=results)

