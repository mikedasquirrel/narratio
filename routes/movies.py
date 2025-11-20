"""
Movie Narrative Routes
Flask routes for movie narrative analysis dashboard

Narrative Optimization System
Date: November 10, 2025
"""

from flask import Blueprint, render_template, jsonify, request
import json
import pickle
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

movies_bp = Blueprint('movies', __name__, url_prefix='/movies')

# Global cache for data
_cache = {}

def load_movie_data():
    """Load movie data (cached)"""
    if 'movies' not in _cache:
        try:
            _cache['movies'] = pd.read_pickle('data/movies_with_features.pkl')
        except:
            _cache['movies'] = None
    return _cache['movies']

def load_results():
    """Load formula results (cached, unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('movies')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                with open('data/movie_formula_results.json', 'r') as f:
                    _cache['results'] = json.load(f)
            except:
                _cache['results'] = None
    return _cache['results']

def load_validation():
    """Load validation results (cached)"""
    if 'validation' not in _cache:
        try:
            with open('data/movie_validation_results.json', 'r') as f:
                _cache['validation'] = json.load(f)
        except:
            _cache['validation'] = None
    return _cache['validation']

def load_models():
    """Load trained models (cached)"""
    if 'models' not in _cache:
        try:
            with open('data/movies_formula.pkl', 'rb') as f:
                _cache['models'] = pickle.load(f)
        except:
            _cache['models'] = None
    return _cache['models']


@movies_bp.route('/')
def dashboard():
    """Main movie narrative dashboard"""
    movies = load_movie_data()
    results = load_results()
    validation = load_validation()
    
    # Check if data available
    data_available = movies is not None and results is not None
    
    # Generate stats
    stats = {}
    if data_available:
        # Try unified format first
        if 'pi' in results and 'analysis' in results:
            unified_stats = extract_stats_from_results(results)
            stats = {
                'total_movies': unified_stats.get('n_organisms', len(movies) if movies is not None else 0),
                'genres': len(movies['primary_genre'].unique()) if movies is not None else 0,
                'year_range': f"{movies['release_year'].min()}-{movies['release_year'].max()}" if movies is not None else 'N/A',
                'avg_success': f"{movies['success_score'].mean():.3f}" if movies is not None else 'N/A',
                'narrative_r2': unified_stats.get('r_narrative', 0),
                'statistical_r2': results.get('comparison', {}).get('statistical_r2', 0),
                'combined_r2': results.get('comparison', {}).get('combined_r2', 0),
                'optimal_alpha': results.get('optimal_alpha', 0),
                'cv_score': validation['cross_validation']['mean_r2'] if validation else 0,
                'pi': unified_stats.get('pi', 0),
                'Д': unified_stats.get('Д', 0),
                'efficiency': unified_stats.get('efficiency', 0),
                'has_comprehensive': 'comprehensive_ю' in results
            }
        else:
            # Legacy format
            stats = {
                'total_movies': len(movies) if movies is not None else 0,
                'genres': len(movies['primary_genre'].unique()) if movies is not None else 0,
                'year_range': f"{movies['release_year'].min()}-{movies['release_year'].max()}" if movies is not None else 'N/A',
                'avg_success': f"{movies['success_score'].mean():.3f}" if movies is not None else 'N/A',
                'narrative_r2': results.get('comparison', {}).get('narrative_r2', 0),
                'statistical_r2': results.get('comparison', {}).get('statistical_r2', 0),
                'combined_r2': results.get('comparison', {}).get('combined_r2', 0),
                'optimal_alpha': results.get('optimal_alpha', 0),
                'cv_score': validation['cross_validation']['mean_r2'] if validation else 0,
                'has_comprehensive': False
            }
    
    return render_template('movies_dashboard.html',
                         data_available=data_available,
                         stats=stats)


@movies_bp.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    movies = load_movie_data()
    
    if movies is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Genre distribution
    genre_dist = movies['primary_genre'].value_counts().head(10).to_dict()
    
    # Year distribution
    year_dist = movies.groupby('release_year').size().to_dict()
    
    # Weight class distribution
    weight_dist = movies['weight_class'].value_counts().to_dict()
    
    # Success distribution
    success_hist = np.histogram(movies['success_score'].dropna(), bins=20)
    
    stats = {
        'total_movies': len(movies),
        'genre_distribution': genre_dist,
        'year_distribution': year_dist,
        'weight_distribution': weight_dist,
        'success_histogram': {
            'counts': success_hist[0].tolist(),
            'bins': success_hist[1].tolist()
        },
        'top_movies': movies.nlargest(10, 'success_score')[
            ['title', 'success_score', 'roi', 'avg_rating', 'primary_genre']
        ].to_dict('records')
    }
    
    return jsonify(stats)


@movies_bp.route('/api/formula')
def api_formula():
    """Get discovered formula details"""
    results = load_results()
    validation = load_validation()
    
    if results is None:
        return jsonify({'error': 'Results not available'}), 404
    
    formula_data = {
        'comparison': results['comparison'],
        'optimal_alpha': results['optimal_alpha'],
        'top_narrative_features': list(zip(
            results['feature_importance']['narrative_features'][:20],
            results['feature_importance']['narrative_importance'][:20]
        )),
        'top_statistical_features': list(zip(
            results['feature_importance']['statistical_features'][:10],
            results['feature_importance']['statistical_importance'][:10]
        )),
        'genre_performance': results.get('genre_results', {}),
        'validation': validation if validation else {}
    }
    
    return jsonify(formula_data)


@movies_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict success for a movie description"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # Load models and transformers
    models = load_models()
    
    if models is None:
        return jsonify({'error': 'Models not available'}), 404
    
    # TODO: Extract features from text using transformers
    # For now, return placeholder
    
    prediction = {
        'success_score': 0.5,
        'confidence': 0.3,
        'narrative_score': 0.45,
        'statistical_score': 0.55,
        'interpretation': 'Analysis requires feature extraction'
    }
    
    return jsonify(prediction)


@movies_bp.route('/api/compare')
def api_compare():
    """Compare two movies"""
    movie_id_1 = request.args.get('id1', type=int)
    movie_id_2 = request.args.get('id2', type=int)
    
    movies = load_movie_data()
    
    if movies is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Find movies
    movie1 = movies[movies['tmdb_id'] == movie_id_1]
    movie2 = movies[movies['tmdb_id'] == movie_id_2]
    
    if movie1.empty or movie2.empty:
        return jsonify({'error': 'Movies not found'}), 404
    
    # Extract comparison data
    comparison = {
        'movie1': {
            'title': movie1.iloc[0]['title'],
            'success_score': float(movie1.iloc[0]['success_score']),
            'roi': float(movie1.iloc[0]['roi']),
            'rating': float(movie1.iloc[0]['avg_rating']),
            'genre': movie1.iloc[0]['primary_genre']
        },
        'movie2': {
            'title': movie2.iloc[0]['title'],
            'success_score': float(movie2.iloc[0]['success_score']),
            'roi': float(movie2.iloc[0]['roi']),
            'rating': float(movie2.iloc[0]['avg_rating']),
            'genre': movie2.iloc[0]['primary_genre']
        }
    }
    
    return jsonify(comparison)


@movies_bp.route('/api/search')
def api_search():
    """Search for movies"""
    query = request.args.get('q', '').lower()
    limit = request.args.get('limit', 10, type=int)
    
    movies = load_movie_data()
    
    if movies is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Search in titles
    matches = movies[movies['title'].str.lower().str.contains(query, na=False)]
    
    # Limit results
    matches = matches.head(limit)
    
    results = matches[['tmdb_id', 'title', 'release_year', 'primary_genre', 'success_score']].to_dict('records')
    
    return jsonify({'results': results})


@movies_bp.route('/genre/<genre_name>')
def genre_analysis(genre_name):
    """Genre-specific analysis page"""
    movies = load_movie_data()
    results = load_results()
    
    if movies is None or results is None:
        return render_template('error.html', error='Data not available')
    
    # Filter by genre
    genre_movies = movies[movies['primary_genre'] == genre_name]
    
    if genre_movies.empty:
        return render_template('error.html', error=f'Genre {genre_name} not found')
    
    # Genre stats
    genre_stats = {
        'name': genre_name,
        'count': len(genre_movies),
        'avg_success': float(genre_movies['success_score'].mean()),
        'avg_roi': float(genre_movies['roi'].mean()),
        'avg_rating': float(genre_movies['avg_rating'].mean()),
        'top_movies': genre_movies.nlargest(5, 'success_score')[
            ['title', 'success_score', 'release_year']
        ].to_dict('records')
    }
    
    # Genre-specific performance
    genre_performance = results.get('genre_results', {}).get(genre_name, {})
    
    return render_template('genre_analysis.html',
                         genre_stats=genre_stats,
                         genre_performance=genre_performance)


@movies_bp.route('/formula')
def formula_page():
    """Detailed formula explanation page"""
    results = load_results()
    validation = load_validation()
    
    if results is None:
        return render_template('error.html', error='Formula results not available')
    
    return render_template('movie_formula.html',
                         results=results,
                         validation=validation)


@movies_bp.route('/explore')
def explore():
    """Interactive movie explorer"""
    return render_template('movie_explore.html')

