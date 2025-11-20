"""
Oscar Routes
Flask routes for Oscar Best Picture narrative analysis dashboard
"""

from flask import Blueprint, render_template, jsonify, request
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

oscars_bp = Blueprint('oscars', __name__, url_prefix='/oscars')

# Global cache for data
_cache = {}

def load_dataset():
    """Load Oscar dataset (cached)"""
    if 'dataset' not in _cache:
        try:
            data_path = Path(__file__).parent.parent / 'data' / 'domains' / 'oscar_nominees_complete.json'
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Flatten to list of films
            films = []
            for year, year_films in raw_data.items():
                for film in year_films:
                    film['year'] = int(year)
                    films.append(film)
            
            _cache['dataset'] = films
            _cache['by_year'] = raw_data
        except:
            _cache['dataset'] = None
            _cache['by_year'] = None
    return _cache['dataset'], _cache['by_year']

def load_results():
    """Load analysis results (cached, unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('oscars')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                # Try full pipeline results first
                results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'oscars' / 'full_pipeline_results.json'
                if not results_path.exists():
                    # Fallback to old results
                    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'oscars' / 'oscar_results_complete.json'
                
                with open(results_path, 'r', encoding='utf-8') as f:
                    _cache['results'] = json.load(f)
            except:
                _cache['results'] = None
    return _cache['results']


@oscars_bp.route('/')
def dashboard():
    """Main Oscar narrative dashboard"""
    dataset, by_year = load_dataset()
    results = load_results()
    
    # Check if data available
    data_available = dataset is not None
    results_available = results is not None
    
    # Generate stats
    stats = {}
    if data_available:
        df = pd.DataFrame(dataset)
        winners = df[df['won_oscar'] == True]
        
        stats = {
            'total_films': len(dataset),
            'total_years': df['year'].nunique(),
            'total_winners': len(winners),
            'year_range': f"{df['year'].min()}-{df['year'].max()}",
            'avg_nominees_per_year': len(dataset) / df['year'].nunique()
        }
    
    if results_available:
        stats.update({
            'narrativity': results['narrativity'],
            'optimal_alpha': results['alpha_analysis']['optimal_alpha'],
            'test_accuracy': results['competitive_model']['test_accuracy'],
            'test_auc': results['competitive_model']['test_auc'],
            'D_test': results['bridge_results']['D_test']
        })
    
    return render_template('oscars_dashboard.html',
                         data_available=data_available,
                         results_available=results_available,
                         stats=stats)


@oscars_bp.route('/api/stats')
def api_stats():
    """Get dataset statistics"""
    dataset, by_year = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    df = pd.DataFrame(dataset)
    
    # Genre distribution
    genre_counts = {}
    for film in dataset:
        for genre in film.get('genres', []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    genre_dist = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Winners by year
    winners_by_year = {}
    for year, films in by_year.items():
        winner = next((f['title'] for f in films if f.get('won_oscar')), None)
        if winner:
            winners_by_year[year] = winner
    
    # Director frequency
    director_counts = {}
    for film in dataset:
        for director in film.get('director', []):
            director_counts[director] = director_counts.get(director, 0) + 1
    top_directors = dict(sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    stats = {
        'total_films': len(dataset),
        'total_years': int(df['year'].nunique()),
        'total_winners': int(df['won_oscar'].sum()),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'avg_nominees_per_year': float(len(dataset) / df['year'].nunique()),
        'genre_distribution': genre_dist,
        'winners_by_year': winners_by_year,
        'top_directors': top_directors,
        'year_distribution': df['year'].value_counts().sort_index().to_dict()
    }
    
    return jsonify(stats)


@oscars_bp.route('/api/formula')
def api_formula():
    """Get discovered formula details"""
    results = load_results()
    
    if results is None:
        return jsonify({'error': 'Results not available'}), 404
    
    formula_data = {
        'narrativity': results['narrativity'],
        'alpha_analysis': results['alpha_analysis'],
        'competitive_model': results['competitive_model'],
        'bridge_results': results['bridge_results'],
        'top_features': results['feature_importance']['top_features'][:30],
        'transformer_counts': results['transformer_counts']
    }
    
    return jsonify(formula_data)


@oscars_bp.route('/api/year/<int:year>')
def api_year_competition(year):
    """Get competitive analysis for a specific year"""
    dataset, by_year = load_dataset()
    results = load_results()
    
    if dataset is None or by_year is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Get films for this year
    year_str = str(year)
    if year_str not in by_year:
        return jsonify({'error': f'Year {year} not found'}), 404
    
    films = by_year[year_str]
    
    # Find winner
    winner = next((f for f in films if f.get('won_oscar')), None)
    nominees = [f for f in films if not f.get('won_oscar')]
    
    # Get gravitational clustering data if available
    gravitational_data = {}
    if results and 'gravitational_clustering' in results:
        gravitational_data = results['gravitational_clustering'].get(str(year), {})
    
    # Get temporal prediction if available
    temporal_data = {}
    if results and 'temporal_trends' in results:
        temporal_data = results['temporal_trends'].get(str(year), {})
    
    competition_data = {
        'year': year,
        'num_nominees': len(films),
        'winner': {
            'title': winner['title'],
            'director': winner.get('director', []),
            'cast': [c.get('actor', '') for c in winner.get('cast', [])[:10]],
            'genres': winner.get('genres', []),
            'overview': winner.get('overview', '')
        } if winner else None,
        'nominees': [
            {
                'title': f['title'],
                'director': f.get('director', []),
                'cast': [c.get('actor', '') for c in f.get('cast', [])[:5]],
                'genres': f.get('genres', [])
            }
            for f in nominees
        ],
        'gravitational_analysis': gravitational_data,
        'prediction': temporal_data
    }
    
    return jsonify(competition_data)


@oscars_bp.route('/api/winners')
def api_winners():
    """Get all winners"""
    dataset, by_year = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    winners = []
    for year, films in sorted(by_year.items()):
        winner = next((f for f in films if f.get('won_oscar')), None)
        if winner:
            winners.append({
                'year': int(year),
                'title': winner['title'],
                'director': winner.get('director', []),
                'cast': [c.get('actor', '') for c in winner.get('cast', [])[:5]],
                'genres': winner.get('genres', []),
                'num_competitors': len(films) - 1
            })
    
    return jsonify({'winners': winners, 'count': len(winners)})


@oscars_bp.route('/api/nominees')
def api_all_nominees():
    """Get all nominees (optionally filtered)"""
    dataset, by_year = load_dataset()
    
    if dataset is None:
        return jsonify({'error': 'Data not available'}), 404
    
    # Optional filters
    year = request.args.get('year', type=int)
    won = request.args.get('won', type=str)
    
    filtered = dataset
    
    if year:
        filtered = [f for f in filtered if f['year'] == year]
    
    if won is not None:
        won_bool = won.lower() == 'true'
        filtered = [f for f in filtered if f.get('won_oscar') == won_bool]
    
    # Format results
    results = [
        {
            'year': f['year'],
            'title': f['title'],
            'original_title': f.get('original_title', f['title']),
            'won_oscar': f.get('won_oscar', False),
            'director': f.get('director', []),
            'genres': f.get('genres', []),
            'cast_count': len(f.get('cast', []))
        }
        for f in filtered
    ]
    
    return jsonify({'nominees': results, 'count': len(results)})


@oscars_bp.route('/api/gravitational')
def api_gravitational_analysis():
    """Get gravitational clustering analysis"""
    results = load_results()
    
    if results is None or 'gravitational_clustering' not in results:
        return jsonify({'error': 'Analysis not available'}), 404
    
    return jsonify(results['gravitational_clustering'])


@oscars_bp.route('/api/temporal')
def api_temporal_trends():
    """Get temporal trends analysis"""
    results = load_results()
    
    if results is None or 'temporal_trends' not in results:
        return jsonify({'error': 'Analysis not available'}), 404
    
    return jsonify(results['temporal_trends'])


@oscars_bp.route('/year/<int:year>')
def year_page(year):
    """Year-specific competition page"""
    dataset, by_year = load_dataset()
    
    if dataset is None or by_year is None:
        return "Data not available", 404
    
    # Check if year exists
    if str(year) not in by_year:
        return f"Year {year} not found", 404
    
    return render_template('oscars_year.html', year=year)


@oscars_bp.route('/winners')
def winners_page():
    """All winners page"""
    return render_template('oscars_winners.html')


@oscars_bp.route('/competitive')
def competitive_analysis():
    """Competitive dynamics visualization"""
    return render_template('oscars_competitive.html')


@oscars_bp.route('/formula')
def formula():
    """Detailed formula page"""
    results = load_results()
    
    if results is None:
        return "Results not available", 404
    
    return render_template('oscars_formula.html', results=results)


@oscars_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Live Oscar winner predictor"""
    if request.method == 'POST':
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Load predictor
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
        
        from domains.oscars.quick_predictor import OscarPredictor
        
        predictor = OscarPredictor()
        if predictor.load_model():
            result = predictor.predict(text, breakdown=True)
            return jsonify(result)
        else:
            return jsonify({'error': 'Model not trained'}), 500
    
    return render_template('oscars_predict.html')

