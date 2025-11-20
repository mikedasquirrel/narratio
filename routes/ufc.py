"""
UFC Narrative Analysis Routes
Flask routes for UFC narrative analysis dashboard

Shows UFC as a high-narrativity performance domain
Date: November 11, 2025
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

ufc_bp = Blueprint('ufc', __name__, url_prefix='/ufc')

# Global cache
_cache = {}

def load_ufc_results():
    """Load UFC comprehensive analysis results (unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('ufc')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                # Try comprehensive results first
                results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'ufc' / 'ufc_REAL_DATA_results.json'
                with open(results_path, 'r') as f:
                    _cache['results'] = json.load(f)
            except:
                try:
                    # Fallback to fast results
                    results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'ufc' / 'ufc_fast_results.json'
                    with open(results_path, 'r') as f:
                        _cache['results'] = json.load(f)
                except Exception as e:
                    print(f"Error loading UFC results: {e}")
                    _cache['results'] = None
    return _cache['results']

def load_context_discovery():
    """Load context discovery results"""
    if 'contexts' not in _cache:
        try:
            ctx_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'ufc' / 'ufc_context_discovery.json'
            with open(ctx_path, 'r') as f:
                _cache['contexts'] = json.load(f)
        except Exception as e:
            print(f"Error loading contexts: {e}")
            _cache['contexts'] = None
    return _cache['contexts']

def load_peak_contexts():
    """Load peak context analysis"""
    if 'peaks' not in _cache:
        try:
            peak_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'ufc' / 'ufc_peak_contexts.json'
            with open(peak_path, 'r') as f:
                _cache['peaks'] = json.load(f)
        except Exception as e:
            print(f"Error loading peaks: {e}")
            _cache['peaks'] = None
    return _cache['peaks']

def load_betting_model():
    """Load pre-flight betting model results"""
    if 'betting' not in _cache:
        try:
            betting_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'ufc' / 'ufc_prefight_betting_model.json'
            with open(betting_path, 'r') as f:
                _cache['betting'] = json.load(f)
        except Exception as e:
            print(f"Error loading betting model: {e}")
            _cache['betting'] = None
    return _cache['betting']


@ufc_bp.route('/')
def dashboard():
    """Main UFC narrative dashboard"""
    results = load_ufc_results()
    contexts = load_context_discovery()
    peaks = load_peak_contexts()
    betting = load_betting_model()
    
    data_available = results is not None
    
    # Generate stats
    stats = {}
    if data_available:
        # Try unified format first
        if 'pi' in results and 'analysis' in results:
            unified_stats = extract_stats_from_results(results)
            stats = {
                'total_fights': unified_stats.get('n_organisms', 7756),
                'features': unified_stats.get('n_features', 149),
                
                # Framework metrics (from unified format)
                'narrativity': unified_stats.get('pi', 0.722),
                'r_narrative': unified_stats.get('r_narrative', 0),
                'Д': unified_stats.get('Д', 0),
                'efficiency': unified_stats.get('efficiency', 0),
                'physical_auc': results.get('physical_only_auc', 0.938),
                'combined_auc': results.get('combined_auc', 0.918),
                'narrative_delta': results.get('delta', -0.020),
                
                # Context results
                'contexts_tested': 62 if contexts else 0,
                'contexts_passing': len(contexts['passing_contexts']) if contexts else 0,
                'pass_rate': len(contexts['passing_contexts']) / 62 if contexts else 0,
                
                # Top contexts
                'top_context': contexts['top_by_efficiency'][0]['context'] if contexts else 'Unknown',
                'top_efficiency': contexts['top_by_efficiency'][0]['efficiency'] if contexts else 0,
                
                # Betting model
                'betting_valid': betting is not None,
                'betting_auc': betting['performance']['full_model_auc'] if betting else 0,
                'betting_edge': betting['betting']['edge'] * 100 if betting else 0,
                'betting_roi': betting['betting']['profitable'] if betting else False,
                'betting_coverage': betting['coverage'] * 100 if betting else 0,
                
                # Comparisons
                'nba_narrativity': 0.49,
                'nfl_narrativity': 0.48,
                'ufc_multiplier': unified_stats.get('pi', 0.722) / 0.49,
                
                # Multi-perspective flag
                'has_comprehensive': 'comprehensive_ю' in results
            }
        else:
            # Legacy format
            stats = {
                'total_fights': results.get('total_fights', results.get('total_dataset', 7756)),
                'features': 149,
                
                # Framework metrics
                'narrativity': 0.722,
                'physical_auc': results.get('physical_only_auc', 0.938),
                'combined_auc': results.get('combined_auc', 0.918),
                'narrative_delta': results.get('delta', -0.020),
                
                # Context results
                'contexts_tested': 62 if contexts else 0,
                'contexts_passing': len(contexts['passing_contexts']) if contexts else 0,
                'pass_rate': len(contexts['passing_contexts']) / 62 if contexts else 0,
                
                # Top contexts
                'top_context': contexts['top_by_efficiency'][0]['context'] if contexts else 'Unknown',
                'top_efficiency': contexts['top_by_efficiency'][0]['efficiency'] if contexts else 0,
                
                # Betting model
                'betting_valid': betting is not None,
                'betting_auc': betting['performance']['full_model_auc'] if betting else 0,
                'betting_edge': betting['betting']['edge'] * 100 if betting else 0,
                'betting_roi': betting['betting']['profitable'] if betting else False,
                'betting_coverage': betting['coverage'] * 100 if betting else 0,
                
                # Comparisons
                'nba_narrativity': 0.49,
                'nfl_narrativity': 0.48,
                'ufc_multiplier': 0.722 / 0.49,
                
                'has_comprehensive': False
            }
    
    return render_template('ufc_dashboard.html', 
                         data_available=data_available,
                         stats=stats,
                         results=results,
                         contexts=contexts,
                         peaks=peaks,
                         betting=betting)


@ufc_bp.route('/betting')
def betting():
    """UFC betting model page"""
    betting_results = load_betting_model()
    contexts = load_context_discovery()
    
    return render_template('ufc_betting.html',
                         betting=betting_results,
                         contexts=contexts)

@ufc_bp.route('/api/stats')
def api_stats():
    """API endpoint for UFC statistics"""
    results = load_ufc_results()
    
    if results is None:
        return jsonify({'error': 'No results available'}), 404
    
    return jsonify(results)

@ufc_bp.route('/api/perspectives')
def api_perspectives():
    """Get multi-perspective ю scores"""
    results = load_ufc_results()
    if results and 'comprehensive_ю' in results:
        perspectives = results['comprehensive_ю'].get('ю_perspectives', {})
        return jsonify(perspectives)
    return jsonify({'error': 'Perspective data not available'}), 404


@ufc_bp.route('/api/methods')
def api_methods():
    """Get multi-method ю scores"""
    results = load_ufc_results()
    if results and 'comprehensive_ю' in results:
        methods = results['comprehensive_ю'].get('ю_methods', {})
        return jsonify(methods)
    return jsonify({'error': 'Method data not available'}), 404


@ufc_bp.route('/api/scales')
def api_scales():
    """Get multi-scale ю scores"""
    results = load_ufc_results()
    if results and 'comprehensive_ю' in results:
        scales = results['comprehensive_ю'].get('ю_scales', {})
        return jsonify(scales)
    return jsonify({'error': 'Scale data not available'}), 404


@ufc_bp.route('/api/comprehensive')
def api_comprehensive():
    """Get full comprehensive analysis"""
    results = load_ufc_results()
    if results and 'comprehensive_ю' in results:
        return jsonify(results['comprehensive_ю'])
    return jsonify({'error': 'Comprehensive data not available'}), 404


@ufc_bp.route('/api/betting')
def api_betting():
    """API endpoint for betting model stats"""
    betting = load_betting_model()
    
    if betting is None:
        return jsonify({'error': 'No betting model available'}), 404
    
    return jsonify(betting)

