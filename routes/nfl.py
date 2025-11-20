"""
NFL Narrative Analysis Routes
Flask routes for NFL narrative optimization dashboard

Complete framework analysis with optimization
Date: November 10, 2025
"""

from flask import Blueprint, render_template, jsonify, request
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

try:
    from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data
except ImportError:
    # Fallback: create dummy functions
    def load_unified_results(domain): return {}
    def extract_stats_from_results(results): return {}
    def get_chart_data(results): return []

try:
    from utils.market_edge_calculator import MarketEdgeCalculator
except ImportError:
    MarketEdgeCalculator = None

try:
    from utils.nfl_betting_engine import (
        discover_live_opportunities,
        evaluate_game,
        get_patterns,
        load_historical_games,
    )
except ImportError:
    # Fallback functions
    def discover_live_opportunities(**kwargs): return []
    def evaluate_game(**kwargs): return {}
    def get_patterns(): return {'patterns': [], 'summary': {}, 'baseline_ats': 0.50}
    def load_historical_games(): return []

nfl_bp = Blueprint('nfl', __name__, url_prefix='/nfl')

# Global cache
_cache = {}
    
def load_processing_summary():
    """Load transformer processing summary for NFL."""
    if 'processing_summary' not in _cache:
        summary_path = features_dir / 'nfl_processing_results.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                _cache['processing_summary'] = json.load(f)
        else:
            _cache['processing_summary'] = {}
    return _cache['processing_summary']


def get_team_catalog(limit: int = 64):
    """Return sorted list of teams for UI dropdowns."""
    if 'teams' in _cache:
        return _cache['teams']
    games = load_historical_games()
    teams = set()
    for game in games:
        if len(teams) >= limit:
            break
        home = game.get('home_team')
        away = game.get('away_team')
        if home:
            teams.add(home.upper())
        if away:
            teams.add(away.upper())
    team_list = sorted(teams)
    _cache['teams'] = team_list
    return team_list
project_root = Path(__file__).parent.parent
features_dir = project_root / 'narrative_optimization' / 'data' / 'features'
context_results_dir = (
    project_root / 'narrative_optimization' / 'results' / 'context_stratification'
)
EDGE_CALCULATOR = MarketEdgeCalculator(min_edge=0.04)

def load_nfl_results():
    """Load NFL complete results (unified format preferred)"""
    if 'results' not in _cache:
        # Try unified format first
        unified_results = load_unified_results('nfl')
        if unified_results:
            _cache['results'] = unified_results
        else:
            # Fallback to legacy format
            try:
                results_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nfl' / 'nfl_results.json'
                with open(results_path, 'r') as f:
                    _cache['results'] = json.load(f)
            except Exception as e:
                print(f"Error loading NFL results: {e}")
                _cache['results'] = None
    return _cache['results']

def load_optimized_formula():
    """Load optimized formula results"""
    if 'optimized' not in _cache:
        try:
            opt_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nfl' / 'nfl_optimized_formula.json'
            with open(opt_path, 'r') as f:
                _cache['optimized'] = json.load(f)
        except Exception as e:
            print(f"Error loading optimized formula: {e}")
            _cache['optimized'] = None
    return _cache['optimized']

def load_context_discoveries():
    """Load context discovery results"""
    if 'contexts' not in _cache:
        try:
            ctx_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nfl' / 'nfl_context_discoveries.json'
            with open(ctx_path, 'r') as f:
                _cache['contexts'] = json.load(f)
        except Exception as e:
            print(f"Error loading context discoveries: {e}")
            _cache['contexts'] = None
    return _cache['contexts']

def load_betting_analysis():
    """Load betting edge analysis"""
    if 'betting' not in _cache:
        try:
            bet_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nfl' / 'nfl_betting_edge_results.json'
            with open(bet_path, 'r') as f:
                _cache['betting'] = json.load(f)
        except Exception as e:
            print(f"Error loading betting analysis: {e}")
            _cache['betting'] = None
    return _cache['betting']


def load_context_stratification() -> Dict:
    if 'nfl_context_patterns' in _cache:
        return _cache['nfl_context_patterns']
    path = context_results_dir / 'nfl_contexts.json'
    if not path.exists():
        _cache['nfl_context_patterns'] = {}
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    _cache['nfl_context_patterns'] = data
    return data


def _build_edge_rows_from_opportunities(opportunities: List[dict]) -> List[dict]:
    rows = []
    for opp in opportunities:
        cover_prob = opp.get('cover_probability', 0.5)
        pattern_name = (opp.get('best_pattern') or {}).get('name', '')
        edges = EDGE_CALCULATOR.evaluate_matchup(
            cover_prob,
            -110,
            110,
            context_label=pattern_name,
            pi_effective=None,
        )
        rows.append(
            {
                'matchup': opp.get('matchup'),
                'confidence': opp.get('confidence'),
                'spread': opp.get('spread'),
                'home': edges['home'],
                'away': edges['away'],
                'pattern': opp.get('best_pattern'),
            }
        )
    return rows


@nfl_bp.route('/')
def dashboard():
    """NFL unified page - Analysis and Betting in one place."""
    processing = load_processing_summary()
    patterns = get_patterns()
    top_patterns = sorted(
        [p for p in patterns.get('patterns', []) if p.get('profitable')],
        key=lambda x: x.get('roi_pct', 0),
        reverse=True
    )[:5]
    live_snapshot = discover_live_opportunities(limit=3, season=2025, min_week=12)
    context_data = load_context_stratification()
    context_cards = (context_data.get('patterns') or [])[:4]
    edge_rows = _build_edge_rows_from_opportunities(live_snapshot)
    
    stats = {
        'model_accuracy': 0.607,
        'ats_roi': patterns.get('summary', {}).get('best_pattern', {}).get('roi_pct', 0),
        'games_analyzed': processing.get('sample_size', 0),
        'features': processing.get('total_features', 0),
        'transformers': processing.get('transformers_completed', 0),
        'pi': 0.57,
        'baseline_ats': patterns.get('baseline_ats', 0.58),
        'best_pattern': patterns.get('summary', {}).get('best_pattern', {}),
    }
    
    return render_template(
        'nfl_unified.html',
        stats=stats,
        top_patterns=top_patterns,
        live_snapshot=live_snapshot,
        processing=processing,
        patterns_summary=patterns.get('summary', {}),
        team_options=get_team_catalog(),
        context_cards=context_cards,
        edge_rows=edge_rows,
    )

@nfl_bp.route('/betting')
def nfl_betting():
    """NFL betting - redirects to unified page."""
    return render_template('nfl_unified.html')

@nfl_bp.route('/dashboard')
def nfl_dashboard_legacy():
    """Legacy NFL narrative dashboard"""
    results = load_nfl_results()
    optimized = load_optimized_formula()
    contexts = load_context_discoveries()
    betting = load_betting_analysis()
    
    data_available = results is not None
    
    # Generate stats
    stats = {}
    if data_available:
        # Try unified format first
        if 'pi' in results and 'analysis' in results:
            unified_stats = extract_stats_from_results(results)
            stats = {
                'total_games': unified_stats.get('n_organisms', 0),
                'seasons': '2010-2024',
                'transformers': results.get('metadata', {}).get('pipeline_info', {}).get('feature_count', 0),
                'features_extracted': unified_stats.get('n_features', 0),
                
                # Framework metrics (from unified format)
                'narrativity': unified_stats.get('pi', 0),
                'correlation': unified_stats.get('r_narrative', 0),
                'correlation_raw': unified_stats.get('r_narrative', 0),
                'pattern': 'positive' if unified_stats.get('r_narrative', 0) > 0 else 'negative',
                'bridge': unified_stats.get('Д', 0),
                'efficiency': unified_stats.get('efficiency', 0),
                'validation': 'PASS' if unified_stats.get('efficiency', 0) > 0.5 else 'FAIL',
                
                # Optimization metrics
                'basic_r2': 0.0001,
                'optimized_train_r2': optimized['performance']['train']['r2'] if optimized else 0,
                'optimized_test_r2': optimized['performance']['test']['r2'] if optimized else 0,
                'improvement': optimized['performance']['improvement'] if optimized else '0%',
                
                # Context discovery
                'contexts_measured': contexts['summary']['total_contexts_measured'] if contexts else 0,
                'strongest_context_r': contexts['summary']['strongest_context']['abs_r'] if contexts and contexts.get('summary', {}).get('strongest_context') else 0,
                
                # Betting
                'narrative_accuracy': betting['narrative_only']['accuracy'] if betting else 0,
                'odds_accuracy': betting['odds_only']['accuracy'] if betting else 0,
                'combined_accuracy': betting['combined']['accuracy'] if betting else 0,
                
                # Multi-perspective flag
                'has_comprehensive': 'comprehensive_ю' in results
            }
        else:
            # Legacy format
            framework = results.get('framework_analysis', {})
            stats = {
                'total_games': results.get('dataset', {}).get('games', 0),
                'seasons': results.get('dataset', {}).get('seasons', 'N/A'),
                'transformers': results.get('dataset', {}).get('transformers_applied', 0),
                'features_extracted': results.get('dataset', {}).get('features_extracted', 0),
                
                # Framework metrics
                'narrativity': framework.get('narrativity', {}).get('п', 0),
                'correlation': abs(framework.get('correlation', {}).get('abs_r', 0)),
                'correlation_raw': framework.get('correlation', {}).get('r', 0),
                'pattern': framework.get('correlation', {}).get('pattern', 'unknown'),
                'bridge': framework.get('bridge', {}).get('Д', 0),
                'efficiency': framework.get('bridge', {}).get('efficiency', 0),
                'validation': framework.get('validation', {}).get('result', 'UNKNOWN'),
                
                # Optimization metrics
                'basic_r2': 0.0001,
                'optimized_train_r2': optimized['performance']['train']['r2'] if optimized else 0,
                'optimized_test_r2': optimized['performance']['test']['r2'] if optimized else 0,
                'improvement': optimized['performance']['improvement'] if optimized else '0%',
                
                # Context discovery
                'contexts_measured': contexts['summary']['total_contexts_measured'] if contexts else 0,
                'strongest_context_r': contexts['summary']['strongest_context']['abs_r'] if contexts and contexts.get('summary', {}).get('strongest_context') else 0,
                
                # Betting
                'narrative_accuracy': betting['narrative_only']['accuracy'] if betting else 0,
                'odds_accuracy': betting['odds_only']['accuracy'] if betting else 0,
                'combined_accuracy': betting['combined']['accuracy'] if betting else 0,
                
                'has_comprehensive': False
            }
    
    return render_template('nfl_dashboard.html',
                         data_available=data_available,
                         stats=stats)


@nfl_bp.route('/api/framework')
def api_framework():
    """Get framework analysis data"""
    results = load_nfl_results()
    
    if not results:
        return jsonify({'error': 'Data not available'}), 404
    
    return jsonify({
        'narrativity': results['framework_analysis']['narrativity'],
        'genome': results['framework_analysis']['genome'],
        'story_quality': results['framework_analysis']['story_quality'],
        'correlation': results['framework_analysis']['correlation'],
        'bridge': results['framework_analysis']['bridge'],
        'validation': results['framework_analysis']['validation']
    })


@nfl_bp.route('/api/optimization')
def api_optimization():
    """Get optimization results"""
    optimized = load_optimized_formula()
    
    if not optimized:
        return jsonify({'error': 'Data not available'}), 404
    
    return jsonify(optimized)


@nfl_bp.route('/api/contexts')
def api_contexts():
    """Get context discovery data"""
    contexts = load_context_discoveries()
    
    if not contexts:
        return jsonify({'error': 'Data not available'}), 404
    
    # Return top 50 contexts
    return jsonify({
        'baseline': contexts['baseline'],
        'top_contexts': contexts['ranked_contexts'][:50],
        'summary': contexts['summary']
    })


@nfl_bp.route('/api/betting')
def api_betting():
    """Get betting edge analysis"""
    betting = load_betting_analysis()
    
    if not betting:
        return jsonify({'error': 'Data not available'}), 404
    
    return jsonify(betting)


@nfl_bp.route('/api/chart_data/<chart_type>')
def api_chart_data(chart_type):
    """Generate data for specific charts"""
    results = load_nfl_results()
    optimized = load_optimized_formula()
    contexts = load_context_discoveries()
    
    if not results:
        return jsonify({'error': 'Data not available'}), 404
    
    # Try unified format chart data first
    if 'comprehensive_ю' in results:
        chart_data = get_chart_data(results, chart_type)
        if chart_data:
            return jsonify(chart_data)
    
    # Fallback to legacy chart types
    if chart_type == 'narrativity_components':
        # Narrativity breakdown
        components = results['framework_analysis']['narrativity']['components']
        return jsonify({
            'labels': ['Structural', 'Temporal', 'Agency', 'Interpretation', 'Format'],
            'values': [
                components['п_structural'],
                components['п_temporal'],
                components['п_agency'],
                components['п_interpretation'],
                components['п_format']
            ]
        })
    
    elif chart_type == 'optimization_comparison':
        # Basic vs Optimized
        return jsonify({
            'labels': ['Basic Measurement', 'Optimized (Train)', 'Optimized (Test)', 'Movie Domain'],
            'r2_values': [
                0.01,
                optimized['performance']['train']['r2'] * 100 if optimized else 0,
                optimized['performance']['test']['r2'] * 100 if optimized else 0,
                59.7
            ]
        })
    
    elif chart_type == 'top_contexts':
        # Top 20 contexts
        if not contexts:
            return jsonify({'error': 'Context data not available'}), 404
        
        top_20 = contexts['ranked_contexts'][:20]
        return jsonify({
            'labels': [ctx['context_value'][:30] for ctx in top_20],
            'abs_r': [ctx['abs_r'] for ctx in top_20],
            'n': [ctx['n'] for ctx in top_20]
        })
    
    elif chart_type == 'golden_ratio':
        # Feature type distribution
        if not optimized:
            return jsonify({'error': 'Optimization data not available'}), 404
        
        ratio = optimized['golden_ratio']
        # Sort by count
        sorted_ratio = sorted(ratio.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'labels': [k for k, v in sorted_ratio],
            'values': [v for k, v in sorted_ratio]
        })
    
    elif chart_type == 'betting_comparison':
        # Betting model comparison
        betting = load_betting_analysis()
        if not betting:
            return jsonify({'error': 'Betting data not available'}), 404
        
        return jsonify({
            'labels': ['Narrative Only', 'Odds Only', 'Combined', 'Baseline'],
            'accuracies': [
                betting['narrative_only']['accuracy'] * 100,
                betting['odds_only']['accuracy'] * 100,
                betting['combined']['accuracy'] * 100,
                betting['baseline']['home_win_rate'] * 100
            ]
        })
    
    else:
        return jsonify({'error': 'Unknown chart type'}), 400


@nfl_bp.route('/api/perspectives')
def api_perspectives():
    """Get multi-perspective ю scores"""
    results = load_nfl_results()
    if results and 'comprehensive_ю' in results:
        perspectives = results['comprehensive_ю'].get('ю_perspectives', {})
        return jsonify(perspectives)
    return jsonify({'error': 'Perspective data not available'}), 404


@nfl_bp.route('/api/methods')
def api_methods():
    """Get multi-method ю scores"""
    results = load_nfl_results()
    if results and 'comprehensive_ю' in results:
        methods = results['comprehensive_ю'].get('ю_methods', {})
        return jsonify(methods)
    return jsonify({'error': 'Method data not available'}), 404


@nfl_bp.route('/api/scales')
def api_scales():
    """Get multi-scale ю scores"""
    results = load_nfl_results()
    if results and 'comprehensive_ю' in results:
        scales = results['comprehensive_ю'].get('ю_scales', {})
        return jsonify(scales)
    return jsonify({'error': 'Scale data not available'}), 404


@nfl_bp.route('/api/comprehensive')
def api_comprehensive():
    """Get full comprehensive analysis"""
    results = load_nfl_results()
    if results and 'comprehensive_ю' in results:
        return jsonify(results['comprehensive_ю'])
    return jsonify({'error': 'Comprehensive data not available'}), 404


@nfl_bp.route('/findings')
def findings():
    """Detailed findings page"""
    results = load_nfl_results()
    optimized = load_optimized_formula()
    
    if not results:
        return render_template('error.html', message='NFL data not yet analyzed')
    
    return render_template('nfl_findings.html',
                         results=results,
                         optimized=optimized)


@nfl_bp.route('/contexts')
def contexts_explorer():
    """Interactive context exploration"""
    contexts = load_context_discoveries()
    
    if not contexts:
        return render_template('error.html', message='Context data not available')
    
    return render_template('nfl_contexts.html',
                         baseline=contexts['baseline'],
                         top_contexts=contexts['ranked_contexts'][:100])


@nfl_bp.route('/optimization')
def optimization_details():
    """Optimization analysis details"""
    optimized = load_optimized_formula()
    
    if not optimized:
        return render_template('error.html', message='Optimization data not available')
    
    return render_template('nfl_optimization.html',
                         optimized=optimized)


@nfl_bp.route('/api/upcoming-games')
def api_upcoming_games():
    """Return upcoming NFL games (prefer live data if available)."""
    games = []
    source = 'historical'
    try:
        import nfl_data_py as nfl
        current_year = datetime.now().year
        schedule = nfl.import_schedules([current_year])
        today = datetime.now()
        horizon = today + timedelta(days=7)
        mask = (schedule['gameday'] >= today.strftime('%Y-%m-%d')) & (
            schedule['gameday'] <= horizon.strftime('%Y-%m-%d')
        )
        upcoming = schedule[mask]
        for _, row in upcoming.iterrows():
            games.append({
                'game_id': row.get('game_id'),
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'gameday': row.get('gameday'),
                'week': int(row.get('week', 0)),
                'spread_line': 0,
            })
        if games:
            source = 'nfl_data_py'
    except Exception:
        games = []
    
    if not games:
        historical = load_historical_games()
        latest_season = max((g.get('season', 0) for g in historical), default=2024)
        games = [
            {
                'game_id': g.get('game_id'),
                'home_team': g.get('home_team'),
                'away_team': g.get('away_team'),
                'gameday': g.get('gameday'),
                'week': g.get('week'),
                'spread_line': g.get('spread_line', 0),
            }
            for g in historical
            if g.get('season') == latest_season and g.get('week', 0) >= 10
        ][:20]
    
    return jsonify({
        'games': games,
        'count': len(games),
        'source': source,
        'fetched_at': datetime.now().isoformat()
    })


def _build_game_payload(data: dict) -> dict:
    """Normalize arbitrary payload for evaluation."""
    payload = {
        'game_id': data.get('game_id'),
        'home_team': data.get('home_team'),
        'away_team': data.get('away_team'),
        'spread_line': data.get('spread_line', data.get('spread', 0)),
        'week': data.get('week', 0),
        'playoff': data.get('playoff'),
        'division_game': data.get('division_game'),
        'rivalry': data.get('rivalry'),
        'primetime': data.get('primetime'),
        'late_season': data.get('late_season'),
        'matchup_history': data.get('matchup_history'),
        'home_record_before': data.get('home_record') or data.get('home_record_before'),
        'away_record_before': data.get('away_record') or data.get('away_record_before'),
        'betting_odds': data.get('betting_odds'),
        'context': data.get('context'),
    }
    if 'rivalry_games' in data and payload['matchup_history'] is None:
        payload['matchup_history'] = {'total_games': data['rivalry_games']}
    return payload


@nfl_bp.route('/api/predict-game', methods=['POST'])
def api_predict_game():
    """Predict ATS + prop edges for any NFL matchup."""
    data = request.get_json() or {}
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    if not home_team or not away_team:
        return jsonify({'error': 'home_team and away_team are required'}), 400
    
    analysis = evaluate_game(_build_game_payload(data))
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'generated_at': datetime.now().isoformat()
    })


@nfl_bp.route('/api/live-eval', methods=['POST'])
def api_live_eval():
    """Alias to run evaluation for live betting style payloads."""
    return api_predict_game()


@nfl_bp.route('/api/live-opportunities')
def api_live_opportunities():
    """Expose top live opportunities for dynamic UI sections."""
    season = request.args.get('season', type=int) or 2025
    min_week = request.args.get('min_week', type=int) or 10
    limit = request.args.get('limit', type=int) or 10
    opportunities = discover_live_opportunities(limit=limit, season=season, min_week=min_week)
    return jsonify({
        'opportunities': opportunities,
        'count': len(opportunities),
        'season': season,
        'min_week': min_week,
        'generated_at': datetime.now().isoformat()
    })

