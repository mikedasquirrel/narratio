"""
NBA Live Betting Dashboard Routes
===================================

Real-time betting predictions interface for NBA games.
Shows high-confidence picks, model reasoning, and performance tracking.

Author: AI Coding Assistant
Date: November 16, 2025
"""

from flask import Blueprint, render_template, jsonify, request
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob

nba_betting_live_bp = Blueprint('nba_betting_live', __name__, url_prefix='/nba/betting')


@nba_betting_live_bp.route('/live')
def betting_live():
    """Main live betting dashboard with upcoming games"""
    
    # Try both optimized and regular predictions
    today = datetime.now().strftime('%Y%m%d')
    
    # Check for optimized predictions first (pattern-enhanced)
    pred_path_optimized = Path(f'data/predictions/nba_optimized_{today}.json')
    pred_path_regular = Path(f'data/predictions/nba_daily_{today}.json')
    
    predictions = None
    is_optimized = False
    
    if pred_path_optimized.exists():
        with open(pred_path_optimized) as f:
            predictions = json.load(f)
        is_optimized = True
    elif pred_path_regular.exists():
        with open(pred_path_regular) as f:
            predictions = json.load(f)
    
    # Load model summary (try optimized first)
    model_summary = None
    summary_path_opt = Path('narrative_optimization/betting/nba_ensemble_summary.json')
    if summary_path_opt.exists():
        with open(summary_path_opt) as f:
            model_summary = json.load(f)
    
    # Load backtest results (try optimized first)
    backtest = None
    backtest_path_opt = Path('narrative_optimization/betting/nba_optimized_results.json')
    backtest_path_reg = Path('narrative_optimization/betting/nba_backtest_results.json')
    
    if backtest_path_opt.exists():
        with open(backtest_path_opt) as f:
            backtest = json.load(f)
    elif backtest_path_reg.exists():
        with open(backtest_path_reg) as f:
            backtest = json.load(f)
    
    # Load upcoming games from 2024-25 season data
    upcoming_games = None
    upcoming_path = Path('data/domains/nba_2024_2025_season.json')
    if upcoming_path.exists():
        try:
            with open(upcoming_path) as f:
                season_data = json.load(f)
            # Take first 10 as "upcoming" sample
            upcoming_games = season_data[:10] if isinstance(season_data, list) else []
        except:
            upcoming_games = []
    
    # Parse high-confidence bets to show detailed info
    high_conf_detailed = []
    if predictions and predictions.get('high_confidence_bets'):
        for bet in predictions['high_confidence_bets']:
            detailed = {
                'narrative': bet.get('narrative', 'Unknown game'),
                'win_prob': bet.get('win_probability', 0.5),
                'confidence_level': bet.get('confidence_level', 'STANDARD'),
                'predicted_outcome': bet.get('predicted_outcome', 'UNKNOWN'),
                'pattern_matched': bet.get('pattern_matched', False),
                'method': bet.get('method', 'TRANSFORMER'),
                'betting': bet.get('betting', {})
            }
            
            # Extract team info from narrative
            narrative_text = bet.get('narrative', '')
            if 'Team' in narrative_text:
                parts = narrative_text.split('.')
                detailed['team'] = parts[0].replace('Team ', '').strip() if parts else 'Unknown'
                detailed['matchup'] = parts[1].replace('Matchup ', '').strip() if len(parts) > 1 else 'vs Opponent'
                detailed['location'] = parts[2].replace('Location ', '').strip() if len(parts) > 2 else 'unknown'
            
            high_conf_detailed.append(detailed)
    
    return render_template('nba_betting_live.html',
                         predictions=predictions,
                         high_confidence_bets=high_conf_detailed,
                         model_summary=model_summary,
                         backtest=backtest,
                         upcoming_games=upcoming_games,
                         is_optimized=is_optimized,
                         today=datetime.now().strftime('%Y-%m-%d'),
                         time=datetime.now().strftime('%H:%M:%S'))


@nba_betting_live_bp.route('/api/todays-picks')
def api_todays_picks():
    """API endpoint for today's high-confidence picks"""
    
    today = datetime.now().strftime('%Y%m%d')
    pred_path = Path(f'data/predictions/nba_daily_{today}.json')
    
    if not pred_path.exists():
        return jsonify({'error': 'No predictions for today', 'date': today}), 404
    
    with open(pred_path) as f:
        data = json.load(f)
    
    return jsonify({
        'date': data['date'],
        'generated_at': data['generated_at'],
        'n_games': data['n_games_analyzed'],
        'n_bets': data['n_high_confidence_bets'],
        'bets': data['high_confidence_bets'],
        'config': data['model_config']
    })


@nba_betting_live_bp.route('/api/recent-performance')
def api_recent_performance():
    """API endpoint for recent prediction performance"""
    
    # Load all recent predictions
    pred_dir = Path('data/predictions')
    if not pred_dir.exists():
        return jsonify({'error': 'No predictions directory'}), 404
    
    pred_files = sorted(glob(str(pred_dir / 'nba_daily_*.json')), reverse=True)[:30]
    
    if len(pred_files) == 0:
        return jsonify({'error': 'No recent predictions'}), 404
    
    daily_summaries = []
    
    for filepath in pred_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            daily_summaries.append({
                'date': data['date'],
                'n_bets': data['n_high_confidence_bets'],
                'total_ev': sum(b['betting']['expected_value'] for b in data['high_confidence_bets']) if data['n_high_confidence_bets'] > 0 else 0
            })
        except:
            continue
    
    return jsonify({
        'n_days': len(daily_summaries),
        'daily_summaries': daily_summaries,
        'total_bets': sum(d['n_bets'] for d in daily_summaries),
        'avg_bets_per_day': sum(d['n_bets'] for d in daily_summaries) / len(daily_summaries) if daily_summaries else 0
    })


@nba_betting_live_bp.route('/api/model-info')
def api_model_info():
    """API endpoint for model information"""
    
    summary_path = Path('narrative_optimization/betting/nba_ensemble_summary.json')
    if not summary_path.exists():
        return jsonify({'error': 'Model not trained'}), 404
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return jsonify(summary)


@nba_betting_live_bp.route('/api/backtest-results')
def api_backtest_results():
    """API endpoint for backtest results"""
    
    backtest_path = Path('narrative_optimization/betting/nba_backtest_results.json')
    if not backtest_path.exists():
        return jsonify({'error': 'No backtest results'}), 404
    
    with open(backtest_path) as f:
        backtest = json.load(f)
    
    return jsonify(backtest)

