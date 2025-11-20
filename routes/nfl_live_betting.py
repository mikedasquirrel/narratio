"""
NFL Live Betting Route
Real-time betting recommendations for NFL games
"""

from flask import Blueprint, render_template, jsonify
from datetime import datetime

from utils.nfl_betting_engine import (
    discover_live_opportunities,
    evaluate_game,
    get_patterns,
    load_historical_games,
)

nfl_live_betting_bp = Blueprint('nfl_live_betting', __name__)

@nfl_live_betting_bp.route('/nfl/betting/live')
def nfl_live_betting():
    """Main NFL live betting page"""
    
    opportunities = discover_live_opportunities(limit=20, season=2025, min_week=10)
    
    return render_template('nfl_live_betting.html',
                         opportunities=opportunities,
                         total_games=len(opportunities),
                         flagged=len(opportunities),
                         timestamp=datetime.now().strftime('%B %d, %Y %H:%M'))

@nfl_live_betting_bp.route('/nfl/betting/api/opportunities')
def api_opportunities():
    """API endpoint for betting opportunities"""
    
    opportunities = discover_live_opportunities(limit=20, season=2025, min_week=10)
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'total_opportunities': len(opportunities),
        'opportunities': opportunities,
    })

@nfl_live_betting_bp.route('/nfl/betting/api/patterns')
def api_patterns():
    """API endpoint for pattern library"""
    patterns = get_patterns()
    
    return jsonify({
        'total_patterns': len(patterns['patterns']),
        'profitable': [p for p in patterns['patterns'] if p['profitable']],
        'baseline_ats': patterns['baseline_ats'],
    })

@nfl_live_betting_bp.route('/nfl/betting/patterns')
def patterns_page():
    """Pattern library page"""
    patterns_data = get_patterns()
    
    profitable = [p for p in patterns_data['patterns'] if p['profitable']]
    all_patterns = sorted(patterns_data['patterns'], key=lambda x: x['roi_pct'], reverse=True)
    
    return render_template('nfl_betting_patterns.html',
                         patterns=all_patterns,
                         profitable_count=len(profitable),
                         total_tested=len(all_patterns),
                         baseline_ats=patterns_data['baseline_ats'])

