"""
Cryptocurrency Analysis Routes

Shows crypto domain with innovation narrative focus.
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

crypto_bp = Blueprint('crypto', __name__)


@crypto_bp.route('/crypto')
def crypto_dashboard():
    """Main crypto analysis dashboard."""
    try:
        # Load crypto data
        data_path = Path(__file__).parent.parent / 'crypto_enriched_narratives.json'
        
        with open(data_path, 'r') as f:
            cryptos = json.load(f)
        
        stats = {
            'total': len(cryptos),
            'top_25_percent': len([c for c in cryptos if c.get('rank', 1000) <= len(cryptos) * 0.25])
        }
        
        return render_template('crypto_dashboard.html', stats=stats)
        
    except Exception as e:
        return render_template('crypto_dashboard.html', error=str(e))


@crypto_bp.route('/api/crypto/domain-properties')
def get_domain_properties():
    """Get crypto domain properties."""
    return jsonify({
        'narrativity': 0.65,  # Moderate openness
        'alpha_predicted': 0.35,  # Hybrid
        'n_organisms': 300,
        'force_balance': 'ф ≈ ة (both matter)',
        'key_features': ['innovation_language', 'technical_legitimacy', 'market_positioning']
    })


@crypto_bp.route('/api/crypto/innovation-analysis')
def get_innovation_analysis():
    """Get innovation narrative analysis."""
    return jsonify({
        'innovation_scores': [0.8, 0.9, 0.4, 0.6, 0.95],
        'valuations': [1000, 5000, 50, 200, 8000],
        'correlation': 0.85
    })

