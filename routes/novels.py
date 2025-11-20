"""
Novels Domain Route
"""

from flask import Blueprint, render_template, jsonify
from pathlib import Path
import json

novels_bp = Blueprint('novels', __name__)

@novels_bp.route('/novels')
def novels():
    """Novels domain overview page."""
    
    # Load novels analysis if available
    novels_analysis_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'novels' / 'novels_complete_analysis.json'
    
    analysis_data = None
    if novels_analysis_path.exists():
        try:
            with open(novels_analysis_path, 'r') as f:
                analysis_data = json.load(f)
        except:
            pass
    
    return render_template('novels.html', analysis=analysis_data)


@novels_bp.route('/nonfiction')
def nonfiction():
    """Nonfiction domain overview page."""
    
    # Load nonfiction analysis if available
    nonfiction_analysis_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'nonfiction' / 'nonfiction_complete_analysis.json'
    
    analysis_data = None
    if nonfiction_analysis_path.exists():
        try:
            with open(nonfiction_analysis_path, 'r') as f:
                analysis_data = json.load(f)
        except:
            pass
    
    return render_template('nonfiction.html', analysis=analysis_data)


@novels_bp.route('/books-combined')
def books_combined():
    """Combined books domain page."""
    
    # Load combined analysis if available
    combined_analysis_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'books_combined' / 'combined_analysis.json'
    
    analysis_data = None
    if combined_analysis_path.exists():
        try:
            with open(combined_analysis_path, 'r') as f:
                analysis_data = json.load(f)
        except:
            pass
    
    return render_template('books_combined.html', analysis=analysis_data)


@novels_bp.route('/api/novels/transformer-interactions')
def novels_transformer_interactions():
    """API endpoint for transformer interactions."""
    
    interactions_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'novels' / 'transformer_interactions.json'
    
    if interactions_path.exists():
        try:
            with open(interactions_path, 'r') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Data not available'}), 404


@novels_bp.route('/api/novels/multi-scale')
def novels_multi_scale():
    """API endpoint for multi-scale analysis."""
    
    multi_scale_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'novels' / 'multi_scale_analysis.json'
    
    if multi_scale_path.exists():
        try:
            with open(multi_scale_path, 'r') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Data not available'}), 404


@novels_bp.route('/api/novels/feature-attribution')
def novels_feature_attribution():
    """API endpoint for feature attribution."""
    
    attribution_path = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'novels' / 'feature_attribution.json'
    
    if attribution_path.exists():
        try:
            with open(attribution_path, 'r') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Data not available'}), 404

