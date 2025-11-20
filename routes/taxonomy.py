"""
Taxonomy Routes - Cross-Domain Browser and Navigation

Provides comprehensive taxonomy browser with:
- Visibility matrix visualization
- Domain registry explorer
- Theory navigator
- Meta-analysis dashboards

Author: Narrative Optimization Research
Date: November 2025
"""

from flask import Blueprint, render_template, request, jsonify
import sys
from pathlib import Path
import json

# Add narrative_optimization to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.taxonomy.domain_registry import DomainRegistry
from src.taxonomy.taxonomy_builder import TaxonomyBuilder
from src.taxonomy.visibility_classifier import VisibilityClassifier
from src.analysis.visibility_calculator import VisibilityCalculator

taxonomy_bp = Blueprint('taxonomy', __name__)

# Global cache
_registry = None
_calculator = None


def _get_registry():
    """Get or create domain registry."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


def _get_calculator():
    """Get or create visibility calculator."""
    global _calculator
    if _calculator is None:
        _calculator = VisibilityCalculator()
    return _calculator


@taxonomy_bp.route('/')
@taxonomy_bp.route('/taxonomy_explorer')
def taxonomy_explorer():
    """Main taxonomy browser page."""
    try:
        registry = _get_registry()
        calculator = _get_calculator()
        
        # Get summary statistics
        stats = registry.get_summary_statistics()
        
        # Get all domains
        all_domains = registry.get_all_domains()
        
        # Calculate model fit
        model_fit = calculator.calculate_model_fit()
        
        return render_template('taxonomy/explorer.html',
                             domains=all_domains,
                             stats=stats,
                             model_fit=model_fit)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@taxonomy_bp.route('/visibility-matrix')
def visibility_matrix():
    """Interactive visibility × effect matrix visualization."""
    try:
        registry = _get_registry()
        calculator = _get_calculator()
        
        # Get domains with observed effects
        domains = [d for d in registry.get_all_domains() 
                  if d.effect_size_observed is not None]
        
        # Prepare data for visualization
        viz_data = []
        for domain in domains:
            viz_data.append({
                'name': domain.display_name,
                'visibility': domain.visibility,
                'effect': domain.effect_size_observed,
                'narrative_importance': domain.narrative_importance,
                'status': domain.status,
                'type': domain.domain_type
            })
        
        return render_template('taxonomy/visibility_matrix.html',
                             domains=viz_data,
                             model_fit=calculator.calculate_model_fit())
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@taxonomy_bp.route('/domain/<domain_id>')
def domain_detail(domain_id):
    """Detailed view for specific domain."""
    try:
        registry = _get_registry()
        domain = registry.get_domain(domain_id)
        
        if not domain:
            return render_template('error.html', error=f'Domain {domain_id} not found'), 404
        
        calculator = _get_calculator()
        
        # Get similar domains
        builder = TaxonomyBuilder(registry)
        similar = []  # Would use comparator to find similar
        
        # Get validation if observed effect exists
        validation = None
        if domain.effect_size_observed:
            validation = calculator.validate_prediction(
                domain.display_name,
                domain.visibility,
                domain.effect_size_observed,
                0.7 if 'high' in domain.narrative_importance else 0.5
            )
        
        return render_template('taxonomy/domain_detail.html',
                             domain=domain,
                             validation=validation,
                             similar=similar)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@taxonomy_bp.route('/compare')
def compare_domains():
    """Compare multiple domains side-by-side."""
    try:
        registry = _get_registry()
        
        # Get domain IDs from query params
        domain_ids = request.args.getlist('domains')
        
        if not domain_ids:
            # Show selection page
            all_domains = registry.get_all_domains()
            return render_template('taxonomy/compare_select.html',
                                 domains=all_domains)
        
        # Get domains
        domains = [registry.get_domain(did) for did in domain_ids]
        domains = [d for d in domains if d is not None]
        
        if not domains:
            return render_template('error.html', error='No valid domains selected'), 400
        
        return render_template('taxonomy/compare.html',
                             domains=domains)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@taxonomy_bp.route('/api/domains')
def api_domains():
    """Get all domains as JSON."""
    try:
        registry = _get_registry()
        domains = registry.get_all_domains()
        
        return jsonify({
            'success': True,
            'count': len(domains),
            'domains': [d.to_dict() for d in domains]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@taxonomy_bp.route('/api/predict-effect', methods=['POST'])
def api_predict_effect():
    """Predict effect size for a domain."""
    try:
        data = request.get_json()
        visibility = data.get('visibility')
        genre_congruence = data.get('genre_congruence', 0.5)
        
        if visibility is None:
            return jsonify({'success': False, 'error': 'Visibility required'}), 400
        
        calculator = _get_calculator()
        prediction = calculator.predict_effect_size(visibility, genre_congruence)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@taxonomy_bp.route('/api/model-fit')
def api_model_fit():
    """Get visibility model fit statistics."""
    try:
        calculator = _get_calculator()
        fit = calculator.calculate_model_fit()
        
        return jsonify({
            'success': True,
            'fit': fit
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
