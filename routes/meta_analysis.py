"""
Meta-Analysis Routes

Cross-domain meta-regression and comparative analysis dashboards.

Author: Narrative Optimization Research
Date: November 2025
"""

from flask import Blueprint, render_template, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.taxonomy.domain_registry import DomainRegistry
from src.analysis.meta_regression import MetaRegression
from src.analysis.visibility_calculator import VisibilityCalculator

meta_analysis_bp = Blueprint('meta_analysis', __name__)

_registry = None
_meta_regression = None


def _get_registry():
    """Get domain registry."""
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


def _get_meta_regression():
    """Get meta regression tool."""
    global _meta_regression
    if _meta_regression is None:
        _meta_regression = MetaRegression()
    return _meta_regression


@meta_analysis_bp.route('/')
def dashboard():
    """Meta-analysis dashboard."""
    try:
        registry = _get_registry()
        meta = _get_meta_regression()
        
        # Get domains with observed effects
        domains = [d for d in registry.get_all_domains() 
                  if d.effect_size_observed is not None]
        
        # Fit model
        visibilities = [d.visibility for d in domains]
        effects = [d.effect_size_observed for d in domains]
        
        model = meta.fit_visibility_model(visibilities, effects)
        
        # Calculate heterogeneity
        heterogeneity = meta.calculate_heterogeneity(effects)
        
        # Subgroup analysis by domain type
        types = [d.domain_type for d in domains]
        subgroup = meta.subgroup_analysis(effects, types)
        
        return render_template('meta_analysis/dashboard.html',
                             model=model,
                             heterogeneity=heterogeneity,
                             subgroup=subgroup,
                             domains=domains)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500


@meta_analysis_bp.route('/api/model')
def api_model():
    """Get meta-regression model."""
    try:
        registry = _get_registry()
        meta = _get_meta_regression()
        
        domains = [d for d in registry.get_all_domains() 
                  if d.effect_size_observed is not None]
        
        visibilities = [d.visibility for d in domains]
        effects = [d.effect_size_observed for d in domains]
        
        model = meta.fit_visibility_model(visibilities, effects)
        
        return jsonify({
            'success': True,
            'model': model
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

