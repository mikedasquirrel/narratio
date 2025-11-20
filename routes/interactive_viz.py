"""
Interactive visualization routes with Plotly and D3
"""

from flask import Blueprint, render_template, send_file, jsonify
from pathlib import Path
import json

interactive_viz_bp = Blueprint('interactive_viz', __name__)

@interactive_viz_bp.route('/experiment/<experiment_id>/gallery')
def visualization_gallery(experiment_id):
    """Display gallery of all available visualizations."""
    return render_template('viz_gallery.html', experiment_id=experiment_id)

@interactive_viz_bp.route('/experiment/<experiment_id>/interactive')
def interactive_dashboard(experiment_id):
    """Display interactive Plotly dashboard for experiment."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    
    # Check for interactive dashboard
    dashboard_file = exp_path / 'interactive_dashboard.html'
    
    if dashboard_file.exists():
        return send_file(dashboard_file)
    else:
        return "Interactive dashboard not yet generated. Run experiment first.", 404

@interactive_viz_bp.route('/experiment/<experiment_id>/3d')
def results_3d(experiment_id):
    """Display 3D results space."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    viz_file = exp_path / 'results_3d.html'
    
    if viz_file.exists():
        return send_file(viz_file)
    else:
        return "3D visualization not available.", 404

@interactive_viz_bp.route('/experiment/<experiment_id>/radar')
def radar_chart(experiment_id):
    """Display radar comparison chart."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    viz_file = exp_path / 'radar_comparison.html'
    
    if viz_file.exists():
        return send_file(viz_file)
    else:
        return "Radar chart not available.", 404

@interactive_viz_bp.route('/experiment/<experiment_id>/heatmap')
def performance_heatmap(experiment_id):
    """Display performance heatmap (transformers × metrics)."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    viz_file = exp_path / 'heatmap_performance.html'
    
    if viz_file.exists():
        return send_file(viz_file)
    else:
        return "Heatmap not available.", 404

@interactive_viz_bp.route('/experiment/<experiment_id>/density')
def density_plot(experiment_id):
    """Display density distribution plots."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    viz_file = exp_path / 'density_comparison.html'
    
    if viz_file.exists():
        return send_file(viz_file)
    else:
        return "Density plot not available.", 404

@interactive_viz_bp.route('/experiment/<experiment_id>/density-detailed')
def density_detailed(experiment_id):
    """Display detailed density plot for primary metric."""
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    viz_file = exp_path / 'density_detailed.html'
    
    if viz_file.exists():
        return send_file(viz_file)
    else:
        return "Detailed density plot not available.", 404
