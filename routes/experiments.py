"""
Experiment browser routes
"""

from flask import Blueprint, render_template, jsonify, send_file
from pathlib import Path
import json

experiments_bp = Blueprint('experiments', __name__)

@experiments_bp.route('/')
def list_experiments():
    """List all experiments with filtering."""
    experiments_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments'
    
    experiments = []
    if experiments_dir.exists():
        for exp_dir in sorted(experiments_dir.iterdir()):
            if exp_dir.is_dir() and not exp_dir.name.startswith('_'):
                exp_data = {
                    'id': exp_dir.name,
                    'name': exp_dir.name.replace('_', ' ').title(),
                    'status': 'pending',
                    'results': None
                }
                
                # Load metadata
                metadata_file = exp_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        exp_data.update(metadata)
                
                # Check for results
                results_file = exp_dir / 'results.json'
                if results_file.exists():
                    exp_data['has_results'] = True
                    exp_data['status'] = 'completed'
                
                experiments.append(exp_data)
    
    return render_template('experiments.html', experiments=experiments)

@experiments_bp.route('/<experiment_id>')
def view_experiment(experiment_id):
    """View detailed experiment results."""
    exp_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    
    if not exp_dir.exists():
        return "Experiment not found", 404
    
    # Load results
    results_file = exp_dir / 'results.json'
    metadata_file = exp_dir / 'metadata.json'
    report_file = exp_dir / 'report.md'
    
    data = {
        'id': experiment_id,
        'name': experiment_id.replace('_', ' ').title(),
        'results': None,
        'metadata': None,
        'report': None
    }
    
    if results_file.exists():
        with open(results_file) as f:
            data['results'] = json.load(f)
    
    if metadata_file.exists():
        with open(metadata_file) as f:
            data['metadata'] = json.load(f)
    
    if report_file.exists():
        with open(report_file) as f:
            data['report'] = f.read()
    
    return render_template('experiment_detail.html', experiment=data)

@experiments_bp.route('/<experiment_id>/download/<file_type>')
def download_results(experiment_id, file_type):
    """Download experiment results in various formats."""
    exp_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    
    if file_type == 'json':
        file_path = exp_dir / 'results.json'
    elif file_type == 'report':
        file_path = exp_dir / 'report.md'
    elif file_type == 'plot':
        file_path = exp_dir / 'experiment_summary.png'
    else:
        return "Invalid file type", 400
    
    if not file_path.exists():
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)

