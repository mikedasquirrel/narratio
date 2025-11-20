"""
Free Will Route - θ (Awareness Resistance) Analysis
"""

from flask import Blueprint, render_template, jsonify
from pathlib import Path
import json

free_will_bp = Blueprint('free_will', __name__)

@free_will_bp.route('/free-will')
def free_will():
    """Free will / awareness resistance page."""
    
    # Load Phase 7 summary with θ values
    phase7_path = Path(__file__).parent.parent / 'narrative_optimization' / 'data' / 'phase7_extraction_summary.json'
    
    theta_data = None
    if phase7_path.exists():
        try:
            with open(phase7_path, 'r') as f:
                data = json.load(f)
                # Extract θ (awareness resistance) across domains
                theta_data = {
                    'domains': [],
                    'theta_values': [],
                    'correlation_with_lambda': data.get('expertise_pattern', {}).get('correlation', 0.702)
                }
                for result in data.get('results', []):
                    theta_data['domains'].append(result['domain'])
                    theta_data['theta_values'].append(result['theta_mean'])
        except:
            pass
    
    return render_template('free_will.html', theta_data=theta_data)
