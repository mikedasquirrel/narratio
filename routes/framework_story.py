"""
Framework Story Routes

Presents the complete narrative framework story and three-force model
in a beautiful, interactive web experience.
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path
import markdown

framework_story_bp = Blueprint('framework_story', __name__)


@framework_story_bp.route('/framework-story')
def framework_story():
    """Complete narrative framework story."""
    # Load the story markdown
    story_path = Path(__file__).parent.parent / 'NARRATIVE_FRAMEWORK_STORY.md'
    
    with open(story_path) as f:
        story_md = f.read()
    
    # Convert to HTML
    story_html = markdown.markdown(story_md, extensions=['tables', 'fenced_code'])
    
    return render_template('framework/story.html', 
                         story_html=story_html,
                         title="The Narrative Framework: A Discovery")


@framework_story_bp.route('/framework-explorer')
def framework_explorer():
    """Interactive three-force model explorer."""
    # Load three-force analysis results
    data_path = Path(__file__).parent.parent / 'narrative_optimization' / 'data' / 'three_force_analysis.json'
    
    domains_data = {}
    if data_path.exists():
        with open(data_path) as f:
            domains_data = json.load(f)
    
    return render_template('framework/explorer.html',
                         domains=domains_data,
                         title="Framework Explorer: Three-Realm Model")


@framework_story_bp.route('/framework-quickref')
def quickref():
    """One-page quick reference guide."""
    quickref_path = Path(__file__).parent.parent / 'FRAMEWORK_QUICKREF.md'
    
    with open(quickref_path) as f:
        content_md = f.read()
    
    content_html = markdown.markdown(content_md, extensions=['tables', 'fenced_code'])
    
    return render_template('framework/quickref.html',
                         content_html=content_html,
                         title="Framework Quick Reference")


@framework_story_bp.route('/api/three-forces')
def three_forces_api():
    """API endpoint for three-force data."""
    data_path = Path(__file__).parent.parent / 'narrative_optimization' / 'data' / 'three_force_analysis.json'
    
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
        return jsonify(data)
    
    return jsonify({'error': 'Data not found'}), 404


@framework_story_bp.route('/api/domain-comparison')
def domain_comparison():
    """API endpoint for cross-domain comparison."""
    data_path = Path(__file__).parent.parent / 'narrative_optimization' / 'data' / 'three_force_analysis.json'
    
    if not data_path.exists():
        return jsonify({'error': 'Data not found'}), 404
    
    with open(data_path) as f:
        data = json.load(f)
    
    # Prepare comparison data
    comparison = []
    for domain_name, forces in data.items():
        comparison.append({
            'domain': domain_name,
            'narrativity': forces.get('narrativity'),
            'nominative_gravity': forces['nominative_gravity'],
            'awareness': forces['awareness_resistance'],
            'constraints': forces['fundamental_constraints'],
            'predicted': forces['predicted_bridge'],
            'observed': forces['observed_bridge'],
            'error': forces['model_error'],
            'dominant': forces['dominant_force']
        })
    
    return jsonify(comparison)


@framework_story_bp.route('/three-realms')
def three_realms():
    """Visual explanation of the three-realm model."""
    return render_template('framework/three_realms.html',
                         title="The Three Realms of Reality")

