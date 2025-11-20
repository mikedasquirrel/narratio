"""
Bible Stories Domain Routes
Flask routes for Bible stories narrative analysis
"""

from flask import Blueprint, render_template, jsonify
import json
from pathlib import Path

bible_bp = Blueprint('bible', __name__)

def load_bible_data():
    """Load Bible stories data"""
    try:
        # Load narrative forces
        forces_path = Path(__file__).parent.parent / "data" / "bible" / "bible_narrative_forces.json"
        with open(forces_path, 'r') as f:
            forces = json.load(f)
        
        # Note: External experiment results removed during cleanup
        experiment_results = None
        format_c = None
        
        return forces, experiment_results, format_c
    except Exception as e:
        print(f"Error loading Bible data: {e}")
        return None, None, None

@bible_bp.route('/')
def dashboard():
    """Bible stories dashboard"""
    forces, experiment_results, format_c = load_bible_data()
    
    if not forces:
        return "Data not available", 404
    
    # Note: Story details require external data that was cleaned up
    top_stories = []
    bottom_stories = []
    total_stories = forces.get('n_stories', 47)
    
    return render_template('bible_dashboard.html',
                          forces=forces,
                          top_stories=top_stories,
                          bottom_stories=bottom_stories,
                          total_stories=total_stories)

@bible_bp.route('/results')
def results():
    """Bible stories results page"""
    forces, experiment_results, format_c = load_bible_data()
    
    if not experiment_results:
        return "Data not available", 404
    
    # Get best transformers
    best_transformers = experiment_results['results'][:5]
    
    # Get stories with extreme persistence
    stories_with_persistence = [(format_c['metadata'][i], format_c['labels'][i]) 
                                 for i in range(len(format_c['labels']))]
    stories_sorted = sorted(stories_with_persistence, key=lambda x: x[1], reverse=True)
    
    return render_template('bible_results.html',
                          results=experiment_results,
                          best_transformers=best_transformers,
                          top_10=stories_sorted[:10],
                          bottom_10=stories_sorted[-10:])

@bible_bp.route('/leaderboard')
def leaderboard():
    """Bible stories cultural persistence leaderboard"""
    # Story details require external data that was cleaned up
    return "Story leaderboard data not available", 404

@bible_bp.route('/forces')
def forces_viz():
    """Bible stories narrative forces visualization"""
    forces, experiment_results, format_c = load_bible_data()
    
    if not forces:
        return "Data not available", 404
    
    return render_template('bible_forces.html',
                          forces=forces)

@bible_bp.route('/api/story/<story_id>')
def story_detail(story_id):
    """Get detailed info for a specific story"""
    forces, experiment_results, format_c = load_bible_data()
    
    if not format_c:
        return jsonify({"error": "Data not available"}), 404
    
    # Find story
    for i, meta in enumerate(format_c['metadata']):
        if meta['story_id'] == story_id:
            return jsonify({
                'story': meta,
                'persistence': format_c['labels'][i],
                'text': format_c['texts'][i],
                'features': format_c['features'][i]
            })
    
    return jsonify({"error": "Story not found"}), 404

