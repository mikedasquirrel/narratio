"""
Supreme Court Route

Web interface for Supreme Court narrative analysis.

Shows:
- Domain formula results (π, Δ, r, κ)
- Theoretical findings (π variance, adversarial dynamics)
- Famous cases analyzed
- Predictive models for citations and precedent status

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

from flask import Blueprint, render_template, jsonify, request
from pathlib import Path
import json
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

supreme_court_bp = Blueprint('supreme_court', __name__)

# Load analysis results
RESULTS_PATH = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'supreme_court' / 'results' / 'supreme_court_analysis_complete.json'


def load_results():
    """Load Supreme Court analysis results."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return None


@supreme_court_bp.route('/')
def dashboard():
    """Supreme Court analysis dashboard."""
    results = load_results()
    
    if not results:
        return render_template('supreme_court/no_data.html')
    
    return render_template('supreme_court/dashboard.html', results=results)


@supreme_court_bp.route('/breakthrough')
def breakthrough_explanation():
    """Comprehensive explanation of Supreme Court breakthrough findings."""
    return render_template('supreme_court/breakthrough.html')


@supreme_court_bp.route('/theoretical')
def theoretical_findings():
    """Theoretical extensions page."""
    results = load_results()
    
    if not results:
        return jsonify({'error': 'No results available'}), 404
    
    return render_template('supreme_court/theoretical.html', results=results)


@supreme_court_bp.route('/cases')
def famous_cases():
    """Famous cases analyzed."""
    # Load actual case data with analysis
    data_path = Path(__file__).parent.parent / 'data' / 'domains' / 'supreme_court_complete.json'
    
    if not data_path.exists():
        return render_template('supreme_court/no_data.html')
    
    with open(data_path) as f:
        cases = json.load(f)
    
    # Filter for famous cases
    famous = [
        'roe v', 'brown v', 'miranda v', 'marbury v', 'gideon v',
        'loving v', 'obergefell v', 'griswold v', 'plessy v', 'korematsu v'
    ]
    
    famous_cases = []
    for case in cases:
        case_name_lower = case.get('case_name', '').lower()
        if any(f in case_name_lower for f in famous):
            famous_cases.append(case)
    
    return render_template('supreme_court/cases.html', cases=famous_cases)


@supreme_court_bp.route('/api/analyze_opinion', methods=['POST'])
def analyze_opinion_api():
    """
    API endpoint to analyze a Supreme Court opinion text.
    
    Accepts opinion text, returns narrative quality and predictions.
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if len(text) < 500:
            return jsonify({'error': 'Opinion text too short (min 500 chars)'}), 400
        
        # Import transformers
        from narrative_optimization.src.transformers.legal.argumentative_structure import ArgumentativeStructureTransformer
        from narrative_optimization.src.transformers.legal.precedential_narrative import PrecedentialNarrativeTransformer
        from narrative_optimization.src.transformers.legal.persuasive_framing import PersuasiveFramingTransformer
        from narrative_optimization.src.transformers.legal.judicial_rhetoric import JudicialRhetoricTransformer
        
        # Extract features
        arg_transformer = ArgumentativeStructureTransformer()
        prec_transformer = PrecedentialNarrativeTransformer()
        pers_transformer = PersuasiveFramingTransformer()
        rhet_transformer = JudicialRhetoricTransformer()
        
        arg_features = arg_transformer.transform([text])[0]
        prec_features = prec_transformer.transform([text])[0]
        pers_features = pers_transformer.transform([text])[0]
        rhet_features = rhet_transformer.transform([text])[0]
        
        # Calculate narrative quality
        all_features = np.concatenate([arg_features, prec_features, pers_features, rhet_features])
        
        narrative_quality = float(np.mean(all_features))
        
        # Predictions (based on correlations from analysis)
        results = load_results()
        if results and 'domain_formula' in results:
            r_citations = results['domain_formula'].get('citations', {}).get('r', 0.3)
            
            # Predict citation count (rough estimate)
            predicted_citations = int(1000 * (narrative_quality ** 2) * abs(r_citations))
            
            return jsonify({
                'success': True,
                'narrative_quality': narrative_quality,
                'quality_score': int(narrative_quality * 100),
                'predicted_citations': predicted_citations,
                'features': {
                    'argumentative_strength': float(np.mean(arg_features)),
                    'precedential_grounding': float(np.mean(prec_features)),
                    'persuasive_power': float(np.mean(pers_features)),
                    'rhetorical_quality': float(np.mean(rhet_features))
                }
            })
        
        return jsonify({
            'success': True,
            'narrative_quality': narrative_quality,
            'quality_score': int(narrative_quality * 100)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

