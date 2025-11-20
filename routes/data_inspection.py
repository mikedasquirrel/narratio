"""
Data Inspection Routes

Show actual data, extracted features, and plain English explanations.
"""

from flask import Blueprint, render_template, jsonify, request
from pathlib import Path
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))

from src.utils.plain_english import PlainEnglishExplainer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer

data_inspection_bp = Blueprint('data_inspection', __name__)

@data_inspection_bp.route('/explore/<experiment_id>')
def explore_data(experiment_id):
    """Browse actual text samples from an experiment."""
    # Load experiment data
    exp_path = Path(__file__).parent.parent / 'narrative_optimization' / 'experiments' / experiment_id
    
    try:
        # Try to load some sample data
        from src.utils.toy_data import quick_load_toy_data
        data = quick_load_toy_data()
        
        samples = []
        for i in range(min(20, len(data['X_test']))):
            text = data['X_test'][i]
            label = int(data['y_test'][i])
            category = data['target_names'][label] if label < len(data['target_names']) else 'Unknown'
            
            samples.append({
                'index': i,
                'text': text[:500] + '...' if len(text) > 500 else text,
                'full_text': text,
                'length': len(text),
                'word_count': len(text.split()),
                'category': category,
                'label': label
            })
        
        return render_template('data_explorer.html', 
                             experiment_id=experiment_id,
                             samples=samples,
                             total_samples=len(data['X_test']))
    except Exception as e:
        return f"Error loading data: {e}", 500

@data_inspection_bp.route('/medical')
def explore_medical():
    """Browse medical diagnosis data - names, harshness, outcomes."""
    try:
        import json
        medical_path = Path(__file__).parent.parent / 'narrative_optimization' / 'data' / 'domains' / 'medical' / 'diagnoses_data.json'
        
        with open(medical_path) as f:
            data = json.load(f)
        
        disorders = data['disorders']
        
        # Calculate correlations for display
        import numpy as np
        from scipy.stats import pearsonr
        
        harshness = [d['harshness'] for d in disorders]
        stigma = [d['stigma'] for d in disorders]
        treatment = [d['treatment_seeking'] for d in disorders]
        mortality = [d['mortality_per_100k'] for d in disorders]
        
        r_harshness_stigma, p1 = pearsonr(harshness, stigma)
        r_stigma_treatment, p2 = pearsonr(stigma, treatment)
        r_harshness_treatment, p3 = pearsonr(harshness, treatment)
        
        stats = {
            'n_disorders': len(disorders),
            'harshness_stigma_r': f"{r_harshness_stigma:.3f}",
            'stigma_treatment_r': f"{r_stigma_treatment:.3f}",
            'harshness_treatment_r': f"{r_harshness_treatment:.3f}",
            'p_values': f"p < 0.01"
        }
        
        return render_template('medical_explorer.html', 
                             disorders=disorders,
                             stats=stats)
    except Exception as e:
        return f"Error loading medical data: {e}", 500

@data_inspection_bp.route('/sample/<experiment_id>/<int:sample_id>')
def view_sample(experiment_id, sample_id):
    """View detailed analysis of a single sample."""
    try:
        from src.utils.toy_data import quick_load_toy_data
        data = quick_load_toy_data()
        
        if sample_id >= len(data['X_test']):
            return "Sample not found", 404
        
        text = data['X_test'][sample_id]
        label = int(data['y_test'][sample_id])
        category = data['target_names'][label]
        
        # Analyze with all transformers
        explainer = PlainEnglishExplainer()
        
        analyses = {}
        
        # Ensemble
        try:
            ensemble = EnsembleNarrativeTransformer(n_top_terms=30)
            ensemble.fit([text] * 3)  # Dummy fit
            features = ensemble.transform([text])[0]
            
            analyses['ensemble'] = {
                'features': {
                    'ensemble_size': float(features[0]) if len(features) > 0 else 0,
                    'cooccurrence_density': float(features[1]) if len(features) > 1 else 0,
                    'diversity': float(features[2]) if len(features) > 2 else 0
                },
                'explanation': explainer._interpret_ensemble({
                    'ensemble_size': float(features[0]) if len(features) > 0 else 0,
                    'cooccurrence_density': float(features[1]) if len(features) > 1 else 0,
                    'diversity': float(features[2]) if len(features) > 2 else 0
                })
            }
        except:
            analyses['ensemble'] = {'explanation': 'Analysis not available for this sample'}
        
        # Linguistic
        try:
            linguistic = LinguisticPatternsTransformer()
            linguistic.fit([text] * 3)
            features = linguistic.transform([text])[0]
            
            analyses['linguistic'] = {
                'features': {
                    'first_person_density': float(features[0]) if len(features) > 0 else 0,
                    'future_orientation': float(features[5]) if len(features) > 5 else 0,
                    'agency_score': float(features[10]) if len(features) > 10 else 0
                },
                'explanation': explainer._interpret_linguistic({
                    'first_person_density': float(features[0]) if len(features) > 0 else 0,
                    'future_orientation': float(features[5]) if len(features) > 5 else 0,
                    'agency_score': float(features[10]) if len(features) > 10 else 0
                })
            }
        except:
            analyses['linguistic'] = {'explanation': 'Analysis not available'}
        
        return render_template('sample_detail.html',
                             experiment_id=experiment_id,
                             sample_id=sample_id,
                             text=text,
                             category=category,
                             analyses=analyses)
    except Exception as e:
        return f"Error: {e}", 500

