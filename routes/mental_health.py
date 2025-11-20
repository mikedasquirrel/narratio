"""
Mental Health Nomenclature Routes

Web interface for diagnostic naming analysis.
"""

from flask import Blueprint, render_template, request, jsonify
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization'))
from utils.result_loader import load_unified_results, extract_stats_from_results, get_chart_data

from domains.mental_health.data_loader import MentalHealthDataLoader
from domains.mental_health.stigma_database import StigmaDatabase
from domains.mental_health.clinical_outcomes_database import ClinicalOutcomesDatabase
from src.transformers.mental_health.treatment_seeking_transformer import TreatmentSeekingTransformer

mental_health_bp = Blueprint('mental_health', __name__)

_data_cache = None
_transformer = None

def _load_data():
    global _data_cache
    if _data_cache is None:
        data_file = Path(__file__).parent.parent / 'narrative_optimization' / 'domains' / 'mental_health' / 'data' / 'integrated_disorders_complete.json'
        if data_file.exists():
            with open(data_file, 'r') as f:
                _data_cache = json.load(f)
        else:
            loader = MentalHealthDataLoader()
            _data_cache = {'disorders': loader.load_disorders()}
    return _data_cache

def _get_transformer():
    global _transformer
    if _transformer is None:
        data = _load_data()
        disorders = data['disorders']
        names = [d.get('disorder_name', '') for d in disorders if d.get('disorder_name')]
        stigma = [d.get('social_impact', {}).get('stigma_score') for d in disorders]
        stigma = [s for s in stigma if s is not None]
        
        _transformer = TreatmentSeekingTransformer()
        if len(names) > 3 and len(stigma) > 3:
            _transformer.fit(names[:len(stigma)], stigma)
    return _transformer

@mental_health_bp.route('/')
def dashboard():
    try:
        data = _load_data()
        disorders = data['disorders']
        
        n_disorders = len(disorders)
        
        stigma_db = StigmaDatabase()
        stats = stigma_db.get_coverage_statistics()
        
        return render_template('mental_health/dashboard.html',
                             n_disorders=n_disorders,
                             stats=stats,
                             sample=disorders[:10])
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@mental_health_bp.route('/browser')
def browser():
    try:
        data = _load_data()
        disorders = data['disorders']
        return render_template('mental_health/browser.html', disorders=disorders)
    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@mental_health_bp.route('/predictor')
def predictor():
    return render_template('mental_health/predictor.html')

@mental_health_bp.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        disorder_name = data.get('name')
        
        if not disorder_name:
            return jsonify({'success': False, 'error': 'Name required'}), 400
        
        transformer = _get_transformer()
        if transformer and transformer.is_fitted_:
            prediction = transformer.predict_treatment_barrier(disorder_name)
            return jsonify({'success': True, 'prediction': prediction})
        else:
            return jsonify({'success': False, 'error': 'Model not fitted'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@mental_health_bp.route('/api/data')
def api_data():
    try:
        data = _load_data()
        return jsonify({'success': True, 'disorders': data['disorders']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

