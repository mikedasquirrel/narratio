"""
Transformer Analysis Routes for Flask App

Add these routes to app.py to enable transformer analysis dashboard.
"""

from flask import render_template, jsonify
from pathlib import Path
import json
import numpy as np

def register_transformer_routes(app):
    """Register transformer analysis routes with Flask app"""
    
    @app.route('/transformers/analysis')
    def transformer_analysis():
        """Transformer analysis dashboard"""
        try:
            # Load completed domains
            features_dir = Path('narrative_optimization/data/features')
            domain_files = list(features_dir.glob('*_all_features.npz'))
            
            # Load catalog
            catalog_path = Path('narrative_optimization/TRANSFORMER_CATALOG.json')
            with open(catalog_path) as f:
                catalog = json.load(f)
            
            # Compile stats
            domain_stats = []
            for domain_file in domain_files:
                domain_name = domain_file.stem.replace('_all_features', '')
                try:
                    data = np.load(domain_file, allow_pickle=True)
                    domain_stats.append({
                        'name': domain_name,
                        'samples': int(data['features'].shape[0]),
                        'features': int(data['features'].shape[1]),
                        'has_outcomes': 'outcomes' in data.files and len(data['outcomes']) > 0,
                        'file': domain_file.name
                    })
                except Exception as e:
                    print(f"Error loading {domain_name}: {e}")
                    continue
            
            # Sort by name
            domain_stats.sort(key=lambda x: x['name'])
            
            # Calculate totals
            total_samples = sum(d['samples'] for d in domain_stats)
            total_features = sum(d['features'] for d in domain_stats)
            
            return render_template('transformer_analysis.html',
                                 domains=domain_stats,
                                 catalog=catalog,
                                 total_transformers=catalog['summary']['total_transformers'],
                                 total_samples=total_samples,
                                 total_features=total_features,
                                 domains_processed=len(domain_stats))
        except Exception as e:
            return f"Error loading transformer analysis: {str(e)}", 500

    @app.route('/transformers/catalog')
    def transformer_catalog():
        """Browse transformer catalog"""
        try:
            catalog_path = Path('narrative_optimization/TRANSFORMER_CATALOG.json')
            with open(catalog_path) as f:
                catalog = json.load(f)
            
            return render_template('transformer_catalog.html',
                                 catalog=catalog)
        except Exception as e:
            return f"Error loading catalog: {str(e)}", 500

    @app.route('/api/domains/<domain>/features')
    def domain_features_api(domain):
        """API endpoint for domain feature data"""
        try:
            features_path = Path(f'narrative_optimization/data/features/{domain}_all_features.npz')
            if not features_path.exists():
                return jsonify({'error': 'Domain not found'}), 404
            
            data = np.load(features_path, allow_pickle=True)
            
            return jsonify({
                'domain': domain,
                'samples': int(data['features'].shape[0]),
                'features': int(data['features'].shape[1]),
                'has_outcomes': 'outcomes' in data.files,
                'feature_names': data['feature_names'].tolist() if 'feature_names' in data.files else []
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    print("✅ Transformer analysis routes registered")
    print("   - /transformers/analysis")
    print("   - /transformers/catalog")
    print("   - /api/domains/<domain>/features")

