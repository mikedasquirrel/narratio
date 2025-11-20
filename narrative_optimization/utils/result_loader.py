"""
Result Loader Utility

Helper functions for loading unified pipeline results in Flask routes.
"""

from pathlib import Path
import json
from typing import Dict, Optional, Any, List
import numpy as np


def load_unified_results(domain_name: str, project_root: Path = None) -> Optional[Dict[str, Any]]:
    """
    Load unified pipeline results for a domain.
    
    Parameters
    ----------
    domain_name : str
        Domain name (e.g., 'nba', 'tennis')
    project_root : Path, optional
        Project root directory
        
    Returns
    -------
    results : dict or None
        Unified results format, or None if not found
    """
    if project_root is None:
        # Auto-detect: go up from utils to narrative_optimization
        project_root = Path(__file__).parent.parent
    
    results_path = project_root / 'domains' / domain_name / f'{domain_name}_results.json'
    
    if not results_path.exists():
        return None
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results for {domain_name}: {e}")
        return None


def extract_stats_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract dashboard statistics from unified results.
    
    Parameters
    ----------
    results : dict
        Unified results format
        
    Returns
    -------
    stats : dict
        Dashboard statistics
    """
    if results is None:
        return {}
    
    analysis = results.get('analysis', {})
    comprehensive = results.get('comprehensive_ю', {})
    
    stats = {
        'pi': results.get('pi', 0),
        'domain_type': results.get('domain_type', 'unknown'),
        'r_narrative': analysis.get('r_narrative', 0),
        'Д': analysis.get('Д', 0),
        'efficiency': analysis.get('efficiency', 0),
        'n_organisms': len(analysis.get('ю', [])) if 'ю' in analysis else 0,
        'n_features': len(analysis.get('feature_names', [])) if 'feature_names' in analysis else 0,
    }
    
    # Add comprehensive stats
    if comprehensive:
        perspectives = comprehensive.get('ю_perspectives', {})
        methods = comprehensive.get('ю_methods', {})
        scales = comprehensive.get('ю_scales', {})
        
        stats['n_perspectives'] = len(perspectives)
        stats['n_methods'] = len(methods)
        stats['n_scales'] = len(scales)
        
        # Calculate perspective means
        if perspectives:
            stats['perspective_means'] = {
                name: float(np.mean(scores)) if isinstance(scores, list) else 0
                for name, scores in perspectives.items()
            }
        
        # Calculate method means
        if methods:
            stats['method_means'] = {
                name: float(np.mean(scores)) if isinstance(scores, list) else 0
                for name, scores in methods.items()
            }
        
        # Get importance scores
        importance = comprehensive.get('importance', {})
        if importance:
            stats['perspective_importance'] = importance.get('perspectives', {})
            stats['method_importance'] = importance.get('methods', {})
            stats['scale_importance'] = importance.get('scales', {})
    
    return stats


def get_chart_data(results: Dict[str, Any], chart_type: str) -> Dict[str, Any]:
    """
    Extract chart data from unified results.
    
    Parameters
    ----------
    results : dict
        Unified results format
    chart_type : str
        Type of chart: 'perspective_comparison', 'method_comparison', 
                      'scale_comparison', 'importance_heatmap', etc.
        
    Returns
    -------
    chart_data : dict
        Chart data in format expected by Chart.js/Plotly
    """
    if results is None:
        return {}
    
    comprehensive = results.get('comprehensive_ю', {})
    
    if chart_type == 'perspective_comparison':
        perspectives = comprehensive.get('ю_perspectives', {})
        if not perspectives:
            return {}
        
        return {
            'labels': list(perspectives.keys()),
            'datasets': [{
                'label': 'Mean ю',
                'data': [
                    float(np.mean(scores)) if isinstance(scores, list) else 0
                    for scores in perspectives.values()
                ],
                'backgroundColor': [
                    f'rgba({i*30}, {100+i*20}, {200-i*10}, 0.8)'
                    for i in range(len(perspectives))
                ]
            }]
        }
    
    elif chart_type == 'method_comparison':
        methods = comprehensive.get('ю_methods', {})
        if not methods:
            return {}
        
        return {
            'labels': list(methods.keys()),
            'datasets': [{
                'label': 'Mean ю',
                'data': [
                    float(np.mean(scores)) if isinstance(scores, list) else 0
                    for scores in methods.values()
                ],
                'borderColor': 'rgba(75, 192, 192, 1)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'fill': True
            }]
        }
    
    elif chart_type == 'scale_comparison':
        scales = comprehensive.get('ю_scales', {})
        if not scales:
            return {}
        
        return {
            'labels': list(scales.keys()),
            'datasets': [{
                'label': 'Mean ю',
                'data': [
                    float(np.mean(scores)) if isinstance(scores, list) else 0
                    for scores in scales.values()
                ],
                'backgroundColor': [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)'
                ][:len(scales)]
            }]
        }
    
    elif chart_type == 'importance_heatmap':
        importance = comprehensive.get('importance', {})
        if not importance:
            return {}
        
        # Create correlation matrix
        perspective_importance = importance.get('perspectives', {})
        method_importance = importance.get('methods', {})
        
        data = []
        labels = []
        
        for name, imp_data in perspective_importance.items():
            labels.append(f'Perspective: {name}')
            data.append([imp_data.get('correlation', 0)])
        
        for name, imp_data in method_importance.items():
            labels.append(f'Method: {name}')
            data.append([imp_data.get('correlation', 0)])
        
        return {
            'labels': labels,
            'data': data,
            'type': 'heatmap'
        }
    
    elif chart_type == 'efficiency_breakdown':
        analysis = results.get('analysis', {})
        comprehensive = results.get('comprehensive_ю', {})
        
        pi = results.get('pi', 0)
        perspectives = comprehensive.get('ю_perspectives', {})
        
        if not perspectives:
            return {}
        
        # Calculate efficiency for each perspective
        outcomes = analysis.get('outcomes', [])
        if not outcomes:
            return {}
        
        efficiency_data = {}
        for name, ю_scores in perspectives.items():
            if isinstance(ю_scores, list) and len(ю_scores) == len(outcomes):
                # Calculate correlation
                r = float(np.corrcoef(ю_scores, outcomes)[0, 1])
                Д = pi * r * 0.5  # Default coupling
                efficiency = Д / pi if pi > 0 else 0
                efficiency_data[name] = efficiency
        
        return {
            'labels': list(efficiency_data.keys()),
            'datasets': [{
                'label': 'Efficiency (Д/п)',
                'data': list(efficiency_data.values()),
                'backgroundColor': 'rgba(40, 167, 69, 0.8)'
            }]
        }
    
    return {}


def get_top_features(results: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top features from results.
    
    Parameters
    ----------
    results : dict
        Unified results format
    n : int
        Number of top features to return
        
    Returns
    -------
    top_features : list
        List of top feature dictionaries
    """
    if results is None:
        return []
    
    top_features = results.get('top_features', [])
    if isinstance(top_features, list):
        return top_features[:n]
    
    return []

