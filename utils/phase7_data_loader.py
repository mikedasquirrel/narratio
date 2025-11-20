"""
Phase 7 Data Loader

Utility to load Phase 7 features (θ and λ) for website display.
Maps website domain names to Phase 7 extraction files.

Author: Narrative Integration System
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Domain mapping: website domain name → Phase 7 file name
DOMAIN_MAPPING = {
    # Sports
    'nba': 'nba_all_seasons_real_phase7.npz',
    'nfl': 'nflset_phase7.npz',
    'tennis': 'tennisset_phase7.npz',
    'golf': 'golf_enhanced_narratives_phase7.npz',
    'ufc': 'ufc_with_narratives_phase7.npz',
    'mlb': 'mlbset_phase7.npz',
    
    # Entertainment & Media
    'mental_health': 'mental_health_phase7.npz',
    'crypto': 'crypto_with_competitive_context_phase7.npz',
    'startups': 'startups_real_phase7.npz',
    
    # Benchmarks
    'lottery': 'coin_flips_benchmark_phase7.npz',  # Closest proxy
    'aviation': 'airports_with_narratives_phase7.npz',
    
    # Note: Some domains don't have direct Phase 7 files
    # These will return None and use domain-level estimates
}


def load_phase7_summary() -> Dict:
    """Load Phase 7 extraction summary"""
    summary_path = PROJECT_ROOT / 'narrative_optimization' / 'data' / 'phase7_extraction_summary.json'
    
    if not summary_path.exists():
        return {'results': []}
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_phase7_for_domain(domain_name: str) -> Optional[Dict]:
    """
    Load Phase 7 features for a specific domain.
    
    Parameters
    ----------
    domain_name : str
        Website domain name (e.g., 'nba', 'mental_health')
    
    Returns
    -------
    phase7_data : dict or None
        Dictionary with theta_mean, lambda_mean, samples, etc.
        None if no Phase 7 data available for this domain
    """
    if domain_name not in DOMAIN_MAPPING:
        return None
    
    phase7_file = DOMAIN_MAPPING[domain_name]
    phase7_path = PROJECT_ROOT / 'narrative_optimization' / 'data' / 'features' / 'phase7' / phase7_file
    
    if not phase7_path.exists():
        return None
    
    try:
        data = np.load(phase7_path)
        
        theta_values = data['theta_values']
        lambda_values = data['lambda_values']
        n_samples = int(data['n_samples'])
        
        return {
            'theta_mean': float(theta_values.mean()),
            'theta_std': float(theta_values.std()),
            'theta_min': float(theta_values.min()),
            'theta_max': float(theta_values.max()),
            'lambda_mean': float(lambda_values.mean()),
            'lambda_std': float(lambda_values.std()),
            'lambda_min': float(lambda_values.min()),
            'lambda_max': float(lambda_values.max()),
            'samples': n_samples,
            'phase7_coverage': True
        }
    except Exception as e:
        print(f"Error loading Phase 7 data for {domain_name}: {e}")
        return None


def get_all_phase7_domains() -> Dict[str, Dict]:
    """
    Get Phase 7 data for all domains that have it.
    
    Returns
    -------
    phase7_data : dict
        Dictionary mapping domain names to their Phase 7 data
    """
    result = {}
    
    for domain_name in DOMAIN_MAPPING.keys():
        data = load_phase7_for_domain(domain_name)
        if data:
            result[domain_name] = data
    
    return result


def get_force_interpretation(theta: float, lambda_val: float) -> Dict[str, str]:
    """
    Interpret force balance.
    
    Parameters
    ----------
    theta : float
        Awareness resistance
    lambda_val : float
        Fundamental constraints
    
    Returns
    -------
    interpretation : dict
        Interpretation of force balance
    """
    interpretation = {}
    
    # Theta interpretation
    if theta < 0.3:
        interpretation['theta'] = "Low awareness - minimal conscious resistance"
    elif theta < 0.5:
        interpretation['theta'] = "Moderate awareness - some conscious resistance"
    elif theta < 0.7:
        interpretation['theta'] = "High awareness - significant meta-awareness"
    else:
        interpretation['theta'] = "Very high awareness - strong deliberate resistance"
    
    # Lambda interpretation
    if lambda_val < 0.3:
        interpretation['lambda'] = "Low constraints - minimal barriers"
    elif lambda_val < 0.5:
        interpretation['lambda'] = "Moderate constraints - some requirements"
    elif lambda_val < 0.7:
        interpretation['lambda'] = "High constraints - significant barriers"
    else:
        interpretation['lambda'] = "Very high constraints - physics/training dominates"
    
    # Force balance
    if lambda_val > theta + 0.2:
        interpretation['dominant'] = "λ dominates - physics/training overwhelms narrative"
    elif theta > lambda_val + 0.2:
        interpretation['dominant'] = "θ dominates - awareness suppresses narrative effects"
    elif lambda_val > 0.6 and theta > 0.6:
        interpretation['dominant'] = "Both high - aware + constrained"
    else:
        interpretation['dominant'] = "Balanced forces - ة can operate"
    
    return interpretation


def format_for_api(domain_name: str, phase7_data: Optional[Dict]) -> Dict:
    """
    Format Phase 7 data for API response.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    phase7_data : dict or None
        Phase 7 data or None
    
    Returns
    -------
    formatted : dict
        Formatted for API response
    """
    if not phase7_data:
        return {
            'phase7_coverage': False,
            'theta': None,
            'lambda': None
        }
    
    interpretation = get_force_interpretation(
        phase7_data['theta_mean'],
        phase7_data['lambda_mean']
    )
    
    return {
        'phase7_coverage': True,
        'theta': phase7_data['theta_mean'],
        'theta_std': phase7_data['theta_std'],
        'lambda': phase7_data['lambda_mean'],
        'lambda_std': phase7_data['lambda_std'],
        'phase7_samples': phase7_data['samples'],
        'force_interpretation': interpretation
    }

