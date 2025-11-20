"""
Stigma Analyzer for Mental Health Disorders

Analyzes the relationship between disorder names and stigma outcomes:
- Phonetic harshness → perceived severity
- Clinical framing → accessibility barriers
- Combined effects → treatment seeking behavior

Author: Narrative Optimization Research
Date: November 2025
"""

from typing import Dict, List
import numpy as np
from scipy import stats


class StigmaAnalyzer:
    """
    Analyze stigma patterns in mental health nomenclature.
    
    Tests the pathway: Harsh name → High stigma → Low treatment seeking
    """
    
    def __init__(self):
        """Initialize stigma analyzer."""
        pass
    
    def analyze_name_stigma_correlation(self, disorders: List[Dict]) -> Dict:
        """
        Analyze correlation between name features and stigma.
        
        Parameters
        ----------
        disorders : list of dict
            Disorder records with phonetic and stigma data
        
        Returns
        -------
        dict
            Correlation analysis results
        """
        # Extract features
        harshness_scores = []
        stigma_scores = []
        syllable_counts = []
        
        for disorder in disorders:
            phonetic = disorder.get('phonetic_analysis', {})
            social = disorder.get('social_impact', {})
            
            if 'harshness_score' in phonetic and 'stigma_score' in social:
                harshness_scores.append(phonetic['harshness_score'])
                stigma_scores.append(social['stigma_score'])
                
                if 'syllables' in phonetic:
                    syllable_counts.append(phonetic['syllables'])
        
        if len(harshness_scores) < 3:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Calculate correlations
        r_harshness, p_harshness = stats.pearsonr(harshness_scores, stigma_scores)
        
        results = {
            'n_disorders': len(harshness_scores),
            'harshness_stigma_correlation': r_harshness,
            'harshness_stigma_pvalue': p_harshness,
            'harshness_mean': np.mean(harshness_scores),
            'harshness_std': np.std(harshness_scores),
            'stigma_mean': np.mean(stigma_scores),
            'stigma_std': np.std(stigma_scores)
        }
        
        if syllable_counts:
            r_syllables, p_syllables = stats.pearsonr(syllable_counts, stigma_scores)
            results['syllables_stigma_correlation'] = r_syllables
            results['syllables_stigma_pvalue'] = p_syllables
        
        return results
    
    def analyze_treatment_seeking_pathway(self, disorders: List[Dict]) -> Dict:
        """
        Test the mediation pathway: Name → Stigma → Treatment seeking.
        
        Parameters
        ----------
        disorders : list of dict
            Disorder records with complete data
        
        Returns
        -------
        dict
            Mediation analysis results
        """
        # Extract complete pathway data
        harshness = []
        stigma = []
        treatment_seeking = []
        
        for disorder in disorders:
            phonetic = disorder.get('phonetic_analysis', {})
            social = disorder.get('social_impact', {})
            clinical = disorder.get('clinical_outcomes', {})
            
            if (('harshness_score' in phonetic) and 
                ('stigma_score' in social) and
                ('treatment_seeking_rate' in clinical)):
                
                harshness.append(phonetic['harshness_score'])
                stigma.append(social['stigma_score'])
                treatment_seeking.append(clinical['treatment_seeking_rate'])
        
        if len(harshness) < 3:
            return {'error': 'Insufficient data for mediation analysis'}
        
        # Path analysis
        # Path a: Harshness → Stigma
        r_a, p_a = stats.pearsonr(harshness, stigma)
        
        # Path b: Stigma → Treatment seeking
        r_b, p_b = stats.pearsonr(stigma, treatment_seeking)
        
        # Path c: Harshness → Treatment seeking (total effect)
        r_c, p_c = stats.pearsonr(harshness, treatment_seeking)
        
        # Indirect effect (mediation): a × b
        indirect_effect = r_a * r_b
        
        return {
            'n_disorders': len(harshness),
            'path_a_harshness_to_stigma': {
                'correlation': r_a,
                'p_value': p_a,
                'interpretation': 'Harsh names → Higher stigma'
            },
            'path_b_stigma_to_seeking': {
                'correlation': r_b,
                'p_value': p_b,
                'interpretation': 'Higher stigma → Lower treatment seeking'
            },
            'path_c_total_effect': {
                'correlation': r_c,
                'p_value': p_c,
                'interpretation': 'Harsh names → Lower treatment seeking (total effect)'
            },
            'indirect_effect': indirect_effect,
            'mediation_supported': (abs(r_a) > 0.15 and abs(r_b) > 0.15 and 
                                   p_a < 0.05 and p_b < 0.05)
        }
    
    def compare_severity_levels(self, disorders: List[Dict]) -> Dict:
        """
        Compare stigma across clinical severity levels.
        
        Tests: Do harsh names amplify stigma even for mild conditions?
        
        Parameters
        ----------
        disorders : list of dict
            Disorder records
        
        Returns
        -------
        dict
            Comparison results
        """
        # Group by severity
        high_severity = []
        low_severity = []
        
        for disorder in disorders:
            clinical = disorder.get('clinical_outcomes', {})
            social = disorder.get('social_impact', {})
            
            mortality = clinical.get('mortality_rate_per_100k', 0)
            stigma = social.get('stigma_score')
            
            if stigma is not None:
                if mortality > 100:  # High severity
                    high_severity.append(stigma)
                else:  # Lower severity
                    low_severity.append(stigma)
        
        if not high_severity or not low_severity:
            return {'error': 'Insufficient data in severity categories'}
        
        # T-test
        t_stat, p_val = stats.ttest_ind(high_severity, low_severity)
        
        return {
            'high_severity_stigma': {
                'mean': np.mean(high_severity),
                'std': np.std(high_severity),
                'n': len(high_severity)
            },
            'low_severity_stigma': {
                'mean': np.mean(low_severity),
                'std': np.std(low_severity),
                'n': len(low_severity)
            },
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

