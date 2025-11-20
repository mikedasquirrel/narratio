"""
Formula Validation: Test π/λ/θ Predictions Against Classical Theory

Validates that π/λ/θ framework recovers known classical patterns:
- High π → High Hero's Journey completion
- High λ → High structural adherence
- High θ → High irony/meta-narrative
- Frye's mythoi cluster in θ/λ space

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformers.archetypes import (
    HeroJourneyTransformer,
    CharacterArchetypeTransformer,
    PlotArchetypeTransformer,
    StructuralBeatTransformer,
    ThematicArchetypeTransformer
)
from config.domain_archetypes import DOMAIN_ARCHETYPES


class FormulaValidator:
    """
    Validate π/λ/θ formulas against classical narrative theory predictions.
    """
    
    def __init__(self):
        self.transformers = {
            'journey': HeroJourneyTransformer(),
            'character': CharacterArchetypeTransformer(),
            'plot': PlotArchetypeTransformer(),
            'structural': StructuralBeatTransformer(),
            'thematic': ThematicArchetypeTransformer()
        }
        
        self.validation_results = {}
    
    def validate_pi_journey_correlation(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 1: High π → High Hero's Journey completion
        
        Args:
            domain_data: {domain_name: {'texts': [...], 'outcomes': [...], 'pi': float}}
            
        Returns:
            Validation results with correlation and significance
        """
        pi_values = []
        journey_completions = []
        domain_names = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            pi = data.get('pi') or DOMAIN_ARCHETYPES.get(domain_name, {}).get('pi', 0.5)
            
            # Extract journey completion
            transformer = self.transformers['journey']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # Feature index 2 is typically journey_completion_mean
            journey_completion = np.mean(features[:, 2])
            
            pi_values.append(pi)
            journey_completions.append(journey_completion)
            domain_names.append(domain_name)
        
        # Calculate correlation
        corr, pval = pearsonr(pi_values, journey_completions)
        
        # Linear regression R²
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = np.array(pi_values).reshape(-1, 1)
        y = np.array(journey_completions)
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        result = {
            'hypothesis': 'High π → High Hero\'s Journey completion',
            'correlation': corr,
            'p_value': pval,
            'r_squared': r2,
            'validated': corr > 0.70 and pval < 0.05,
            'domains': list(zip(domain_names, pi_values, journey_completions)),
            'interpretation': self._interpret_correlation(corr, pval, 0.70)
        }
        
        self.validation_results['pi_journey'] = result
        return result
    
    def validate_lambda_structure_correlation(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 2: High λ → High structural adherence
        
        Structural adherence = beat timing + act structure + Aristotelian principles
        """
        lambda_values = []
        structure_scores = []
        domain_names = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            
            # Get λ from domain config
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            lambda_val = np.mean(config.get('lambda_range', (0.5, 0.5)))
            
            # Extract structural adherence
            transformer = self.transformers['structural']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # Overall structure quality (typically last feature)
            structure_score = np.mean(features[:, -1])
            
            lambda_values.append(lambda_val)
            structure_scores.append(structure_score)
            domain_names.append(domain_name)
        
        corr, pval = pearsonr(lambda_values, structure_scores)
        
        result = {
            'hypothesis': 'High λ → High structural adherence',
            'correlation': corr,
            'p_value': pval,
            'validated': corr > 0.60 and pval < 0.05,
            'domains': list(zip(domain_names, lambda_values, structure_scores)),
            'interpretation': self._interpret_correlation(corr, pval, 0.60)
        }
        
        self.validation_results['lambda_structure'] = result
        return result
    
    def validate_theta_irony_correlation(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 3: High θ → High irony/satire (Frye)
        """
        theta_values = []
        irony_scores = []
        domain_names = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            
            # Get θ
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            theta = np.mean(config.get('theta_range', (0.5, 0.5)))
            
            # Extract irony score
            transformer = self.transformers['thematic']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # Frye irony is 4th mythos (index 3)
            irony_score = np.mean(features[:, 3])
            
            theta_values.append(theta)
            irony_scores.append(irony_score)
            domain_names.append(domain_name)
        
        corr, pval = pearsonr(theta_values, irony_scores)
        
        result = {
            'hypothesis': 'High θ → High irony/satire (Frye)',
            'correlation': corr,
            'p_value': pval,
            'validated': corr > 0.65 and pval < 0.05,
            'domains': list(zip(domain_names, theta_values, irony_scores)),
            'interpretation': self._interpret_correlation(corr, pval, 0.65)
        }
        
        self.validation_results['theta_irony'] = result
        return result
    
    def validate_frye_theta_lambda_clustering(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 4: Frye's mythoi cluster distinctly in θ/λ space
        
        Expected clusters:
        - Comedy: θ≈0.30, λ≈0.50
        - Romance: θ≈0.20, λ≈0.30
        - Tragedy: θ≈0.55, λ≈0.75
        - Irony: θ≈0.85, λ≈0.50
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        coordinates = []  # (θ, λ) pairs
        mythos_labels = []  # Dominant mythos per sample
        sample_info = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            
            # Get θ and λ
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            theta = np.mean(config.get('theta_range', (0.5, 0.5)))
            lambda_val = np.mean(config.get('lambda_range', (0.5, 0.5)))
            
            # Extract mythoi scores
            transformer = self.transformers['thematic']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # First 4 features are mythoi
            mythoi = features[:, :4]
            
            for i in range(len(texts)):
                coordinates.append([theta, lambda_val])
                dominant_mythos = np.argmax(mythoi[i])
                mythos_labels.append(dominant_mythos)
                sample_info.append({
                    'domain': domain_name,
                    'theta': theta,
                    'lambda': lambda_val,
                    'mythos_scores': mythoi[i].tolist()
                })
        
        if len(coordinates) < 10:
            return {'error': 'Insufficient samples for clustering test'}
        
        # K-means with 4 clusters
        coordinates_array = np.array(coordinates)
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(coordinates_array)
        
        # Calculate silhouette score (clustering quality)
        silhouette = silhouette_score(coordinates_array, clusters)
        
        # Map clusters to mythoi
        cluster_mythoi_map = {}
        for cluster_id in range(4):
            cluster_mask = clusters == cluster_id
            cluster_mythoi = [mythos_labels[i] for i in range(len(mythos_labels)) if cluster_mask[i]]
            
            if cluster_mythoi:
                # Most common mythos in this cluster
                from collections import Counter
                most_common = Counter(cluster_mythoi).most_common(1)[0]
                cluster_mythoi_map[cluster_id] = {
                    'dominant_mythos': most_common[0],
                    'purity': most_common[1] / len(cluster_mythoi)
                }
        
        result = {
            'hypothesis': 'Frye\'s mythoi cluster in θ/λ space',
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_mythoi_mapping': cluster_mythoi_map,
            'validated': silhouette > 0.40,  # Reasonable clustering
            'interpretation': 'Strong clustering' if silhouette > 0.50 else 
                             'Moderate clustering' if silhouette > 0.30 else 'Weak clustering'
        }
        
        self.validation_results['frye_clustering'] = result
        return result
    
    def validate_aristotelian_principles(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 5: Aristotelian principles predict success in high-λ domains
        """
        high_lambda_results = []
        low_lambda_results = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            outcomes = data['outcomes']
            
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            lambda_val = np.mean(config.get('lambda_range', (0.5, 0.5)))
            
            # Extract structural features (proxy for Aristotelian principles)
            transformer = self.transformers['structural']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # Overall structure quality
            structure_quality = features[:, -1]
            
            # Correlation with outcomes
            if len(outcomes) == len(structure_quality):
                corr, pval = pearsonr(structure_quality, outcomes)
                
                result = {
                    'domain': domain_name,
                    'lambda': lambda_val,
                    'correlation': corr,
                    'p_value': pval,
                    'r_squared': corr ** 2 if corr > 0 else 0
                }
                
                if lambda_val > 0.65:
                    high_lambda_results.append(result)
                else:
                    low_lambda_results.append(result)
        
        # Compare high vs low λ domains
        high_lambda_r2 = np.mean([r['r_squared'] for r in high_lambda_results]) if high_lambda_results else 0
        low_lambda_r2 = np.mean([r['r_squared'] for r in low_lambda_results]) if low_lambda_results else 0
        
        result = {
            'hypothesis': 'Aristotelian principles predict success when λ > 0.65',
            'high_lambda_r2': high_lambda_r2,
            'low_lambda_r2': low_lambda_r2,
            'difference': high_lambda_r2 - low_lambda_r2,
            'validated': high_lambda_r2 > 0.50 and (high_lambda_r2 - low_lambda_r2) > 0.20,
            'high_lambda_domains': high_lambda_results,
            'low_lambda_domains': low_lambda_results
        }
        
        self.validation_results['aristotelian'] = result
        return result
    
    def validate_archetype_clarity_ta_marbuta(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Test Hypothesis 6: Jung's archetype clarity correlates with ة (nominative gravity)
        
        Clear archetypes have memorable, iconic names.
        """
        clarity_values = []
        ta_marbuta_values = []
        domain_names = []
        
        for domain_name, data in domain_data.items():
            texts = data['texts']
            
            config = DOMAIN_ARCHETYPES.get(domain_name, {})
            ta_marbuta = 0.70  # Default, would compute from nominative features
            
            # Extract archetype clarity
            transformer = self.transformers['character']
            transformer.fit(texts)
            features = transformer.transform(texts)
            
            # Jung archetype clarity (index 12 typically)
            clarity = np.mean(features[:, 12])
            
            clarity_values.append(clarity)
            ta_marbuta_values.append(ta_marbuta)
            domain_names.append(domain_name)
        
        corr, pval = pearsonr(clarity_values, ta_marbuta_values)
        
        result = {
            'hypothesis': 'Archetype clarity correlates with ة (nominative gravity)',
            'correlation': corr,
            'p_value': pval,
            'validated': corr > 0.60 and pval < 0.05,
            'domains': list(zip(domain_names, clarity_values, ta_marbuta_values)),
            'interpretation': self._interpret_correlation(corr, pval, 0.60)
        }
        
        self.validation_results['clarity_nominative'] = result
        return result
    
    def run_all_validations(self, domain_data: Dict[str, Dict]) -> Dict:
        """
        Run all validation tests.
        
        Args:
            domain_data: {domain_name: {'texts': [...], 'outcomes': [...], 'pi': float}}
            
        Returns:
            Complete validation results
        """
        results = {
            'summary': {},
            'validations': {}
        }
        
        # Run each validation
        print("Running validation 1/6: π → Journey correlation...")
        results['validations']['pi_journey'] = self.validate_pi_journey_correlation(domain_data)
        
        print("Running validation 2/6: λ → Structure correlation...")
        results['validations']['lambda_structure'] = self.validate_lambda_structure_correlation(domain_data)
        
        print("Running validation 3/6: θ → Irony correlation...")
        results['validations']['theta_irony'] = self.validate_theta_irony_correlation(domain_data)
        
        print("Running validation 4/6: Frye clustering...")
        results['validations']['frye_clustering'] = self.validate_frye_theta_lambda_clustering(domain_data)
        
        print("Running validation 5/6: Aristotelian principles...")
        results['validations']['aristotelian'] = self.validate_aristotelian_principles(domain_data)
        
        print("Running validation 6/6: Clarity-Nominative correlation...")
        results['validations']['clarity_nominative'] = self.validate_archetype_clarity_ta_marbuta(domain_data)
        
        # Summary statistics
        validated_count = sum([1 for v in results['validations'].values() 
                              if v.get('validated', False)])
        
        results['summary'] = {
            'total_tests': 6,
            'tests_validated': validated_count,
            'validation_rate': validated_count / 6,
            'overall_validated': validated_count >= 4,  # 4 of 6 is success
            'domains_analyzed': len(domain_data),
            'total_samples': sum([len(d['texts']) for d in domain_data.values()])
        }
        
        return results
    
    def _interpret_correlation(self, corr: float, pval: float, threshold: float) -> str:
        """Interpret correlation result."""
        if pval >= 0.05:
            return f"Not significant (p={pval:.3f})"
        
        if corr >= threshold + 0.10:
            return f"Strong validation (r={corr:.3f}, exceeds threshold {threshold})"
        elif corr >= threshold:
            return f"Validated (r={corr:.3f}, meets threshold {threshold})"
        elif corr >= threshold - 0.10:
            return f"Weak validation (r={corr:.3f}, near threshold {threshold})"
        else:
            return f"Not validated (r={corr:.3f}, below threshold {threshold})"
    
    def generate_validation_report(self, output_path: str) -> None:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            raise ValueError("No validation results. Run run_all_validations first.")
        
        report = {
            'validation_results': self.validation_results,
            'timestamp': pd.Timestamp.now().isoformat(),
            'framework': 'π/λ/θ/ة archetype integration'
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report saved to: {output_path}")


# Quick validation function
def quick_validate(domain_datasets: Dict[str, Dict]) -> Dict:
    """
    Quick validation of all hypotheses.
    
    Args:
        domain_datasets: {domain_name: {'texts': [...], 'outcomes': [...], 'pi': float}}
        
    Returns:
        Validation results
    """
    validator = FormulaValidator()
    results = validator.run_all_validations(domain_datasets)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(f"Tests validated: {results['summary']['tests_validated']}/6")
    print(f"Overall validated: {results['summary']['overall_validated']}")
    print(f"Domains analyzed: {results['summary']['domains_analyzed']}")
    print(f"Total samples: {results['summary']['total_samples']}")
    print("="*60 + "\n")
    
    return results

