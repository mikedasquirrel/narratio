"""
Constant Detector

Searches for mathematical constants across narrative data.
Like your 1.338 discovery in NBA player names.

Looks for:
- Universal ratios
- Scaling factors
- Decay constants
- Golden ratios
- Recurring mathematical relationships
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
from itertools import combinations


class ConstantDetector:
    """
    Detects mathematical constants in narrative data.
    
    Your discovery: NBA name constant = 1.338
    Question: What other constants exist?
    """
    
    def __init__(self):
        self.discovered_constants = {}
        self.candidate_constants = []
    
    def search_for_ratios(
        self,
        features: Dict[str, np.ndarray],
        significance_threshold: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Search for significant ratios between features.
        
        Looks for: feature_i / feature_j ≈ constant across samples
        """
        print(f"\n{'='*70}")
        print(f"SEARCHING FOR MATHEMATICAL CONSTANTS")
        print(f"{'='*70}")
        
        constants_found = []
        
        feature_pairs = list(combinations(features.keys(), 2))
        
        print(f"\nTesting {len(feature_pairs)} feature pairs for constant ratios...")
        
        for feat1, feat2 in feature_pairs:
            values1 = features[feat1]
            values2 = features[feat2]
            
            # Skip if either has zeros
            if np.any(values2 == 0) or np.any(values1 == 0):
                continue
            
            # Calculate ratios
            ratios = values1 / (values2 + 1e-10)
            
            # Check if ratio is approximately constant
            ratio_mean = np.mean(ratios)
            ratio_std = np.std(ratios)
            cv = ratio_std / (ratio_mean + 1e-10)  # Coefficient of variation
            
            # If coefficient of variation is low, ratio is constant
            if cv < 0.3 and ratio_mean > 0:  # Low variance relative to mean
                # Statistical test
                _, p_value = stats.ttest_1samp(ratios, ratio_mean)
                
                if p_value < significance_threshold:  # Significantly consistent
                    constants_found.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'ratio': float(ratio_mean),
                        'std': float(ratio_std),
                        'cv': float(cv),
                        'p_value': float(p_value),
                        'interpretation': f"{feat1} / {feat2} ≈ {ratio_mean:.3f}"
                    })
        
        # Sort by consistency (lowest CV)
        constants_found.sort(key=lambda x: x['cv'])
        
        # Display top 10
        print(f"\nTOP 10 MOST CONSISTENT RATIOS:")
        print(f"{'Rank':<6} {'Ratio':<50} {'Value':<10} {'CV':<8}")
        print(f"{'-'*80}")
        
        for i, const in enumerate(constants_found[:10], 1):
            ratio_str = f"{const['feature_1'][:20]} / {const['feature_2'][:20]}"
            print(f"{i:<6} {ratio_str:<50} {const['ratio']:<10.3f} {const['cv']:<8.3f}")
        
        self.candidate_constants = constants_found
        
        return constants_found
    
    def search_for_scaling_factors(
        self,
        feature: np.ndarray,
        outcome: np.ndarray
    ) -> Dict[str, float]:
        """
        Find scaling factor: outcome ≈ k * feature
        
        Returns optimal k and fit quality.
        """
        # Linear regression to find k
        k = np.dot(feature, outcome) / (np.dot(feature, feature) + 1e-10)
        
        # Predicted values
        predicted = k * feature
        
        # R-squared
        ss_res = np.sum((outcome - predicted) ** 2)
        ss_tot = np.sum((outcome - np.mean(outcome)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'scaling_factor': float(k),
            'r_squared': float(r_squared),
            'interpretation': f"outcome ≈ {k:.3f} * feature"
        }
    
    def detect_golden_ratios(self, features: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Search for golden ratio (φ ≈ 1.618) or other special constants.
        
        Famous constants to check:
        - φ (phi): 1.618 (golden ratio)
        - e: 2.718 (Euler's number)
        - Your constant: 1.338
        """
        famous_constants = {
            'phi': 1.618,
            'sqrt_2': 1.414,
            'sqrt_3': 1.732,
            'e': 2.718,
            'user_constant': 1.338
        }
        
        golden_ratio_matches = []
        
        for feat1, feat2 in combinations(features.keys(), 2):
            values1 = features[feat1]
            values2 = features[feat2]
            
            if np.any(values2 == 0):
                continue
            
            ratios = values1 / (values2 + 1e-10)
            ratio_mean = np.mean(ratios)
            
            # Check proximity to famous constants
            for const_name, const_value in famous_constants.items():
                if abs(ratio_mean - const_value) < 0.1:  # Within 0.1
                    golden_ratio_matches.append({
                        'constant_name': const_name,
                        'constant_value': const_value,
                        'observed_ratio': float(ratio_mean),
                        'difference': float(abs(ratio_mean - const_value)),
                        'features': (feat1, feat2),
                        'interpretation': f"{feat1}/{feat2} ≈ {const_name} ({const_value})"
                    })
        
        if golden_ratio_matches:
            print(f"\n🌟 SPECIAL CONSTANTS DETECTED:")
            for match in golden_ratio_matches:
                print(f"   {match['interpretation']}")
                print(f"   Observed: {match['observed_ratio']:.3f} vs Expected: {match['constant_value']:.3f}")
        
        return golden_ratio_matches
    
    def comprehensive_constant_search(
        self,
        features: Dict[str, np.ndarray],
        outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run comprehensive search for all types of constants.
        """
        results = {
            'ratios': self.search_for_ratios(features),
            'golden_ratios': self.detect_golden_ratios(features),
            'scaling_factors': {},
            'summary': []
        }
        
        # Test each feature for scaling
        for feat_name, feat_values in list(features.items())[:10]:  # Test first 10
            scaling = self.search_for_scaling_factors(feat_values, outcomes)
            if scaling['r_squared'] > 0.1:  # Significant relationship
                results['scaling_factors'][feat_name] = scaling
        
        # Summarize
        n_ratios = len(results['ratios'])
        n_golden = len(results['golden_ratios'])
        n_scaling = len(results['scaling_factors'])
        
        results['summary'] = [
            f"Found {n_ratios} consistent feature ratios",
            f"Detected {n_golden} matches to special constants",
            f"Identified {n_scaling} scaling relationships"
        ]
        
        if n_golden > 0:
            results['summary'].append("🌟 Special constants present - mathematical structure detected!")
        
        return results


def create_constant_detector():
    """Factory function."""
    return ConstantDetector()

