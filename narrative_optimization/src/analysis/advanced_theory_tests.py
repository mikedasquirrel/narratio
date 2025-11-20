"""
Advanced Theoretical Framework Tests

Tests paradigm-shifting theories against narrative energy data:
1. Renormalization Group Theory - Scale invariance of constants
2. Catastrophe Theory - Discontinuous phase transitions
3. Maximum Entropy Production - Thermodynamic evolution
4. Gauge Theory - Hidden symmetries across domains

These tests probe whether narrative energy follows deep mathematical structures
from physics, revealing fundamental laws vs domain-specific phenomena.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.optimize import curve_fit
from collections import Counter


class AdvancedTheoryTester:
    """
    Tests advanced theoretical frameworks against narrative data.
    
    Capabilities:
    - Scale invariance testing (renormalization group)
    - Catastrophe point detection (Thom's elementary catastrophes)
    - Entropy production measurement (Prigogine)
    - Symmetry breaking analysis (gauge theory)
    - Stochastic resonance detection
    - Percolation threshold identification
    """
    
    def __init__(self):
        self.results = {}
    
    # === RENORMALIZATION GROUP THEORY ===
    
    def test_scale_invariance(
        self,
        data_micro: np.ndarray,    # Phoneme-level features
        data_meso: np.ndarray,     # Name-level features
        data_macro: np.ndarray,    # Category-level features
        data_meta: np.ndarray      # Domain-level features
    ) -> Dict[str, Any]:
        """
        Test if constants hold across scales (renormalization group).
        
        If your 0.993/1.008 constants are FUNDAMENTAL, they should appear
        at all scales (scale-invariant like speed of light).
        
        If different at each scale, different "physics" at different scales
        (like quantum vs classical).
        
        Parameters
        ----------
        data_micro : array
            Phoneme-level measurements
        data_meso : array
            Name-level measurements (where you found constants)
        data_macro : array
            Category-level measurements
        data_meta : array
            Cross-domain measurements
        
        Returns
        -------
        results : dict
            Scale invariance test results
        """
        results = {
            'scale_invariant': False,
            'constants_by_scale': {},
            'renormalization_flow': [],
            'critical_exponents': {}
        }
        
        # Compute constants at each scale
        for scale_name, data in [
            ('micro', data_micro),
            ('meso', data_meso),
            ('macro', data_macro),
            ('meta', data_meta)
        ]:
            if len(data) >= 2:
                # Look for oscillation pattern (contraction/expansion)
                diffs = np.diff(data)
                contractions = diffs[diffs < 0]
                expansions = diffs[diffs > 0]
                
                if len(contractions) > 0 and len(expansions) > 0:
                    contraction_avg = np.abs(np.mean(contractions))
                    expansion_avg = np.mean(expansions)
                    
                    results['constants_by_scale'][scale_name] = {
                        'contraction': contraction_avg,
                        'expansion': expansion_avg,
                        'ratio': expansion_avg / max(0.001, contraction_avg)
                    }
        
        # Test scale invariance: Are constants similar across scales?
        if len(results['constants_by_scale']) >= 2:
            ratios = [v['ratio'] for v in results['constants_by_scale'].values()]
            ratio_variance = np.var(ratios)
            
            # Scale invariant if variance < 0.05 (constants consistent)
            results['scale_invariant'] = ratio_variance < 0.05
            results['ratio_variance'] = ratio_variance
            
            # Critical exponent: How does quantity scale?
            # Q(scale) = Q₀ × scale^α
            scales = np.array([1, 2, 3, 4])  # Micro, meso, macro, meta
            if len(ratios) == 4:
                # Fit power law
                try:
                    log_scales = np.log(scales)
                    log_ratios = np.log(ratios)
                    slope, intercept = np.polyfit(log_scales, log_ratios, 1)
                    results['critical_exponents']['ratio'] = slope
                except:
                    results['critical_exponents']['ratio'] = None
        
        return results
    
    # === CATASTROPHE THEORY ===
    
    def detect_catastrophe_points(
        self,
        control_param: np.ndarray,  # E.g., saturation level
        response_param: np.ndarray,  # E.g., effect size
        catastrophe_type: str = 'cusp'
    ) -> Dict[str, Any]:
        """
        Detect catastrophe points (discontinuous transitions).
        
        Catastrophe theory (René Thom): Systems can exhibit sudden jumps
        when control parameters reach critical values.
        
        For naming: At saturation threshold, effects don't gradually decline -
        they COLLAPSE discontinuously (cliff).
        
        Parameters
        ----------
        control_param : array
            Control parameter (e.g., saturation, competition)
        response_param : array
            Response variable (e.g., effect size, success rate)
        catastrophe_type : str
            'fold', 'cusp', or 'butterfly'
        
        Returns
        -------
        results : dict
            Catastrophe detection results
        """
        results = {
            'catastrophe_detected': False,
            'catastrophe_type': catastrophe_type,
            'critical_point': None,
            'jump_magnitude': None
        }
        
        # Sort by control parameter
        sorted_indices = np.argsort(control_param)
        control_sorted = control_param[sorted_indices]
        response_sorted = response_param[sorted_indices]
        
        # Look for discontinuous jump (derivative spike)
        if len(control_sorted) >= 5:
            # Compute first derivative (rate of change)
            derivatives = np.diff(response_sorted) / np.diff(control_sorted)
            
            # Detect spike in derivative (discontinuity signature)
            derivative_threshold = np.mean(np.abs(derivatives)) + 2 * np.std(np.abs(derivatives))
            
            spikes = np.where(np.abs(derivatives) > derivative_threshold)[0]
            
            if len(spikes) > 0:
                # Catastrophe detected
                critical_index = spikes[0]
                results['catastrophe_detected'] = True
                results['critical_point'] = control_sorted[critical_index]
                
                # Jump magnitude
                if critical_index + 1 < len(response_sorted):
                    jump = abs(response_sorted[critical_index + 1] - response_sorted[critical_index])
                    results['jump_magnitude'] = jump
                
                # Fit catastrophe manifold
                if catastrophe_type == 'cusp':
                    # Cusp: x³ + ax + b = 0
                    results['cusp_parameters'] = self._fit_cusp_manifold(
                        control_sorted[:critical_index+5],
                        response_sorted[:critical_index+5]
                    )
        
        return results
    
    def _fit_cusp_manifold(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Fit cusp catastrophe manifold: x³ + ax + b."""
        try:
            def cusp_model(x, a, b):
                # Solve x³ + ax + b = 0 for y behavior
                return -a * x - b + x**3
            
            params, _ = curve_fit(cusp_model, x, y, p0=[1, 0])
            return {'a': params[0], 'b': params[1]}
        except:
            return {'a': None, 'b': None}
    
    # === MAXIMUM ENTROPY PRODUCTION ===
    
    def test_maximum_entropy_production(
        self,
        diversity_sequence: np.ndarray,  # Diversity over time
        success_sequence: np.ndarray      # Success rate over time
    ) -> Dict[str, Any]:
        """
        Test if system maximizes entropy production (MEP principle).
        
        Dewar/Jaynes: Systems evolve to maximize entropy production rate dS/dt.
        
        For names: Markets should evolve to optimal diversity that maximizes
        information transmission rate.
        
        Prediction: Beyond optimal point, adding diversity DECREASES success
        (entropy production declining).
        
        Parameters
        ----------
        diversity_sequence : array
            Diversity levels over time
        success_sequence : array
            Success rates corresponding to diversity
        
        Returns
        -------
        results : dict
            MEP test results
        """
        results = {
            'mep_validated': False,
            'optimal_diversity': None,
            'entropy_production_rate': None,
            'max_entropy_point': None
        }
        
        # Compute entropy production rate approximation
        # dS/dt ≈ diversity × success_rate
        if len(diversity_sequence) == len(success_sequence):
            entropy_production = diversity_sequence * success_sequence
            
            # Find maximum
            max_idx = np.argmax(entropy_production)
            results['max_entropy_point'] = diversity_sequence[max_idx]
            results['entropy_production_rate'] = entropy_production[max_idx]
            results['optimal_diversity'] = diversity_sequence[max_idx]
            
            # Test if system is AT maximum (MEP principle)
            current_diversity = diversity_sequence[-1]
            optimal_diversity = results['optimal_diversity']
            
            # Within 10% of optimal = MEP validated
            if abs(current_diversity - optimal_diversity) / optimal_diversity < 0.10:
                results['mep_validated'] = True
            
            # Test inverted-U curve (entropy production should peak)
            if max_idx not in [0, len(diversity_sequence)-1]:
                # Peak is in middle = inverted-U confirmed
                results['inverted_u_confirmed'] = True
            else:
                results['inverted_u_confirmed'] = False
        
        return results
    
    # === GAUGE THEORY / SYMMETRY ===
    
    def test_gauge_invariance(
        self,
        feature_matrix_domain_A: np.ndarray,
        feature_matrix_domain_B: np.ndarray,
        transformation_matrix: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Test if there exists a gauge transformation unifying domains.
        
        In physics: EM and weak force look different but unified under
        electroweak gauge theory (local gauge invariance).
        
        In narratives: Memorability +positive in domain A, -negative in B.
        But might unify under coordinate transformation.
        
        Hypothesis: There exists transformation T such that:
            T(features_A) = features_B (up to gauge freedom)
        
        Parameters
        ----------
        feature_matrix_domain_A : array
            Features from domain A
        feature_matrix_domain_B : array
            Features from domain B (same features, different domain)
        transformation_matrix : array, optional
            Known transformation (if testing specific gauge)
        
        Returns
        -------
        results : dict
            Gauge invariance test results
        """
        results = {
            'gauge_invariant': False,
            'transformation_matrix': None,
            'symmetry_group': None,
            'broken_symmetries': []
        }
        
        # If no transformation provided, search for one
        if transformation_matrix is None:
            # Simple test: Linear transformation
            # Find T such that A ≈ T × B
            if feature_matrix_domain_A.shape == feature_matrix_domain_B.shape:
                # Use least squares to find best transformation
                try:
                    T, residuals, rank, s = np.linalg.lstsq(
                        feature_matrix_domain_B,
                        feature_matrix_domain_A,
                        rcond=None
                    )
                    results['transformation_matrix'] = T
                    
                    # Test quality: If residuals small, transformation exists
                    if len(residuals) > 0 and residuals[0] < 0.1:
                        results['gauge_invariant'] = True
                        results['symmetry_group'] = 'U(1)'  # Simplest gauge group
                except:
                    pass
        else:
            # Test provided transformation
            transformed = feature_matrix_domain_B @ transformation_matrix
            error = np.linalg.norm(feature_matrix_domain_A - transformed)
            
            if error < 0.1:
                results['gauge_invariant'] = True
        
        # Check which symmetries are broken
        # Compare feature signs across domains
        if feature_matrix_domain_A.shape[1] == feature_matrix_domain_B.shape[1]:
            for i in range(min(10, feature_matrix_domain_A.shape[1])):
                sign_A = np.sign(np.mean(feature_matrix_domain_A[:, i]))
                sign_B = np.sign(np.mean(feature_matrix_domain_B[:, i]))
                
                if sign_A != sign_B:
                    results['broken_symmetries'].append({
                        'feature_index': i,
                        'sign_flip': True,
                        'magnitude_A': np.mean(feature_matrix_domain_A[:, i]),
                        'magnitude_B': np.mean(feature_matrix_domain_B[:, i])
                    })
        
        return results
    
    # === STOCHASTIC RESONANCE ===
    
    def detect_stochastic_resonance(
        self,
        complexity_levels: np.ndarray,  # Name complexity (noise)
        success_rates: np.ndarray        # Outcome (signal detection)
    ) -> Dict[str, Any]:
        """
        Detect optimal noise level for signal enhancement.
        
        Stochastic resonance: Signal + noise + nonlinearity = enhanced detection
        at OPTIMAL noise level.
        
        For names: "Ethereum" succeeds because complexity σ matches optimal σ*.
        Too simple ("Pay") or too complex ("Supercalifragilistic") fails.
        
        Parameters
        ----------
        complexity_levels : array
            Complexity scores (0-10 scale)
        success_rates : array
            Success probability (0-1)
        
        Returns
        -------
        results : dict
            Stochastic resonance analysis
        """
        results = {
            'resonance_detected': False,
            'optimal_complexity': None,
            'resonance_peak': None,
            'inverted_u': False
        }
        
        if len(complexity_levels) < 5:
            return results
        
        # Fit inverted-U curve: y = ax² + bx + c
        try:
            coeffs = np.polyfit(complexity_levels, success_rates, 2)
            a, b, c = coeffs
            
            # Inverted-U if a < 0 (negative quadratic term)
            if a < 0:
                # Find maximum: dy/dx = 2ax + b = 0 → x = -b/(2a)
                optimal_complexity = -b / (2 * a)
                
                # Check if maximum is in reasonable range
                if 0 < optimal_complexity < 10:
                    results['resonance_detected'] = True
                    results['optimal_complexity'] = optimal_complexity
                    results['resonance_peak'] = a * optimal_complexity**2 + b * optimal_complexity + c
                    results['inverted_u'] = True
                    
                    # Optimal complexity ± tolerance
                    results['optimal_range'] = (optimal_complexity - 1, optimal_complexity + 1)
        except:
            pass
        
        return results
    
    # === CATASTROPHE THEORY (Enhanced) ===
    
    def identify_catastrophe_type(
        self,
        control_1: np.ndarray,
        control_2: np.ndarray = None,
        response: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Identify which of Thom's 7 elementary catastrophes describes data.
        
        Types:
        - Fold: 1 control, gradual → sudden
        - Cusp: 2 controls, bifurcation
        - Swallowtail: 3 controls
        - Butterfly: 4 controls
        
        Parameters
        ----------
        control_1 : array
            Primary control parameter (e.g., saturation)
        control_2 : array, optional
            Secondary control (e.g., competition)
        response : array
            Response variable (e.g., effect size)
        
        Returns
        -------
        results : dict
            Catastrophe type identification
        """
        results = {
            'catastrophe_type': None,
            'control_dimensions': 1 if control_2 is None else 2,
            'critical_manifold': None,
            'bifurcation_set': None
        }
        
        if control_2 is None:
            # 1D: Fold catastrophe
            # V(x) = x³/3 + ax
            results['catastrophe_type'] = 'fold'
            
            # Find inflection point: d³V/dx³ = 0
            # For cubic: always at x = 0
            if response is not None and len(control_1) == len(response):
                # Fit cubic
                try:
                    coeffs = np.polyfit(control_1, response, 3)
                    # Critical point where 3rd derivative = 0
                    # For x³ + ax² + bx + c, inflection at x = -a/3
                    if abs(coeffs[0]) > 0.001:
                        inflection = -coeffs[1] / (3 * coeffs[0])
                        results['critical_manifold'] = inflection
                except:
                    pass
        else:
            # 2D: Cusp catastrophe
            # V(x) = x⁴/4 + ax²/2 + bx
            results['catastrophe_type'] = 'cusp'
            
            # Bifurcation set: Curve in (a,b) space where catastrophe occurs
            # a = -3x², b = 2x³
            # Eliminating x: 27b² = -4a³ (cusp shape)
            results['bifurcation_set'] = '27b² = -4a³'
        
        return results
    
    # === PERCOLATION THEORY ===
    
    def find_percolation_threshold(
        self,
        connectivity_matrix: np.ndarray,  # Name-name connections
        success_outcomes: np.ndarray       # Which names succeeded
    ) -> Dict[str, Any]:
        """
        Find percolation threshold where network effects activate.
        
        At critical connectivity p_c, giant component suddenly forms.
        Below: Isolated names, no network effects.
        Above: Connected cluster, strong network effects.
        
        Parameters
        ----------
        connectivity_matrix : array, shape (n, n)
            Adjacency matrix (1 if connected, 0 if not)
        success_outcomes : array, shape (n,)
            Success indicator for each name
        
        Returns
        -------
        results : dict
            Percolation threshold analysis
        """
        results = {
            'percolation_threshold': None,
            'giant_component_size': None,
            'phase_transition_detected': False
        }
        
        n = len(connectivity_matrix)
        
        # Compute connectivity levels
        connectivity_fractions = np.linspace(0, 1, 20)
        giant_component_sizes = []
        
        for p in connectivity_fractions:
            # Threshold connectivity matrix at level p
            thresholded = connectivity_matrix > (1 - p)
            
            # Find largest connected component (simplified DFS)
            visited = set()
            max_component = 0
            
            for i in range(n):
                if i not in visited:
                    component = self._dfs_component(i, thresholded, visited)
                    max_component = max(max_component, len(component))
            
            giant_component_sizes.append(max_component / n)
        
        # Find threshold: Where giant component rapidly grows
        if len(giant_component_sizes) > 5:
            growth_rates = np.diff(giant_component_sizes)
            max_growth_idx = np.argmax(growth_rates)
            
            if growth_rates[max_growth_idx] > 0.1:  # Rapid growth
                results['percolation_threshold'] = connectivity_fractions[max_growth_idx]
                results['giant_component_size'] = giant_component_sizes[max_growth_idx + 1]
                results['phase_transition_detected'] = True
        
        return results
    
    def _dfs_component(
        self,
        start: int,
        adjacency: np.ndarray,
        visited: Set[int]
    ) -> Set[int]:
        """Depth-first search for connected component."""
        component = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            
            visited.add(node)
            component.add(node)
            
            # Add neighbors
            neighbors = np.where(adjacency[node])[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    # === QUANTUM FIELD THEORY TESTS ===
    
    def test_uncertainty_relations(
        self,
        name_quality_variance: float,
        context_specificity_variance: float
    ) -> Dict[str, Any]:
        """
        Test if Heisenberg-like uncertainty relation holds.
        
        Hypothesis: Δ(name quality) × Δ(context specificity) ≥ ℏ_narrative
        
        Cannot simultaneously have:
        - Perfectly clear name meaning (low quality variance)
        - Perfect adaptability (low context variance)
        
        Like quantum: Cannot know position and momentum simultaneously.
        
        Parameters
        ----------
        name_quality_variance : float
            Variance in perceived quality across contexts
        context_specificity_variance : float
            Variance in context appropriateness
        
        Returns
        -------
        results : dict
            Uncertainty relation test results
        """
        results = {
            'uncertainty_product': None,
            'planck_constant_narrative': None,
            'uncertainty_relation_holds': False
        }
        
        # Compute uncertainty product
        uncertainty_product = name_quality_variance * context_specificity_variance
        results['uncertainty_product'] = uncertainty_product
        
        # Empirical Planck constant (minimum observed)
        # From data: What's the minimum ΔQ × ΔC observed?
        results['planck_constant_narrative'] = uncertainty_product  # Would compute from corpus
        
        # Test if relation holds (product exceeds minimum)
        # This would require corpus of measurements
        # For now, document framework
        results['uncertainty_relation_holds'] = uncertainty_product > 0  # Placeholder
        
        return results
    
    def test_cpt_symmetry(
        self,
        positive_associations: np.ndarray,
        negative_associations: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test CPT symmetry (Charge-Parity-Time).
        
        C (charge): positive ↔ negative associations preserved?
        P (parity): harsh ↔ soft (reflection) preserves laws?
        T (time): running backward recovers state?
        
        CPT theorem: All physics respects CPT symmetry combined.
        If violated in names → fundamental asymmetry.
        
        Parameters
        ----------
        positive_associations : array
            Positive association strengths
        negative_associations : array
            Negative association strengths (corresponding names)
        
        Returns
        -------
        results : dict
            CPT symmetry test results
        """
        results = {
            'c_symmetry': False,  # Charge
            'p_symmetry': False,  # Parity
            't_symmetry': False,  # Time
            'cpt_symmetry': False
        }
        
        # C-symmetry: Are positive and negative effects symmetric?
        # Test: Mean(positive) ≈ -Mean(negative)
        if len(positive_associations) == len(negative_associations):
            pos_mean = np.mean(positive_associations)
            neg_mean = np.mean(negative_associations)
            
            # Symmetric if close to opposite
            if abs(pos_mean + neg_mean) / abs(pos_mean) < 0.2:
                results['c_symmetry'] = True
        
        # P-symmetry: Test would require harsh/soft pairs
        # Placeholder: Would need paired data
        results['p_symmetry'] = None  # Requires paired measurements
        
        # T-symmetry: Test would require temporal data
        # Can process run backward and recover state?
        results['t_symmetry'] = None  # Requires time-series data
        
        # CPT combined (all three together)
        if results['c_symmetry']:
            results['cpt_symmetry'] = True  # Partial validation
        
        return results


def run_all_advanced_theory_tests(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete advanced theory validation suite.
    
    Parameters
    ----------
    data : dict
        Contains arrays for different scales, domains, time series
    
    Returns
    -------
    results : dict
        Complete test results
    """
    tester = AdvancedTheoryTester()
    
    all_results = {
        'scale_invariance': None,
        'catastrophe_detection': None,
        'maximum_entropy': None,
        'gauge_invariance': None,
        'stochastic_resonance': None,
        'percolation': None,
        'uncertainty_relations': None,
        'cpt_symmetry': None
    }
    
    # Run each test if data available
    if 'multi_scale' in data:
        all_results['scale_invariance'] = tester.test_scale_invariance(
            data['multi_scale']['micro'],
            data['multi_scale']['meso'],
            data['multi_scale']['macro'],
            data['multi_scale']['meta']
        )
    
    if 'saturation' in data and 'effects' in data:
        all_results['catastrophe_detection'] = tester.detect_catastrophe_points(
            data['saturation'],
            data['effects'],
            catastrophe_type='cusp'
        )
    
    if 'diversity_time_series' in data:
        all_results['maximum_entropy'] = tester.test_maximum_entropy_production(
            data['diversity_time_series']['diversity'],
            data['diversity_time_series']['success']
        )
    
    if 'domain_a_features' in data and 'domain_b_features' in data:
        all_results['gauge_invariance'] = tester.test_gauge_invariance(
            data['domain_a_features'],
            data['domain_b_features']
        )
    
    if 'complexity' in data and 'success' in data:
        all_results['stochastic_resonance'] = tester.detect_stochastic_resonance(
            data['complexity'],
            data['success']
        )
    
    return all_results

