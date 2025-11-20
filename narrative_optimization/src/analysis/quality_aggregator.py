"""
Quality Aggregator

Aggregate ю from multiple perspectives, methods, and scales into final score.
Provides multiple aggregation strategies.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union
from scipy import stats

from .multi_perspective_quality import MultiPerspectiveQualityCalculator, NarrativePerspective
from .quality_methods import QualityMethodRegistry
from .multi_scale_quality import MultiScaleQualityCalculator


class QualityAggregator:
    """
    Aggregate ю from multiple dimensions (perspectives, methods, scales).
    
    Provides sophisticated aggregation strategies:
    - Simple averaging
    - Weighted by importance (correlation with outcomes)
    - Maximum (best dimension wins)
    - Context-dependent selection
    - Ensemble voting
    """
    
    def __init__(self, п: float):
        """
        Initialize aggregator.
        
        Parameters
        ----------
        п : float
            Domain narrativity
        """
        self.п = п
        self.perspective_calculator = MultiPerspectiveQualityCalculator(п)
        self.method_registry = QualityMethodRegistry()
        self.scale_calculator = MultiScaleQualityCalculator(п)
    
    def compute_comprehensive_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        outcomes: Optional[np.ndarray] = None,
        perspectives: Optional[List[NarrativePerspective]] = None,
        methods: Optional[List[str]] = None,
        scales: Optional[List[str]] = None,
        scale_genomes: Optional[Dict[str, np.ndarray]] = None,
        context_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute comprehensive ю from all dimensions.
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж)
        feature_names : list of str
            Feature names
        outcomes : ndarray, optional
            Outcomes (for importance weighting)
        perspectives : list, optional
            Which perspectives to compute
        methods : list, optional
            Which methods to compute
        scales : list, optional
            Which scales to compute
        scale_genomes : dict, optional
            Scale-specific genomes
        context_features : ndarray, optional
            Context features for context-dependent methods
        **kwargs
            Additional parameters
            
        Returns
        -------
        results : dict
            Complete analysis with:
            - ю_perspectives: {perspective: ю_array}
            - ю_methods: {method: ю_array}
            - ю_scales: {scale: ю_array}
            - ю_final: Aggregated final score
            - importance: Importance scores
            - summary: Human-readable summary
        """
        results = {}
        
        # Compute perspectives
        if perspectives is not None:
            self.perspective_calculator.perspectives = perspectives
        
        ю_perspectives = self.perspective_calculator.compute_all_perspectives(
            genome, feature_names
        )
        results['ю_perspectives'] = ю_perspectives
        
        # Compute methods
        if methods is None:
            methods = ['weighted_mean', 'ensemble']
        
        ю_methods = self.method_registry.compute_all_methods(
            genome, feature_names, self.п,
            method_names=methods,
            context_features=context_features,
            scale_features=scale_genomes,
            **kwargs
        )
        results['ю_methods'] = ю_methods
        
        # Compute scales
        if scales is not None:
            self.scale_calculator.scales = scales
        
        ю_scales = self.scale_calculator.compute_all_scales(
            genome, feature_names, scale_genomes=scale_genomes
        )
        results['ю_scales'] = ю_scales
        
        # Compute importance (if outcomes provided)
        importance = {}
        if outcomes is not None:
            # Perspective importance
            perspective_importance = self.perspective_calculator.get_perspective_importance(
                ю_perspectives, outcomes
            )
            importance['perspectives'] = perspective_importance
            
            # Method importance
            method_importance = {}
            for method_name, ю_m in ю_methods.items():
                r, p = stats.pearsonr(ю_m, outcomes)
                method_importance[method_name] = {
                    'correlation': float(r),
                    'p_value': float(p),
                    'abs_correlation': float(abs(r))
                }
            importance['methods'] = method_importance
            
            # Scale importance
            scale_importance = {}
            for scale_name, ю_s in ю_scales.items():
                r, p = stats.pearsonr(ю_s, outcomes)
                scale_importance[scale_name] = {
                    'correlation': float(r),
                    'p_value': float(p),
                    'abs_correlation': float(abs(r))
                }
            importance['scales'] = scale_importance
        
        results['importance'] = importance
        
        # Aggregate to final ю
        ю_final = self.aggregate_all_dimensions(
            ю_perspectives, ю_methods, ю_scales,
            importance=importance,
            aggregation_method=kwargs.get('aggregation_method', 'importance_weighted')
        )
        results['ю_final'] = ю_final
        
        # Generate summary
        summary = self.generate_summary(
            ю_perspectives, ю_methods, ю_scales, ю_final, importance
        )
        results['summary'] = summary
        
        return results
    
    def aggregate_all_dimensions(
        self,
        ю_perspectives: Dict[str, np.ndarray],
        ю_methods: Dict[str, np.ndarray],
        ю_scales: Dict[str, np.ndarray],
        importance: Optional[Dict[str, Dict[str, Any]]] = None,
        aggregation_method: str = 'importance_weighted'
    ) -> np.ndarray:
        """
        Aggregate across all dimensions (perspectives, methods, scales).
        
        Parameters
        ----------
        ю_perspectives : dict
            {perspective: ю_array}
        ю_methods : dict
            {method: ю_array}
        ю_scales : dict
            {scale: ю_array}
        importance : dict, optional
            Importance scores
        aggregation_method : str
            How to aggregate:
            - 'importance_weighted': Weight by correlation with outcomes
            - 'equal_weight': Equal weight to all dimensions
            - 'hierarchical': Scales → Methods → Perspectives
            - 'maximum': Maximum across all dimensions
            
        Returns
        -------
        ю_final : ndarray
            Final aggregated score
        """
        # Collect all ю arrays
        all_ю = {}
        all_ю.update({f"perspective_{k}": v for k, v in ю_perspectives.items()})
        all_ю.update({f"method_{k}": v for k, v in ю_methods.items()})
        all_ю.update({f"scale_{k}": v for k, v in ю_scales.items()})
        
        if not all_ю:
            raise ValueError("No ю arrays provided")
        
        n_organisms = len(list(all_ю.values())[0])
        
        if aggregation_method == 'equal_weight':
            # Simple average
            ю_final = np.mean(list(all_ю.values()), axis=0)
        
        elif aggregation_method == 'maximum':
            # Maximum across all
            ю_final = np.max(list(all_ю.values()), axis=0)
        
        elif aggregation_method == 'importance_weighted':
            if importance is None:
                # Fallback to equal weight
                ю_final = np.mean(list(all_ю.values()), axis=0)
            else:
                # Weight by importance
                weights = {}
                
                # Perspective weights
                if 'perspectives' in importance:
                    for persp_name, imp_data in importance['perspectives'].items():
                        weights[f"perspective_{persp_name}"] = imp_data['abs_correlation']
                
                # Method weights
                if 'methods' in importance:
                    for method_name, imp_data in importance['methods'].items():
                        weights[f"method_{method_name}"] = imp_data['abs_correlation']
                
                # Scale weights
                if 'scales' in importance:
                    for scale_name, imp_data in importance['scales'].items():
                        weights[f"scale_{scale_name}"] = imp_data['abs_correlation']
                
                if weights:
                    # Normalize weights
                    total_weight = sum(weights.values())
                    normalized_weights = {k: v / total_weight for k, v in weights.items()}
                    
                    ю_final = np.zeros(n_organisms)
                    for name, ю_array in all_ю.items():
                        weight = normalized_weights.get(name, 0.0)
                        ю_final += weight * ю_array
                else:
                    ю_final = np.mean(list(all_ю.values()), axis=0)
        
        elif aggregation_method == 'hierarchical':
            # Hierarchical: scales → methods → perspectives
            # Start with scales (most fundamental)
            if ю_scales:
                ю_base = self.scale_calculator.aggregate_scales(
                    ю_scales, method='weighted_average'
                )
            else:
                ю_base = np.mean(list(all_ю.values())[0])
            
            # Adjust with methods
            if ю_methods:
                method_avg = np.mean(list(ю_methods.values()), axis=0)
                ю_base = 0.6 * ю_base + 0.4 * method_avg
            
            # Adjust with perspectives
            if ю_perspectives:
                persp_avg = np.mean(list(ю_perspectives.values()), axis=0)
                ю_final = 0.7 * ю_base + 0.3 * persp_avg
            else:
                ю_final = ю_base
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Normalize to [0, 1]
        ю_final = np.clip(ю_final, 0, 1)
        
        return ю_final
    
    def generate_summary(
        self,
        ю_perspectives: Dict[str, np.ndarray],
        ю_methods: Dict[str, np.ndarray],
        ю_scales: Dict[str, np.ndarray],
        ю_final: np.ndarray,
        importance: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Generate comprehensive summary of all dimensions.
        
        Parameters
        ----------
        ю_perspectives : dict
            Perspective scores
        ю_methods : dict
            Method scores
        ю_scales : dict
            Scale scores
        ю_final : ndarray
            Final aggregated score
        importance : dict, optional
            Importance scores
            
        Returns
        -------
        summary : str
            Formatted summary
        """
        lines = [
            "=" * 80,
            "COMPREHENSIVE NARRATIVE QUALITY ANALYSIS",
            "=" * 80,
            f"\nNarrativity (п): {self.п:.3f}",
            f"\nFinal ю: mean={np.mean(ю_final):.3f} ± {np.std(ю_final):.3f}",
            f"  range=[{np.min(ю_final):.3f}, {np.max(ю_final):.3f}]"
        ]
        
        # Perspectives
        if ю_perspectives:
            lines.append(f"\nPerspectives ({len(ю_perspectives)}):")
            lines.append("-" * 80)
            for persp_name, ю_p in sorted(ю_perspectives.items()):
                mean_ю = np.mean(ю_p)
                line = f"  {persp_name:15s}: {mean_ю:.3f}"
                if importance and 'perspectives' in importance and persp_name in importance['perspectives']:
                    corr = importance['perspectives'][persp_name]['correlation']
                    line += f" (r={corr:.3f})"
                lines.append(line)
        
        # Methods
        if ю_methods:
            lines.append(f"\nMethods ({len(ю_methods)}):")
            lines.append("-" * 80)
            for method_name, ю_m in sorted(ю_methods.items()):
                mean_ю = np.mean(ю_m)
                line = f"  {method_name:15s}: {mean_ю:.3f}"
                if importance and 'methods' in importance and method_name in importance['methods']:
                    corr = importance['methods'][method_name]['correlation']
                    line += f" (r={corr:.3f})"
                lines.append(line)
        
        # Scales
        if ю_scales:
            lines.append(f"\nScales ({len(ю_scales)}):")
            lines.append("-" * 80)
            for scale_name in ['nano', 'micro', 'meso', 'macro']:
                if scale_name in ю_scales:
                    ю_s = ю_scales[scale_name]
                    mean_ю = np.mean(ю_s)
                    line = f"  {scale_name:15s}: {mean_ю:.3f}"
                    if importance and 'scales' in importance and scale_name in importance['scales']:
                        corr = importance['scales'][scale_name]['correlation']
                        line += f" (r={corr:.3f})"
                    lines.append(line)
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

