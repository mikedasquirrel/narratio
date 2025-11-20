"""
Multi-Perspective Narrative Quality Calculator

Calculate ю (story quality) from multiple perspectives.
Each perspective emphasizes different aspects of narrative quality.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler

from .story_quality import StoryQualityCalculator
from .perspective_weights import PerspectiveWeightSchemas, NarrativePerspective


class MultiPerspectiveQualityCalculator:
    """
    Calculate ю from multiple perspectives.
    
    Each perspective has:
    - Different feature weights
    - Different evaluation criteria
    - Different outcome correlations
    
    Perspectives:
    - director: Creator's intent, vision, execution
    - audience: Viewer engagement, emotion, accessibility
    - critic: Craft, innovation, cultural significance
    - character: Character development, authenticity, arc
    - cultural: Cultural resonance, relevance, zeitgeist
    - meta: Self-awareness, genre play, innovation
    - authority: Leadership, strategic vision
    - star: Individual excellence, heroism
    - collective: Team/organization identity
    - supporting: Ensemble, role fulfillment
    """
    
    def __init__(self, п: float, perspectives: Optional[List[NarrativePerspective]] = None):
        """
        Initialize multi-perspective calculator.
        
        Parameters
        ----------
        п : float
            Domain narrativity [0, 1]
        perspectives : list of NarrativePerspective, optional
            Which perspectives to compute. If None, uses all available.
        """
        self.п = п
        
        if perspectives is None:
            # Default: all perspectives
            self.perspectives = list(NarrativePerspective)
        else:
            self.perspectives = perspectives
        
        # Initialize weight schemas
        self.weight_schemas = PerspectiveWeightSchemas()
        
        # Store perspective-specific calculators
        self.perspective_calculators = {}
        for perspective in self.perspectives:
            weights = self.weight_schemas.get_weights(perspective, п)
            self.perspective_calculators[perspective] = StoryQualityCalculator(
                п=п,
                method='weighted_mean'
            )
            # Override weights with perspective-specific weights
            self.perspective_calculators[perspective].weights = weights
    
    def compute_all_perspectives(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        standardize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute ю for all configured perspectives.
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж) (n_organisms, n_features)
        feature_names : list of str
            Feature names matching ж columns
        standardize : bool
            Whether to standardize features before aggregating
            
        Returns
        -------
        ю_perspectives : dict
            {perspective_name: ю_array} for each perspective
        """
        ю_perspectives = {}
        
        for perspective in self.perspectives:
            calculator = self.perspective_calculators[perspective]
            ю_p = calculator.compute_ю(genome, feature_names, standardize=standardize)
            ю_perspectives[perspective.value] = ю_p
        
        return ю_perspectives
    
    def compute_single_perspective(
        self,
        perspective: NarrativePerspective,
        genome: np.ndarray,
        feature_names: List[str],
        standardize: bool = True
    ) -> np.ndarray:
        """
        Compute ю for a single perspective.
        
        Parameters
        ----------
        perspective : NarrativePerspective
            Which perspective to compute
        genome : ndarray
            Genome features (ж)
        feature_names : list of str
            Feature names
        standardize : bool
            Whether to standardize
            
        Returns
        -------
        ю : ndarray
            Story quality from this perspective
        """
        if perspective not in self.perspective_calculators:
            # Create calculator on the fly
            weights = self.weight_schemas.get_weights(perspective, self.п)
            calculator = StoryQualityCalculator(self.п, method='weighted_mean')
            calculator.weights = weights
        else:
            calculator = self.perspective_calculators[perspective]
        
        return calculator.compute_ю(genome, feature_names, standardize=standardize)
    
    def get_perspective_importance(
        self,
        ю_perspectives: Dict[str, np.ndarray],
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Determine which perspectives correlate most with outcomes.
        
        Parameters
        ----------
        ю_perspectives : dict
            {perspective_name: ю_array}
        outcomes : ndarray
            Outcomes (❊)
            
        Returns
        -------
        importance : dict
            {perspective_name: correlation_with_outcomes}
        """
        from scipy import stats
        
        importance = {}
        
        for perspective_name, ю_p in ю_perspectives.items():
            r, p_value = stats.pearsonr(ю_p, outcomes)
            importance[perspective_name] = {
                'correlation': float(r),
                'p_value': float(p_value),
                'abs_correlation': float(abs(r))
            }
        
        return importance
    
    def aggregate_perspectives(
        self,
        ю_perspectives: Dict[str, np.ndarray],
        method: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None,
        importance: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Aggregate ю from multiple perspectives into single score.
        
        Parameters
        ----------
        ю_perspectives : dict
            {perspective_name: ю_array}
        method : str
            Aggregation method:
            - 'weighted_average': Weight by provided weights
            - 'importance_weighted': Weight by correlation with outcomes
            - 'maximum': Take maximum across perspectives
            - 'average': Simple average
            - 'median': Median across perspectives
        weights : dict, optional
            Manual weights for each perspective
        importance : dict, optional
            Importance scores (from get_perspective_importance)
            
        Returns
        -------
        ю_final : ndarray
            Aggregated story quality
        """
        if not ю_perspectives:
            raise ValueError("No perspectives provided")
        
        # Convert to array format
        perspective_arrays = list(ю_perspectives.values())
        n_organisms = len(perspective_arrays[0])
        
        if method == 'average':
            ю_final = np.mean(perspective_arrays, axis=0)
        
        elif method == 'median':
            ю_final = np.median(perspective_arrays, axis=0)
        
        elif method == 'maximum':
            ю_final = np.max(perspective_arrays, axis=0)
        
        elif method == 'weighted_average':
            if weights is None:
                # Equal weights
                weights = {name: 1.0 / len(ю_perspectives) 
                          for name in ю_perspectives.keys()}
            
            # Normalize weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            
            ю_final = np.zeros(n_organisms)
            for perspective_name, ю_p in ю_perspectives.items():
                weight = normalized_weights.get(perspective_name, 0.0)
                ю_final += weight * ю_p
        
        elif method == 'importance_weighted':
            if importance is None:
                raise ValueError("importance required for importance_weighted method")
            
            # Use absolute correlation as weights
            weights = {
                name: importance[name]['abs_correlation']
                for name in ю_perspectives.keys()
                if name in importance
            }
            
            if not weights:
                # Fallback to average
                ю_final = np.mean(perspective_arrays, axis=0)
            else:
                # Normalize weights
                total_weight = sum(weights.values())
                normalized_weights = {k: v / total_weight for k, v in weights.items()}
                
                ю_final = np.zeros(n_organisms)
                for perspective_name, ю_p in ю_perspectives.items():
                    weight = normalized_weights.get(perspective_name, 0.0)
                    ю_final += weight * ю_p
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return ю_final
    
    def get_perspective_summary(
        self,
        ю_perspectives: Dict[str, np.ndarray],
        importance: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Get human-readable summary of perspective results.
        
        Parameters
        ----------
        ю_perspectives : dict
            {perspective_name: ю_array}
        importance : dict, optional
            Importance scores
            
        Returns
        -------
        summary : str
            Formatted summary
        """
        lines = [
            "=" * 80,
            "MULTI-PERSPECTIVE NARRATIVE QUALITY ANALYSIS",
            "=" * 80,
            f"\nNarrativity (п): {self.п:.3f}",
            f"Perspectives: {len(ю_perspectives)}",
            "\nPerspective Scores:",
            "-" * 80
        ]
        
        for perspective_name, ю_p in sorted(ю_perspectives.items()):
            mean_ю = np.mean(ю_p)
            std_ю = np.std(ю_p)
            min_ю = np.min(ю_p)
            max_ю = np.max(ю_p)
            
            line = f"  {perspective_name:15s}: mean={mean_ю:.3f} ± {std_ю:.3f}, range=[{min_ю:.3f}, {max_ю:.3f}]"
            
            if importance and perspective_name in importance:
                corr = importance[perspective_name]['correlation']
                line += f", r={corr:.3f}"
            
            lines.append(line)
        
        if importance:
            lines.append("\nPerspective Importance (correlation with outcomes):")
            lines.append("-" * 80)
            sorted_importance = sorted(
                importance.items(),
                key=lambda x: x[1]['abs_correlation'],
                reverse=True
            )
            
            for perspective_name, imp_data in sorted_importance:
                lines.append(
                    f"  {perspective_name:15s}: r={imp_data['correlation']:.3f} "
                    f"(p={imp_data['p_value']:.4f})"
                )
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

