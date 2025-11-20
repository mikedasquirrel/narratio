"""
Quality Calculation Methods Registry

Multiple methods for computing ю (story quality) from ж (genome).
Each method provides different insights and robustness.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod

from .story_quality import StoryQualityCalculator
from .perspective_weights import PerspectiveWeightSchemas


class QualityMethod(ABC):
    """Base class for ю calculation methods"""
    
    @abstractmethod
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute ю using this method.
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж) (n_organisms, n_features)
        feature_names : list of str
            Feature names
        п : float
            Domain narrativity
        **kwargs
            Method-specific parameters
            
        Returns
        -------
        ю : ndarray
            Story quality scores
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get method name"""
        pass


class WeightedMeanMethod(QualityMethod):
    """Method 1: Weighted Mean (current approach)"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        **kwargs
    ) -> np.ndarray:
        """Standard weighted mean approach"""
        calculator = StoryQualityCalculator(п, method='weighted_mean')
        return calculator.compute_ю(genome, feature_names, standardize=True)
    
    def get_name(self) -> str:
        return "weighted_mean"


class EnsembleMethod(QualityMethod):
    """Method 2: Ensemble Voting - Multiple weight schemes vote"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        n_schemes: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        Ensemble method: Multiple weight schemes vote.
        
        Creates multiple weight schemes (conservative, aggressive, balanced)
        and averages their ю calculations.
        """
        # Create multiple weight schemes
        schemes = self._create_weight_schemes(п, n_schemes)
        
        ю_schemes = []
        for scheme_name, weights in schemes.items():
            calculator = StoryQualityCalculator(п, method='weighted_mean')
            calculator.weights = weights
            ю_s = calculator.compute_ю(genome, feature_names, standardize=True)
            ю_schemes.append(ю_s)
        
        # Average across schemes
        ю_ensemble = np.mean(ю_schemes, axis=0)
        
        return ю_ensemble
    
    def _create_weight_schemes(self, п: float, n_schemes: int) -> Dict[str, Dict[str, float]]:
        """Create multiple weight schemes for ensemble"""
        schemes = {}
        
        # Base weights
        base_calculator = StoryQualityCalculator(п, method='weighted_mean')
        schemes['base'] = base_calculator.weights.copy()
        
        # Conservative: Emphasize statistical/objective features
        conservative = base_calculator.weights.copy()
        if 'statistical' in conservative:
            conservative['statistical'] *= 1.5
        schemes['conservative'] = conservative
        
        # Aggressive: Emphasize narrative features
        aggressive = base_calculator.weights.copy()
        narrative_features = ['nominative', 'self_perception', 'narrative_potential', 
                            'emotional_semantic', 'authenticity']
        for feat in narrative_features:
            if feat in aggressive:
                aggressive[feat] *= 1.3
        schemes['aggressive'] = aggressive
        
        # Balanced: Equal emphasis
        balanced = {}
        total_features = len(base_calculator.weights)
        equal_weight = 1.0 / total_features
        for feat in base_calculator.weights.keys():
            balanced[feat] = equal_weight
        schemes['balanced'] = balanced
        
        # Character-focused: Emphasize character features
        character_focused = base_calculator.weights.copy()
        char_features = ['nominative', 'self_perception', 'authenticity']
        for feat in char_features:
            if feat in character_focused:
                character_focused[feat] *= 1.4
        schemes['character_focused'] = character_focused
        
        # Plot-focused: Emphasize structural features
        plot_focused = base_calculator.weights.copy()
        plot_features = ['conflict', 'suspense', 'statistical', 'linguistic']
        for feat in plot_features:
            if feat in plot_focused:
                plot_focused[feat] *= 1.4
        schemes['plot_focused'] = plot_focused
        
        # Return requested number
        scheme_names = list(schemes.keys())[:n_schemes]
        return {name: schemes[name] for name in scheme_names}
    
    def get_name(self) -> str:
        return "ensemble"


class TemporalDynamicsMethod(QualityMethod):
    """Method 3: Temporal Dynamics - ю(t) over time"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        temporal_indices: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Temporal dynamics: ю(t) - narrative quality over time.
        
        Parameters
        ----------
        temporal_indices : ndarray, optional
            Temporal ordering indices. If None, assumes sequential.
        """
        # Base calculation
        base_calculator = StoryQualityCalculator(п, method='weighted_mean')
        ю_base = base_calculator.compute_ю(genome, feature_names, standardize=True)
        
        if temporal_indices is None:
            # Assume sequential
            temporal_indices = np.arange(len(ю_base))
        
        # Sort by temporal order
        sorted_indices = np.argsort(temporal_indices)
        ю_sorted = ю_base[sorted_indices]
        
        # Compute momentum (derivative)
        ю_momentum = np.gradient(ю_sorted)
        
        # Compute acceleration (second derivative)
        ю_acceleration = np.gradient(ю_momentum)
        
        # Combine: ю = base + momentum_weight * momentum + accel_weight * acceleration
        momentum_weight = kwargs.get('momentum_weight', 0.1)
        accel_weight = kwargs.get('accel_weight', 0.05)
        
        ю_temporal = ю_sorted + momentum_weight * ю_momentum + accel_weight * ю_acceleration
        
        # Normalize back to [0, 1]
        ю_temporal = np.clip(ю_temporal, 0, 1)
        
        # Map back to original order
        ю_final = np.zeros_like(ю_base)
        ю_final[sorted_indices] = ю_temporal
        
        return ю_final
    
    def get_name(self) -> str:
        return "temporal"


class ContextDependentMethod(QualityMethod):
    """Method 4: Context-Dependent - ю(context) varies by context"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        context_features: Optional[np.ndarray] = None,
        context_labels: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Context-dependent: ю(context) - quality varies by context.
        
        Parameters
        ----------
        context_features : ndarray, optional
            Context feature vectors
        context_labels : list of str, optional
            Context labels (e.g., 'high_stakes', 'routine', 'playoffs')
        """
        # Base calculation
        base_calculator = StoryQualityCalculator(п, method='weighted_mean')
        ю_base = base_calculator.compute_ю(genome, feature_names, standardize=True)
        
        if context_features is None and context_labels is None:
            # No context info, return base
            return ю_base
        
        # Detect context for each organism
        contexts = self._detect_contexts(
            genome, context_features, context_labels, **kwargs
        )
        
        # Compute context-specific adjustments
        ю_contextual = ю_base.copy()
        
        for i, context in enumerate(contexts):
            adjustment = self._get_context_adjustment(context, п)
            ю_contextual[i] = ю_base[i] * adjustment
        
        # Normalize
        ю_contextual = np.clip(ю_contextual, 0, 1)
        
        return ю_contextual
    
    def _detect_contexts(
        self,
        genome: np.ndarray,
        context_features: Optional[np.ndarray],
        context_labels: Optional[List[str]],
        **kwargs
    ) -> List[str]:
        """Detect context for each organism"""
        n_organisms = len(genome)
        
        if context_labels:
            return context_labels[:n_organisms]
        
        if context_features is not None:
            # Use context features to infer context
            # Simple heuristic: high values = high-stakes
            context_scores = np.mean(context_features, axis=1)
            median_score = np.median(context_scores)
            
            contexts = []
            for score in context_scores:
                if score > median_score * 1.2:
                    contexts.append('high_stakes')
                elif score < median_score * 0.8:
                    contexts.append('routine')
                else:
                    contexts.append('normal')
            
            return contexts
        
        # Default: all normal
        return ['normal'] * n_organisms
    
    def _get_context_adjustment(self, context: str, п: float) -> float:
        """Get adjustment factor for context"""
        adjustments = {
            'high_stakes': 1.15,  # Narrative matters more in high-stakes
            'routine': 0.90,  # Narrative matters less in routine
            'normal': 1.0,
            'playoffs': 1.20,  # Sports: playoffs amplify narrative
            'awards_season': 1.25,  # Movies: awards season amplifies
            'crisis': 1.30,  # Crisis amplifies narrative effects
        }
        
        return adjustments.get(context, 1.0)
    
    def get_name(self) -> str:
        return "context_dependent"


class MultiScaleMethod(QualityMethod):
    """Method 5: Multi-Scale Aggregation - Aggregate across scales"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        scale_features: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Multi-scale: Aggregate ю across scales (nano, micro, meso, macro).
        
        Parameters
        ----------
        scale_features : dict, optional
            {scale_name: feature_array} for each scale
        """
        # Base calculation (macro scale)
        base_calculator = StoryQualityCalculator(п, method='weighted_mean')
        ю_macro = base_calculator.compute_ю(genome, feature_names, standardize=True)
        
        if scale_features is None:
            # No scale-specific features, return macro
            return ю_macro
        
        # Compute ю for each scale
        ю_scales = {'macro': ю_macro}
        
        scale_weights = kwargs.get('scale_weights', {
            'nano': 0.1,
            'micro': 0.2,
            'meso': 0.3,
            'macro': 0.4
        })
        
        for scale_name, scale_genome in scale_features.items():
            if scale_name != 'macro':
                # Compute ю for this scale
                ю_scale = base_calculator.compute_ю(
                    scale_genome, feature_names, standardize=True
                )
                ю_scales[scale_name] = ю_scale
        
        # Aggregate scales
        ю_final = np.zeros_like(ю_macro)
        total_weight = 0.0
        
        for scale_name, ю_scale in ю_scales.items():
            weight = scale_weights.get(scale_name, 0.0)
            ю_final += weight * ю_scale
            total_weight += weight
        
        if total_weight > 0:
            ю_final /= total_weight
        
        return ю_final
    
    def get_name(self) -> str:
        return "multi_scale"


class QualityMethodRegistry:
    """Registry of all quality calculation methods"""
    
    def __init__(self):
        """Initialize registry with all methods"""
        self.methods = {
            'weighted_mean': WeightedMeanMethod(),
            'ensemble': EnsembleMethod(),
            'temporal': TemporalDynamicsMethod(),
            'context_dependent': ContextDependentMethod(),
            'multi_scale': MultiScaleMethod(),
        }
    
    def get_method(self, method_name: str) -> QualityMethod:
        """Get method by name"""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.methods.keys())}")
        return self.methods[method_name]
    
    def compute_with_method(
        self,
        method_name: str,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        **kwargs
    ) -> np.ndarray:
        """Compute ю using specified method"""
        method = self.get_method(method_name)
        return method.compute_ю(genome, feature_names, п, **kwargs)
    
    def compute_all_methods(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        method_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Compute ю using all methods (or specified subset).
        
        Returns
        -------
        ю_methods : dict
            {method_name: ю_array} for each method
        """
        if method_names is None:
            method_names = list(self.methods.keys())
        
        ю_methods = {}
        for method_name in method_names:
            try:
                ю = self.compute_with_method(
                    method_name, genome, feature_names, п, **kwargs
                )
                ю_methods[method_name] = ю
            except Exception as e:
                print(f"Warning: Method {method_name} failed: {e}")
        
        return ю_methods
    
    def list_methods(self) -> List[str]:
        """List all available methods"""
        return list(self.methods.keys())

