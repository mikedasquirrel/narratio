"""
Story Quality Calculator

Computes ю (story quality) from ж (genome) using п-based weighting.

Implements: ю = Σ w_k × ж_k where w_k = f(п, feature_type_k)

Follows formal variable system exactly.

ENHANCED: Now supports multi-perspective and multi-method calculations
via integration with MultiPerspectiveQualityCalculator and QualityMethodRegistry.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler


class StoryQualityCalculator:
    """
    Calculate ю (story quality) from ж (genome).
    
    Key principle: Feature weights determined by domain's п (narrativity).
    
    BREAKTHROUGH (Nov 2025): Now supports π_effective (instance-specific narrativity).
    π can vary by instance complexity within domain.
    
    - Low п (<0.3): Weight plot/content features heavily
    - High п (>0.7): Weight character/identity features heavily
    - Mid п (0.3-0.7): Balanced, discover optimal α
    """
    
    def __init__(self, п: float, method='weighted_mean', use_dynamic_pi: bool = True):
        """
        Initialize story quality calculator.
        
        Parameters
        ----------
        п : float
            Domain narrativity [0, 1] (base or effective)
        method : str
            'weighted_mean' - Use п-based weights
            'alpha_discovery' - Discover optimal α empirically
        use_dynamic_pi : bool
            If True, accepts π_effective per instance (default True)
            If False, uses fixed domain π for all instances
        """
        self.п = п
        self.п_base = п  # Store original
        self.method = method
        self.use_dynamic_pi = use_dynamic_pi
        self.weights = {}
        self.optimal_alpha = None
        
        self._determine_weights()
    
    def _determine_weights(self):
        """
        Determine feature weights based on п.
        
        Implements theoretical principle:
        - п < 0.3: Plot features dominate (w = 0.7)
        - п > 0.7: Character features dominate (w = 0.7)
        - 0.3 ≤ п ≤ 0.7: Balanced (discover α)
        """
        
        if self.п < 0.3:
            # Constrained/plot-driven domains
            self.weights = {
                'statistical': 0.35,
                'quantitative': 0.15,
                'linguistic': 0.15,
                'ensemble': 0.10,
                'temporal_evolution': 0.10,
                'conflict': 0.10,
                'suspense': 0.05
            }
            self.interpretation = "Plot-driven: Content and structure dominate"
        
        elif self.п > 0.7:
            # Open/character-driven domains
            self.weights = {
                'nominative': 0.25,
                'self_perception': 0.20,
                'narrative_potential': 0.15,
                'emotional_semantic': 0.15,
                'authenticity': 0.10,
                'phonetic': 0.08,
                'social_status': 0.05,
                'linguistic': 0.02
            }
            self.interpretation = "Character-driven: Identity and narrative dominate"
        
        else:
            # Balanced domains - discover optimal α
            self.method = 'alpha_discovery'
            self.weights = {
                'nominative': 0.15,
                'self_perception': 0.12,
                'narrative_potential': 0.10,
                'linguistic': 0.12,
                'ensemble': 0.10,
                'emotional_semantic': 0.12,
                'statistical': 0.15,
                'conflict': 0.08,
                'cultural_context': 0.06
            }
            self.interpretation = "Balanced: Discover optimal narrative-statistical mix"
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        standardize: bool = True
    ) -> np.ndarray:
        """
        Compute ю (story quality) from ж (genome).
        
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
        story_quality : ndarray
            Story quality scores (ю) (n_organisms,) in range [0, 1]
        """
        # Group features by transformer
        feature_groups = self._group_features_by_transformer(feature_names)
        
        # Standardize if requested
        if standardize:
            from scipy.sparse import issparse
            
            # Handle sparse matrices properly
            if issparse(genome):
                # Convert to dense if small enough (< 100MB), otherwise scale without centering
                size_mb = (genome.shape[0] * genome.shape[1] * 8) / (1024 * 1024)
                if size_mb < 100:
                    print(f"  Converting sparse matrix to dense ({size_mb:.1f} MB)...")
                    genome_dense = genome.toarray()
                    scaler = StandardScaler()
                    genome_scaled = scaler.fit_transform(genome_dense)
                else:
                    print(f"  Sparse matrix too large ({size_mb:.1f} MB), scaling without centering...")
                    scaler = StandardScaler(with_mean=False)
                    genome_scaled = scaler.fit_transform(genome)
            else:
                scaler = StandardScaler()
                genome_scaled = scaler.fit_transform(genome)
        else:
            genome_scaled = genome
        
        # Compute weighted aggregate
        weighted_scores = []
        
        for transformer_name, weight in self.weights.items():
            # Get indices for this transformer
            indices = feature_groups.get(transformer_name, [])
            
            if len(indices) > 0:
                # Extract features for this transformer
                transformer_features = genome_scaled[:, indices]
                
                # Aggregate (mean across transformer's features)
                transformer_score = np.mean(transformer_features, axis=1)
                
                # Apply weight
                weighted_score = transformer_score * weight
                weighted_scores.append(weighted_score)
        
        # Sum all weighted scores
        if len(weighted_scores) > 0:
            story_quality = np.sum(weighted_scores, axis=0)
        else:
            # Fallback: simple mean
            story_quality = np.mean(genome_scaled, axis=1)
        
        # Normalize to [0, 1]
        sq_min = story_quality.min()
        sq_max = story_quality.max()
        
        if sq_max > sq_min:
            story_quality_norm = (story_quality - sq_min) / (sq_max - sq_min)
        else:
            story_quality_norm = np.full_like(story_quality, 0.5)
        
        return story_quality_norm
    
    def compute_ю_with_dynamic_pi(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        pi_effective_values: np.ndarray,
        standardize: bool = True
    ) -> np.ndarray:
        """
        Compute ю (story quality) with instance-specific π_effective values.
        
        BREAKTHROUGH: Uses different feature weights for each instance based on
        its unique π_effective (which varies by complexity).
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж) (n_organisms, n_features)
        feature_names : list of str
            Feature names matching ж columns
        pi_effective_values : ndarray
            Instance-specific π_effective values (n_organisms,)
        standardize : bool
            Whether to standardize features before aggregating
            
        Returns
        -------
        story_quality : ndarray
            Story quality scores (ю) (n_organisms,) in range [0, 1]
        """
        if not self.use_dynamic_pi:
            # Fall back to standard computation
            return self.compute_ю(genome, feature_names, standardize)
        
        # Group features by transformer
        feature_groups = self._group_features_by_transformer(feature_names)
        
        # Standardize if requested
        if standardize:
            from scipy.sparse import issparse
            
            if issparse(genome):
                size_mb = (genome.shape[0] * genome.shape[1] * 8) / (1024 * 1024)
                if size_mb < 100:
                    genome_scaled = StandardScaler().fit_transform(genome.toarray())
                else:
                    genome_scaled = StandardScaler(with_mean=False).fit_transform(genome)
            else:
                genome_scaled = StandardScaler().fit_transform(genome)
        else:
            genome_scaled = genome
        
        # Compute quality for each instance with its specific π_effective
        n_instances = genome_scaled.shape[0]
        story_qualities = np.zeros(n_instances)
        
        for i in range(n_instances):
            pi_eff = pi_effective_values[i]
            
            # Determine weights for this specific π_effective
            instance_weights = self._get_weights_for_pi(pi_eff)
            
            # Compute weighted score for this instance
            weighted_scores = []
            
            for transformer_name, weight in instance_weights.items():
                indices = feature_groups.get(transformer_name, [])
                
                if len(indices) > 0:
                    transformer_features = genome_scaled[i, indices]
                    transformer_score = np.mean(transformer_features)
                    weighted_score = transformer_score * weight
                    weighted_scores.append(weighted_score)
            
            if weighted_scores:
                story_qualities[i] = np.sum(weighted_scores)
            else:
                story_qualities[i] = np.mean(genome_scaled[i, :])
        
        # Normalize to [0, 1]
        sq_min = story_qualities.min()
        sq_max = story_qualities.max()
        
        if sq_max > sq_min:
            story_quality_norm = (story_qualities - sq_min) / (sq_max - sq_min)
        else:
            story_quality_norm = np.full_like(story_qualities, 0.5)
        
        return story_quality_norm
    
    def _get_weights_for_pi(self, pi: float) -> Dict[str, float]:
        """
        Get feature weights for a specific π value.
        
        Allows dynamic weight adjustment based on instance-specific narrativity.
        
        Parameters
        ----------
        pi : float
            Narrativity value (0-1)
        
        Returns
        -------
        dict
            Feature weights by transformer
        """
        if pi < 0.3:
            # Constrained/plot-driven
            return {
                'statistical': 0.35,
                'quantitative': 0.15,
                'linguistic': 0.15,
                'ensemble': 0.10,
                'temporal_evolution': 0.10,
                'conflict': 0.10,
                'suspense': 0.05
            }
        elif pi > 0.7:
            # Open/character-driven
            return {
                'nominative': 0.25,
                'self_perception': 0.20,
                'narrative_potential': 0.15,
                'emotional_semantic': 0.15,
                'authenticity': 0.10,
                'phonetic': 0.08,
                'social_status': 0.05,
                'linguistic': 0.02
            }
        else:
            # Balanced - interpolate between extremes
            # Linear interpolation based on how close to 0.3 vs 0.7
            balance = (pi - 0.3) / 0.4  # 0 at π=0.3, 1 at π=0.7
            
            return {
                'nominative': 0.15 * (1 + balance * 0.67),
                'self_perception': 0.12 * (1 + balance * 0.67),
                'narrative_potential': 0.10 * (1 + balance * 0.50),
                'linguistic': 0.12 * (1 - balance * 0.83),
                'ensemble': 0.10,
                'emotional_semantic': 0.12 * (1 + balance * 0.25),
                'statistical': 0.15 * (1 - balance * 0.57),
                'conflict': 0.08,
                'cultural_context': 0.06
            }
    
    def _group_features_by_transformer(
        self,
        feature_names: List[str]
    ) -> Dict[str, List[int]]:
        """
        Group feature indices by transformer name.
        
        Parameters
        ----------
        feature_names : list of str
            Feature names like 'nominative_field_density', 'emotional_joy', etc.
            
        Returns
        -------
        groups : dict
            {transformer_name: [feature_indices]}
        """
        groups = {}
        
        # Map prefixes to transformer names
        prefix_map = {
            'nominative': 'nominative',
            'self_perception': 'self_perception',
            'narrative_potential': 'narrative_potential',
            'linguistic': 'linguistic',
            'ensemble': 'ensemble',
            'relational': 'relational',
            'statistical': 'statistical',
            'emotional': 'emotional_semantic',
            'authenticity': 'authenticity',
            'conflict': 'conflict',
            'suspense': 'suspense',
            'expertise': 'expertise',
            'cultural': 'cultural_context',
            'phonetic': 'phonetic',
            'social_status': 'social_status',
            'temporal': 'temporal_evolution',
            'quantitative': 'quantitative',
            'visual': 'visual'
        }
        
        for idx, feat_name in enumerate(feature_names):
            # Extract transformer prefix
            feat_lower = feat_name.lower()
            
            assigned = False
            for prefix, trans_name in prefix_map.items():
                if feat_lower.startswith(prefix):
                    if trans_name not in groups:
                        groups[trans_name] = []
                    groups[trans_name].append(idx)
                    assigned = True
                    break
            
            if not assigned:
                # Unknown transformer - assign to 'other'
                if 'other' not in groups:
                    groups['other'] = []
                groups['other'].append(idx)
        
        return groups
    
    def get_feature_importance(
        self,
        genome: np.ndarray,
        outcomes: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Get importance of each transformer in computing ю (story quality).
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж)
        outcomes : ndarray
            Outcomes (❊)
        feature_names : list
            Feature names
            
        Returns
        -------
        importance : dict
            {transformer_name: importance_score}
        """
        from scipy import stats
        
        # Compute ю
        story_quality = self.compute_ю(genome, feature_names)
        
        # Group features
        feature_groups = self._group_features_by_transformer(feature_names)
        
        # Compute correlation of each transformer group with outcomes
        importance = {}
        
        for trans_name, indices in feature_groups.items():
            if len(indices) > 0:
                # Get this transformer's features
                trans_features = genome[:, indices]
                trans_score = np.mean(trans_features, axis=1)
                
                # Correlate with outcomes
                r, p = stats.pearsonr(trans_score, outcomes)
                
                importance[trans_name] = {
                    'correlation': float(r),
                    'p_value': float(p),
                    'weight_assigned': self.weights.get(trans_name, 0.0),
                    'n_features': len(indices)
                }
        
        return importance
    
    def explain_weights(self) -> str:
        """Explain why these weights were chosen"""
        explanation = f"""
Story Quality (ю) Calculation for п = {self.п:.2f}

{self.interpretation}

Feature Weights:
"""
        for transformer, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            explanation += f"  {transformer:25s}: {weight:.2f}\n"
        
        explanation += f"\nTotal weight: {sum(self.weights.values()):.2f}"
        
        return explanation
    
    def compute_multi_perspective(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        perspectives: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute ю from multiple perspectives (enhanced method).
        
        This integrates with MultiPerspectiveQualityCalculator for
        sophisticated multi-perspective analysis.
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж)
        feature_names : list of str
            Feature names
        perspectives : list of str, optional
            Which perspectives to compute
            
        Returns
        -------
        ю_perspectives : dict
            {perspective_name: ю_array}
        """
        from .multi_perspective_quality import MultiPerspectiveQualityCalculator, NarrativePerspective
        
        if perspectives:
            persp_enum = [NarrativePerspective(p) for p in perspectives]
        else:
            persp_enum = None
        
        calculator = MultiPerspectiveQualityCalculator(self.п, perspectives=persp_enum)
        return calculator.compute_all_perspectives(genome, feature_names)
    
    def compute_multi_method(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        methods: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute ю using multiple calculation methods (enhanced method).
        
        This integrates with QualityMethodRegistry for robustness.
        
        Parameters
        ----------
        genome : ndarray
            Genome features (ж)
        feature_names : list of str
            Feature names
        methods : list of str, optional
            Which methods to use
            
        Returns
        -------
        ю_methods : dict
            {method_name: ю_array}
        """
        from .quality_methods import QualityMethodRegistry
        
        registry = QualityMethodRegistry()
        return registry.compute_all_methods(
            genome, feature_names, self.п, method_names=methods
        )

