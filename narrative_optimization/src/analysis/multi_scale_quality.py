"""
Multi-Scale Narrative Quality Calculator

Calculate ю at multiple scales (nano, micro, meso, macro) and aggregate.
Each scale captures different aspects of narrative structure.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from .story_quality import StoryQualityCalculator


class MultiScaleQualityCalculator:
    """
    Calculate ю at multiple scales and aggregate.
    
    Scales:
    - Nano: Sentence/phrase level (linguistic patterns, word choice)
    - Micro: Scene/sequence level (local narrative structure)
    - Meso: Act/chapter level (mid-level arcs, character development)
    - Macro: Full narrative level (overall arc, resolution)
    
    Each scale has:
    - Nominative features (names at that scale)
    - Narrative features (stories at that scale)
    - Scale-specific ю calculation
    """
    
    def __init__(self, п: float, scales: Optional[List[str]] = None):
        """
        Initialize multi-scale calculator.
        
        Parameters
        ----------
        п : float
            Domain narrativity [0, 1]
        scales : list of str, optional
            Which scales to compute. If None, uses ['nano', 'micro', 'meso', 'macro']
        """
        self.п = п
        
        if scales is None:
            self.scales = ['nano', 'micro', 'meso', 'macro']
        else:
            self.scales = scales
        
        # Scale-specific calculators
        self.scale_calculators = {}
        for scale in self.scales:
            self.scale_calculators[scale] = StoryQualityCalculator(п, method='weighted_mean')
    
    def compute_all_scales(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        scale_genomes: Optional[Dict[str, np.ndarray]] = None,
        standardize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute ю for all scales.
        
        Parameters
        ----------
        genome : ndarray
            Main genome features (macro scale) (n_organisms, n_features)
        feature_names : list of str
            Feature names
        scale_genomes : dict, optional
            {scale_name: genome_array} for scale-specific features
        standardize : bool
            Whether to standardize
            
        Returns
        -------
        ю_scales : dict
            {scale_name: ю_array} for each scale
        """
        ю_scales = {}
        
        # Macro scale (main genome)
        if 'macro' in self.scales:
            macro_calc = self.scale_calculators['macro']
            ю_macro = macro_calc.compute_ю(genome, feature_names, standardize=standardize)
            ю_scales['macro'] = ю_macro
        
        # Other scales (if scale-specific genomes provided)
        if scale_genomes:
            for scale_name in self.scales:
                if scale_name != 'macro' and scale_name in scale_genomes:
                    scale_genome = scale_genomes[scale_name]
                    scale_calc = self.scale_calculators[scale_name]
                    ю_scale = scale_calc.compute_ю(
                        scale_genome, feature_names, standardize=standardize
                    )
                    ю_scales[scale_name] = ю_scale
        else:
            # No scale-specific genomes: use main genome for all scales
            # but with different weight schemes
            for scale_name in self.scales:
                if scale_name != 'macro':
                    scale_weights = self._get_scale_weights(scale_name)
                    scale_calc = StoryQualityCalculator(self.п, method='weighted_mean')
                    scale_calc.weights = scale_weights
                    ю_scale = scale_calc.compute_ю(genome, feature_names, standardize=standardize)
                    ю_scales[scale_name] = ю_scale
        
        return ю_scales
    
    def _get_scale_weights(self, scale: str) -> Dict[str, float]:
        """
        Get weight scheme for a specific scale.
        
        Different scales emphasize different features:
        - Nano: Linguistic, phonetic, word-level
        - Micro: Local structure, conflict, suspense
        - Meso: Character development, arcs, ensemble
        - Macro: Overall arc, resolution, cultural
        """
        if scale == 'nano':
            # Nano: Linguistic patterns, word choice
            return {
                'linguistic': 0.35,
                'phonetic': 0.25,
                'statistical': 0.20,
                'nominative': 0.20
            }
        elif scale == 'micro':
            # Micro: Local narrative structure
            return {
                'conflict': 0.25,
                'suspense': 0.20,
                'framing': 0.18,
                'linguistic': 0.15,
                'ensemble': 0.12,
                'statistical': 0.10
            }
        elif scale == 'meso':
            # Meso: Mid-level arcs, character development
            return {
                'narrative_potential': 0.22,
                'self_perception': 0.20,
                'ensemble': 0.18,
                'authenticity': 0.15,
                'conflict': 0.12,
                'emotional_semantic': 0.13
            }
        elif scale == 'macro':
            # Macro: Overall arc, resolution, cultural
            return {
                'narrative_potential': 0.20,
                'cultural_context': 0.18,
                'authenticity': 0.15,
                'emotional_semantic': 0.15,
                'nominative': 0.12,
                'ensemble': 0.10,
                'statistical': 0.10
            }
        else:
            # Default: balanced
            return {
                'nominative': 0.15,
                'narrative_potential': 0.15,
                'linguistic': 0.15,
                'ensemble': 0.15,
                'statistical': 0.15,
                'conflict': 0.10,
                'authenticity': 0.10,
                'emotional_semantic': 0.05
            }
    
    def aggregate_scales(
        self,
        ю_scales: Dict[str, np.ndarray],
        method: str = 'weighted_average',
        scale_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Aggregate ю across scales into single score.
        
        Parameters
        ----------
        ю_scales : dict
            {scale_name: ю_array}
        method : str
            Aggregation method:
            - 'weighted_average': Weight by scale importance
            - 'average': Simple average
            - 'maximum': Maximum across scales
            - 'hierarchical': Macro dominates, others adjust
        scale_weights : dict, optional
            Manual weights for each scale
            
        Returns
        -------
        ю_final : ndarray
            Aggregated story quality
        """
        if not ю_scales:
            raise ValueError("No scales provided")
        
        if method == 'average':
            ю_final = np.mean(list(ю_scales.values()), axis=0)
        
        elif method == 'maximum':
            ю_final = np.max(list(ю_scales.values()), axis=0)
        
        elif method == 'weighted_average':
            if scale_weights is None:
                # Default weights: macro > meso > micro > nano
                scale_weights = {
                    'macro': 0.40,
                    'meso': 0.30,
                    'micro': 0.20,
                    'nano': 0.10
                }
            
            # Normalize weights
            total_weight = sum(scale_weights.values())
            normalized_weights = {k: v / total_weight for k, v in scale_weights.items()}
            
            ю_final = np.zeros(len(list(ю_scales.values())[0]))
            for scale_name, ю_scale in ю_scales.items():
                weight = normalized_weights.get(scale_name, 0.0)
                ю_final += weight * ю_scale
        
        elif method == 'hierarchical':
            # Macro dominates, others provide adjustments
            if 'macro' not in ю_scales:
                raise ValueError("Macro scale required for hierarchical method")
            
            ю_macro = ю_scales['macro']
            adjustments = []
            
            for scale_name, ю_scale in ю_scales.items():
                if scale_name != 'macro':
                    # Adjustment = difference from macro
                    adjustment = ю_scale - ю_macro
                    adjustments.append(adjustment)
            
            if adjustments:
                avg_adjustment = np.mean(adjustments, axis=0)
                ю_final = ю_macro + 0.2 * avg_adjustment  # 20% weight on adjustments
            else:
                ю_final = ю_macro
            
            # Normalize
            ю_final = np.clip(ю_final, 0, 1)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return ю_final
    
    def get_scale_summary(
        self,
        ю_scales: Dict[str, np.ndarray]
    ) -> str:
        """
        Get human-readable summary of scale results.
        
        Parameters
        ----------
        ю_scales : dict
            {scale_name: ю_array}
            
        Returns
        -------
        summary : str
            Formatted summary
        """
        lines = [
            "=" * 80,
            "MULTI-SCALE NARRATIVE QUALITY ANALYSIS",
            "=" * 80,
            f"\nNarrativity (п): {self.п:.3f}",
            f"Scales: {len(ю_scales)}",
            "\nScale Scores:",
            "-" * 80
        ]
        
        for scale_name in ['nano', 'micro', 'meso', 'macro']:
            if scale_name in ю_scales:
                ю_scale = ю_scales[scale_name]
                mean_ю = np.mean(ю_scale)
                std_ю = np.std(ю_scale)
                min_ю = np.min(ю_scale)
                max_ю = np.max(ю_scale)
                
                lines.append(
                    f"  {scale_name:8s}: mean={mean_ю:.3f} ± {std_ю:.3f}, "
                    f"range=[{min_ю:.3f}, {max_ю:.3f}]"
                )
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

