"""
Narrative Context Weighter

Implements NARRATIVE_WEIGHTING_THEORY principles:
- Context determines weight (stakes, rivalry, timing, momentum, significance)
- Temporal dynamics (better stories win over time)
- Story accumulation (narratives compound)
- Multi-dimensional story quality (coherence, confidence, momentum, identity, stakes, context, timing)
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import numpy as np


class NarrativeContextWeighter:
    """
    Assigns narrative weights based on context, stakes, and temporal factors.
    
    Implements the theory that not all comparisons are equal narratively:
    - Championship ≠ Regular season
    - Rivalry ≠ Random matchup
    - Elimination ≠ Meaningless event
    
    Narrative_Impact = Narrative_Quality × Context_Weight
    """
    
    def __init__(self):
        # Stakes markers (highest weight factors)
        self.stakes_markers = {
            'championship': 2.5,
            'finals': 2.3,
            'playoff': 2.0,
            'elimination': 2.3,
            'semifinal': 1.9,
            'quarterfinal': 1.7,
            'title': 2.2,
            'crown': 2.1,
            'trophy': 2.0
        }
        
        # Rivalry markers
        self.rivalry_markers = {
            'rival': 1.8,
            'rivalry': 1.8,
            'historic': 1.7,
            'classic': 1.6,
            'traditional': 1.5,
            'legendary': 1.7,
            'bitter': 1.6
        }
        
        # Momentum markers
        self.momentum_markers = {
            'streak': 1.6,
            'winning streak': 1.7,
            'hot': 1.5,
            'momentum': 1.5,
            'surging': 1.6,
            'rolling': 1.5,
            'unstoppable': 1.8
        }
        
        # Timing markers
        self.timing_markers = {
            'crucial': 1.5,
            'critical': 1.5,
            'decisive': 1.6,
            'pivotal': 1.6,
            'must-win': 1.8,
            'do-or-die': 1.9,
            'win-or-go-home': 2.0
        }
        
        # Low-stakes markers (reduce weight)
        self.low_stakes_markers = {
            'exhibition': 0.5,
            'preseason': 0.6,
            'friendly': 0.5,
            'scrimmage': 0.4,
            'practice': 0.3,
            'meaningless': 0.4,
            'routine': 0.7
        }
    
    def compute_narrative_weight(
        self,
        text_a: str,
        text_b: str,
        user_question: str = '',
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Compute narrative context weight for a comparison.
        
        Parameters
        ----------
        text_a, text_b : str
            Texts being compared
        user_question : str
            User's question (provides context)
        context : dict
            Additional context from conversation
        
        Returns
        -------
        weighting : dict
            Weight score, components, and explanation
        """
        context = context or {}
        
        # Combine all text for analysis
        combined_text = f"{text_a} {text_b} {user_question}".lower()
        
        # Also check context dictionary
        context_text = ' '.join(str(v) for v in context.values()).lower()
        all_text = combined_text + ' ' + context_text
        
        # Start with baseline
        weight = 1.0
        components = {}
        
        # Check stakes
        stakes_weight, stakes_found = self._check_markers(all_text, self.stakes_markers)
        if stakes_weight > 1.0:
            weight = max(weight, stakes_weight)
            components['stakes'] = {'weight': stakes_weight, 'markers': stakes_found}
        
        # Check rivalry
        rivalry_weight, rivalry_found = self._check_markers(all_text, self.rivalry_markers)
        if rivalry_weight > 1.0:
            weight *= rivalry_weight
            components['rivalry'] = {'weight': rivalry_weight, 'markers': rivalry_found}
        
        # Check momentum
        momentum_weight, momentum_found = self._check_markers(all_text, self.momentum_markers)
        if momentum_weight > 1.0:
            weight *= momentum_weight
            components['momentum'] = {'weight': momentum_weight, 'markers': momentum_found}
        
        # Check timing
        timing_weight, timing_found = self._check_markers(all_text, self.timing_markers)
        if timing_weight > 1.0:
            weight *= timing_weight
            components['timing'] = {'weight': timing_weight, 'markers': timing_found}
        
        # Check low-stakes (reduces weight)
        low_stakes_weight, low_stakes_found = self._check_markers(all_text, self.low_stakes_markers)
        if low_stakes_weight < 1.0:
            weight *= low_stakes_weight
            components['low_stakes'] = {'weight': low_stakes_weight, 'markers': low_stakes_found}
        
        # Cap weight at reasonable bounds
        weight = max(0.3, min(3.0, weight))
        
        # Classify weight level
        if weight >= 2.0:
            level = 'VERY HIGH'
            description = 'High-stakes, narratively critical comparison'
        elif weight >= 1.5:
            level = 'HIGH'
            description = 'Significant narrative context present'
        elif weight >= 1.2:
            level = 'ELEVATED'
            description = 'Some narrative importance'
        elif weight >= 0.8:
            level = 'MODERATE'
            description = 'Standard narrative context'
        else:
            level = 'LOW'
            description = 'Low-stakes or routine comparison'
        
        return {
            'weight': float(weight),
            'level': level,
            'description': description,
            'components': components,
            'explanation': self._generate_explanation(weight, components),
            'should_prioritize': weight >= 1.5
        }
    
    def _check_markers(self, text: str, markers: Dict[str, float]) -> Tuple[float, List[str]]:
        """Check for presence of markers and compute weight."""
        max_weight = 1.0
        found_markers = []
        
        for marker, marker_weight in markers.items():
            if marker in text:
                max_weight = max(max_weight, marker_weight)
                found_markers.append(marker)
        
        return max_weight, found_markers
    
    def _generate_explanation(self, weight: float, components: Dict) -> str:
        """Generate human-readable explanation of weighting."""
        if not components:
            return "Standard narrative context. No special weighting factors detected."
        
        parts = []
        
        if 'stakes' in components:
            markers = ', '.join(components['stakes']['markers'])
            parts.append(f"High stakes detected ({markers})")
        
        if 'rivalry' in components:
            markers = ', '.join(components['rivalry']['markers'])
            parts.append(f"Rivalry context ({markers})")
        
        if 'momentum' in components:
            markers = ', '.join(components['momentum']['markers'])
            parts.append(f"Momentum factors ({markers})")
        
        if 'timing' in components:
            markers = ', '.join(components['timing']['markers'])
            parts.append(f"Critical timing ({markers})")
        
        if 'low_stakes' in components:
            markers = ', '.join(components['low_stakes']['markers'])
            parts.append(f"Low-stakes context ({markers})")
        
        explanation = '. '.join(parts) + f". Overall narrative weight: {weight:.2f}x"
        return explanation
    
    def compute_temporal_dynamics(
        self,
        comparison_data: Dict,
        narrative_weight: float
    ) -> Dict[str, Any]:
        """
        Compute temporal prediction dynamics.
        
        Implements: "Better stories win over time, better ones over longer periods"
        """
        # Extract narrative quality indicators
        narrative_quality = self._assess_narrative_quality(comparison_data)
        
        # Predict accuracy across time horizons
        # Base accuracy starts around 52-55% (baseline)
        base_accuracy = 0.53
        
        # Narrative contribution increases with time
        # Immediate: minimal (noise dominates)
        # Short: emerging patterns
        # Medium: clear trends
        # Long: narrative prevails
        
        immediate_boost = (narrative_quality * narrative_weight) * 0.02
        short_boost = (narrative_quality * narrative_weight) * 0.05
        medium_boost = (narrative_quality * narrative_weight) * 0.08
        long_boost = (narrative_quality * narrative_weight) * 0.12
        
        return {
            'immediate': {
                'horizon': '1 event',
                'expected_accuracy': base_accuracy + immediate_boost,
                'confidence': 'LOW',
                'explanation': 'Single event predictions dominated by noise'
            },
            'short': {
                'horizon': '5 events',
                'expected_accuracy': base_accuracy + short_boost,
                'confidence': 'MODERATE',
                'explanation': 'Short-term patterns beginning to emerge'
            },
            'medium': {
                'horizon': '10 events',
                'expected_accuracy': base_accuracy + medium_boost,
                'confidence': 'GOOD',
                'explanation': 'Medium-term trends visible and reliable'
            },
            'long': {
                'horizon': 'Season/campaign',
                'expected_accuracy': base_accuracy + long_boost,
                'confidence': 'HIGH',
                'explanation': 'Long-term: better stories win consistently'
            },
            'narrative_quality': narrative_quality,
            'context_multiplier': narrative_weight
        }
    
    def _assess_narrative_quality(self, comparison_data: Dict) -> float:
        """
        Assess multi-dimensional narrative quality.
        
        Quality = f(Coherence, Confidence, Momentum, Identity, Stakes, Context, Timing)
        """
        if 'transformers' not in comparison_data:
            return 0.5
        
        transformers = comparison_data['transformers']
        
        # Extract quality indicators from each transformer
        quality_scores = []
        
        # Coherence (from linguistic and ensemble)
        if 'linguistic' in transformers:
            ling_diff = transformers['linguistic'].get('difference', 0)
            # Lower difference = more coherent comparison
            coherence = 1.0 - min(ling_diff / 10, 1.0)
            quality_scores.append(coherence)
        
        # Confidence/Identity (from self-perception and nominative)
        if 'self_perception' in transformers:
            sp_diff = transformers['self_perception'].get('difference', 0)
            # Strong identity = quality narrative
            identity_strength = min(sp_diff / 5, 1.0)
            quality_scores.append(identity_strength)
        
        # Momentum/Potential (from narrative_potential)
        if 'narrative_potential' in transformers:
            np_diff = transformers['narrative_potential'].get('difference', 0)
            # High potential difference = strong momentum narrative
            momentum = min(np_diff / 5, 1.0)
            quality_scores.append(momentum)
        
        # Overall differentiation
        overall_sim = comparison_data.get('overall_similarity', 0.5)
        # More differentiated = clearer narrative
        differentiation = 1.0 - overall_sim
        quality_scores.append(differentiation)
        
        # Average quality
        if quality_scores:
            return np.mean(quality_scores)
        return 0.5
    
    def should_bet_on_comparison(
        self,
        narrative_weight: float,
        prediction_confidence: float,
        model_edge: float,
        min_weight: float = 1.5,
        min_confidence: float = 0.25,
        min_edge: float = 0.10
    ) -> Dict[str, Any]:
        """
        Determine if comparison merits action (betting/decision).
        
        Context-aware strategy: Only act when narratives actually matter.
        """
        # All conditions must be met
        weight_ok = narrative_weight >= min_weight
        confidence_ok = prediction_confidence >= min_confidence
        edge_ok = model_edge >= min_edge
        
        should_act = weight_ok and confidence_ok and edge_ok
        
        reasons = []
        if not weight_ok:
            reasons.append(f"Narrative weight too low ({narrative_weight:.2f} < {min_weight})")
        if not confidence_ok:
            reasons.append(f"Confidence too low ({prediction_confidence:.2f} < {min_confidence})")
        if not edge_ok:
            reasons.append(f"Edge too small ({model_edge:.2f} < {min_edge})")
        
        if should_act:
            recommendation = "ACT"
            reasoning = f"High narrative context (weight={narrative_weight:.2f}) with sufficient confidence and edge"
        else:
            recommendation = "PASS"
            reasoning = '; '.join(reasons) if reasons else "Conditions not met"
        
        return {
            'recommendation': recommendation,
            'should_act': should_act,
            'reasoning': reasoning,
            'checks': {
                'narrative_weight': {'value': narrative_weight, 'threshold': min_weight, 'passed': weight_ok},
                'confidence': {'value': prediction_confidence, 'threshold': min_confidence, 'passed': confidence_ok},
                'edge': {'value': model_edge, 'threshold': min_edge, 'passed': edge_ok}
            }
        }


def create_narrative_weighter():
    """Factory function to create narrative weighter."""
    return NarrativeContextWeighter()

