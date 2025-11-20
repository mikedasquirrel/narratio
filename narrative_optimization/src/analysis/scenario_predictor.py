"""
Multi-Scenario Predictor

Generates multiple prediction scenarios with confidence intervals:
- Optimistic (best narrative alignment)
- Pessimistic (worst narrative alignment)
- Realistic (most likely given context)
- Context-weighted (narrative theory integrated)

Provides probability distributions instead of point predictions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class MultiScenarioPredictor:
    """
    Generates multiple prediction scenarios with uncertainty quantification.
    
    Instead of single prediction: "Text A: 65%"
    Provides scenarios:
    - Optimistic: "Text A: 75% (if narrative aligns perfectly)"
    - Realistic: "Text A: 65% (most likely scenario)"
    - Pessimistic: "Text A: 55% (if narrative factors disappoint)"
    """
    
    def __init__(self):
        pass
    
    def generate_scenarios(
        self,
        base_prediction: Dict,
        comparison_data: Dict,
        narrative_weight: float,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate multi-scenario predictions.
        
        Parameters
        ----------
        base_prediction : dict
            Initial prediction from comprehensive_compare
        comparison_data : dict
            Full comparison results
        narrative_weight : float
            Context weight from narrative weighter
        context : dict
            Additional context
        
        Returns
        -------
        scenarios : dict
            Multiple prediction scenarios with reasoning
        """
        # Extract base probabilities
        if 'prediction' in base_prediction and base_prediction['prediction'].get('can_predict'):
            base_prob_a = base_prediction['prediction']['text_a_probability']
        else:
            # If no prediction, use similarity-based estimate
            similarity = comparison_data.get('overall_similarity', 0.5)
            # Less similar = more differentiated = one likely better
            base_prob_a = 0.5 + (1 - similarity) * 0.2
        
        # Calculate uncertainty based on domain confidence
        expertise = comparison_data.get('domain_expertise', {}).get('expertise', {})
        base_confidence = expertise.get('overall_confidence', 0.5)
        
        # Higher confidence = narrower scenarios, lower = wider
        uncertainty_factor = 1.0 - base_confidence
        
        # Generate scenario probabilities
        scenarios = {
            'optimistic': self._generate_optimistic_scenario(
                base_prob_a, comparison_data, narrative_weight, uncertainty_factor
            ),
            'realistic': self._generate_realistic_scenario(
                base_prob_a, comparison_data, narrative_weight, base_confidence
            ),
            'pessimistic': self._generate_pessimistic_scenario(
                base_prob_a, comparison_data, narrative_weight, uncertainty_factor
            ),
            'context_weighted': self._generate_context_weighted_scenario(
                base_prob_a, comparison_data, narrative_weight, context
            )
        }
        
        # Add distribution information
        scenarios['distribution'] = self._compute_probability_distribution(
            scenarios, base_confidence
        )
        
        # Add meta-information
        scenarios['meta'] = {
            'base_probability_a': float(base_prob_a),
            'narrative_weight': float(narrative_weight),
            'uncertainty_factor': float(uncertainty_factor),
            'confidence': float(base_confidence)
        }
        
        return scenarios
    
    def _generate_optimistic_scenario(
        self,
        base_prob: float,
        comparison_data: Dict,
        narrative_weight: float,
        uncertainty: float
    ) -> Dict[str, Any]:
        """Generate best-case scenario."""
        # Optimistic: narrative factors all align favorably
        # Push probability further from 50% (more confident prediction)
        
        deviation = base_prob - 0.5
        optimistic_boost = 1.0 + (narrative_weight * 0.15) + (uncertainty * 0.1)
        optimistic_deviation = deviation * optimistic_boost
        
        prob_a = 0.5 + optimistic_deviation
        # Clamp to reasonable bounds
        prob_a = max(0.15, min(0.85, prob_a))
        
        prob_b = 1.0 - prob_a
        winner = "Text A" if prob_a > prob_b else "Text B"
        
        # Extract favorable factors
        favorable_factors = self._extract_favorable_factors(comparison_data, prob_a > 0.5)
        
        return {
            'name': 'Optimistic',
            'text_a_probability': float(prob_a),
            'text_b_probability': float(prob_b),
            'winner': winner,
            'confidence_level': 'HIGH' if narrative_weight >= 1.5 else 'MODERATE',
            'description': 'Best-case scenario where narrative factors align favorably',
            'reasoning': f"If {', '.join(favorable_factors[:3])}, {winner} has strong advantage",
            'likelihood': 'POSSIBLE' if narrative_weight >= 1.5 else 'OPTIMISTIC'
        }
    
    def _generate_realistic_scenario(
        self,
        base_prob: float,
        comparison_data: Dict,
        narrative_weight: float,
        confidence: float
    ) -> Dict[str, Any]:
        """Generate most-likely scenario."""
        # Realistic: use base prediction with slight context adjustment
        
        # Apply narrative weight moderately
        deviation = base_prob - 0.5
        realistic_adjustment = 1.0 + (narrative_weight - 1.0) * 0.5
        realistic_deviation = deviation * realistic_adjustment
        
        prob_a = 0.5 + realistic_deviation
        prob_a = max(0.20, min(0.80, prob_a))
        
        prob_b = 1.0 - prob_a
        winner = "Text A" if prob_a > prob_b else "Text B"
        
        # Determine confidence
        if abs(prob_a - 0.5) > 0.15 and confidence > 0.6:
            conf_level = 'MODERATE-HIGH'
        elif abs(prob_a - 0.5) > 0.10:
            conf_level = 'MODERATE'
        else:
            conf_level = 'LOW'
        
        return {
            'name': 'Realistic',
            'text_a_probability': float(prob_a),
            'text_b_probability': float(prob_b),
            'winner': winner,
            'confidence_level': conf_level,
            'description': 'Most likely outcome based on current analysis',
            'reasoning': self._generate_realistic_reasoning(comparison_data, winner),
            'likelihood': 'MOST LIKELY'
        }
    
    def _generate_pessimistic_scenario(
        self,
        base_prob: float,
        comparison_data: Dict,
        narrative_weight: float,
        uncertainty: float
    ) -> Dict[str, Any]:
        """Generate worst-case scenario."""
        # Pessimistic: narrative factors don't help or work against
        # Push probability toward 50% (less confident)
        
        deviation = base_prob - 0.5
        pessimistic_factor = 1.0 - (uncertainty * 0.3)
        pessimistic_deviation = deviation * pessimistic_factor
        
        prob_a = 0.5 + pessimistic_deviation
        prob_a = max(0.30, min(0.70, prob_a))
        
        prob_b = 1.0 - prob_a
        winner = "Text A" if prob_a > prob_b else "Text B"
        
        # Extract risk factors
        risk_factors = self._extract_risk_factors(comparison_data)
        
        return {
            'name': 'Pessimistic',
            'text_a_probability': float(prob_a),
            'text_b_probability': float(prob_b),
            'winner': winner,
            'confidence_level': 'LOW',
            'description': 'Worst-case scenario where predictions are less reliable',
            'reasoning': f"If narrative factors disappoint or {', '.join(risk_factors[:2])}, outcome is uncertain",
            'likelihood': 'POSSIBLE'
        }
    
    def _generate_context_weighted_scenario(
        self,
        base_prob: float,
        comparison_data: Dict,
        narrative_weight: float,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate scenario using narrative weighting theory."""
        # Apply full narrative weighting theory
        
        # High weight = trust narrative more
        # Low weight = regress toward uncertainty
        
        if narrative_weight >= 1.5:
            # High-context: narratives matter, trust the analysis
            deviation = base_prob - 0.5
            weighted_deviation = deviation * (1.0 + (narrative_weight - 1.0) * 0.7)
            prob_a = 0.5 + weighted_deviation
            trust_level = 'HIGH'
            reasoning = "High narrative context detected. Story-based prediction reliable."
        elif narrative_weight >= 1.0:
            # Moderate context: some narrative signal
            deviation = base_prob - 0.5
            weighted_deviation = deviation * 1.0
            prob_a = 0.5 + weighted_deviation
            trust_level = 'MODERATE'
            reasoning = "Moderate narrative context. Prediction has value but uncertain."
        else:
            # Low context: narratives don't matter much, high uncertainty
            prob_a = 0.5 + (base_prob - 0.5) * 0.5
            trust_level = 'LOW'
            reasoning = "Low narrative context. Outcome largely unpredictable from stories alone."
        
        prob_a = max(0.25, min(0.75, prob_a))
        prob_b = 1.0 - prob_a
        winner = "Text A" if prob_a > prob_b else "Text B"
        
        return {
            'name': 'Context-Weighted',
            'text_a_probability': float(prob_a),
            'text_b_probability': float(prob_b),
            'winner': winner,
            'confidence_level': trust_level,
            'description': 'Prediction weighted by narrative context importance',
            'reasoning': reasoning,
            'likelihood': 'THEORY-BASED',
            'narrative_weight_applied': float(narrative_weight)
        }
    
    def _compute_probability_distribution(
        self,
        scenarios: Dict,
        confidence: float
    ) -> Dict[str, Any]:
        """Compute full probability distribution across scenarios."""
        # Extract probabilities from each scenario
        probs_a = [
            scenarios['optimistic']['text_a_probability'],
            scenarios['realistic']['text_a_probability'],
            scenarios['pessimistic']['text_a_probability'],
            scenarios['context_weighted']['text_a_probability']
        ]
        
        mean_prob = np.mean(probs_a)
        std_prob = np.std(probs_a)
        
        # Confidence intervals
        # Higher confidence = narrower intervals
        ci_width = std_prob * (2.0 - confidence)
        
        ci_lower = max(0.0, mean_prob - ci_width)
        ci_upper = min(1.0, mean_prob + ci_width)
        
        return {
            'mean': float(mean_prob),
            'std': float(std_prob),
            'confidence_interval_95': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'range': {
                'min': float(min(probs_a)),
                'max': float(max(probs_a))
            }
        }
    
    def _extract_favorable_factors(self, comparison_data: Dict, favor_a: bool) -> List[str]:
        """Extract factors that favor one text over the other."""
        factors = []
        
        if 'transformers' not in comparison_data:
            return ['narrative factors align', 'context supports', 'momentum present']
        
        transformers = comparison_data['transformers']
        
        # Check which transformers show strong differentiation
        for name, data in transformers.items():
            if data.get('difference', 0) > 3.0:
                factors.append(f"{name.replace('_', ' ')} strongly differentiates")
        
        if not factors:
            factors = ['key narrative elements align', 'context supports outcome', 'patterns are clear']
        
        return factors
    
    def _extract_risk_factors(self, comparison_data: Dict) -> List[str]:
        """Extract uncertainty/risk factors."""
        risks = []
        
        # Low differentiation = high uncertainty
        if comparison_data.get('overall_similarity', 0) > 0.7:
            risks.append('texts are very similar')
        
        # Check domain confidence
        expertise = comparison_data.get('domain_expertise', {}).get('expertise', {})
        if expertise.get('overall_confidence', 1.0) < 0.5:
            risks.append('limited domain expertise')
        
        if not risks:
            risks = ['context is ambiguous', 'outcome is uncertain', 'variables are unclear']
        
        return risks
    
    def _generate_realistic_reasoning(self, comparison_data: Dict, winner: str) -> str:
        """Generate reasoning for realistic scenario."""
        # Use AI insights if available
        if 'ai_insights' in comparison_data:
            insights = comparison_data['ai_insights']
            if 'reasoning' in insights:
                return insights['reasoning']
        
        # Use most different dimension
        most_diff = comparison_data.get('most_different_dimension', 'narrative')
        return f"Based on {most_diff.replace('_', ' ')} analysis, {winner} shows stronger narrative position"


def create_scenario_predictor():
    """Factory function to create scenario predictor."""
    return MultiScenarioPredictor()

