"""
Temporal Dynamics Analyzer

Analyzes how predictions evolve across time horizons.
Implements: "Better stories win over time, better ones over longer periods"

Provides predictions for:
- Immediate (single event)
- Short-term (5 events)
- Medium-term (10 events)
- Long-term (season/campaign)
"""

from typing import Dict, Any, List
import numpy as np


class TemporalDynamicsAnalyzer:
    """
    Analyzes temporal dynamics of narrative predictions.
    
    Core insight: Narrative effects compound over time.
    - Immediate: Noise dominates (~52-55% accuracy)
    - Short: Patterns emerge (~56-59%)
    - Medium: Trends clear (~60-64%)
    - Long: Stories prevail (~64-68%)
    """
    
    def __init__(self):
        # Time horizons
        self.horizons = {
            'immediate': {
                'name': 'Immediate',
                'description': 'Single event/interaction',
                'events': 1,
                'noise_factor': 0.95,  # Very high noise
                'narrative_factor': 0.05  # Minimal narrative effect
            },
            'short': {
                'name': 'Short-term',
                'description': '5 events/interactions',
                'events': 5,
                'noise_factor': 0.65,
                'narrative_factor': 0.35
            },
            'medium': {
                'name': 'Medium-term',
                'description': '10 events/interactions',
                'events': 10,
                'noise_factor': 0.40,
                'narrative_factor': 0.60
            },
            'long': {
                'name': 'Long-term',
                'description': 'Season/campaign (20+ events)',
                'events': 20,
                'noise_factor': 0.20,
                'narrative_factor': 0.80
            }
        }
    
    def analyze_temporal_dynamics(
        self,
        comparison_data: Dict,
        base_prediction: Dict,
        narrative_weight: float
    ) -> Dict[str, Any]:
        """
        Analyze predictions across time horizons.
        
        Parameters
        ----------
        comparison_data : dict
            Full comparison results
        base_prediction : dict
            Current prediction
        narrative_weight : float
            Context weight from narrative weighter
        
        Returns
        -------
        temporal_analysis : dict
            Predictions and confidence across time horizons
        """
        # Extract base probability
        if 'prediction' in base_prediction and base_prediction['prediction'].get('can_predict'):
            base_prob_a = base_prediction['prediction']['text_a_probability']
        else:
            # Use similarity-based estimate
            similarity = comparison_data.get('overall_similarity', 0.5)
            base_prob_a = 0.5 + (1 - similarity) * 0.2
        
        # Calculate narrative quality
        narrative_quality = self._assess_narrative_quality(comparison_data)
        
        # Generate predictions for each horizon
        horizon_predictions = {}
        
        for horizon_key, horizon_config in self.horizons.items():
            prediction = self._predict_for_horizon(
                base_prob_a,
                horizon_config,
                narrative_quality,
                narrative_weight,
                comparison_data
            )
            horizon_predictions[horizon_key] = prediction
        
        # Add evolution analysis
        evolution = self._analyze_evolution(horizon_predictions)
        
        # Generate insights
        insights = self._generate_temporal_insights(
            horizon_predictions,
            evolution,
            narrative_quality,
            narrative_weight
        )
        
        return {
            'horizons': horizon_predictions,
            'evolution': evolution,
            'insights': insights,
            'narrative_quality': float(narrative_quality),
            'narrative_weight': float(narrative_weight)
        }
    
    def _predict_for_horizon(
        self,
        base_prob: float,
        horizon_config: Dict,
        narrative_quality: float,
        narrative_weight: float,
        comparison_data: Dict
    ) -> Dict[str, Any]:
        """Generate prediction for specific time horizon."""
        # Base accuracy (around coin flip for immediate)
        base_accuracy = 0.53
        
        # Narrative contribution increases with time
        narrative_boost = (
            narrative_quality *
            narrative_weight *
            horizon_config['narrative_factor'] *
            0.15  # Max boost of ~15%
        )
        
        # Expected accuracy at this horizon
        expected_accuracy = base_accuracy + narrative_boost
        expected_accuracy = min(0.75, expected_accuracy)  # Cap at 75%
        
        # Adjust prediction based on horizon
        # Longer horizon = more confident if narrative is strong
        noise_factor = horizon_config['noise_factor']
        narrative_factor = horizon_config['narrative_factor']
        
        # Calculate probability
        deviation = base_prob - 0.5
        
        # Apply temporal dynamics
        # Short term: mostly noise, some narrative
        # Long term: mostly narrative, less noise
        temporal_deviation = (
            deviation * noise_factor * 0.3 +  # Noise component
            deviation * narrative_factor * 1.2  # Narrative component
        )
        
        prob_a = 0.5 + temporal_deviation
        prob_a = max(0.20, min(0.80, prob_a))
        
        prob_b = 1.0 - prob_a
        winner = "Text A" if prob_a > prob_b else "Text B"
        
        # Confidence increases with time if narrative is strong
        if narrative_quality > 0.6 and narrative_weight >= 1.5:
            if narrative_factor > 0.6:
                confidence = 'HIGH'
            elif narrative_factor > 0.4:
                confidence = 'MODERATE-HIGH'
            else:
                confidence = 'MODERATE'
        else:
            if narrative_factor > 0.6:
                confidence = 'MODERATE'
            else:
                confidence = 'LOW'
        
        return {
            'name': horizon_config['name'],
            'description': horizon_config['description'],
            'events': horizon_config['events'],
            'text_a_probability': float(prob_a),
            'text_b_probability': float(prob_b),
            'winner': winner,
            'confidence': confidence,
            'expected_accuracy': float(expected_accuracy),
            'noise_factor': float(noise_factor),
            'narrative_factor': float(narrative_factor),
            'explanation': self._explain_horizon(horizon_config, narrative_quality, narrative_weight)
        }
    
    def _explain_horizon(
        self,
        horizon_config: Dict,
        narrative_quality: float,
        narrative_weight: float
    ) -> str:
        """Generate explanation for this time horizon."""
        name = horizon_config['name']
        narrative_factor = horizon_config['narrative_factor']
        
        if narrative_factor < 0.2:
            return f"{name}: Single event dominated by noise and randomness. Narrative barely matters."
        elif narrative_factor < 0.5:
            return f"{name}: Narrative patterns beginning to emerge but noise still significant."
        elif narrative_factor < 0.7:
            return f"{name}: Narrative trends visible and increasingly reliable. Better stories winning."
        else:
            return f"{name}: Narrative fully expressed. Better stories win consistently over time."
    
    def _assess_narrative_quality(self, comparison_data: Dict) -> float:
        """Assess overall narrative quality of comparison."""
        if 'transformers' not in comparison_data:
            return 0.5
        
        transformers = comparison_data['transformers']
        
        # Check differentiation across dimensions
        differences = []
        for trans_name, trans_data in transformers.items():
            diff = trans_data.get('difference', 0)
            differences.append(diff)
        
        # High differentiation = strong narrative quality
        avg_diff = np.mean(differences) if differences else 0
        
        # Normalize to 0-1
        quality = min(avg_diff / 5.0, 1.0)
        
        return quality
    
    def _analyze_evolution(self, horizon_predictions: Dict) -> Dict[str, Any]:
        """Analyze how prediction evolves across time."""
        # Extract probabilities across horizons
        horizons_ordered = ['immediate', 'short', 'medium', 'long']
        
        probs_a = []
        confidences = []
        
        for h in horizons_ordered:
            if h in horizon_predictions:
                probs_a.append(horizon_predictions[h]['text_a_probability'])
                conf_map = {'LOW': 1, 'MODERATE': 2, 'MODERATE-HIGH': 3, 'HIGH': 4}
                confidences.append(conf_map.get(horizon_predictions[h]['confidence'], 2))
        
        # Calculate trends
        prob_trend = 'increasing' if probs_a[-1] > probs_a[0] + 0.05 else 'decreasing' if probs_a[-1] < probs_a[0] - 0.05 else 'stable'
        conf_trend = 'increasing' if confidences[-1] > confidences[0] else 'stable'
        
        # Calculate range
        prob_range = max(probs_a) - min(probs_a)
        
        return {
            'probability_trend': prob_trend,
            'confidence_trend': conf_trend,
            'probability_range': float(prob_range),
            'convergence': 'high' if prob_range < 0.10 else 'moderate' if prob_range < 0.20 else 'low',
            'interpretation': self._interpret_evolution(prob_trend, conf_trend, prob_range)
        }
    
    def _interpret_evolution(self, prob_trend: str, conf_trend: str, prob_range: float) -> str:
        """Interpret evolutionary pattern."""
        if conf_trend == 'increasing' and prob_range < 0.15:
            return "Prediction strengthens and stabilizes over time. Strong narrative signal."
        elif conf_trend == 'increasing':
            return "Confidence increases over time but prediction varies. Narrative emerging."
        elif prob_range < 0.10:
            return "Prediction stable across horizons. Consistent narrative throughout."
        else:
            return "Prediction shifts across time horizons. Context-dependent narrative."
    
    def _generate_temporal_insights(
        self,
        horizon_predictions: Dict,
        evolution: Dict,
        narrative_quality: float,
        narrative_weight: float
    ) -> List[str]:
        """Generate insights about temporal dynamics."""
        insights = []
        
        # Insight 1: Quality assessment
        if narrative_quality > 0.7:
            insights.append(
                "Strong narrative differentiation detected. "
                "Predictions become more reliable over longer time horizons."
            )
        elif narrative_quality > 0.4:
            insights.append(
                "Moderate narrative signals present. "
                "Medium to long-term predictions have value but expect some uncertainty."
            )
        else:
            insights.append(
                "Weak narrative differentiation. "
                "Predictions uncertain across all time horizons. Outcomes largely unpredictable."
            )
        
        # Insight 2: Context weight
        if narrative_weight >= 2.0:
            insights.append(
                "Very high narrative context weight detected (championship/critical stakes). "
                "Stories matter enormously in this comparison."
            )
        elif narrative_weight >= 1.5:
            insights.append(
                "High narrative context weight (important matchup). "
                "Narrative analysis highly relevant for prediction."
            )
        elif narrative_weight >= 1.0:
            insights.append(
                "Moderate narrative context. "
                "Stories matter but other factors also important."
            )
        else:
            insights.append(
                "Low narrative context weight. "
                "This comparison has limited narrative significance."
            )
        
        # Insight 3: Evolution pattern
        if evolution['confidence_trend'] == 'increasing':
            insights.append(
                "Confidence increases with time horizon, following 'better stories win over time' theory. "
                "Long-term prediction more trustworthy than immediate."
            )
        
        if evolution['convergence'] == 'high':
            insights.append(
                "Prediction highly stable across time horizons. "
                "Outcome expected to be consistent regardless of timeframe."
            )
        elif evolution['convergence'] == 'low':
            insights.append(
                "Prediction varies significantly across time horizons. "
                "Short-term and long-term outcomes may differ."
            )
        
        # Insight 4: Actionable recommendation
        long_pred = horizon_predictions.get('long', {})
        immediate_pred = horizon_predictions.get('immediate', {})
        
        if long_pred.get('confidence') in ['HIGH', 'MODERATE-HIGH']:
            insights.append(
                f"For long-term outlook: {long_pred.get('winner', 'Unknown')} favored. "
                f"Better story wins over time."
            )
        
        if immediate_pred.get('confidence') == 'LOW':
            insights.append(
                "Immediate outcome highly uncertain (dominated by noise). "
                "Single event predictions unreliable regardless of narrative strength."
            )
        
        return insights


def create_temporal_analyzer():
    """Factory function to create temporal analyzer."""
    return TemporalDynamicsAnalyzer()

