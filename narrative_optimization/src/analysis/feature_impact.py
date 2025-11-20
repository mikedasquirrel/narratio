"""
Feature Impact Analyzer

Analyzes which features drive predictions and enables sensitivity analysis.
Shows narrative flexibility: "What if we reframe the story?"
Provides counterfactual analysis and explains reasoning.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class FeatureImpactAnalyzer:
    """
    Analyzes feature importance and sensitivity for narrative predictions.
    
    Answers questions like:
    - Which features matter most?
    - How sensitive is the prediction to each variable?
    - What if we reframe the narrative?
    - What needs to change to flip the prediction?
    """
    
    def __init__(self):
        # Feature categories for interpretation
        self.feature_categories = {
            'identity': ['nominative', 'self_perception'],
            'potential': ['narrative_potential'],
            'style': ['linguistic'],
            'relationships': ['relational', 'ensemble']
        }
    
    def analyze_feature_impact(
        self,
        comparison_data: Dict,
        base_prediction: Dict
    ) -> Dict[str, Any]:
        """
        Analyze which features drive the prediction.
        
        Parameters
        ----------
        comparison_data : dict
            Full comparison results
        base_prediction : dict
            Current prediction
        
        Returns
        -------
        impact_analysis : dict
            Feature importance, sensitivity, and insights
        """
        if 'transformers' not in comparison_data:
            return {'error': 'No transformer data available'}
        
        # Calculate feature importances
        feature_importances = self._calculate_feature_importances(comparison_data)
        
        # Identify top drivers
        top_drivers = self._identify_top_drivers(feature_importances, n=10)
        
        # Calculate sensitivities
        sensitivities = self._calculate_sensitivities(comparison_data, feature_importances)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            comparison_data, base_prediction, feature_importances
        )
        
        # Category-level impact
        category_impact = self._calculate_category_impact(feature_importances)
        
        return {
            'top_drivers': top_drivers,
            'sensitivities': sensitivities,
            'counterfactuals': counterfactuals,
            'category_impact': category_impact,
            'narrative_flexibility': self._assess_narrative_flexibility(sensitivities)
        }
    
    def _calculate_feature_importances(self, comparison_data: Dict) -> List[Dict[str, Any]]:
        """Calculate importance of each feature."""
        importances = []
        
        transformers = comparison_data['transformers']
        
        for transformer_name, transformer_data in transformers.items():
            if 'feature_names' not in transformer_data:
                continue
            
            features_a = np.array(transformer_data['text_a']['features'])
            features_b = np.array(transformer_data['text_b']['features'])
            feature_names = transformer_data['feature_names']
            
            # Calculate absolute differences
            differences = np.abs(features_a - features_b)
            
            # Normalize by transformer's overall difference
            transformer_diff = transformer_data.get('difference', 1.0)
            
            for idx, name in enumerate(feature_names):
                if idx < len(differences):
                    importance = {
                        'feature': name,
                        'transformer': transformer_name,
                        'value_a': float(features_a[idx]),
                        'value_b': float(features_b[idx]),
                        'difference': float(differences[idx]),
                        'normalized_importance': float(differences[idx] / (transformer_diff + 0.01)),
                        'raw_importance': float(differences[idx])
                    }
                    importances.append(importance)
        
        # Sort by raw importance
        importances.sort(key=lambda x: x['raw_importance'], reverse=True)
        
        return importances
    
    def _identify_top_drivers(
        self,
        importances: List[Dict],
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify top N most important features."""
        top = importances[:n]
        
        # Add interpretations
        for item in top:
            item['interpretation'] = self._interpret_feature_impact(item)
            item['direction'] = 'Text A higher' if item['value_a'] > item['value_b'] else 'Text B higher'
        
        return top
    
    def _interpret_feature_impact(self, feature_impact: Dict) -> str:
        """Generate human-readable interpretation of feature impact."""
        feature = feature_impact['feature']
        diff = feature_impact['difference']
        val_a = feature_impact['value_a']
        val_b = feature_impact['value_b']
        
        # Generate interpretation based on feature name
        if 'future' in feature.lower():
            if val_a > val_b:
                return "Text A more future-oriented"
            else:
                return "Text B more future-oriented"
        
        elif 'confidence' in feature.lower() or 'agency' in feature.lower():
            if val_a > val_b:
                return "Text A shows more confidence/agency"
            else:
                return "Text B shows more confidence/agency"
        
        elif 'growth' in feature.lower() or 'potential' in feature.lower():
            if val_a > val_b:
                return "Text A emphasizes growth more"
            else:
                return "Text B emphasizes growth more"
        
        elif 'emotion' in feature.lower() or 'sentiment' in feature.lower():
            if val_a > val_b:
                return "Text A more emotionally expressive"
            else:
                return "Text B more emotionally expressive"
        
        elif 'complexity' in feature.lower():
            if val_a > val_b:
                return "Text A linguistically more complex"
            else:
                return "Text B linguistically more complex"
        
        else:
            # Generic interpretation
            higher = "A" if val_a > val_b else "B"
            return f"Text {higher} scores higher on {feature.replace('_', ' ')}"
    
    def _calculate_sensitivities(
        self,
        comparison_data: Dict,
        importances: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate prediction sensitivity to each feature."""
        # Group by transformer
        transformer_sensitivities = {}
        
        for imp in importances:
            trans = imp['transformer']
            if trans not in transformer_sensitivities:
                transformer_sensitivities[trans] = {
                    'total_impact': 0.0,
                    'features': []
                }
            
            transformer_sensitivities[trans]['total_impact'] += imp['raw_importance']
            transformer_sensitivities[trans]['features'].append(imp)
        
        # Calculate relative sensitivities
        total_impact = sum(t['total_impact'] for t in transformer_sensitivities.values())
        
        for trans, data in transformer_sensitivities.items():
            data['relative_sensitivity'] = data['total_impact'] / (total_impact + 0.01)
            data['sensitivity_level'] = self._sensitivity_level(data['relative_sensitivity'])
        
        return transformer_sensitivities
    
    def _sensitivity_level(self, sensitivity: float) -> str:
        """Convert sensitivity score to level."""
        if sensitivity > 0.25:
            return 'VERY HIGH'
        elif sensitivity > 0.18:
            return 'HIGH'
        elif sensitivity > 0.12:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _generate_counterfactuals(
        self,
        comparison_data: Dict,
        base_prediction: Dict,
        importances: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios: "What if X changed?"."""
        counterfactuals = []
        
        # Get top 5 most important features
        top_features = importances[:5]
        
        for feature_imp in top_features:
            feature = feature_imp['feature']
            transformer = feature_imp['transformer']
            val_a = feature_imp['value_a']
            val_b = feature_imp['value_b']
            diff = feature_imp['difference']
            
            # Generate counterfactual: what if values were swapped?
            cf = {
                'feature': feature,
                'transformer': transformer,
                'scenario': f"What if '{feature.replace('_', ' ')}' values were reversed?",
                'current_difference': float(diff),
                'impact': self._estimate_prediction_impact(diff),
                'explanation': self._explain_counterfactual(feature, val_a, val_b)
            }
            counterfactuals.append(cf)
        
        return counterfactuals
    
    def _estimate_prediction_impact(self, feature_diff: float) -> str:
        """Estimate how much a feature impacts the prediction."""
        if feature_diff > 0.5:
            return "MAJOR IMPACT - Could significantly change prediction"
        elif feature_diff > 0.2:
            return "MODERATE IMPACT - Would shift prediction meaningfully"
        elif feature_diff > 0.1:
            return "MINOR IMPACT - Would slightly adjust prediction"
        else:
            return "MINIMAL IMPACT - Small change to prediction"
    
    def _explain_counterfactual(self, feature: str, val_a: float, val_b: float) -> str:
        """Explain what a counterfactual means narratively."""
        feature_clean = feature.replace('_', ' ')
        
        if 'future' in feature.lower():
            return f"If the future orientation were reversed, the narrative momentum would shift"
        elif 'confidence' in feature.lower():
            return f"If confidence markers were swapped, the perceived strength would change"
        elif 'growth' in feature.lower():
            return f"If growth language were flipped, the developmental trajectory would invert"
        elif 'emotion' in feature.lower():
            return f"If emotional intensity were reversed, the resonance would shift"
        else:
            return f"If {feature_clean} were different, the narrative balance would change"
    
    def _calculate_category_impact(self, importances: List[Dict]) -> Dict[str, Any]:
        """Calculate impact at category level (identity, potential, style, relationships)."""
        category_impacts = {
            'identity': 0.0,
            'potential': 0.0,
            'style': 0.0,
            'relationships': 0.0,
            'other': 0.0
        }
        
        for imp in importances:
            transformer = imp['transformer']
            impact = imp['raw_importance']
            
            categorized = False
            for category, transformers in self.feature_categories.items():
                if transformer in transformers:
                    category_impacts[category] += impact
                    categorized = True
                    break
            
            if not categorized:
                category_impacts['other'] += impact
        
        # Normalize
        total = sum(category_impacts.values())
        if total > 0:
            for category in category_impacts:
                category_impacts[category] = category_impacts[category] / total
        
        # Sort by impact
        sorted_categories = sorted(
            category_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'impacts': dict(sorted_categories),
            'primary_driver': sorted_categories[0][0] if sorted_categories else 'unknown',
            'secondary_driver': sorted_categories[1][0] if len(sorted_categories) > 1 else 'unknown'
        }
    
    def _assess_narrative_flexibility(self, sensitivities: Dict) -> Dict[str, Any]:
        """Assess how flexible/robust the narrative is."""
        # High flexibility = many features matter = unstable prediction
        # Low flexibility = few features matter = robust prediction
        
        # Count transformers with high sensitivity
        high_sensitivity_count = sum(
            1 for data in sensitivities.values()
            if data['relative_sensitivity'] > 0.18
        )
        
        if high_sensitivity_count >= 4:
            flexibility = 'HIGH'
            description = 'Prediction highly sensitive to narrative framing'
            robustness = 'LOW'
        elif high_sensitivity_count >= 2:
            flexibility = 'MODERATE'
            description = 'Prediction moderately sensitive to key narrative elements'
            robustness = 'MODERATE'
        else:
            flexibility = 'LOW'
            description = 'Prediction robust to narrative variations'
            robustness = 'HIGH'
        
        return {
            'flexibility': flexibility,
            'robustness': robustness,
            'description': description,
            'high_sensitivity_count': high_sensitivity_count,
            'interpretation': self._interpret_flexibility(flexibility)
        }
    
    def _interpret_flexibility(self, flexibility: str) -> str:
        """Interpret what flexibility means."""
        if flexibility == 'HIGH':
            return "Many narrative dimensions matter equally. How you frame the story significantly impacts the prediction. This comparison is empirically sensitive."
        elif flexibility == 'MODERATE':
            return "A few key narrative dimensions drive the prediction. Framing matters but outcome is somewhat stable."
        else:
            return "One or two narrative dimensions dominate. Prediction is robust regardless of how story is told."


def create_feature_impact_analyzer():
    """Factory function to create feature impact analyzer."""
    return FeatureImpactAnalyzer()

