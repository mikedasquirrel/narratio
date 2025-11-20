"""
Explanation Generator

Creates human-readable explanations of learned archetypes and predictions.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Any
from collections import defaultdict


class ExplanationGenerator:
    """
    Generates natural language explanations for:
    - Learned archetypes
    - Pattern importance
    - Predictions
    - Feature attribution
    
    Makes the learning system interpretable.
    """
    
    def __init__(self):
        self.pattern_templates = {
            'universal_underdog': "The underdog narrative: {entity} is lower-ranked or unexpected",
            'universal_comeback': "The comeback story: {entity} recovered from a difficult position",
            'universal_rivalry': "Rivalry narrative: repeated competition with historical context",
            'universal_dominance': "Dominance pattern: consistent superiority and strong performance",
            'universal_pressure': "High-pressure moment: critical, high-stakes situation",
        }
    
    def explain_pattern(
        self,
        pattern_name: str,
        pattern_data: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate natural language explanation of a pattern.
        
        Parameters
        ----------
        pattern_name : str
            Pattern name
        pattern_data : dict
            Pattern data
        context : dict, optional
            Additional context
        
        Returns
        -------
        str
            Human-readable explanation
        """
        # Check for template
        if pattern_name in self.pattern_templates:
            template = self.pattern_templates[pattern_name]
            explanation = template.format(**(context or {}))
        else:
            # Generate explanation from data
            pattern_type = pattern_data.get('type', 'pattern')
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            frequency = pattern_data.get('frequency', 0.0)
            
            if pattern_type == 'narrative_archetype':
                desc = pattern_data.get('description', '')
                explanation = f"{desc}. This pattern appears in {frequency:.1%} of narratives."
            elif pattern_type == 'domain_specific':
                domain = pattern_data.get('domain', 'unknown')
                explanation = f"Domain-specific pattern for {domain}: characterized by {', '.join(keywords[:3])}. "
                explanation += f"Appears in {frequency:.1%} of {domain} narratives."
            else:
                explanation = f"Pattern characterized by: {', '.join(keywords[:3])}. "
                explanation += f"Frequency: {frequency:.1%}."
        
        # Add validation info if available
        if 'validation' in pattern_data:
            val = pattern_data['validation']
            if val.get('validated', False):
                correlation = val.get('correlation', 0.0)
                effect_size = val.get('effect_size', 0.0)
                explanation += f" Validated: correlation={correlation:.2f}, effect size={effect_size:.2f}."
        
        return explanation
    
    def explain_prediction(
        self,
        entity_name: str,
        story_quality: float,
        contributing_patterns: List[Dict[str, Any]],
        outcome_prediction: float
    ) -> str:
        """
        Explain a prediction.
        
        Parameters
        ----------
        entity_name : str
            Entity being analyzed
        story_quality : float
            Computed story quality (ю)
        contributing_patterns : list of dict
            Patterns that contributed
        outcome_prediction : float
            Predicted outcome
        
        Returns
        -------
        str
            Explanation
        """
        explanation = f"Analysis for {entity_name}:\n\n"
        explanation += f"Story Quality (ю): {story_quality:.3f}\n"
        explanation += f"Predicted Outcome: {outcome_prediction:.1%}\n\n"
        
        if contributing_patterns:
            explanation += "Contributing Patterns:\n"
            for pattern in contributing_patterns[:5]:  # Top 5
                pattern_name = pattern['name']
                contribution = pattern.get('contribution', 0.0)
                explanation += f"  • {pattern_name}: {contribution:+.3f}\n"
                
                # Add pattern description
                if 'description' in pattern:
                    explanation += f"    {pattern['description']}\n"
        
        return explanation
    
    def explain_feature_importance(
        self,
        feature_importances: Dict[str, float],
        top_n: int = 10
    ) -> str:
        """
        Explain feature importance.
        
        Parameters
        ----------
        feature_importances : dict
            Feature name -> importance score
        top_n : int
            Number of top features to explain
        
        Returns
        -------
        str
            Explanation
        """
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        explanation = "Most Important Patterns:\n\n"
        
        for i, (feature, importance) in enumerate(sorted_features[:top_n]):
            direction = "increases" if importance > 0 else "decreases"
            explanation += f"{i+1}. {feature}: {direction} outcome probability by {abs(importance):.2%}\n"
        
        return explanation
    
    def explain_archetype_evolution(
        self,
        pattern_name: str,
        versions: List[Dict[str, Any]]
    ) -> str:
        """
        Explain how an archetype evolved over time.
        
        Parameters
        ----------
        pattern_name : str
            Pattern name
        versions : list of dict
            Version history
        
        Returns
        -------
        str
            Explanation
        """
        explanation = f"Evolution of {pattern_name}:\n\n"
        
        for i, version in enumerate(versions):
            version_id = version.get('version', f'v{i+1}')
            performance = version.get('performance', 0.0)
            created = version.get('created_at', 'unknown')
            
            explanation += f"Version {version_id} ({created}):\n"
            explanation += f"  Performance: {performance:+.3f}\n"
            
            if i > 0:
                prev_perf = versions[i-1].get('performance', 0.0)
                change = performance - prev_perf
                direction = "improved" if change > 0 else "declined"
                explanation += f"  Change: {direction} by {abs(change):.3f}\n"
            
            explanation += "\n"
        
        return explanation
    
    def generate_report(
        self,
        domain: str,
        patterns: Dict[str, Dict],
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        Generate comprehensive report.
        
        Parameters
        ----------
        domain : str
            Domain name
        patterns : dict
            All patterns
        performance_metrics : dict
            Performance metrics
        
        Returns
        -------
        str
            Formatted report
        """
        report = f"="*80 + "\n"
        report += f"ARCHETYPE LEARNING REPORT: {domain.upper()}\n"
        report += f"="*80 + "\n\n"
        
        # Performance summary
        report += "Performance Metrics:\n"
        for metric_name, metric_value in performance_metrics.items():
            report += f"  {metric_name}: {metric_value:.3f}\n"
        report += "\n"
        
        # Patterns summary
        report += f"Discovered Patterns: {len(patterns)}\n\n"
        
        # Group patterns by type
        by_type = defaultdict(list)
        for pattern_name, pattern_data in patterns.items():
            pattern_type = pattern_data.get('type', 'unknown')
            by_type[pattern_type].append((pattern_name, pattern_data))
        
        for pattern_type, pattern_list in by_type.items():
            report += f"{pattern_type.replace('_', ' ').title()}:\n"
            for pattern_name, pattern_data in pattern_list:
                frequency = pattern_data.get('frequency', 0.0)
                report += f"  • {pattern_name} (frequency: {frequency:.1%})\n"
                
                # Add validation status
                if 'validation' in pattern_data:
                    if pattern_data['validation'].get('validated', False):
                        corr = pattern_data['validation'].get('correlation', 0.0)
                        report += f"    Validated (r={corr:.2f})\n"
                    else:
                        report += f"    Not validated\n"
            
            report += "\n"
        
        report += "="*80 + "\n"
        
        return report
    
    def visualize_pattern_network(
        self,
        patterns: Dict[str, Dict],
        relationships: Dict[str, List[str]]
    ) -> str:
        """
        Create text-based visualization of pattern relationships.
        
        Parameters
        ----------
        patterns : dict
            Patterns
        relationships : dict
            Pattern relationships (pattern -> related patterns)
        
        Returns
        -------
        str
            ASCII visualization
        """
        viz = "Pattern Network:\n\n"
        
        for pattern_name, related in relationships.items():
            viz += f"{pattern_name}\n"
            for i, related_pattern in enumerate(related):
                is_last = i == len(related) - 1
                connector = "└──" if is_last else "├──"
                viz += f"  {connector} {related_pattern}\n"
            viz += "\n"
        
        return viz

