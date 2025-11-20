"""
Nominative Domain Type

Template for nominative domains (housing, hurricanes, pure nominative tests).
Focuses on name-based features and pure nominative effects.
"""

from typing import List, Dict, Any
from .base import BaseDomainType
from pipelines.domain_config import DomainConfig, DomainType


class NominativeDomain(BaseDomainType):
    """Template for nominative domains (housing, hurricanes, pure nominative)"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Nominative domains emphasize character, cultural, and meta perspectives"""
        return ['character', 'cultural', 'meta']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Nominative domain ADDITIONAL transformers (beyond core).
        
        NOTE: Nominative features (nominative, phonetic, etc.) are available
        to ALL domains as core narrative features. This domain type adds
        emphasis on pure nominative effect analysis.
        """
        # Additional transformers for nominative-focused analysis
        # (Core transformers including nominative are already included)
        additional = [
            'cultural_context',  # Cultural associations with names
            'statistical'  # Baseline for comparison
        ]
        
        # Add based on narrativity
        if п > 0.8:
            # Highly nominative (housing #13, self-rated)
            additional.extend([
                'authenticity',  # Name authenticity
                'narrative_potential'  # Growth potential
            ])
        else:
            # Mixed nominative (hurricanes)
            additional.extend(['self_perception'])
        
        return additional
    
    def get_validation_metrics(self) -> List[str]:
        """Nominative domains: pure nominative effect, phonetic vs semantic, cultural variation"""
        return [
            'r2',
            'nominative_effect',
            'phonetic_contribution',
            'semantic_contribution',
            'cultural_variation'
        ]
    
    def get_baseline_comparison(self) -> Dict[str, Any]:
        """Nominative: compare against random/control baseline"""
        return {
            'method': 'random_control',
            'features': [],
            'control_group': 'random_names'
        }
    
    def get_domain_specific_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate nominative-specific insights"""
        insights = []
        
        # Pure nominative effect
        if 'nominative_effect' in results:
            effect = results['nominative_effect']
            insights.append(
                f"Pure nominative effect: {effect:.3f} - names alone predict outcomes"
            )
        
        # Phonetic vs semantic
        if 'phonetic_contribution' in results and 'semantic_contribution' in results:
            phonetic = results['phonetic_contribution']
            semantic = results['semantic_contribution']
            
            if phonetic > semantic:
                insights.append(
                    f"Phonetic effects ({phonetic:.3f}) > semantic ({semantic:.3f}) - "
                    f"sound patterns matter more than meaning"
                )
            else:
                insights.append(
                    f"Semantic effects ({semantic:.3f}) > phonetic ({phonetic:.3f}) - "
                    f"meaning matters more than sound"
                )
        
        # Cultural variation
        if 'cultural_variation' in results:
            variation = results['cultural_variation']
            insights.append(
                f"Cultural variation: {variation:.3f} - effects vary by cultural context"
            )
        
        return insights
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Nominative-specific validation: pure effect isolation, confound control"""
        checks = []
        
        # Check pure nominative effect
        if 'nominative_effect' in results:
            effect = results['nominative_effect']
            checks.append({
                'check': 'pure_nominative_effect',
                'status': 'pass' if effect > 0.1 else 'warn',
                'message': f'Pure nominative effect: {effect:.3f}'
            })
        
        # Check confound control
        if 'confound_control' in results:
            control = results['confound_control']
            checks.append({
                'check': 'confound_control',
                'status': 'pass' if control < 0.05 else 'warn',
                'message': f'Confound control: {control:.3f}'
            })
        
        return {
            'domain_specific_checks': checks,
            'all_passed': all(c['status'] == 'pass' for c in checks)
        }
    
    def get_reporting_template(self) -> str:
        """Nominative domains use pure-effect reporting"""
        return 'nominative_pure'

