"""
Sports Domain Types

Templates for sports domains (team and individual sports).
Customizes transformer selection and validation for sports-specific needs.
"""

from typing import List, Dict, Any
from .base import BaseDomainType
from pipelines.domain_config import DomainConfig, DomainType


class SportsDomain(BaseDomainType):
    """Template for all sports domains"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Sports domains emphasize audience, authority (coach), and star perspectives"""
        return ['audience', 'authority', 'star', 'collective']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Sports domain ADDITIONAL transformers (beyond core).
        
        NOTE: Core transformers (including nominative, self_perception, etc.)
        are available to ALL domains. This adds sports-specific transformers.
        """
        # Sports-specific transformers (core already included)
        sports_specific = [
            'ensemble',  # Critical for team sports
            'conflict',  # Competitive tension
            'statistical'  # Performance baseline
        ]
        
        # Add based on narrativity
        if п > 0.6:
            # More narrative-driven sports (tennis, golf)
            sports_specific.extend(['authenticity', 'emotional_semantic'])
        else:
            # More performance-driven (NBA, NFL)
            sports_specific.extend(['framing'])  # How story is framed matters
        
        return sports_specific
    
    def get_validation_metrics(self) -> List[str]:
        """Sports domains validate via betting accuracy + R²"""
        return ['r2', 'betting_roi', 'prediction_accuracy', 'calibration']
    
    def get_baseline_comparison(self) -> Dict[str, Any]:
        """Sports domains compare against performance statistics"""
        return {
            'method': 'performance_stats',
            'features': ['context_features'] if self.config.data.context_fields else [],
            'include_betting_odds': True
        }
    
    def get_domain_specific_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate sports-specific insights"""
        insights = []
        
        r_narrative = results.get('r_narrative', 0)
        pi = results.get('п', 0)
        
        if r_narrative > 0.5:
            insights.append(
                f"Strong narrative effect (r={r_narrative:.3f}) - story quality "
                f"predicts outcomes better than performance stats alone"
            )
        
        if pi < 0.5:
            insights.append(
                f"Performance-dominated domain (п={pi:.3f}) - physical skill "
                f"matters more than narrative, but narrative provides edge"
            )
        
        return insights
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sports-specific validation: betting ROI, context sensitivity"""
        checks = []
        
        # Check if betting ROI is available
        if 'betting_roi' in results:
            roi = results['betting_roi']
            if roi > 0.1:
                checks.append({
                    'check': 'betting_roi',
                    'status': 'pass',
                    'message': f'Positive betting ROI: {roi:.1%}'
                })
            else:
                checks.append({
                    'check': 'betting_roi',
                    'status': 'warn',
                    'message': f'Low betting ROI: {roi:.1%}'
                })
        
        # Check context sensitivity (playoffs vs regular season)
        if 'context_sensitivity' in results:
            sensitivity = results['context_sensitivity']
            checks.append({
                'check': 'context_sensitivity',
                'status': 'pass' if sensitivity > 0.1 else 'warn',
                'message': f'Context sensitivity: {sensitivity:.3f}'
            })
        
        return {
            'domain_specific_checks': checks,
            'all_passed': all(c['status'] == 'pass' for c in checks)
        }


class SportsIndividualDomain(SportsDomain):
    """Template for individual sports (tennis, golf, UFC)"""
    
    def get_default_transformers(self, п: float) -> List[str]:
        """Individual sports focus on self-perception and narrative potential"""
        # Start with base sports transformers
        base = super().get_default_transformers(п)
        
        # Emphasize individual-focused transformers
        individual_specific = [
            'self_perception',  # Individual identity
            'narrative_potential',  # Growth arc
            'authenticity'  # Personal authenticity
        ]
        
        # Remove ensemble (not relevant for individual sports)
        base = [t for t in base if t != 'ensemble']
        
        # Add individual-specific
        for trans in individual_specific:
            if trans not in base:
                base.append(trans)
        
        return base
    
    def get_validation_metrics(self) -> List[str]:
        """Individual sports: R², ROI, head-to-head prediction"""
        return ['r2', 'betting_roi', 'head_to_head_accuracy', 'prediction_accuracy']


class SportsTeamDomain(SportsDomain):
    """Template for team sports (NBA, NFL, soccer)"""
    
    def get_default_transformers(self, п: float) -> List[str]:
        """Team sports emphasize ensemble and relational dynamics"""
        # Start with base sports transformers
        base = super().get_default_transformers(п)
        
        # Ensure ensemble is included (critical for teams)
        if 'ensemble' not in base:
            base.insert(0, 'ensemble')
        
        # Add relational transformer
        if 'relational' not in base:
            base.append('relational')
        
        return base
    
    def get_validation_metrics(self) -> List[str]:
        """Team sports: R², betting accuracy, team chemistry metrics"""
        return ['r2', 'betting_roi', 'team_chemistry_score', 'prediction_accuracy']
    
    def get_domain_specific_insights(self, results: Dict[str, Any]) -> List[str]:
        """Team sports insights: chemistry, ensemble effects"""
        insights = super().get_domain_specific_insights(results)
        
        if 'ensemble_effect' in results:
            effect = results['ensemble_effect']
            insights.append(
                f"Ensemble effect: {effect:.3f} - team chemistry matters "
                f"more than individual talent"
            )
        
        return insights

