"""
Entertainment Domain Type

Template for entertainment domains (movies, music, WWE).
Focuses on story structure, audience engagement, and award prediction.
"""

from typing import List, Dict, Any
from .base import BaseDomainType
from pipelines.domain_config import DomainConfig, DomainType


class EntertainmentDomain(BaseDomainType):
    """Template for entertainment domains (movies, music, WWE)"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Entertainment domains emphasize director, audience, critic, and cultural perspectives"""
        return ['director', 'audience', 'critic', 'cultural', 'meta']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Entertainment domain ADDITIONAL transformers (beyond core).
        
        NOTE: Core transformers (including nominative, self_perception, etc.)
        are available to ALL domains. This adds entertainment-specific transformers.
        """
        # Entertainment-specific: story structure is critical
        entertainment_specific = [
            'conflict',  # Story tension
            'suspense',  # Narrative tension
            'framing',  # How story is presented
            'emotional_semantic',  # Emotional resonance
            'cultural_context',  # Cultural fit
            'statistical'  # Baseline
        ]
        
        # Add based on narrativity
        if п > 0.7:
            # Highly narrative (WWE, awards)
            entertainment_specific.extend(['authenticity', 'expertise'])
        
        return entertainment_specific
    
    def get_validation_metrics(self) -> List[str]:
        """Entertainment domains: genre-specific effects, award prediction, audience engagement"""
        return ['r2', 'genre_specific_r2', 'award_prediction_auc', 'audience_engagement']
    
    def get_baseline_comparison(self) -> Dict[str, Any]:
        """Entertainment: compare against genre, budget, production value"""
        return {
            'method': 'genre_budget',
            'features': ['genre', 'budget', 'production_value'] if self.config.data.context_fields else []
        }
    
    def get_domain_specific_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate entertainment-specific insights"""
        insights = []
        
        # Genre-specific effects
        if 'genre_effects' in results:
            genre_effects = results['genre_effects']
            top_genre = max(genre_effects.items(), key=lambda x: x[1]) if genre_effects else None
            if top_genre:
                insights.append(
                    f"Strongest narrative effect in {top_genre[0]} genre "
                    f"(r={top_genre[1]:.3f})"
                )
        
        # Award prediction
        if 'award_auc' in results:
            auc = results['award_auc']
            if auc > 0.65:
                insights.append(
                    f"Strong award prediction (AUC={auc:.3f}) - narrative quality "
                    f"predicts awards better than genre/budget alone"
                )
        
        return insights
    
    def validate_domain_specific(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Entertainment-specific validation: genre effects, award accuracy"""
        checks = []
        
        # Check genre-specific effects
        if 'genre_effects' in results:
            genre_effects = results['genre_effects']
            if genre_effects:
                max_effect = max(genre_effects.values())
                checks.append({
                    'check': 'genre_specific_effects',
                    'status': 'pass' if max_effect > 0.3 else 'warn',
                    'message': f'Max genre effect: {max_effect:.3f}'
                })
        
        # Check award prediction
        if 'award_auc' in results:
            auc = results['award_auc']
            checks.append({
                'check': 'award_prediction',
                'status': 'pass' if auc > 0.65 else 'warn',
                'message': f'Award prediction AUC: {auc:.3f}'
            })
        
        return {
            'domain_specific_checks': checks,
            'all_passed': all(c['status'] == 'pass' for c in checks)
        }
    
    def get_reporting_template(self) -> str:
        """Entertainment domains use genre-focused reporting"""
        return 'entertainment_genre'

