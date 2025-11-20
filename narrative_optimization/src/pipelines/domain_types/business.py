"""
Business Domain Type

Template for business domains (startups, crypto).
Focuses on credibility, growth potential, and market dynamics.
"""

from typing import List, Dict, Any
from .base import BaseDomainType


class BusinessDomain(BaseDomainType):
    """Template for business domains (startups, crypto)"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Business domains emphasize authority (CEO), audience (market), and cultural perspectives"""
        return ['authority', 'audience', 'cultural', 'meta']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Business domain ADDITIONAL transformers (beyond core).
        
        NOTE: Core transformers are available to ALL domains.
        """
        additional = [
            'authenticity',  # Trust
            'expertise',  # Credibility
            'statistical'  # Baseline
        ]
        
        if п > 0.7:
            additional.extend(['emotional_semantic', 'cultural_context'])
        
        return additional
    
    def get_validation_metrics(self) -> List[str]:
        return ['r2', 'market_correlation', 'growth_prediction', 'credibility_score']
    
    def get_reporting_template(self) -> str:
        return 'business_growth'


class MedicalDomain(BaseDomainType):
    """Template for medical domains (mental health)"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Medical domains emphasize authority (clinician), character (patient), and cultural perspectives"""
        return ['authority', 'character', 'cultural']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """
        Medical domain ADDITIONAL transformers (beyond core).
        
        NOTE: Core transformers are available to ALL domains.
        """
        additional = [
            'authenticity',
            'expertise',
            'statistical'  # Baseline
        ]
        
        if п > 0.5:
            additional.extend(['emotional_semantic'])
        
        return additional
    
    def get_validation_metrics(self) -> List[str]:
        return ['r2', 'clinical_correlation', 'stigma_prediction', 'treatment_outcome']
    
    def get_reporting_template(self) -> str:
        return 'medical_clinical'

