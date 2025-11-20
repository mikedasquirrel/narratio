"""
Visibility Classifier - Automatically classify domain visibility
"""

from typing import Dict


class VisibilityClassifier:
    """Classify domains into visibility categories."""
    
    @staticmethod
    def classify_visibility_level(visibility: float) -> str:
        """Classify visibility into discrete levels."""
        if visibility >= 80:
            return 'ultra_high'
        elif visibility >= 60:
            return 'high'
        elif visibility >= 40:
            return 'medium'
        elif visibility >= 20:
            return 'low'
        else:
            return 'very_low'
    
    @staticmethod
    def classify_effect_size(effect: float) -> str:
        """Classify effect size magnitude."""
        if effect < 0.10:
            return 'negligible'
        elif effect < 0.20:
            return 'small'
        elif effect < 0.30:
            return 'medium'
        else:
            return 'large'
    
    @staticmethod
    def get_visibility_description(visibility: float) -> str:
        """Get description for visibility level."""
        descriptions = {
            'ultra_high': 'Direct performance observation (80-100%)',
            'high': 'Stats available, persona matters (60-80%)',
            'medium': 'Mixed data and story signals (40-60%)',
            'low': 'Story fills large information gaps (20-40%)',
            'very_low': 'Narrative dominates evaluation (0-20%)'
        }
        level = VisibilityClassifier.classify_visibility_level(visibility)
        return descriptions[level]

