"""
Immigration Identity Transformation Domain

Analyzes how name adaptation patterns reflect integration:
- Original → adapted name changes
- Toponymic markers (place-based naming)
- Assimilation indicators
- Generational progression

Expected effect: r ~ 0.20 (visibility=30%)

Dataset: 200 immigration studies with temporal trends
"""

from .data_loader import ImmigrationDataLoader
from .adaptation_analyzer import AdaptationAnalyzer

__all__ = [
    'ImmigrationDataLoader',
    'AdaptationAnalyzer'
]

