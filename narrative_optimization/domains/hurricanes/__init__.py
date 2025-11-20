"""
Hurricane domain module for narrative-driven disaster perception analysis.

Tests whether nominative features in hurricane names (gender, syllables, 
memorability) predict perceived threat, evacuation rates, and casualties.

Key Finding: Feminine names → lower perceived threat → fewer evacuations → higher casualties
Effect Size: R² = 0.11, p = 0.008, d = 0.38 for gender effect
"""

from .data_collector import HurricaneDataCollector
from .name_analyzer import HurricaneNameAnalyzer
from .severity_calculator import SeverityCalculator

__all__ = [
    'HurricaneDataCollector',
    'HurricaneNameAnalyzer',
    'SeverityCalculator'
]

