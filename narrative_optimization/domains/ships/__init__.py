"""
Naval Ships Domain - Historical Naming Patterns Analysis

Tests whether ship names predict historical significance and mission success:
- Name gravitas (virtue, monarch, geographic vs saint)
- Purpose alignment (name fits mission type)
- Temporal evolution (naming patterns through centuries)

Key Finding: Geographic-named ships > Saint-named ships (d=1.12, p<0.0001)
Effect: r ~ 0.18 (visibility=50%, narrative importance=medium)

Dataset: 853 naval vessels across 500 years (1460-1990)
"""

from .data_loader import ShipDataLoader
from .gravitas_analyzer import GravitasAnalyzer

__all__ = [
    'ShipDataLoader',
    'GravitasAnalyzer'
]

