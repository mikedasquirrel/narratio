"""
Marriage Compatibility Domain

Tests whether phonetic/semantic interactions between partners' names
predict relationship outcomes.

Four Competing Theories:
1. Similarity: Similar names → compatibility
2. Complementarity: Opposite names → balance  
3. Golden Ratio: φ relationship → harmony (syllables, length)
4. Resonance: Harmonic ratios → success

Expected effect: r ~ 0.15-0.25

Dataset: 500 couples with relative success metrics
"""

from .data_loader import MarriageDataLoader
from .compatibility_analyzer import CompatibilityAnalyzer

__all__ = [
    'MarriageDataLoader',
    'CompatibilityAnalyzer'
]

