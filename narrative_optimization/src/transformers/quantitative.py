"""
Quantitative Transformer

Captures numerical precision, statistical language, and quantitative specificity.
Numbers make claims concrete, enable verification, and signal credibility.

Features extracted (10):
- Number density, precision level
- Statistical language (percentage, average, median)
- Measurement units, comparative quantification
- Range language, magnitude markers
"""

from typing import List, Dict, Any
import numpy as np
import re
from .base import NarrativeTransformer


class QuantitativeTransformer(NarrativeTransformer):
    """Analyzes quantitative and numerical patterns."""
    
    def __init__(self):
        super().__init__(
            narrative_id="quantitative",
            description="Quantitative analysis: numbers, precision, statistical language"
        )
    
    def fit(self, X, y=None):
        """Learn quantitative patterns from corpus."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform to quantitative features."""
        self._validate_fitted()
        return np.array([self._extract_quantitative_features(text) for text in X])
    
    def _extract_quantitative_features(self, text: str) -> np.ndarray:
        """Extract 10 quantitative features."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = max(1, len(words))
        
        features = []
        
        # Number density
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        features.append(len(numbers) / word_count)
        
        # Precision (decimal places)
        decimals = [n for n in numbers if '.' in n]
        precision = len(decimals) / max(1, len(numbers))
        features.append(precision)
        
        # Statistical language
        stats_words = ['percent', 'average', 'median', 'mean', 'std', 'variance', 'correlation']
        stats_count = sum(1 for w in stats_words if w in text.lower())
        features.append(stats_count / word_count)
        
        # Measurement units
        units = ['kg', 'lb', 'meter', 'mile', 'dollar', 'euro', 'hour', 'year', 'percent']
        unit_count = sum(1 for u in units if u in text.lower())
        features.append(unit_count / word_count)
        
        # Comparative quantification
        comparatives = [r'\d+x', r'\d+%\s+more', r'\d+%\s+less', r'double', r'triple', r'half']
        comp_count = sum(len(re.findall(p, text.lower())) for p in comparatives)
        features.append(comp_count / word_count)
        
        # Range language
        ranges = [r'between\s+\d+\s+and\s+\d+', r'from\s+\d+\s+to\s+\d+', r'\d+-\d+']
        range_count = sum(len(re.findall(p, text)) for p in ranges)
        features.append(range_count / word_count)
        
        # Magnitude markers
        magnitudes = ['thousand', 'million', 'billion', 'trillion', 'hundred']
        mag_count = sum(1 for m in magnitudes if m in text.lower())
        features.append(mag_count / word_count)
        
        # Rounding indicators
        rounding = ['about', 'approximately', 'roughly', 'around', 'nearly', 'almost']
        round_count = sum(1 for r in rounding if r in text.lower())
        features.append(round_count / word_count)
        
        # Exact vs approximate
        exact_words = ['exactly', 'precise', 'specific', 'exact']
        exact_count = sum(1 for e in exact_words if e in text.lower())
        total_precision = exact_count + round_count
        exactness_ratio = exact_count / total_precision if total_precision > 0 else 0.5
        features.append(exactness_ratio)
        
        # Overall quantitative density
        total_quant = len(numbers) + stats_count + unit_count + comp_count
        features.append(total_quant / word_count)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        return ['number_density', 'precision_level', 'statistical_language', 'measurement_units',
                'comparative_quantification', 'range_language', 'magnitude_markers', 'rounding_indicators',
                'exactness_ratio', 'overall_quantitative_density']

