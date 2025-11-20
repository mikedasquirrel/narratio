"""
Attentional Structure Transformer

Measures how narratives direct audience attention via emphasis markers,
questions, and list structures.
"""

import numpy as np
import re
from .base import NarrativeTransformer


class AttentionalStructureTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='attentional_structure',
            description='Captures emphasis markers, rhetorical questions, enumerations.'
        )
        self.emphasis_words = ['notably', 'importantly', 'crucially', 'remember', 'focus']

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            lowered = text.lower()
            emphasis = sum(lowered.count(w) for w in self.emphasis_words)
            questions = text.count('?')
            exclamations = text.count('!')
            enumerations = len(re.findall(r'\\b(?:first|second|third|1\\.|2\\.|3\\.)', lowered))
            total_words = max(len(text.split()), 1)
            rows.append([
                emphasis / total_words,
                questions / 5.0,
                exclamations / 5.0,
                enumerations / 3.0
            ])
        return np.array(rows, dtype=float)

