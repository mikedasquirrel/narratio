"""
Cognitive Load Transformer

Approximates language complexity via lexical diversity, word length, punctuation density.
"""

import numpy as np
import re
from .base import NarrativeTransformer


class CognitiveLoadTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='cognitive_load',
            description='Measures text complexity indicators tied to perceived cognitive load.'
        )

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            tokens = re.findall(r"[A-Za-z']+", text.lower())
            total = len(tokens) or 1
            avg_word_len = sum(len(t) for t in tokens) / total
            unique_ratio = len(set(tokens)) / total
            punctuation = len(re.findall(r'[,:;()]', text)) / max(len(text), 1)
            subordinate = sum(text.lower().count(w) for w in ('which', 'that', 'because', 'while'))
            subordinate_ratio = subordinate / total
            rows.append([
                avg_word_len / 10.0,
                unique_ratio,
                punctuation * 10,
                subordinate_ratio
            ])
        return np.array(rows, dtype=float)

