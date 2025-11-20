"""
Multi-Perspective Transformer

Estimates perspective balance using pronoun ratios (first/second/third person).
"""

import numpy as np
from .base import NarrativeTransformer


class MultiPerspectiveTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='multi_perspective',
            description='Tracks first/second/third person pronoun ratios.'
        )
        self.first = {'i', 'me', 'we', 'us', 'our', 'ours'}
        self.second = {'you', 'your', 'yours'}
        self.third = {'he', 'she', 'they', 'him', 'her', 'them', 'their', 'theirs'}

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            tokens = [t.lower() for t in text.split()]
            total = max(len(tokens), 1)
            first_ratio = sum(1 for t in tokens if t in self.first) / total
            second_ratio = sum(1 for t in tokens if t in self.second) / total
            third_ratio = sum(1 for t in tokens if t in self.third) / total
            diversity = len({cat for cat, ratio in zip(['f', 's', 't'], [first_ratio, second_ratio, third_ratio]) if ratio > 0}) / 3
            rows.append([first_ratio, second_ratio, third_ratio, diversity])
        return np.array(rows, dtype=float)

