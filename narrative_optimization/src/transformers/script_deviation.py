"""
Script Deviation Transformer

Estimates how much a narrative describes expected vs unexpected events,
based on simple cue words.
"""

import numpy as np
from .base import NarrativeTransformer


class ScriptDeviationTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='script_deviation',
            description='Compares expectation markers vs surprise markers to estimate deviation.'
        )
        self.expectation_words = ['expected', 'typical', 'routine', 'standard', 'predictable']
        self.surprise_words = ['upset', 'shock', 'surprise', 'chaos', 'unthinkable', 'wild']

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            lowered = text.lower()
            exp_count = sum(lowered.count(w) for w in self.expectation_words)
            surprise_count = sum(lowered.count(w) for w in self.surprise_words)
            total = max(len(text.split()), 1)
            deviation = (surprise_count - exp_count) / total
            rows.append([
                exp_count / total,
                surprise_count / total,
                deviation
            ])
        return np.array(rows, dtype=float)

