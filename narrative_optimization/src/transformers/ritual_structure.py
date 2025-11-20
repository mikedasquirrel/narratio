"""
Ritual Structure Transformer

Detects ritual/tradition language signaling ceremony, repetition, preparation.
"""

import numpy as np
from .base import NarrativeTransformer


class RitualStructureTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='ritual_structure',
            description='Counts tradition/ceremony/preparation vocabulary.'
        )
        self.lexicons = {
            'tradition': ['tradition', 'legacy', 'heritage', 'ritual', 'custom'],
            'ceremony': ['ceremony', 'anthem', 'banner', 'moment of silence', 'ceremonial'],
            'preparation': ['pregame', 'warmup', 'walkthrough', 'drill', 'practice'],
            'communal': ['fans', 'crowd', 'community', 'supporters', 'faithful']
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            lowered = text.lower()
            counts = [
                sum(lowered.count(word) for word in words)
                for words in self.lexicons.values()
            ]
            total = max(len(text.split()), 1)
            rows.append([c / total for c in counts])
        return np.array(rows, dtype=float)

