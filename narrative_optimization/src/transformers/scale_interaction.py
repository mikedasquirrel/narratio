"""
Scale Interaction Transformer

Tracks micro/meso/macro scale language (moments vs season vs legacy)
and interactions between them.
"""

import numpy as np
from .base import NarrativeTransformer


class ScaleInteractionTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='scale_interaction',
            description='Measures references across micro (play), meso (season), and macro (legacy) scales.'
        )
        self.micro = {'shift', 'play', 'possession', 'drive', 'sequence', 'moment'}
        self.meso = {'streak', 'season', 'series', 'month', 'stretch'}
        self.macro = {'career', 'legacy', 'dynasty', 'history', 'decade', 'generation'}

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            tokens = text.lower().split()
            total = max(len(tokens), 1)
            micro = sum(1 for t in tokens if t in self.micro) / total
            meso = sum(1 for t in tokens if t in self.meso) / total
            macro = sum(1 for t in tokens if t in self.macro) / total
            span = macro - micro
            rows.append([micro, meso, macro, span])
        return np.array(rows, dtype=float)

