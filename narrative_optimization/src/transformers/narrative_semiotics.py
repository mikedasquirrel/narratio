"""
Narrative Semiotics Transformer

Heuristic counts of protagonist/antagonist/moral language inspired by Greimas semiotic square.
"""

import numpy as np
from .base import NarrativeTransformer


class NarrativeSemioticsTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='narrative_semiotics',
            description='Counts semiotic markers (protagonist/opponent/moral/evaluation cues).'
        )
        self.lexicons = {
            'protagonist': ['hero', 'captain', 'leader', 'star', 'franchise'],
            'opponent': ['villain', 'nemesis', 'rival', 'antagonist', 'spoiler'],
            'moral': ['justice', 'honor', 'respect', 'loyalty', 'ethic'],
            'evaluation': ['crucial', 'pivotal', 'iconic', 'legendary', 'historic']
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            lowered = text.lower()
            counts = []
            for words in self.lexicons.values():
                counts.append(sum(lowered.count(w) for w in words))
            total_words = max(len(text.split()), 1)
            normalized = [c / total_words for c in counts]
            rows.append(normalized)
        return np.array(rows, dtype=float)

