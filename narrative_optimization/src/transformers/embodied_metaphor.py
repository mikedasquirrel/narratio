"""
Embodied Metaphor Transformer

Counts sensory/body-motion references to capture embodied framing.
"""

import numpy as np
from .base import NarrativeTransformer


class EmbodiedMetaphorTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='embodied_metaphor',
            description='Tracks body/sensory/metabolic vocabulary indicating embodied metaphors.'
        )
        self.lexicons = {
            'body': ['heart', 'mind', 'hand', 'shoulder', 'muscle', 'vein'],
            'motion': ['drive', 'push', 'pull', 'surge', 'collapse', 'burst'],
            'senses': ['see', 'hear', 'feel', 'taste', 'touch', 'smell'],
            'balance': ['center', 'pivot', 'balance', 'stance', 'posture']
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
            rows.append([c / total_words for c in counts])
        return np.array(rows, dtype=float)

