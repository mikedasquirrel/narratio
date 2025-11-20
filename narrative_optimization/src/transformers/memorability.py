"""
Memorability Transformer

Uses simple heuristics (alliteration, repetition, question/exclamation usage)
to estimate narrative memorability cues.
"""

import numpy as np
import re
from collections import Counter
from .base import NarrativeTransformer


class MemorabilityTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='memorability',
            description='Estimates memorability via phonetic repetition and rhetorical flourish.'
        )

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        rows = []
        for text in X:
            tokens = re.findall(r"[A-Za-z']+", text.lower())
            total = max(len(tokens), 1)
            initials = [t[0] for t in tokens if t]
            initial_counts = Counter(initials)
            if initial_counts:
                alliteration = max(initial_counts.values()) / total
            else:
                alliteration = 0.0
            repeats = self._repetition_ratio(tokens)
            rhetorical = (text.count('?') + text.count('!')) / max(len(text), 1)
            vivid_words = sum(1 for t in tokens if len(t) >= 8)
            rows.append([
                alliteration,
                repeats,
                rhetorical * 10,
                vivid_words / total
            ])
        return np.array(rows, dtype=float)

    def _repetition_ratio(self, tokens):
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        repeated = sum(c for c in counts.values() if c > 1)
        return repeated / len(tokens)

