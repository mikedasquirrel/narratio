"""
Lightweight Discourse Analysis Transformer

Approximates discourse structure using simple sentence statistics
so we can capture coherence and transition signals without heavy models.
"""

import numpy as np
import re
from .base import NarrativeTransformer


class DiscourseAnalysisTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='discourse_analysis',
            description='Estimates discourse coherence via sentence length/variance and transition cues.'
        )
        self.transition_markers = {'however', 'therefore', 'meanwhile', 'because', 'although', 'but', 'then'}

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        features = []
        for text in X:
            sentences = self._split_sentences(text)
            n_sent = max(len(sentences), 1)
            lengths = [len(s.split()) for s in sentences] or [0]
            avg_len = sum(lengths) / n_sent
            std_len = float(np.std(lengths)) if n_sent > 1 else 0.0
            transitions = sum(1 for s in sentences if self._has_transition(s))
            transition_ratio = transitions / n_sent
            intro_len = lengths[0] if sentences else 0
            outro_len = lengths[-1] if sentences else 0
            features.append([
                n_sent / 20.0,  # normalized sentence count
                avg_len / 40.0,
                std_len / 20.0,
                transition_ratio,
                (intro_len - outro_len) / 40.0
            ])
        return np.array(features, dtype=float)

    def _split_sentences(self, text):
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def _has_transition(self, sentence: str) -> bool:
        tokens = sentence.lower().split()
        return any(token in self.transition_markers for token in tokens)

