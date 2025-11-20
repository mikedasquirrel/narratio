"""
Conceptual Metaphor Transformer

Scores narratives on common conceptual metaphors (BATTLE, JOURNEY, MACHINE, WEATHER)
using intuitive keyword families.
"""

import numpy as np
from .base import NarrativeTransformer


class ConceptualMetaphorTransformer(NarrativeTransformer):
    """Counts metaphor families to capture framing intensity."""

    def __init__(self):
        super().__init__(
            narrative_id='conceptual_metaphor',
            description='Measures prevalence of battle/journey/machine/weather metaphors.'
        )
        self.metaphor_keywords = {
            'battle': ['battle', 'fight', 'war', 'onslaught', 'siege', 'attack', 'defense'],
            'journey': ['journey', 'path', 'road', 'chapter', 'milestone', 'checkpoint'],
            'machine': ['engine', 'gears', 'machine', 'mechanism', 'systematic'],
            'weather': ['storm', 'cold', 'heat', 'blizzard', 'gust', 'pressure front']
        }
        self.feature_names_ = list(self.metaphor_keywords.keys())

    def fit(self, X, y=None):
        self.metadata['metaphors_tracked'] = self.feature_names_
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        n_samples = len(X)
        features = np.zeros((n_samples, len(self.feature_names_)))

        for idx, text in enumerate(X):
            lowered = text.lower()
            for j, metaphor in enumerate(self.feature_names_):
                features[idx, j] = sum(lowered.count(keyword) for keyword in self.metaphor_keywords[metaphor])

        return features

    def _generate_interpretation(self) -> str:
        return "Highlights which conceptual metaphors dominate the narrative framing."

