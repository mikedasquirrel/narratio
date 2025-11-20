"""
Cross-Cultural Archetype Transformer

Captures archetypal story energy (hero, mentor, trickster, warrior, caretaker)
across sports/business narratives by counting grounded language markers.
"""

import numpy as np
from .base import NarrativeTransformer


class CrossCulturalArchetypeTransformer(NarrativeTransformer):
    """Quantifies archetype references using lightweight linguistic heuristics."""

    def __init__(self):
        super().__init__(
            narrative_id='cross_cultural_archetype',
            description='Tracks archetypal language (hero, mentor, trickster, warrior, caretaker).'
        )
        self.archetype_keywords = {
            'hero': ['hero', 'clutch', 'savior', 'go-to', 'leader'],
            'mentor': ['mentor', 'coach', 'guide', 'veteran', 'captain'],
            'trickster': ['trick', 'surprise', 'wildcard', 'chaos', 'unpredictable'],
            'warrior': ['battle', 'war', 'fight', 'grit', 'warrior'],
            'caretaker': ['support', 'glue', 'steady', 'reliable', 'anchor']
        }
        self.feature_names_ = list(self.archetype_keywords.keys())

    def fit(self, X, y=None):
        self.metadata['archetypes_tracked'] = self.feature_names_
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        n_samples = len(X)
        features = np.zeros((n_samples, len(self.feature_names_)))

        for idx, text in enumerate(X):
            lowered = text.lower()
            for j, archetype in enumerate(self.feature_names_):
                features[idx, j] = sum(lowered.count(keyword) for keyword in self.archetype_keywords[archetype])

        return features

    def _generate_interpretation(self) -> str:
        return "Higher values indicate narratives leaning on specific archetypal framing (hero, mentor, etc.)."

