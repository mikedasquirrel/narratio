"""
Actantial Structure Transformer

Estimates presence of Greimas actantial roles (hero, opponent, helper,
object, sender) using lightweight keyword heuristics.
"""

import numpy as np
from .base import NarrativeTransformer


class ActantialStructureTransformer(NarrativeTransformer):
    """Detects balance between hero/object/helper/opponent language."""

    def __init__(self):
        super().__init__(
            narrative_id='actantial_structure',
            description='Measures Greimas actantial roles (hero/object/helper/opponent/sender).'
        )
        self.role_keywords = {
            'hero_terms': ['hero', 'captain', 'front-line', 'leading scorer', 'primary weapon'],
            'object_terms': ['goal', 'record', 'milestone', 'championship', 'dream'],
            'helper_terms': ['support', 'assist', 'helper', 'line-mate', 'partner'],
            'opponent_terms': ['foe', 'opponent', 'rival', 'adversary', 'villain'],
            'sender_terms': ['fans', 'organization', 'city', 'nation', 'legacy']
        }
        self.feature_names_ = list(self.role_keywords.keys())

    def fit(self, X, y=None):
        self.metadata['roles_tracked'] = self.feature_names_
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        n_samples = len(X)
        features = np.zeros((n_samples, len(self.feature_names_)))

        for i, text in enumerate(X):
            lowered = text.lower()
            for j, role in enumerate(self.feature_names_):
                features[i, j] = sum(lowered.count(keyword) for keyword in self.role_keywords[role])

        return features

    def _generate_interpretation(self) -> str:
        return "Higher values highlight narratives emphasizing specific actantial roles."

