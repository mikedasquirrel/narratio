"""
Labovian Narrative Transformer

Estimates presence of Labov's six-part structure: abstract, orientation,
complicating action, evaluation, resolution, coda.
"""

import numpy as np
from .base import NarrativeTransformer


class LabovianNarrativeTransformer(NarrativeTransformer):
    """Binary/bounded features for Labov narrative components."""

    def __init__(self):
        super().__init__(
            narrative_id='labovian_structure',
            description='Detects Labov narrative components (abstract/orientation/action/evaluation/resolution/coda).'
        )
        self.component_keywords = {
            'abstract': ['headline', 'summary', 'storyline', 'opening'],
            'orientation': ['setting', 'context', 'background', 'scenario'],
            'complicating_action': ['suddenly', 'then', 'after that', 'escalated'],
            'evaluation': ['significant', 'critical', 'importantly', 'notably'],
            'resolution': ['finally', 'ultimately', 'in the end', 'resulted'],
            'coda': ['looking ahead', 'next up', 'moving forward', 'closing']
        }
        self.feature_names_ = list(self.component_keywords.keys())

    def fit(self, X, y=None):
        self.metadata['labov_components'] = self.feature_names_
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._validate_fitted()
        n_samples = len(X)
        features = np.zeros((n_samples, len(self.feature_names_)))

        for idx, text in enumerate(X):
            lowered = text.lower()
            for j, comp in enumerate(self.feature_names_):
                features[idx, j] = 1 if any(keyword in lowered for keyword in self.component_keywords[comp]) else 0

        return features

    def _generate_interpretation(self) -> str:
        return "Values indicate whether Labov narrative beats are explicitly present."

