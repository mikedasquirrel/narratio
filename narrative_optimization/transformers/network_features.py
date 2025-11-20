"""
Network Feature Transformer
===========================

Wraps `NetworkGenomeBuilder` so pipelines can consume graph-derived metrics as
regular features.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from narrative_optimization.analysis.network_genome_builder import (
    NetworkGenomeBuilder,
)


class NetworkFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, domain_name: str, feature_order: Optional[List[str]] = None):
        self.domain_name = domain_name
        self.builder = NetworkGenomeBuilder(domain_name)
        self.feature_order = feature_order
        self.graph_metadata_ = {}

    def fit(self, X: Iterable[Dict], y=None):
        records = list(X)
        annotated = self.builder.annotate_records(records)
        self.graph_metadata_ = annotated.get("graph_metadata", {})
        sample_features = next(
            (
                record.get("network_features")
                for record in annotated.get("records", [])
                if record.get("network_features")
            ),
            {},
        )
        if sample_features and not self.feature_order:
            self.feature_order = sorted(sample_features.keys())
        return self

    def transform(self, X: Iterable[Dict]):
        records = list(X)
        annotated = self.builder.annotate_records(records)
        rows = []
        for record in annotated.get("records", []):
            features = record.get("network_features") or {}
            rows.append(self._vectorize(features))
        if not rows:
            rows = [self._vectorize({}) for _ in range(len(records))]
        return np.array(rows, dtype=float)

    def _vectorize(self, features: Dict[str, float]) -> List[float]:
        order = self.feature_order or sorted(features.keys())
        return [float(features.get(key, 0.0)) for key in order]

    def get_feature_names_out(self, input_features=None):
        order = self.feature_order or []
        return np.array([f"network_{name}" for name in order])


__all__ = ["NetworkFeatureTransformer"]


