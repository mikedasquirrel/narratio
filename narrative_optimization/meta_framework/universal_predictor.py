"""
Universal Predictor
===================

Meta-model that learns f(π, θ, λ, context) → effect size.
Used to validate framework universality and forecast new domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class MetaExample:
    domain: str
    pi: float
    theta: float
    lambda_: float
    context_strength: float
    nominative_richness: float
    effect_size: float


class UniversalPredictor:
    def __init__(self, n_estimators: int = 300, random_state: int = 7):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, max_depth=8
        )
        self.scaler = StandardScaler()
        self.feature_order = [
            "pi",
            "theta",
            "lambda",
            "context_strength",
            "nominative_richness",
        ]
        self.trained = False
        self.r2_ = None

    def fit(self, examples: Iterable[MetaExample]):
        data = list(examples)
        if not data:
            raise ValueError("No meta examples supplied.")
        X = np.array([self._vectorize(example) for example in data])
        y = np.array([example.effect_size for example in data])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        preds = self.model.predict(X_scaled)
        self.r2_ = float(r2_score(y, preds))
        self.trained = True

    def predict(self, meta: MetaExample) -> float:
        if not self.trained:
            raise RuntimeError("Call fit() before predict().")
        X = self._vectorize(meta)
        X_scaled = self.scaler.transform([X])
        return float(self.model.predict(X_scaled)[0])

    def save(self, path: str) -> None:
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")
        payload = {
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "feature_order": self.feature_order,
            "model_params": self.model.get_params(),
            "r2": self.r2_,
        }
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _vectorize(self, example: MetaExample) -> List[float]:
        return [
            example.pi,
            example.theta,
            example.lambda_,
            example.context_strength,
            example.nominative_richness,
        ]

    @classmethod
    def from_result_files(cls, files: List[str]) -> "UniversalPredictor":
        examples: List[MetaExample] = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for domain_entry in data.get("domains", []):
                examples.append(
                    MetaExample(
                        domain=domain_entry["name"],
                        pi=float(domain_entry.get("pi")),
                        theta=float(domain_entry.get("theta")),
                        lambda_=float(domain_entry.get("lambda")),
                        context_strength=float(domain_entry.get("context_strength", 0.5)),
                        nominative_richness=float(
                            domain_entry.get("nominative_richness", 0.5)
                        ),
                        effect_size=float(domain_entry.get("effect_size")),
                    )
                )
        predictor = cls()
        predictor.fit(examples)
        return predictor


__all__ = ["UniversalPredictor", "MetaExample"]


