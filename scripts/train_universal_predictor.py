#!/usr/bin/env python3
"""
Train Universal Predictor
=========================

Fits the cross-domain meta model on the dataset produced by build_meta_dataset.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from narrative_optimization.meta_framework.universal_predictor import (
    MetaExample,
    UniversalPredictor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "narrative_optimization" / "results" / "meta_framework" / "meta_dataset.json"
MODEL_PATH = PROJECT_ROOT / "narrative_optimization" / "results" / "meta_framework" / "universal_meta_model.json"


def _load_examples(dataset_path: Path) -> List[MetaExample]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Meta dataset not found: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("domains") or []
    if not entries:
        raise ValueError("Meta dataset contains no domain entries.")
    examples = []
    for entry in entries:
        examples.append(
            MetaExample(
                domain=entry["name"],
                pi=float(entry.get("pi", 0.5)),
                theta=float(entry.get("theta", 0.0)),
                lambda_=float(entry.get("lambda", 0.0)),
                context_strength=float(entry.get("context_strength", 0.0)),
                nominative_richness=float(entry.get("nominative_richness", 0.0)),
                effect_size=float(entry.get("effect_size", 0.0)),
            )
        )
    return examples


def train_universal_predictor(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(dataset_path)

    predictor = UniversalPredictor()
    predictor.fit(examples)
    predictor.save(model_path)

    predictions = [
        {"domain": ex.domain, "effect_estimate": predictor.predict(ex)} for ex in examples
    ]

    result = {
        "n_domains": len(examples),
        "r2": predictor.r2_,
        "predictions": predictions,
        "model_path": str(model_path),
    }
    print(f"✓ Trained universal predictor on {len(examples)} domains (R²={predictor.r2_:.3f})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Universal Predictor meta-model.")
    parser.add_argument("--dataset", type=str, help="Path to meta dataset JSON.")
    parser.add_argument("--model", type=str, help="Output path for trained model.")
    args = parser.parse_args()

    train_universal_predictor(
        dataset_path=Path(args.dataset) if args.dataset else DATASET_PATH,
        model_path=Path(args.model) if args.model else MODEL_PATH,
    )

