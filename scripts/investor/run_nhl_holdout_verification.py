#!/usr/bin/env python3
"""
NHL Holdout Verification - CLI version of the investor notebook.

Loads the canonical feature matrix, fits a simple logistic regression model
with a temporal split, and evaluates win rate / ROI at key probability
thresholds. Results are written to docs/investor/verification/nhl_holdout_metrics.json
for inclusion in investor materials.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "modeling_datasets"
FEATURE_MATRIX = DATA_DIR / "nhl_feature_matrix.parquet"
SUMMARY_JSON = DATA_DIR / "nhl_feature_matrix_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "investor" / "verification" / "nhl_holdout_metrics.json"


def load_feature_matrix() -> pd.DataFrame:
    if not FEATURE_MATRIX.exists():
        raise FileNotFoundError(f"Feature matrix not found: {FEATURE_MATRIX}")

    df = pd.read_parquet(FEATURE_MATRIX)
    if "date" not in df.columns or "moneyline_result" not in df.columns:
        raise ValueError("Feature matrix missing required columns ('date', 'moneyline_result').")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("perf_", "nom_", "narr_", "ctx_", "odds_")
    feature_cols = [c for c in df.columns if c.startswith(prefixes)]
    if not feature_cols:
        raise ValueError("No feature columns found with expected prefixes.")
    return feature_cols


def fit_holdout(df: pd.DataFrame, feature_cols: List[str], cutoff: str = "2024-09-01") -> Dict:
    cutoff_date = pd.Timestamp(cutoff)
    train_mask = df["date"] < cutoff_date
    test_mask = ~train_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Temporal split produced empty train/test partitions.")

    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values
    y_train = df.loc[train_mask, "moneyline_result"].values
    y_test = df.loc[test_mask, "moneyline_result"].values

    scaler = StandardScaler()
    dX_train = scaler.fit_transform(X_train)
    dX_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(dX_train, y_train)
    probs = model.predict_proba(dX_test)[:, 1]

    thresholds = []
    for threshold in (0.55, 0.60, 0.65, 0.70):
        mask = probs >= threshold
        bets = int(mask.sum())
        if bets == 0:
            win_rate = None
            roi = None
        else:
            wins = int(y_test[mask].sum())
            losses = bets - wins
            win_rate = wins / bets
            roi = (wins - losses) / bets

        thresholds.append(
            {
                "threshold": threshold,
                "bets": bets,
                "win_rate": win_rate,
                "roi": roi,
            }
        )

    return {
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "train_games": int(train_mask.sum()),
        "test_games": int(test_mask.sum()),
        "feature_count": len(feature_cols),
        "probabilities": probs.tolist(),  # optional for downstream audits
        "outcomes": y_test.tolist(),
        "thresholds": thresholds,
    }


def load_feature_summary() -> Dict:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text())
    return {}


def main() -> None:
    print(f"Using feature matrix: {FEATURE_MATRIX}")
    print(f"Writing metrics to: {OUTPUT_PATH}")

    df = load_feature_matrix()
    feature_cols = pick_feature_columns(df)
    holdout_results = fit_holdout(df, feature_cols)
    feature_summary = load_feature_summary()

    metrics = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cutoff_date": holdout_results["cutoff_date"],
        "train_games": holdout_results["train_games"],
        "test_games": holdout_results["test_games"],
        "feature_count": holdout_results["feature_count"],
        "thresholds": holdout_results["thresholds"],
        "source_feature_matrix": str(FEATURE_MATRIX),
        "feature_summary": {
            "total_features": feature_summary.get("total_features"),
            "breakdown": feature_summary.get("feature_breakdown"),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


