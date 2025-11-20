#!/usr/bin/env python3
"""
Train a calibrated NHL moneyline model using the canonical 79-feature narrative stack.

Data source: data/modeling_datasets/nhl_feature_matrix.parquet
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "nhl_feature_matrix.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "nhl_moneyline_logistic.pkl"
REPORT_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "nhl_model_training_report.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def american_to_prob(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return -price / (-price + 100.0)
    return None


def bet_profit(price: float, outcome_win: bool) -> float:
    if price < 0:
        win_return = 100.0 / (-price)
    else:
        win_return = price / 100.0
    return win_return if outcome_win else -1.0


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing NHL feature matrix. Run merge_nhl_features_with_odds.py.")
    df = pd.read_parquet(DATA_PATH)
    df["target"] = df["home_won"].astype(int)
    df["season_year"] = pd.to_datetime(df["date"]).dt.year
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    feat_cols = [c for c in df.columns if c.startswith("perf_") or c.startswith("nom_")]
    if not feat_cols:
        raise ValueError("No feature columns found (expected perf_/nom_).")
    return feat_cols


def simulate_betting(test_df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.03) -> Dict[str, float]:
    total_profit = 0.0
    bets = 0
    wins = 0
    edges: List[float] = []

    for row, prob in zip(test_df.itertuples(index=False), probs):
        implied = american_to_prob(row.closing_moneyline_home)
        if implied is None:
            continue
        edge = prob - implied
        if edge < threshold:
            continue
        result = bool(row.target)
        profit = bet_profit(row.closing_moneyline_home, result)
        total_profit += profit
        bets += 1
        if profit > 0:
            wins += 1
        edges.append(edge)

    roi = total_profit / bets if bets else 0.0
    hit_rate = wins / bets if bets else 0.0
    avg_edge = float(np.mean(edges)) if edges else 0.0
    return {
        "bets": bets,
        "wins": wins,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_edge": avg_edge,
        "threshold": threshold,
        "total_profit_units": total_profit,
    }


def main() -> None:
    df = load_dataset()
    feature_cols = select_features(df)

    train_df = df[df["season_year"] < 2024]
    test_df = df[df["season_year"] >= 2024]
    if train_df.empty or test_df.empty:
        raise ValueError("Training/Test split is empty; not enough seasons.")

    X_train = train_df[feature_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_test = test_df["target"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=1.5, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    def safe_auc(y_true, y_prob):
        try:
            return roc_auc_score(y_true, y_prob)
        except ValueError:
            return None

    def safe_log_loss(y_true, y_prob):
        try:
            return log_loss(y_true, y_prob, eps=1e-9)
        except ValueError:
            return None

    betting = simulate_betting(test_df, test_probs)

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_probs > 0.5),
        "train_auc": safe_auc(y_train, train_probs),
        "train_log_loss": safe_log_loss(y_train, train_probs),
        "test_accuracy": accuracy_score(y_test, test_probs > 0.5),
        "test_auc": safe_auc(y_test, test_probs),
        "test_log_loss": safe_log_loss(y_test, test_probs),
        "betting_strategy": betting,
        "test_samples": int(len(y_test)),
    }

    joblib.dump({"model": model, "feature_cols": feature_cols, "metrics": metrics}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2))

    print(
        f"Trained NHL moneyline model on {len(df)} games "
        f"(train={len(train_df)}, test={len(test_df)})."
    )
    auc_str = f"{metrics['test_auc']:.3f}" if metrics["test_auc"] is not None else "n/a"
    print(
        f"Holdout accuracy={metrics['test_accuracy']:.3f} | "
        f"AUC={auc_str} | "
        f"ROI={metrics['betting_strategy']['roi']:.3f} over "
        f"{metrics['betting_strategy']['bets']} bets"
    )


if __name__ == "__main__":
    main()

