#!/usr/bin/env python3
"""
Train NBA moneyline model using canonical 1,885-feature narrative stack.
"""

from __future__ import annotations

import json
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
DATA_FILE = PROJECT_ROOT / "data" / "modeling_datasets" / "nba_feature_matrix.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "nba_canonical_moneyline.pkl"
REPORT_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "nba_canonical_model_report.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


META_COLUMNS = {
    "game_id",
    "season",
    "date",
    "team_abbreviation",
    "team_name",
    "opponent",
    "matchup",
    "actual_win",
    "team_points",
    "plus_minus",
    "narrative",
    "home_game",
    "closing_moneyline",
    "closing_moneyline_implied",
    "closing_spread",
    "closing_spread_price",
    "closing_total",
    "closing_total_price",
    "total_under_price",
    "moneyline_result",
    "spread_result",
    "total_result",
    "espn_game_id",
    "season_year",
}


def american_to_prob(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    price = float(price)
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
    if not DATA_FILE.exists():
        raise FileNotFoundError("Missing nba_feature_matrix.parquet. Run merge_nba_features_with_odds.py.")
    df = pd.read_parquet(DATA_FILE)
    df = df[df["moneyline_result"].isin([0.0, 1.0])]
    df["season_year"] = pd.to_datetime(df["date"]).dt.year
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return [col for col in numeric_cols if col not in META_COLUMNS]


def simulate_betting(test_df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    total_profit = 0.0
    bets = 0
    wins = 0
    edges: List[float] = []

    for row, prob in zip(test_df.itertuples(index=False), probs):
        implied = american_to_prob(row.closing_moneyline)
        if implied is None:
            continue
        edge = prob - implied
        if edge < threshold:
            continue
        outcome = bool(row.moneyline_result)
        profit = bet_profit(row.closing_moneyline, outcome)
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
    feat_cols = feature_columns(df)
    if not feat_cols:
        raise ValueError("No feature columns found.")

    train_df = df[df["season_year"] < 2024]
    test_df = df[df["season_year"] >= 2024]
    if train_df.empty or test_df.empty:
        raise ValueError("Training or test split is empty.")

    X_train = train_df[feat_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_train = train_df["moneyline_result"].astype(int)
    X_test = test_df[feat_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_test = test_df["moneyline_result"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, C=0.8, class_weight="balanced", solver="lbfgs")),
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

    joblib.dump({"model": model, "feature_cols": feat_cols, "metrics": metrics}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2))

    auc_str = f"{metrics['test_auc']:.3f}" if metrics["test_auc"] is not None else "n/a"
    print(
        f"Trained NBA canonical moneyline model on {len(df)} samples "
        f"(train={len(train_df)}, test={len(test_df)})."
    )
    print(
        f"Holdout accuracy={metrics['test_accuracy']:.3f} | "
        f"AUC={auc_str} | ROI={betting['roi']:.3f} over {betting['bets']} bets"
    )


if __name__ == "__main__":
    main()

