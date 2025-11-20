#!/usr/bin/env python3
"""
Train NBA moneyline model using engineered scoreboard features.
"""

from __future__ import annotations

import json
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
DATA_FILE = PROJECT_ROOT / "data" / "modeling_datasets" / "nba_engineered_features.jsonl"
MODEL_PATH = PROJECT_ROOT / "models" / "nba_engineered_moneyline.pkl"
REPORT_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "nba_engineered_model_report.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


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
        raise FileNotFoundError("Run build_nba_engineered_features.py first.")
    rows: List[Dict[str, object]] = []
    with DATA_FILE.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["target"] = df["moneyline_result"].astype(float)
    df = df[df["target"].isin([0.0, 1.0])]
    df["season_year"] = pd.to_datetime(df["event_time"]).dt.year
    if "closing_moneyline_home_implied" not in df.columns:
        df["closing_moneyline_home_implied"] = df["closing_moneyline_home"].apply(american_to_prob)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    base = [
        "closing_moneyline_home",
        "closing_moneyline_home_implied",
        "closing_spread_home",
        "closing_spread_home_price",
        "closing_total",
        "closing_total_price",
        "neutral_site",
        "home_games_played",
        "away_games_played",
        "home_rest_days",
        "away_rest_days",
        "home_avg_margin_last5",
        "away_avg_margin_last5",
        "home_win_pct_last5",
        "away_win_pct_last5",
        "home_win_pct_last10",
        "away_win_pct_last10",
        "home_points_avg_last5",
        "away_points_avg_last5",
        "home_points_allowed_avg_last5",
        "away_points_allowed_avg_last5",
        "home_streak",
        "away_streak",
    ]
    return [col for col in base if col in df.columns]


def simulate_betting(test_df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    total_profit = 0.0
    bets = 0
    wins = 0
    edges: List[float] = []

    for row, prob in zip(test_df.itertuples(index=False), probs):
        prices = {
            "home": row.closing_moneyline_home,
            "away": row.closing_moneyline_away,
        }
        implied_home = american_to_prob(prices["home"])
        implied_away = american_to_prob(prices["away"])
        candidates = []
        if implied_home is not None:
            candidates.append(("home", prob - implied_home, prices["home"], bool(row.target)))
        if implied_away is not None:
            candidates.append(("away", (1 - prob) - implied_away, prices["away"], not bool(row.target)))
        if not candidates:
            continue
        side, edge, price, outcome = max(candidates, key=lambda x: x[1])
        if edge < threshold:
            continue
        profit = bet_profit(price, outcome)
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
        raise ValueError("No feature columns found for model training.")

    train_df = df[df["season_year"] < 2024]
    test_df = df[df["season_year"] >= 2024]
    if train_df.empty or test_df.empty:
        raise ValueError("Training/test split is empty. Need more seasons.")

    X_train = train_df[feat_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_train = train_df["target"].astype(int)
    X_test = test_df[feat_cols].replace({None: np.nan}).fillna(0.0).astype(float)
    y_test = test_df["target"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.2, class_weight="balanced", solver="lbfgs")),
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
        f"Trained NBA engineered moneyline model on {len(df)} samples "
        f"(train={len(train_df)}, test={len(test_df)})."
    )
    print(
        f"Holdout accuracy={metrics['test_accuracy']:.3f} | "
        f"AUC={auc_str} | ROI={betting['roi']:.3f} over {betting['bets']} bets"
    )


if __name__ == "__main__":
    main()

