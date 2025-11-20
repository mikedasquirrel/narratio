#!/usr/bin/env python3
"""
Train NHL Betting Models with Narrative Features

Loads the merged dataset (structured + narrative),
fits a scaler + trio of models (logistic, gradient boosting, random forest),
and saves them under narrative_optimization/domains/nhl/models/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import joblib


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "nhl_narrative_betting_dataset.parquet"
MODELS_DIR = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "models"
SUMMARY_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "narrative_model_summary.json"
HOLDOUT_SEASONS = None  # use most recent season if None
PREDICTIONS_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "narrative_holdout_predictions.parquet"
STRATEGY_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "narrative_betting_strategy.json"
STRATEGY_TRADES_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "narrative_strategy_trades.parquet"
EDGE_THRESHOLD = 0.03  # Minimum edge over market implied probability to place bet


def load_dataset():
    df = pd.read_parquet(DATA_PATH)
    drop_cols = ["game_id", "season", "date", "home_team", "away_team", "home_won"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df, feature_cols


def split_by_season(df):
    seasons = sorted(df["season"].unique())
    if HOLDOUT_SEASONS:
        holdout = HOLDOUT_SEASONS
    else:
        holdout = [seasons[-1]]
    train_df = df[~df["season"].isin(holdout)]
    test_df = df[df["season"].isin(holdout)]
    if train_df.empty or test_df.empty:
        raise ValueError("Season split produced empty train/test sets")
    return train_df, test_df, holdout


def prepare_xy(df, feature_cols):
    X = df[feature_cols].values
    y = df["home_won"].astype(int).values
    return X, y


def build_models():
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler(with_mean=False)
    stable_logistic = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=8000,
        class_weight="balanced",
        warm_start=True,
        n_jobs=-1,
        tol=1e-4,
        random_state=42,
    )
    logistic_pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("clf", stable_logistic)
    ])
    return {
        "narrative_logistic": logistic_pipeline,
        "narrative_gradient": GradientBoostingClassifier(random_state=42),
        "narrative_forest": RandomForestClassifier(n_estimators=400, random_state=42)
    }


def evaluate_market_baseline(test_df):
    if "odds_implied_prob_home" not in test_df.columns:
        return None
    probs = test_df["odds_implied_prob_home"].astype(float)
    probs = probs.clip(1e-6, 1 - 1e-6)
    preds = (probs >= 0.5).astype(int)
    y_true = test_df["home_won"].astype(int).values
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs))
    }


def _american_payout(odds):
    if odds is None or np.isnan(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    elif odds < 0:
        return 100.0 / abs(odds)
    return None


def evaluate_strategy(model_name, test_df, probas, edge_threshold=EDGE_THRESHOLD):
    bets = []
    total_profit = 0.0
    running_peak = 0.0
    max_drawdown = 0.0
    wins = 0
    stake = 1.0

    for idx, prob in zip(test_df.index, probas):
        row = test_df.loc[idx]
        implied_home = row.get("odds_implied_prob_home")
        implied_away = row.get("odds_implied_prob_away")
        if pd.isna(implied_home) or pd.isna(implied_away):
            continue

        home_edge = prob - implied_home
        away_edge = (1 - prob) - implied_away

        candidates = [
            ("home", home_edge, row.get("odds_moneyline_home"), bool(row.get("home_won"))),
            ("away", away_edge, row.get("odds_moneyline_away"), not bool(row.get("home_won")))
        ]
        bet_side, edge, odds, win = max(candidates, key=lambda x: x[1])

        if edge < edge_threshold or odds is None or pd.isna(odds):
            continue

        payout = _american_payout(odds)
        if payout is None:
            continue

        profit = payout * stake if win else -stake
        total_profit += profit
        running_peak = max(running_peak, total_profit)
        drawdown = running_peak - total_profit
        max_drawdown = max(max_drawdown, drawdown)
        if win:
            wins += 1

        bets.append({
            "model": model_name,
            "game_id": row.get("game_id"),
            "season": row.get("season"),
            "date": row.get("date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "bet_side": bet_side,
            "edge": float(edge),
            "odds": float(odds),
            "payout_units": payout,
            "profit_units": profit,
            "cumulative_profit": total_profit
        })

    if not bets:
        return {
            "bets": 0,
            "win_rate": None,
            "roi": None,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": None,
            "edge_threshold": edge_threshold
        }, bets

    bets_count = len(bets)
    win_rate = wins / bets_count
    roi = total_profit / bets_count
    avg_edge = float(np.mean([b["edge"] for b in bets]))

    return {
        "bets": bets_count,
        "win_rate": win_rate,
        "roi": roi,
        "total_profit": total_profit,
        "max_drawdown": max_drawdown,
        "avg_edge": avg_edge,
        "edge_threshold": edge_threshold
    }, bets


def top_logistic_features(model, feature_cols, top_n=20):
    if not isinstance(model, Pipeline):
        return []
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return []
    coefs = clf.coef_[0]
    indices = np.argsort(np.abs(coefs))[-top_n:][::-1]
    return [
        {"feature": feature_cols[i], "coefficient": float(coefs[i])}
        for i in indices
    ]


def train_models():
    df, feature_cols = load_dataset()
    train_df, test_df, holdout = split_by_season(df)
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    MODELS_DIR.mkdir(exist_ok=True)
    summary = {
        "feature_count": len(feature_cols),
        "train_seasons": sorted(train_df["season"].unique().tolist()),
        "holdout_seasons": holdout,
        "train_samples": int(len(train_df)),
        "holdout_samples": int(len(test_df)),
        "models": {},
        "edge_threshold": EDGE_THRESHOLD
    }

    summary["baseline_market"] = evaluate_market_baseline(test_df)

    predictions_cols = [
        "game_id", "season", "date", "home_team", "away_team", "home_won",
        "odds_moneyline_home", "odds_moneyline_away",
        "odds_implied_prob_home", "odds_implied_prob_away"
    ]
    predictions_cols = [c for c in predictions_cols if c in test_df.columns]
    predictions_output = test_df[predictions_cols].copy()

    all_strategy_rows = []

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)[:, 1]
        else:
            decision = model.decision_function(X_test)
            probas = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)

        model_summary = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probas)),
            "brier": float(brier_score_loss(y_test, probas)),
            "log_loss": float(log_loss(y_test, probas))
        }

        strategy_stats, strategy_rows = evaluate_strategy(name, test_df, probas)
        model_summary["strategy"] = strategy_stats
        all_strategy_rows.extend(strategy_rows)

        if name == "narrative_logistic":
            model_summary["top_features"] = top_logistic_features(model, feature_cols)

        summary["models"][name] = model_summary
        predictions_output[f"{name}_proba"] = probas
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    predictions_output.to_parquet(PREDICTIONS_PATH, index=False)
    summary["predictions_path"] = str(PREDICTIONS_PATH)
    summary["feature_columns"] = feature_cols

    if all_strategy_rows:
        strategy_df = pd.DataFrame(all_strategy_rows)
        strategy_df.to_parquet(STRATEGY_TRADES_PATH, index=False)
        with open(STRATEGY_PATH, "w") as f:
            json.dump(summary["models"], f, indent=2)
        summary["strategy_trades_path"] = str(STRATEGY_TRADES_PATH)
        summary["strategy_summary_path"] = str(STRATEGY_PATH)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Trained models saved to", MODELS_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train_models()

