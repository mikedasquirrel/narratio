"""
Golf Betting Backtest
=====================

Production-ready validation pipeline for golf narrative models.

Steps:
1. Load enriched golf dataset (narratives + structured fields)
2. Engineer tournament-level competitive genome features
3. Train calibrated classifier to estimate win probability per player-entry
4. Evaluate discovery metrics (AUC, log loss, Brier, top-k tournament hit rate)
5. Run Kelly-sized betting simulation with synthetic market odds derived from prestige priors
6. Persist full report (model metrics + bankroll trajectory) for audit/comparison

Usage:
    python scripts/golf_betting_backtest.py --season 2024 --min_edge 0.02 --min_prob 0.03

Author: Narrative Optimization Framework
Date: November 2025
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from narrative_optimization.betting.kelly_criterion import KellyCriterion


# ==============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ==============================================================================

def load_golf_records(data_path: Path) -> List[Dict]:
    """Load golf dataset and flatten potential dict structures."""
    if not data_path.exists():
        raise FileNotFoundError(f"Golf dataset missing: {data_path}")

    with open(data_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw

    flattened: List[Dict] = []
    for value in raw.values():
        if isinstance(value, list):
            flattened.extend(value)
        elif isinstance(value, dict):
            flattened.append(value)
    return flattened


def build_feature_dataframe(records: List[Dict]) -> pd.DataFrame:
    """Construct rich feature frame from raw golf entries."""
    rows = []
    for rec in records:
        rounds = rec.get("rounds") or []
        rounds = [float(r) for r in rounds if r is not None]
        final_round = rounds[-1] if rounds else np.nan
        third_round = rounds[-2] if len(rounds) >= 2 else np.nan
        avg_round = np.mean(rounds) if rounds else np.nan
        best_round = np.min(rounds) if rounds else np.nan
        round_std = np.std(rounds) if len(rounds) > 1 else 0.0
        closing_surge = (third_round - final_round) if np.isfinite(final_round) and np.isfinite(third_round) else 0.0

        prestige = float(rec.get("player_prestige", 0.5) or 0.0)
        ranking = float(rec.get("world_ranking_before", 300) or 300)
        course_diff = float(rec.get("course_difficulty", 0.5) or 0.5)
        majors = float(rec.get("player_majors", 0) or 0)
        to_par = float(rec.get("to_par", 0.0) or 0.0)

        rows.append(
            {
                "player_tournament_id": rec.get("player_tournament_id"),
                "tournament_name": rec.get("tournament_name"),
                "course_name": rec.get("course_name"),
                "year": int(rec.get("year", 0)),
                "is_major": int(bool(rec.get("is_major", False))),
                "won_tournament": int(bool(rec.get("won_tournament", False))),
                "player_name": rec.get("player_name"),
                "world_ranking_before": ranking,
                "player_prestige": prestige,
                "player_majors": majors,
                "course_difficulty": course_diff,
                "to_par": to_par,
                "finish_position": float(rec.get("finish_position", np.nan)),
                "top_10_finish": int(bool(rec.get("top_10_finish", False))),
                "top_3_finish": int(bool(rec.get("top_3_finish", False))),
                "made_cut": int(bool(rec.get("made_cut", False))),
                "final_round": final_round,
                "third_round": third_round,
                "avg_round": avg_round,
                "best_round": best_round,
                "round_std": round_std,
                "closing_surge": closing_surge,
                "prestige_x_course": prestige * course_diff,
                "prestige_x_major": prestige * (1 + int(bool(rec.get("is_major")))),
                "ranking_inverse": 1.0 / (ranking + 1.0),
                "major_experience": majors * int(bool(rec.get("is_major"))),
                "narrative_length": len((rec.get("narrative") or "").split()),
            }
        )

    df = pd.DataFrame(rows)
    df["tournament_id"] = df["tournament_name"].fillna("Unknown") + "_" + df["year"].astype(str)
    return df


# ==============================================================================
# MODELING
# ==============================================================================

def train_probability_model(train_df: pd.DataFrame, feature_cols: List[str]):
    """Train calibrated gradient boosting classifier for win probability."""
    X_train = train_df[feature_cols]
    y_train = train_df["won_tournament"].values

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=400,
        learning_rate=0.08,
        l2_regularization=0.1,
        min_samples_leaf=25,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weight = neg / max(pos, 1)
    sample_weight = np.where(y_train == 1, class_weight, 1.0)

    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    return pipeline


def evaluate_predictions(df: pd.DataFrame, probs: np.ndarray) -> Dict:
    """Compute classification metrics on provided dataframe/probabilities."""
    y_true = df["won_tournament"].values
    metrics = {
        "accuracy": float(accuracy_score(y_true, probs >= 0.5)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs, eps=1e-12)),
        "brier": float(brier_score_loss(y_true, probs)),
    }
    return metrics


def tournament_hit_rates(df: pd.DataFrame, probs: np.ndarray, top_k: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, float]:
    """Measure how often the actual winner sits in the top-k predictions per tournament."""
    enriched = df.copy()
    enriched["prob"] = probs

    results = {}
    groups = enriched.groupby("tournament_id")
    total = len(groups)

    for k in top_k:
        hits = 0
        for _, group in groups:
            sorted_group = group.sort_values("prob", ascending=False).head(k)
            if sorted_group["won_tournament"].max() == 1:
                hits += 1
        results[f"top_{k}_hit_rate"] = hits / total if total else 0.0

    # Probability assigned to true winner (calibration insight)
    winner_probs = (
        enriched.loc[enriched["won_tournament"] == 1, "prob"].mean()
        if enriched["won_tournament"].sum() > 0
        else 0.0
    )
    results["avg_winner_prob"] = float(winner_probs)
    return results


# ==============================================================================
# MARKET / BETTING SIMULATION
# ==============================================================================

@dataclass
class BettingResult:
    roi: float
    cagr: float
    hit_rate: float
    n_bets: int
    total_wagered: float
    final_bankroll: float
    max_drawdown: float
    avg_edge: float
    avg_odds: float


def prestige_to_market_probs(prestige: np.ndarray, vig: float = 1.08) -> np.ndarray:
    """Convert prestige weights to market-style probabilities with vig."""
    prestige = np.clip(prestige, 1e-3, None)
    normalized = prestige / prestige.sum()
    market = normalized * vig
    return np.clip(market, 1e-4, 0.95)


def decimal_to_american(decimal_odds: np.ndarray) -> np.ndarray:
    """Vectorized decimal → American odds conversion."""
    american = np.where(
        decimal_odds >= 2.0,
        (decimal_odds - 1.0) * 100,
        -100 / (decimal_odds - 1.0),
    )
    return american


def simulate_betting(
    df: pd.DataFrame,
    probs: np.ndarray,
    min_edge: float = 0.02,
    min_prob: float = 0.02,
    bankroll: float = 10_000.0,
) -> BettingResult:
    """Run Kelly-sized betting simulation over tournaments."""
    enriched = df.copy().reset_index(drop=True)
    enriched["model_prob"] = probs

    kelly = KellyCriterion(default_fraction=0.5, max_bet_pct=0.01, min_edge=min_edge)

    current_bankroll = bankroll
    peak_bankroll = bankroll
    total_wagered = 0.0
    wins = 0
    bets = 0
    edges = []
    odds_list = []

    grouped = enriched.sort_values(["year", "tournament_id"]).groupby("tournament_id")

    for _, group in grouped:
        prestige = group["player_prestige"].to_numpy(dtype=float)
        market_prob = prestige_to_market_probs(prestige)
        decimal_odds = 1.0 / market_prob
        american_odds = decimal_to_american(decimal_odds)

        for idx, (_, row) in enumerate(group.iterrows()):
            model_prob = float(row["model_prob"])
            if model_prob < min_prob:
                continue

            edge = model_prob - float(market_prob[idx])
            if edge < min_edge:
                continue

            bet = kelly.calculate_bet(
                game_id=row["player_tournament_id"],
                bet_type="outright",
                side=row["player_name"] or "player",
                american_odds=float(american_odds[idx]),
                win_probability=model_prob,
                bankroll=current_bankroll,
            )

            stake = current_bankroll * bet.recommended_fraction
            if stake <= 0:
                continue

            bets += 1
            edges.append(edge)
            odds_list.append(decimal_odds[idx])
            total_wagered += stake

            if row["won_tournament"]:
                profit = stake * (decimal_odds[idx] - 1.0)
                current_bankroll += profit
                wins += 1
            else:
                current_bankroll -= stake

            peak_bankroll = max(peak_bankroll, current_bankroll)
            drawdown = (peak_bankroll - current_bankroll) / peak_bankroll

    roi = (current_bankroll - bankroll) / total_wagered if total_wagered else 0.0
    years = max(1, len(enriched["year"].unique()))
    cagr = (current_bankroll / bankroll) ** (1 / years) - 1 if total_wagered else 0.0
    hit_rate = wins / bets if bets else 0.0
    max_drawdown = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll else 0.0

    return BettingResult(
        roi=float(roi),
        cagr=float(cagr),
        hit_rate=float(hit_rate),
        n_bets=bets,
        total_wagered=float(total_wagered),
        final_bankroll=float(current_bankroll),
        max_drawdown=float(max_drawdown),
        avg_edge=float(np.mean(edges) if edges else 0.0),
        avg_odds=float(np.mean(odds_list) if odds_list else 0.0),
    )


# ==============================================================================
# CLI / EXECUTION
# ==============================================================================

def run_backtest(args: argparse.Namespace):
    data_path = Path("data/domains/golf_with_narratives.json")
    records = load_golf_records(data_path)
    df = build_feature_dataframe(records)

    feature_cols = [
        "world_ranking_before",
        "player_prestige",
        "player_majors",
        "course_difficulty",
        "to_par",
        "finish_position",
        "top_10_finish",
        "top_3_finish",
        "made_cut",
        "final_round",
        "third_round",
        "avg_round",
        "best_round",
        "round_std",
        "closing_surge",
        "prestige_x_course",
        "prestige_x_major",
        "ranking_inverse",
        "major_experience",
        "is_major",
        "narrative_length",
    ]

    season = args.season
    train_df = df[df["year"] < season]
    test_df = df[df["year"] == season]

    if train_df.empty or test_df.empty:
        raise ValueError(f"Insufficient data for season {season}. Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    model = train_probability_model(train_df, feature_cols)
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    metrics = evaluate_predictions(test_df, test_probs)
    tournament_metrics = tournament_hit_rates(test_df, test_probs)
    betting_result = simulate_betting(
        test_df,
        test_probs,
        min_edge=args.min_edge,
        min_prob=args.min_prob,
        bankroll=args.bankroll,
    )

    summary = {
        "season": season,
        "n_records_train": int(len(train_df)),
        "n_records_test": int(len(test_df)),
        "class_balance_test": float(test_df["won_tournament"].mean()),
        "metrics": metrics,
        "tournament_metrics": tournament_metrics,
        "betting": betting_result.__dict__,
        "parameters": {
            "min_edge": args.min_edge,
            "min_prob": args.min_prob,
            "bankroll": args.bankroll,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    print("=" * 80)
    print(f"GOLF BETTING BACKTEST - SEASON {season}")
    print("=" * 80)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    print(f"Positive rate (test): {test_df['won_tournament'].mean():.3%}")
    print("\nModel Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("\nTournament Metrics:")
    for k, v in tournament_metrics.items():
        print(f"  {k:<16}: {v:.4f}")
    print("\nBetting Simulation:")
    print(f"  Bets placed      : {betting_result.n_bets}")
    print(f"  Hit rate         : {betting_result.hit_rate:.2%}")
    print(f"  ROI              : {betting_result.roi:.2%}")
    print(f"  CAGR             : {betting_result.cagr:.2%}")
    print(f"  Avg edge         : {betting_result.avg_edge:.2%}")
    print(f"  Avg decimal odds : {betting_result.avg_odds:.2f}")
    print(f"  Final bankroll   : ${betting_result.final_bankroll:,.2f}")

    results_dir = Path("narrative_optimization/results/betting/golf")
    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / f"golf_backtest_{season}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {outfile}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Golf betting validation pipeline.")
    parser.add_argument("--season", type=int, default=2024, help="Season/year to evaluate (default: 2024)")
    parser.add_argument("--min_edge", type=float, default=0.02, help="Minimum edge threshold for placing bets")
    parser.add_argument("--min_prob", type=float, default=0.02, help="Minimum model probability for consideration")
    parser.add_argument("--bankroll", type=float, default=10_000.0, help="Initial bankroll for simulation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args)

