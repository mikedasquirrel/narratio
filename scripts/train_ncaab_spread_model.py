#!/usr/bin/env python3
"""
Engineer features and train a production-ready NCAAB spread model.

The workflow:
1. Load merged scoreboard + odds records (closing lines, results, context)
2. Build rolling team form features with strict chronological ordering
3. Train a calibrated classifier (HistGradientBoosting) on pre-2024 seasons
4. Evaluate on 2024 holdout, simulate an edge-based betting strategy
5. Persist feature matrix, model artifact, and training report for deployment
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)
FEATURE_MATRIX_PATH = DATA_ROOT / "modeling_datasets" / "ncaab_feature_matrix.jsonl"
MODEL_ARTIFACT_PATH = MODEL_ROOT / "ncaab_spread_hgb.pkl"
TRAINING_REPORT_PATH = DATA_ROOT / "modeling_datasets" / "ncaab_model_training_report.json"
MERGED_DATA_PATH = DATA_ROOT / "modeling_datasets" / "ncaab_games_with_closing_odds.jsonl"
STAT_FEATURE_FIELDS: List[str] = []
BASE_ELO = 1500.0
ELO_K_FACTOR = 20.0
ELO_SEASON_REGRESSION = 0.75


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_datetime(value: str) -> datetime:
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(cleaned).astimezone(timezone.utc)


def american_to_prob(price: Optional[float]) -> Optional[float]:
    if price is None:
        return None
    if price == 0:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def bet_profit(price: float, outcome_win: bool) -> float:
    if price is None:
        return 0.0
    if outcome_win:
        if price < 0:
            return 100.0 / (-price)
        return price / 100.0
    return -1.0


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(mean(values)) if values else None


def normalize_rank(value: Optional[int]) -> int:
    return int(value) if value is not None else 400


def clamp_none(value: Optional[float]) -> float:
    return float(value) if value is not None and not math.isnan(value) else 0.0


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _ensure_team_elo(
    team: str,
    season_year: int,
    elo_ratings: Dict[str, float],
    last_season_seen: Dict[str, int],
) -> float:
    rating = elo_ratings.get(team, BASE_ELO)
    last = last_season_seen.get(team)
    if last is None:
        last_season_seen[team] = season_year
        elo_ratings[team] = rating
        return rating
    if season_year > last:
        rating = rating * ELO_SEASON_REGRESSION + BASE_ELO * (1 - ELO_SEASON_REGRESSION)
        elo_ratings[team] = rating
        last_season_seen[team] = season_year
    return rating


def _update_elo_ratings(
    elo_ratings: Dict[str, float],
    home_team: str,
    away_team: str,
    home_score: Optional[int],
    away_score: Optional[int],
    home_rating: float,
    away_rating: float,
) -> None:
    if home_score is None or away_score is None:
        return
    if home_score == away_score:
        result = 0.5
    else:
        result = 1.0 if home_score > away_score else 0.0
    expected_home = _elo_expected(home_rating, away_rating)
    margin = abs(home_score - away_score)
    margin_multiplier = math.log(max(margin, 1) + 1)
    adjustment = ELO_K_FACTOR * margin_multiplier * (result - expected_home)
    elo_ratings[home_team] = home_rating + adjustment
    elo_ratings[away_team] = away_rating - adjustment


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def load_merged_games() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with MERGED_DATA_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["event_dt"] = parse_datetime(payload["event_time"])
            records.append(payload)
    if not records:
        raise ValueError("Merged NCAAB dataset is empty: run build_ncaab_modeling_dataset.py first.")
    return sorted(records, key=lambda r: r["event_dt"])


def compute_recent_stats(history: List[Dict[str, object]], window: int) -> Dict[str, Optional[float]]:
    subset = history[-window:]
    if not subset:
        return {
            "margin_avg": None,
            "points_avg": None,
            "points_allowed_avg": None,
            "win_pct": None,
            "cover_pct": None,
        }
    margins = [entry["margin"] for entry in subset]
    points_for = [entry["points_for"] for entry in subset]
    points_against = [entry["points_against"] for entry in subset]
    wins = sum(1 for entry in subset if entry["win"])
    cover_results = [entry["cover"] for entry in subset if entry.get("cover") is not None]
    cover_pct = (
        sum(1 for result in cover_results if result) / len(cover_results) if cover_results else None
    )
    return {
        "margin_avg": safe_mean(margins),
        "points_avg": safe_mean(points_for),
        "points_allowed_avg": safe_mean(points_against),
        "win_pct": wins / len(subset),
        "cover_pct": cover_pct,
    }


def compute_streak(history: List[Dict[str, object]]) -> int:
    streak = 0
    for entry in reversed(history[-5:]):
        if entry["win"]:
            streak = streak + 1 if streak >= 0 else 1
        else:
            streak = streak - 1 if streak <= 0 else -1
    return streak


def compute_team_features(
    team_history: List[Dict[str, object]],
    season_stats: Dict[str, float],
    event_dt: datetime,
) -> Dict[str, Optional[float]]:
    last_game_time = team_history[-1]["event_dt"] if team_history else None
    rest_days = (
        (event_dt - last_game_time).total_seconds() / 86400.0 if last_game_time else None
    )

    stats_last3 = compute_recent_stats(team_history, 3)
    stats_last10 = compute_recent_stats(team_history, 10)

    return {
        "games_played": len(team_history),
        "rest_days": rest_days,
        "margin_avg_last3": stats_last3["margin_avg"],
        "margin_avg_last10": stats_last10["margin_avg"],
        "points_avg_last3": stats_last3["points_avg"],
        "points_avg_last10": stats_last10["points_avg"],
        "points_allowed_avg_last3": stats_last3["points_allowed_avg"],
        "points_allowed_avg_last10": stats_last10["points_allowed_avg"],
        "win_pct_last3": stats_last3["win_pct"],
        "win_pct_last10": stats_last10["win_pct"],
        "cover_pct_last3": stats_last3["cover_pct"],
        "cover_pct_last10": stats_last10["cover_pct"],
        "season_win_pct": season_stats.get("win_pct"),
        "season_conference_win_pct": season_stats.get("conference_win_pct"),
        "streak": compute_streak(team_history) if team_history else 0,
    }


def update_team_history(
    history: List[Dict[str, object]],
    event_dt: datetime,
    points_for: Optional[int],
    points_against: Optional[int],
    conference_game: bool,
    covered: Optional[bool],
) -> None:
    if points_for is None or points_against is None:
        return
    history.append(
        {
            "event_dt": event_dt,
            "points_for": points_for,
            "points_against": points_against,
            "margin": points_for - points_against,
            "win": points_for > points_against,
            "conference_game": conference_game,
            "cover": covered,
        }
    )


def update_season_stats(
    season_tracker: Dict[str, Dict[str, float]],
    team: str,
    season_year: int,
    won: Optional[bool],
    conference_game: bool,
) -> Dict[str, float]:
    key = (team, season_year)
    stats = season_tracker.setdefault(key, {})
    stats.setdefault("games", 0)
    stats.setdefault("wins", 0)
    stats.setdefault("conference_games", 0)
    stats.setdefault("conference_wins", 0)

    stats["games"] += 1
    if won:
        stats["wins"] += 1
    if conference_game:
        stats["conference_games"] += 1
        if won:
            stats["conference_wins"] += 1

    stats["win_pct"] = stats["wins"] / stats["games"] if stats["games"] else 0.0
    stats["conference_win_pct"] = (
        stats["conference_wins"] / stats["conference_games"] if stats["conference_games"] else 0.0
    )
    return stats


def get_season_snapshot(
    season_tracker: Dict[Tuple[str, int], Dict[str, float]],
    team: str,
    season_year: int,
) -> Dict[str, float]:
    stats = season_tracker.get((team, season_year), {})
    return {
        "win_pct": stats.get("win_pct", 0.0),
        "conference_win_pct": stats.get("conference_win_pct", 0.0),
    }


def build_feature_rows() -> List[Dict[str, object]]:
    merged_games = load_merged_games()
    team_histories: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    season_tracker: Dict[Tuple[str, int], Dict[str, float]] = {}
    feature_rows: List[Dict[str, object]] = []
    elo_ratings: Dict[str, float] = {}
    last_season_seen: Dict[str, int] = {}

    for record in merged_games:
        season_year = int(record.get("season_year") or record["event_dt"].year)
        home = record["home_team"]
        away = record["away_team"]

        home_stats = get_season_snapshot(season_tracker, home, season_year)
        away_stats = get_season_snapshot(season_tracker, away, season_year)

        home_features = compute_team_features(
            team_histories[home],
            home_stats,
            record["event_dt"],
        )
        away_features = compute_team_features(
            team_histories[away],
            away_stats,
            record["event_dt"],
        )

        home_elo = _ensure_team_elo(home, season_year, elo_ratings, last_season_seen)
        away_elo = _ensure_team_elo(away, season_year, elo_ratings, last_season_seen)

        target = record.get("spread_result")

        feature_row = {
            "event_time": record["event_time"],
            "season_year": season_year,
            "neutral_site": int(bool(record.get("neutral_site"))),
            "conference_game": int(bool(record.get("conference_game"))),
            "closing_spread_home": record.get("closing_spread_home"),
            "closing_spread_home_price": record.get("closing_spread_home_price"),
            "closing_spread_away_price": record.get("closing_spread_away_price"),
            "closing_moneyline_home_implied": record.get("closing_moneyline_home_implied"),
            "closing_total": record.get("closing_total"),
            "closing_total_price": record.get("closing_total_price"),
            "closing_total_under_price": record.get("total_under_price"),
            "rank_diff": normalize_rank(record.get("home_rank")) - normalize_rank(
                record.get("away_rank")
            ),
            "home_rank_value": normalize_rank(record.get("home_rank")),
            "away_rank_value": normalize_rank(record.get("away_rank")),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "target": target,
        }

        for key, value in home_features.items():
            feature_row[f"home_{key}"] = value
        for key, value in away_features.items():
            feature_row[f"away_{key}"] = value

        feature_row["rest_days_diff"] = (
            feature_row["home_rest_days"] - feature_row["away_rest_days"]
            if feature_row.get("home_rest_days") is not None
            and feature_row.get("away_rest_days") is not None
            else None
        )
        feature_row["win_pct_diff"] = (
            feature_row["home_season_win_pct"] - feature_row["away_season_win_pct"]
        )

        for stat_name in STAT_FEATURE_FIELDS:
            home_val = record.get(f"home_stat_{stat_name}")
            away_val = record.get(f"away_stat_{stat_name}")
            feature_row[f"home_stat_{stat_name}"] = home_val
            feature_row[f"away_stat_{stat_name}"] = away_val
            if home_val is not None and away_val is not None:
                feature_row[f"stat_diff_{stat_name}"] = home_val - away_val
            else:
                feature_row[f"stat_diff_{stat_name}"] = None

        feature_rows.append(feature_row)

        # Update histories after feature computation to avoid leakage
        home_cover_flag = None
        if target == 1.0:
            home_cover_flag = True
        elif target == 0.0:
            home_cover_flag = False
        away_cover_flag = None if home_cover_flag is None else not home_cover_flag

        update_team_history(
            team_histories[home],
            record["event_dt"],
            record.get("home_score"),
            record.get("away_score"),
            bool(record.get("conference_game")),
            home_cover_flag,
        )
        update_team_history(
            team_histories[away],
            record["event_dt"],
            record.get("away_score"),
            record.get("home_score"),
            bool(record.get("conference_game")),
            away_cover_flag,
        )

        home_won = record.get("home_margin")
        home_won = home_won > 0 if home_won is not None else None
        update_season_stats(
            season_tracker,
            home,
            season_year,
            home_won,
            bool(record.get("conference_game")),
        )
        update_season_stats(
            season_tracker,
            away,
            season_year,
            None if home_won is None else not home_won,
            bool(record.get("conference_game")),
        )

        _update_elo_ratings(
            elo_ratings,
            home,
            away,
            record.get("home_score"),
            record.get("away_score"),
            home_elo,
            away_elo,
        )

    if not feature_rows:
        raise ValueError("No feature rows generated; check input data integrity.")

    with FEATURE_MATRIX_PATH.open("w") as f:
        for row in feature_rows:
            f.write(json.dumps(row) + "\n")

    return feature_rows


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def prepare_dataframe(feature_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(feature_rows)
    df = df[df["target"].isin([0.0, 1.0])]  # drop pushes
    essential = ["closing_spread_home", "closing_spread_home_price", "closing_spread_away_price"]
    df = df.dropna(subset=essential)
    df = df.replace({None: np.nan}).fillna(0.0)
    return df


def train_model(df: pd.DataFrame):
    feature_cols = [
        "closing_spread_home",
        "closing_spread_home_price",
        "closing_spread_away_price",
        "closing_moneyline_home_implied",
        "closing_total",
        "closing_total_price",
        "closing_total_under_price",
        "neutral_site",
        "conference_game",
        "rank_diff",
        "home_games_played",
        "away_games_played",
        "home_rest_days",
        "away_rest_days",
        "rest_days_diff",
        "home_margin_avg_last3",
        "home_margin_avg_last10",
        "away_margin_avg_last3",
        "away_margin_avg_last10",
        "home_points_avg_last3",
        "home_points_avg_last10",
        "away_points_avg_last3",
        "away_points_avg_last10",
        "home_points_allowed_avg_last3",
        "away_points_allowed_avg_last3",
        "home_points_allowed_avg_last10",
        "away_points_allowed_avg_last10",
        "home_win_pct_last3",
        "away_win_pct_last3",
        "home_win_pct_last10",
        "away_win_pct_last10",
        "home_cover_pct_last3",
        "away_cover_pct_last3",
        "home_cover_pct_last10",
        "away_cover_pct_last10",
        "home_season_win_pct",
        "away_season_win_pct",
        "home_season_conference_win_pct",
        "away_season_conference_win_pct",
        "win_pct_diff",
        "home_streak",
        "away_streak",
        "home_elo",
        "away_elo",
        "elo_diff",
    ]
    for stat_name in STAT_FEATURE_FIELDS:
        feature_cols.extend(
            [
                f"home_stat_{stat_name}",
                f"away_stat_{stat_name}",
                f"stat_diff_{stat_name}",
            ]
        )
    feature_cols = [col for col in feature_cols if col in df.columns]

    train_df = df[df["season_year"] < 2024]
    test_df = df[df["season_year"] >= 2024]

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test split is empty; ensure dataset spans multiple seasons.")

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["target"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, feature_cols, test_df)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "training_metrics": metrics,
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    return model, metrics


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

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

    betting_report = simulate_betting(test_df, test_pred)
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = dict(zip(feature_cols, model.feature_importances_))
    elif isinstance(model, Pipeline) and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            feature_importances = dict(zip(feature_cols, clf.coef_[0].tolist()))

    return {
        "train_accuracy": accuracy_score(y_train, train_pred > 0.5),
        "train_auc": safe_auc(y_train, train_pred),
        "train_log_loss": safe_log_loss(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred > 0.5),
        "test_auc": safe_auc(y_test, test_pred),
        "test_log_loss": safe_log_loss(y_test, test_pred),
        "feature_importances": feature_importances,
        "betting_strategy": betting_report,
        "test_samples": int(len(y_test)),
    }


def simulate_betting(test_df: pd.DataFrame, probabilities: np.ndarray, threshold: float = 0.03) -> Dict[str, float]:
    total_profit = 0.0
    bets = 0
    wins = 0
    edges: List[float] = []

    for row, prob in zip(test_df.itertuples(index=False), probabilities):
        if row.target not in (0, 1):
            continue

        candidates = []
        if row.closing_spread_home_price not in (None, 0):
            implied_home = american_to_prob(row.closing_spread_home_price)
            if implied_home is not None:
                edge_home = prob - implied_home
                candidates.append(("home", edge_home, row.closing_spread_home_price, row.target == 1))
        if row.closing_spread_away_price not in (None, 0):
            implied_away = american_to_prob(row.closing_spread_away_price)
            if implied_away is not None:
                edge_away = (1 - prob) - implied_away
                candidates.append(("away", edge_away, row.closing_spread_away_price, row.target == 0))

        if not candidates:
            continue

        side = max(candidates, key=lambda x: x[1])
        if side[1] < threshold:
            continue

        edges.append(side[1])
        profit = bet_profit(side[2], side[3])
        total_profit += profit
        bets += 1
        if profit > 0:
            wins += 1

    roi = total_profit / bets if bets else 0.0
    hit_rate = wins / bets if bets else 0.0
    avg_edge = float(safe_mean(edges)) if edges else 0.0

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
    feature_rows = build_feature_rows()
    df = prepare_dataframe(feature_rows)
    model, metrics = train_model(df)
    TRAINING_REPORT_PATH.write_text(json.dumps(metrics, indent=2))
    test_auc_display = metrics["test_auc"]
    print(
        f"Trained NCAAB spread model on {len(df)} samples "
        f"(train={len(df[df['season_year'] < 2024])}, test={len(df[df['season_year'] >= 2024])})"
    )
    auc_str = f"{test_auc_display:.3f}" if test_auc_display is not None else "n/a"
    print(f"Holdout accuracy: {metrics['test_accuracy']:.3f} | AUC: {auc_str}")
    print(
        f"ROI (edge strategy): {metrics['betting_strategy']['roi']:.3f} "
        f"over {metrics['betting_strategy']['bets']} bets"
    )


if __name__ == "__main__":
    main()

