#!/usr/bin/env python3
"""
Score upcoming NHL games with the trained narrative models.

Workflow:
1. Load the latest upcoming games file (generated from the Absolute Max snapshot)
2. Build narratives + derived context/odds features
3. Recreate the feature matrix expected by the trained models
4. Load the serialized models and compute win probabilities + betting edges
5. Emit recommendations for bets with configurable edge thresholds
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

# Prevent TensorFlow / tokenizer mutex deadlocks before any heavy imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "narrative_optimization"))

from domain_registry import build_nhl_narrative
from src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline


DEFAULT_UPCOMING_PATH = PROJECT_ROOT / "data" / "domains" / "nhl_upcoming_latest.json"
TRAIN_PARQUET = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "nhl_narrative_betting_dataset.parquet"
SUMMARY_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "narrative_model_summary.json"
METADATA_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "nhl_narrative_betting_metadata.json"
FEATURES_METADATA_PATH = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "nhl_features_metadata.json"
CACHE_DIR = PROJECT_ROOT / "narrative_optimization" / "cache" / "features"
MODELS_DIR = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl" / "models"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "analysis" / "nhl_upcoming_predictions.json"

EDGE_THRESHOLD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score upcoming NHL games using trained narrative models.")
    parser.add_argument(
        "--upcoming-path",
        type=Path,
        default=DEFAULT_UPCOMING_PATH,
        help="Path to the upcoming games JSON file.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.01,
        help="Minimum edge (probability difference) to surface a betting recommendation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON file for the full prediction results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["narrative_logistic", "narrative_gradient", "narrative_forest"],
        help="Model names to load from the NHL models directory.",
    )
    return parser.parse_args()


def american_to_prob(odds: float) -> float:
    if odds is None:
        return np.nan
    if isinstance(odds, str) and odds.lower() == "even":
        odds = 100
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def enrich_game_context(game: Dict) -> Dict:
    """Compute temporal context + implied probs for a single game entry."""
    def win_pct(record):
        wins = record.get("wins", 0)
        losses = record.get("losses", 0)
        total = wins + losses
        return wins / total if total else 0.5

    home_record = game.get("home_record", {})
    away_record = game.get("away_record", {})

    ctx = {
        "home_win_pct": win_pct(home_record),
        "away_win_pct": win_pct(away_record),
        "home_wins": home_record.get("wins", 0),
        "home_losses": home_record.get("losses", 0),
        "away_wins": away_record.get("wins", 0),
        "away_losses": away_record.get("losses", 0),
        "home_l10_wins": home_record.get("wins", 0),  # proxy
        "away_l10_wins": away_record.get("wins", 0),
        "home_rest_days": 2,
        "away_rest_days": 2,
        "home_back_to_back": False,
        "away_back_to_back": False,
        "rest_advantage": 0,
        "record_differential": win_pct(home_record) - win_pct(away_record),
        "form_differential": win_pct(home_record) - win_pct(away_record)
    }
    game["temporal_context"] = ctx

    odds = game.setdefault("betting_odds", {})
    odds["implied_prob_home"] = american_to_prob(odds.get("moneyline_home"))
    odds["implied_prob_away"] = american_to_prob(odds.get("moneyline_away"))
    odds["moneyline_home"] = odds.get("moneyline_home")
    odds["moneyline_away"] = odds.get("moneyline_away")
    return game


def load_upcoming_games(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Upcoming games file not found: {path}")
    with path.open("r") as f:
        games = json.load(f)
    return [enrich_game_context(game) for game in games]


def get_structured_baseline(num_games: int) -> pd.DataFrame:
    df = pd.read_parquet(TRAIN_PARQUET)
    perf_cols = [c for c in df.columns if c.startswith("perf_")]
    nom_cols = [c for c in df.columns if c.startswith("nom_")]
    structured_cols = perf_cols + nom_cols
    means = df[structured_cols].mean()
    return pd.DataFrame([means.values] * num_games, columns=structured_cols)


def expand_odds_and_context(games_df: pd.DataFrame) -> pd.DataFrame:
    odds_records = []
    context_records = []
    for game in games_df.to_dict(orient="records"):
        odds = game.get("betting_odds", {})
        context = game.get("temporal_context", {})
        odds_records.append({
            "odds_moneyline_home": odds.get("moneyline_home"),
            "odds_moneyline_away": odds.get("moneyline_away"),
            "odds_total": odds.get("total"),
            "odds_over_odds": odds.get("over_odds"),
            "odds_under_odds": odds.get("under_odds"),
            "odds_implied_prob_home": odds.get("implied_prob_home"),
            "odds_implied_prob_away": odds.get("implied_prob_away"),
            "odds_implied_edge": (odds.get("implied_prob_home") or 0) - (odds.get("implied_prob_away") or 0),
            "odds_market_efficiency": (odds.get("implied_prob_home") or 0) + (odds.get("implied_prob_away") or 0),
            "odds_moneyline_gap": (odds.get("moneyline_home") or 0) - (odds.get("moneyline_away") or 0),
            "odds_total_pressure": (odds.get("over_odds") or 0) - (odds.get("under_odds") or 0)
        })
        context_records.append({
            "ctx_home_win_pct": context.get("home_win_pct"),
            "ctx_away_win_pct": context.get("away_win_pct"),
            "ctx_home_wins": context.get("home_wins"),
            "ctx_home_losses": context.get("home_losses"),
            "ctx_away_wins": context.get("away_wins"),
            "ctx_away_losses": context.get("away_losses"),
            "ctx_home_l10_wins": context.get("home_l10_wins"),
            "ctx_away_l10_wins": context.get("away_l10_wins"),
            "ctx_home_rest_days": context.get("home_rest_days"),
            "ctx_away_rest_days": context.get("away_rest_days"),
            "ctx_home_back_to_back": int(bool(context.get("home_back_to_back"))),
            "ctx_away_back_to_back": int(bool(context.get("away_back_to_back"))),
            "ctx_rest_advantage": context.get("rest_advantage"),
            "ctx_record_differential": context.get("record_differential"),
            "ctx_form_differential": context.get("form_differential"),
            "ctx_rest_gap": context.get("rest_advantage"),
            "ctx_win_pct_gap": (context.get("home_win_pct") or 0) - (context.get("away_win_pct") or 0)
        })
    return pd.DataFrame(odds_records), pd.DataFrame(context_records)


def load_transformer_names() -> List[str]:
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    cache_key = meta.get("narrative_cache_key")

    # If the narrative features were embedded into the structured block, fall back to the
    # feature metadata file which records the transformer inventory.
    if not cache_key or cache_key == "embedded_in_structured":
        with open(FEATURES_METADATA_PATH, "r") as f:
            features_meta = json.load(f)
        transformers = features_meta.get("universal", {}).get("transformers")
        if not transformers:
            raise ValueError("Unable to determine transformer list from features metadata.")
        return transformers

    cache_meta_path = CACHE_DIR / f"{cache_key}_metadata.json"
    with open(cache_meta_path, "r") as f:
        cache_meta = json.load(f)
    return cache_meta["transformer_names"]


def extract_narrative_features(games: List[Dict], transformer_names: List[str]) -> pd.DataFrame:
    narratives = [build_nhl_narrative({**game, "home_won": None}) for game in games]
    pipeline = FeatureExtractionPipeline(
        transformer_names=transformer_names,
        domain_name="nhl",
        enable_caching=False,
        verbose=False
    )
    features = pipeline.fit_transform(narratives, return_dataframe=True)
    return features


def build_feature_matrix(games: List[Dict], feature_columns: List[str]) -> pd.DataFrame:
    base_df = pd.DataFrame(games)
    structured = get_structured_baseline(len(games))
    odds_df, context_df = expand_odds_and_context(base_df)
    transformer_names = load_transformer_names()
    narrative_df = extract_narrative_features(games, transformer_names)

    frames = [structured, narrative_df, odds_df, context_df]
    features = pd.concat(frames, axis=1)
    features = features.fillna(0.0)

    # Align with training column order, fill missing with zeros
    aligned = pd.DataFrame(index=features.index, columns=feature_columns, dtype=float)
    for col in feature_columns:
        if col in features:
            aligned[col] = features[col]
        else:
            aligned[col] = 0.0

    return aligned.fillna(0.0)


def load_models(model_names: List[str]):
    models = {}
    for name in model_names:
        path = MODELS_DIR / f"{name}.pkl"
        models[name] = joblib.load(path)
    return models


def main():
    args = parse_args()
    global EDGE_THRESHOLD
    EDGE_THRESHOLD = args.edge_threshold

    games = load_upcoming_games(args.upcoming_path)
    if not games:
        print("No upcoming games found in the provided file.")
        return

    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)
    feature_cols = summary["feature_columns"]

    feature_matrix = build_feature_matrix(games, feature_cols)
    models = load_models(args.models)

    predictions = {}
    for name, model in models.items():
        probas = model.predict_proba(feature_matrix.values)[:, 1]
        predictions[name] = probas

    results = []
    for idx, game in enumerate(games):
        odds = game["betting_odds"]
        implied_home = odds["implied_prob_home"]
        implied_away = odds["implied_prob_away"]
        row = {
            "game_id": game["game_id"],
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "date": game["date"],
            "commence_time": game.get("commence_time"),
            "moneyline_home": odds["moneyline_home"],
            "moneyline_away": odds["moneyline_away"]
        }
        recs = []
        for name, probas in predictions.items():
            prob = probas[idx]
            home_edge = prob - implied_home
            away_edge = (1 - prob) - implied_away
            if home_edge >= EDGE_THRESHOLD or away_edge >= EDGE_THRESHOLD:
                if home_edge >= away_edge:
                    side = "home"
                    edge = home_edge
                    ml = odds["moneyline_home"]
                    implied = implied_home
                else:
                    side = "away"
                    edge = away_edge
                    ml = odds["moneyline_away"]
                    implied = implied_away
                recs.append({
                    "model": name,
                    "side": side,
                    "prob": prob if side == "home" else (1 - prob),
                    "edge": edge,
                    "moneyline": ml,
                    "implied_prob": implied
                })
        row["recommendations"] = recs
        results.append(row)

    actionable = [r for r in results if r["recommendations"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(results, f, indent=2)

    if not actionable:
        print(f"No bets cleared the {EDGE_THRESHOLD*100:.1f}% edge threshold. Full results saved to {args.output}")
        return

    print(f"Bets with >= {EDGE_THRESHOLD*100:.1f}% edge (full output saved to {args.output}):")
    for row in actionable:
        print(f"\n{row['matchup']} ({row['date']})")
        for rec in row["recommendations"]:
            prob_pct = rec["prob"] * 100
            edge_pct = rec["edge"] * 100
            print(
                f"  - {rec['model']}: {rec['side'].upper()} "
                f"prob={prob_pct:.2f}% edge={edge_pct:.2f}% ML {rec['moneyline']} "
                f"(market implied {rec['implied_prob']*100:.2f}%)"
            )


if __name__ == "__main__":
    main()

