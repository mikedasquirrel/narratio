#!/usr/bin/env python3
"""
Nominative Enrichment Runner
---------------------------

Applies the DeepNominativeTransformer to each validated domain and measures the
lift vs simple name stats (baseline). Outputs R²/accuracy deltas per domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from narrative_optimization.domain_registry import get_domain
from narrative_optimization.transformers.deep_nominative import (
    DeepNominativeTransformer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "domains"
OUTPUT_DIR = (
    PROJECT_ROOT / "narrative_optimization" / "results" / "nominative_enrichment"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_records(domain: str) -> List[Dict]:
    config = get_domain(domain)
    if not config:
        raise ValueError(f"Domain '{domain}' is not registered.")
    data = config.get_raw_records()
    if isinstance(data, dict):
        if "games" in data:
            data = data["games"]
        else:
            data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unexpected structure for {domain}: {type(data)}")
    return data


def _nhl_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    if record.get("home_won") is None:
        return None
    names = [
        record.get("home_team"),
        record.get("away_team"),
        record.get("home_goalie"),
        record.get("away_goalie"),
        record.get("home_coach"),
        record.get("away_coach"),
    ]
    return [n for n in names if n], float(record.get("home_won"))


def _nfl_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    if record.get("home_won") is None:
        return None
    names = [
        record.get("home_team"),
        record.get("away_team"),
        (record.get("home_qb") or {}).get("name"),
        (record.get("away_qb") or {}).get("name"),
        record.get("home_coach"),
        record.get("away_coach"),
        record.get("stadium"),
    ]
    return [n for n in names if n], float(record.get("home_won"))


def _nba_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    if record.get("won") is None:
        return None
    names = [
        record.get("team_name"),
        record.get("matchup"),
        record.get("team_abbreviation"),
        record.get("pregame_narrative"),
    ]
    return [n for n in names if n], float(record.get("won"))


def _mlb_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    winner = record.get("winner")
    if winner not in {"home", "away"}:
        return None
    names = []
    home = record.get("home_team") or {}
    away = record.get("away_team") or {}
    names.extend([home.get("name"), home.get("nickname"), away.get("name"), away.get("nickname")])
    for lineup in ("home_lineup", "away_lineup"):
        roster = record.get(lineup) or []
        for player in roster:
            if isinstance(player, dict) and player.get("name"):
                names.append(player["name"])
    return [n for n in names if n], float(winner == "home")


def _golf_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    if record.get("won_tournament") is None:
        return None
    names = [
        record.get("player_name"),
        record.get("tournament_name"),
        record.get("course_name"),
    ]
    return [n for n in names if n], float(record.get("won_tournament"))


def _startups_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    if record.get("successful") is None:
        return None
    names = [record.get("name"), record.get("location")]
    founders = record.get("founders") or []
    if isinstance(founders, list):
        names.extend(founders)
    return [n for n in names if n], float(record.get("successful"))


def _supreme_court_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    outcome = record.get("outcome") or {}
    metadata = record.get("metadata") or {}
    names = [
        record.get("case_name"),
        metadata.get("author"),
        metadata.get("dissent_author"),
        metadata.get("concurrence_author"),
    ]
    citation = outcome.get("citation_count")
    if citation is None:
        return None
    return [n for n in names if n], float(citation)


def _wikiplots_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    title = record.get("title", "")
    narrative = record.get("narrative", "")
    import re
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative)
    names.append(title)
    return names[:20], float(impact)


def _stereotropes_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    narrative = record.get("narrative", "")
    import re
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative)
    return names[:15], float(impact)


def _ml_research_names(record: Dict) -> Optional[Tuple[List[str], float]]:
    impact = record.get("impact_score")
    if impact is None:
        return None
    narrative = record.get("narrative", "")
    import re
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', narrative)
    return names[:10], float(impact)


NAME_BUILDERS: Dict[str, Callable[[Dict], Optional[Tuple[List[str], float]]]] = {
    "nhl": _nhl_names,
    "nfl": _nfl_names,
    "nba": _nba_names,
    "mlb": _mlb_names,
    "golf": _golf_names,
    "startups": _startups_names,
    "supreme_court": _supreme_court_names,
    "wikiplots": _wikiplots_names,
    "stereotropes": _stereotropes_names,
    "ml_research": _ml_research_names,
    "cmu_movies": _wikiplots_names,  # Same structure
}


def load_names_and_outcomes(domain: str) -> Tuple[List[List[str]], np.ndarray]:
    builder = NAME_BUILDERS.get(domain)
    if not builder:
        raise ValueError(f"No nominative loader configured for '{domain}'")

    records = _load_records(domain)
    names: List[List[str]] = []
    outcomes: List[float] = []
    for rec in records:
        result = builder(rec)
        if not result:
            continue
        name_list, outcome = result
        if not name_list:
            continue
        names.append(name_list)
        outcomes.append(outcome)
    return names, np.array(outcomes, dtype=float)


def baseline_features(names: List[List[str]]) -> np.ndarray:
    feats = []
    for row in names:
        lengths = [len(name) for name in row] or [0]
        feats.append(
            [
                len(row),
                np.mean(lengths),
                np.std(lengths),
                sum(name.isupper() for name in row) / max(len(row), 1),
            ]
        )
    return np.array(feats)


def run_domain(domain: str):
    print(f"[Nominative] Loading names and outcomes for {domain}...", flush=True)
    X_names, y = load_names_and_outcomes(domain)
    if not X_names:
        print(f"[{domain}] No records available.", flush=True)
        return
    print(f"[Nominative] Loaded {len(X_names)} records for {domain}", flush=True)

    print(f"[Nominative] Extracting deep nominative features...", flush=True)
    transformer = DeepNominativeTransformer()
    X_enriched = transformer.transform(X_names)
    print(f"[Nominative] Deep features extracted: {X_enriched.shape}", flush=True)
    
    print(f"[Nominative] Computing baseline features...", flush=True)
    X_base = baseline_features(X_names)
    print(f"[Nominative] Baseline features computed: {X_base.shape}", flush=True)

    print(f"[Nominative] Splitting data (75/25 train/test)...", flush=True)
    X_train_e, X_test_e, y_train, y_test = train_test_split(
        X_enriched, y, test_size=0.25, random_state=42
    )
    X_train_b, X_test_b, _, _ = train_test_split(
        X_base, y, test_size=0.25, random_state=42
    )
    print(f"[Nominative] Train: {len(X_train_e)}, Test: {len(X_test_e)}", flush=True)

    # Use regression for continuous outcomes, classification for binary
    continuous_domains = {"supreme_court", "wikiplots", "stereotropes", "ml_research", "cmu_movies"}
    
    if domain in continuous_domains:
        print(f"[Nominative] Using regression models (continuous outcome)...", flush=True)
        model_enriched = RandomForestRegressor(random_state=42, n_estimators=400)
        model_baseline = RandomForestRegressor(random_state=42, n_estimators=200)
        print(f"[Nominative] Training enriched model...", flush=True)
        model_enriched.fit(X_train_e, y_train)
        print(f"[Nominative] Training baseline model...", flush=True)
        model_baseline.fit(X_train_b, y_train)
        print(f"[Nominative] Evaluating models...", flush=True)
        preds_e = model_enriched.predict(X_test_e)
        preds_b = model_baseline.predict(X_test_b)
        metric_enriched = r2_score(y_test, preds_e)
        metric_baseline = r2_score(y_test, preds_b)
        metric_name = "r2"
    else:
        print(f"[Nominative] Using classification models (binary outcome)...", flush=True)
        clf_enriched = LogisticRegression(max_iter=1000)
        clf_baseline = LogisticRegression(max_iter=1000)
        print(f"[Nominative] Training enriched classifier...", flush=True)
        clf_enriched.fit(X_train_e, y_train)
        print(f"[Nominative] Training baseline classifier...", flush=True)
        clf_baseline.fit(X_train_b, y_train)
        print(f"[Nominative] Evaluating classifiers...", flush=True)
        preds_e = clf_enriched.predict(X_test_e)
        preds_b = clf_baseline.predict(X_test_b)
        metric_enriched = accuracy_score(y_test, preds_e)
        metric_baseline = accuracy_score(y_test, preds_b)
        metric_name = "accuracy"

    lift = metric_enriched - metric_baseline
    print(f"[Nominative] Results: baseline={metric_baseline:.3f}, enriched={metric_enriched:.3f}, lift={lift:.3f}", flush=True)
    
    print(f"[Nominative] Saving results...", flush=True)
    output = {
        "domain": domain,
        "metric": metric_name,
        "baseline": metric_baseline,
        "enriched": metric_enriched,
        "lift": lift,
        "sample_size": len(X_names),
        "feature_count": X_enriched.shape[1],
    }
    with open(OUTPUT_DIR / f"{domain}_nominative_enrichment.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(
        f"[{domain}] baseline={metric_baseline:.3f} enriched={metric_enriched:.3f} lift={lift:.3f}", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description="Run nominative enrichment tests.")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["nhl", "nfl", "nba", "mlb", "golf", "startups", "supreme_court"],
    )
    args = parser.parse_args()
    print(f"[Nominative Enrichment] Domains queued: {args.domains}", flush=True)

    for domain in args.domains:
        run_domain(domain)


if __name__ == "__main__":
    print("[Nominative Enrichment] CLI entrypoint engaged.", flush=True)
    main()


