"""
Literary Insights Routes
========================

Production-ready dashboard surfacing the newly ingested literary domains
and their downstream impact on the universal narrative framework.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from flask import Blueprint, render_template


literary_bp = Blueprint("literary", __name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "narrative_optimization" / "results"
CONTEXT_DIR = RESULTS_ROOT / "context_stratification"
NOMINATIVE_DIR = RESULTS_ROOT / "nominative_enrichment"
DYNAMIC_PI_DIR = RESULTS_ROOT / "dynamic_pi"
NETWORK_DIR = RESULTS_ROOT / "network_genomes"
META_DATASET_PATH = RESULTS_ROOT / "meta_framework" / "meta_dataset.json"

LITERARY_DOMAINS: Dict[str, Dict[str, str]] = {
    "wikiplots": {
        "name": "WikiPlots",
        "icon": "book",
        "tagline": "Global narrative encyclopedia",
        "description": "106k plot synopses spanning every genre; ideal for capturing archetypal arcs.",
    },
    "stereotropes": {
        "name": "Stereotropes",
        "icon": "masks",
        "tagline": "Trope atlas",
        "description": "High-signal stereotype/trope combinations for stress-testing pattern precision.",
    },
    "cmu_movies": {
        "name": "CMU Movie Summaries",
        "icon": "film",
        "tagline": "Studio-level summaries",
        "description": "42k curated movie storylines with revenue/runtime metadata for grounded benchmarks.",
    },
    "ml_research": {
        "name": "ML Research Papers",
        "icon": "lab",
        "tagline": "Academic narratives",
        "description": "Dense, citation-driven narratives capturing scientific persuasion strategies.",
    },
}


@literary_bp.route("/insights")
def insights_dashboard():
    meta_lookup = _load_meta_lookup()
    cards = []
    for slug, meta in LITERARY_DOMAINS.items():
        cards.append(_build_domain_card(slug, meta, meta_lookup.get(slug)))

    shared_features = _compute_shared_features(cards)
    meta_summary = _build_meta_summary(cards)

    return render_template(
        "literary_insights.html",
        cards=cards,
        shared_features=shared_features,
        meta_summary=meta_summary,
    )


# ---------------------------------------------------------------------- #
# Builders
# ---------------------------------------------------------------------- #


def _build_domain_card(
    slug: str, meta: Dict[str, str], meta_entry: Optional[Dict]
) -> Dict:
    context_payload = _load_json(CONTEXT_DIR / f"{slug}_contexts.json")
    nominative_payload = _load_json(NOMINATIVE_DIR / f"{slug}_nominative_enrichment.json")
    dynamic_payload = _load_json(DYNAMIC_PI_DIR / f"{slug}_dynamic_pi.json")
    network_payload = _load_json(NETWORK_DIR / f"{slug}_network_genome.json")

    patterns = context_payload.get("patterns", [])
    summarized_patterns = _summarize_patterns(patterns)
    dominant_features = _dominant_features(patterns)

    return {
        "slug": slug,
        "icon": meta["icon"],
        "name": meta["name"],
        "tagline": meta["tagline"],
        "description": meta["description"],
        "baseline": context_payload.get("baseline_win_rate"),
        "n_patterns": len(patterns),
        "patterns": summarized_patterns,
        "dominant_features": dominant_features,
        "nominative": nominative_payload,
        "dynamic_pi": dynamic_payload,
        "network": network_payload.get("graph_metadata"),
        "meta": meta_entry,
        "feature_tags": [feature["key"] for feature in dominant_features],
    }


def _summarize_patterns(patterns: List[Dict], limit: int = 4) -> List[Dict]:
    if not patterns:
        return []
    ranked = sorted(
        patterns,
        key=lambda p: p.get("edge_vs_baseline", 0.0),
        reverse=True,
    )[:limit]
    summary = []
    for pattern in ranked:
        summary.append(
            {
                "name": pattern.get("pattern_name"),
                "edge": pattern.get("edge_vs_baseline"),
                "accuracy": pattern.get("accuracy"),
                "samples": pattern.get("sample_size"),
                "pi_mean": pattern.get("pi_mean"),
                "conditions": _humanize_conditions(pattern.get("conditions", {})),
            }
        )
    return summary


def _dominant_features(patterns: List[Dict], limit: int = 3) -> List[Dict]:
    counter: Counter = Counter()
    for pattern in patterns:
        for feature in (pattern.get("conditions") or {}).keys():
            counter[feature] += 1
    most_common = counter.most_common(limit)
    output = []
    for feature, count in most_common:
        output.append(
            {
                "key": feature,
                "label": feature.replace("_", " ").title(),
                "count": count,
            }
        )
    return output


def _compute_shared_features(cards: List[Dict]) -> List[Dict]:
    counter: Counter = Counter()
    for card in cards:
        for tag in card.get("feature_tags", []):
            counter[tag] += 1
    shared = [
        {"key": key, "label": key.replace("_", " ").title(), "count": count}
        for key, count in counter.items()
        if count >= 2
    ]
    shared.sort(key=lambda item: item["count"], reverse=True)
    return shared


def _build_meta_summary(cards: List[Dict]) -> Dict[str, float]:
    total_patterns = sum(card["n_patterns"] for card in cards if card["n_patterns"])
    avg_baseline = _safe_mean([card["baseline"] for card in cards if card["baseline"]])
    avg_pi = _safe_mean(
        [
            card.get("dynamic_pi", {}).get("pi_mean")
            for card in cards
            if card.get("dynamic_pi", {}).get("pi_mean")
        ]
    )
    avg_density = _safe_mean(
        [
            card.get("network", {}).get("density")
            for card in cards
            if card.get("network", {}).get("density")
        ]
    )
    return {
        "domain_count": len(cards),
        "total_patterns": total_patterns,
        "avg_baseline": avg_baseline,
        "avg_pi": avg_pi,
        "avg_density": avg_density,
    }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _humanize_conditions(conditions: Dict[str, Dict]) -> List[str]:
    humanized = []
    for feature, constraint in conditions.items():
        if isinstance(constraint, dict):
            parts = []
            if "min" in constraint:
                parts.append(f">= {constraint['min']:.2f}")
            if "max" in constraint:
                parts.append(f"<= {constraint['max']:.2f}")
            if "eq" in constraint:
                parts.append(f"= {constraint['eq']}")
            clause = ", ".join(parts)
        else:
            clause = f"= {constraint}"
        humanized.append(f"{feature.replace('_', ' ')} ({clause})")
    return humanized


def _load_meta_lookup() -> Dict[str, Dict]:
    payload = _load_json(META_DATASET_PATH)
    lookup = {}
    for entry in payload.get("domains", []):
        lookup[entry["name"]] = entry
    return lookup


def _load_json(path: Path) -> Dict:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    try:
        return float(sum(values) / len(values))
    except ZeroDivisionError:
        return None


__all__ = ["literary_bp"]


