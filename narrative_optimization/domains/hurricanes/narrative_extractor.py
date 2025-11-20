"""
Hurricane Narrative Extractor
=============================

Transforms structured hurricane records into narrative text + outcome targets.

This bridges the dual-π framework:
- π_storm  ≈ 0.30 (physical system, low agency)
- π_response ≈ 0.68 (human evacuation + preparation)

We synthesize narratives that capture both the meteorological arc (formation,
intensification, landfall) and the human behavioral response driven by name
perception research (Jung et al. 2014) on gender + phonetic harshness.
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import shorten
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .name_analyzer import HurricaneNameAnalyzer


WIND_KNOT_TO_MPH = 1.15078
BASE_FATALITIES_BY_CATEGORY = {
    0: 0.4,
    1: 3.2,
    2: 9.5,
    3: 28.0,
    4: 74.0,
    5: 185.0,
}


@dataclass(frozen=True)
class HurricaneSeverity:
    name: str
    year: int
    category: int
    max_wind_mph: float
    min_pressure_mb: float
    track_count: int
    duration_hours: float
    basin: str
    landfall: bool
    landfall_locations: List[str]


def extract_hurricane_narratives(
    raw_data: Iterable[Dict],
) -> Tuple[List[str], np.ndarray, int]:
    """
    Convert hurricane records into narratives + death toll outcomes.
    """
    storms = _coerce_storm_list(raw_data)
    analyzer = HurricaneNameAnalyzer()
    narratives: List[str] = []
    outcomes: List[float] = []

    for storm in storms:
        severity = _summarize_severity(storm)
        if severity is None:
            continue

        name_features = analyzer.analyze_name(severity.name)
        response = _estimate_response_metrics(storm, severity, name_features)
        if response is None:
            continue

        narrative = _compose_narrative(severity, name_features, response)
        if not narrative or len(narrative) < 80:
            continue

        narratives.append(narrative)
        outcomes.append(response["fatalities"])

    return narratives, np.array(outcomes, dtype=float), len(storms)


def _coerce_storm_list(raw_data: Iterable[Dict]) -> List[Dict]:
    if isinstance(raw_data, dict):
        if "storms" in raw_data:
            return list(raw_data["storms"])
        # Flatten 1-level dict-of-lists
        flattened: List[Dict] = []
        for value in raw_data.values():
            if isinstance(value, list):
                flattened.extend(value)
        return flattened
    return list(raw_data)


def _summarize_severity(storm: Dict) -> HurricaneSeverity | None:
    name = (storm.get("name") or "").strip().title()
    if not name:
        return None

    year = int(storm.get("year", 0) or 0)
    category = int(storm.get("category", 0) or 0)
    max_wind = float(storm.get("max_wind", 0) or 0)
    min_pressure = float(storm.get("min_pressure", 1010) or 1010)
    tracks = storm.get("tracks") or storm.get("track") or []
    track_count = len(tracks)
    duration_hours = max(6.0, track_count * 6.0)

    severity = HurricaneSeverity(
        name=name,
        year=year,
        category=max(0, min(5, category)),
        max_wind_mph=max_wind * WIND_KNOT_TO_MPH,
        min_pressure_mb=min_pressure,
        track_count=track_count,
        duration_hours=duration_hours,
        basin=storm.get("basin", "Atlantic"),
        landfall=bool(storm.get("landfall")),
        landfall_locations=list(storm.get("landfall_locations") or []),
    )

    return severity


def _estimate_response_metrics(
    storm: Dict,
    severity: HurricaneSeverity,
    name_features: Dict,
) -> Dict[str, float] | None:
    """
    Deterministically estimate fatalities using physical + nominative signals.
    """
    famous = (storm.get("name_profile") or {}).get("famous_association") or {}
    if (
        famous
        and famous.get("year") == severity.year
        and isinstance(famous.get("deaths"), (int, float))
    ):
        known_deaths = max(0.0, float(famous["deaths"]))
        evacuation_rate = _estimate_evacuation_rate(severity, name_features)
        perceived_threat = _perceived_threat_score(evacuation_rate, name_features)
        return {
            "fatalities": known_deaths,
            "evacuation_rate": evacuation_rate,
            "perceived_threat": perceived_threat,
            "storm_pi": 0.30,
            "response_pi": 0.68,
        }

    category_base = BASE_FATALITIES_BY_CATEGORY.get(severity.category, 5.0)
    if severity.category == 0 and not severity.landfall:
        return None  # Non-landfalling tropical depressions add noise

    landfall_factor = 1.8 if severity.landfall else 0.25
    if severity.landfall and severity.landfall_locations:
        landfall_factor *= 1 + 0.08 * min(3, len(severity.landfall_locations) - 1)

    wind_component = max(0.75, (severity.max_wind_mph / 110.0) ** 1.3)
    pressure_component = 1.0 + max(0.0, (980.0 - severity.min_pressure_mb) / 200.0)
    duration_component = 1.0 + min(0.6, severity.duration_hours / 120.0)

    modernization = (severity.year - 1950) / max(1, 2024 - 1950)
    modernization = min(1.0, max(0.0, modernization))
    infrastructure_factor = 1.25 - 0.45 * modernization

    evacuation_rate = _estimate_evacuation_rate(severity, name_features)
    perceived_threat = _perceived_threat_score(evacuation_rate, name_features)

    non_evacuation_penalty = 1.0 + (0.55 - evacuation_rate) * 0.9
    fatalities = (
        category_base
        * landfall_factor
        * wind_component
        * pressure_component
        * duration_component
        * infrastructure_factor
        * non_evacuation_penalty
    )
    fatalities = max(0.0, float(round(fatalities, 2)))

    return {
        "fatalities": fatalities,
        "evacuation_rate": evacuation_rate,
        "perceived_threat": perceived_threat,
        "storm_pi": 0.30,
        "response_pi": 0.68,
    }


def _estimate_evacuation_rate(
    severity: HurricaneSeverity,
    name_features: Dict,
) -> float:
    base = 0.32 + 0.11 * max(0, severity.category)
    if severity.landfall:
        base += 0.12
    else:
        base -= 0.08
    base += 0.03 * min(3, len(severity.landfall_locations or []))
    base += 0.05 * min(1.0, severity.duration_hours / 96.0)

    # Nominative adjustments (Jung et al. 2014 inspired)
    gender_shift = -0.08 * ((name_features["gender_rating"] - 4.0) / 3.0)
    harshness_shift = 0.05 * (name_features["phonetic_hardness"] - 0.5)
    memorability_shift = 0.04 * (name_features["memorability"] - 0.5)
    syllable_shift = -0.02 * (name_features["syllables"] - 2)

    evac = base + gender_shift + harshness_shift + memorability_shift + syllable_shift
    return float(min(0.98, max(0.05, evac)))


def _perceived_threat_score(evacuation_rate: float, name_features: Dict) -> float:
    threat = 0.55 + (evacuation_rate - 0.5) * 0.8
    threat += 0.05 * (name_features["phonetic_hardness"] - 0.5)
    threat -= 0.04 * ((name_features["gender_rating"] - 4.0) / 3.0)
    return float(min(1.0, max(0.0, threat)))


def _compose_narrative(
    severity: HurricaneSeverity,
    name_features: Dict,
    response: Dict[str, float],
) -> str:
    cat_desc = (
        "tropical storm"
        if severity.category == 0
        else f"Category {severity.category} hurricane"
    )
    landfall_phrase = (
        "stayed offshore"
        if not severity.landfall
        else _format_landfall(severity.landfall_locations)
    )

    gender_label = name_features["gender_category"]
    harshness = name_features["phonetic_hardness"]
    evac_rate = response["evacuation_rate"]

    narrative = (
        f"{severity.year} Hurricane {severity.name} tracked {severity.track_count} advisories across the "
        f"{severity.basin}, peaking as a {cat_desc} with sustained winds near "
        f"{severity.max_wind_mph:.0f} mph and a core pressure around {severity.min_pressure_mb:.0f} mb. "
        f"The storm {landfall_phrase}, keeping the storm-side narrativity low (π≈{response['storm_pi']:.2f}) "
        f"while the human response retained agency (π≈{response['response_pi']:.3f}). "
        f"Name perception flagged {severity.name} as {gender_label} with "
        f"{name_features['syllables']} syllables and phonetic hardness {harshness:.2f}, "
        f"pulling evacuation intent to {evac_rate:.0%}. "
        f"Combined physics + perception modeling yields an estimated {int(response['fatalities']):,} fatalities "
        f"after accounting for modernization, landfall exposure, and nominative bias."
    )

    return shorten(narrative, width=900, placeholder="…")


def _format_landfall(locations: List) -> str:
    if not locations:
        return "never made landfall"

    descriptors = [_describe_location(loc) for loc in locations if loc]
    if not descriptors:
        return "never made landfall"

    if len(descriptors) == 1:
        return f"made landfall near {descriptors[0]}"

    primary = descriptors[0]
    secondary = ", ".join(descriptors[1:3])
    if len(descriptors) > 2:
        return f"impacted {primary} first, then swept across {secondary}, and nearby coasts"
    return f"made landfall near {primary} before clipping {secondary}"


def _describe_location(location) -> str:
    if isinstance(location, str):
        return location

    if isinstance(location, dict):
        lat = location.get("lat")
        lon = location.get("lon")
        if lat and lon:
            return f"{lat} / {lon}"
        if lat:
            return str(lat)
        if lon:
            return str(lon)
        if "name" in location:
            return str(location["name"])

    return "an unspecified coastal stretch"


