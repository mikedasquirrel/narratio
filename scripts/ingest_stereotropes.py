#!/usr/bin/env python3
"""
Ingest Stereotropes narrative data into a unified corpus.

Usage:
    python scripts/ingest_stereotropes.py \
        --source /path/to/stereotropes-data-public \
        --output data/literary_corpus/stereotropes_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literary_corpus" / "stereotropes_corpus.json"


def _load_structured(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            sep = "," if path.suffix.lower() == ".csv" else "\t"
            df = pd.read_csv(path, sep=sep)
            data = df.to_dict(orient="records")
        else:
            return []
    except Exception:
        return []

    if isinstance(data, dict):
        # Only return dict values if they're dicts themselves
        values = list(data.values())
        return [v for v in values if isinstance(v, dict)]
    if isinstance(data, list):
        # Filter out non-dict items
        return [item for item in data if isinstance(item, dict)]
    return []


def _flatten_text(record: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
    if not record:
        return ""
    parts: List[str] = []
    if keys:
        for key in keys:
            value = record.get(key)
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: " + ", ".join(map(str, value[:20])))
    else:
        for key, value in record.items():
            if isinstance(value, (str, int, float)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                preview = ", ".join(map(str, value[:15]))
                parts.append(f"{key}: {preview}")
    return " ".join(parts)


def _estimate_score(record: Dict[str, Any]) -> float:
    numeric_total = 0.0
    for value in record.values():
        if isinstance(value, (int, float)):
            numeric_total += float(value)
        elif isinstance(value, list):
            numeric_total += len(value)
    return numeric_total if numeric_total > 0 else len(record.keys()) or 1.0


def ingest_stereotropes(source: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Stereotropes source directory not found: {source}")

    categories = {
        "films": ["title", "plot", "description", "tropes"],
        "tropes": ["name", "description", "examples"],
        "adjectives": ["adjective", "description", "gender"],
    }

    entries: List[Dict[str, Any]] = []
    for category, preferred_keys in categories.items():
        category_dir = source / category
        if not category_dir.exists():
            continue
        for file_path in category_dir.glob("*"):
            for record in _load_structured(file_path):
                narrative = _flatten_text(record, preferred_keys) or _flatten_text(record)
                if not narrative:
                    continue
                impact_score = _estimate_score(record)
                entry_id = (
                    str(record.get("id"))
                    or record.get("name")
                    or record.get("title")
                    or f"{category}:{file_path.name}:{len(entries)}"
                )
                entries.append(
                    {
                        "id": entry_id,
                        "category": category,
                        "narrative": narrative,
                        "impact_score": impact_score,
                        "labels": record.get("genres")
                        or record.get("tags")
                        or record.get("types"),
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Saved {len(entries)} stereotropes narratives to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest stereotropes dataset.")
    parser.add_argument("--source", type=str, required=True, help="Path to stereotropes-data-public clone.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    args = parser.parse_args()
    ingest_stereotropes(Path(args.source), Path(args.output))

