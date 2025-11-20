#!/usr/bin/env python3
"""
Ingest ML/AI literature corpus from markdown-heavy repositories.

Usage:
    python scripts/ingest_research_corpus.py \
        --source /path/to/Papers-Literature-ML-DL-RL-AI \
        --output data/literary_corpus/ml_research_corpus.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literary_corpus" / "ml_research_corpus.json"


HEADING_PATTERN = re.compile(r"^#+\s+", re.MULTILINE)


def _split_sections(text: str) -> List[str]:
    if not text:
        return []
    if "##" not in text:
        return [text.strip()]
    sections = HEADING_PATTERN.split(text)
    return [section.strip() for section in sections if section.strip()]


def _estimate_impact(text: str) -> float:
    links = text.count("http")
    citations = text.count("[")  # rough proxy
    length = len(text.split())
    score = links + citations * 0.5 + length / 500
    return max(score, 1.0)


def ingest_research_corpus(source: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Research corpus path not found: {source}")

    sections: List[Dict] = []
    for path in source.rglob("*"):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for idx, section in enumerate(_split_sections(text)):
            if len(section) < 120:
                continue
            entry_id = f"{path.stem}-{idx}"
            sections.append(
                {
                    "id": entry_id,
                    "path": str(path.relative_to(source)),
                    "narrative": section,
                    "impact_score": _estimate_impact(section),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    print(f"✓ Saved {len(sections)} research narratives to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ML/AI literature narratives.")
    parser.add_argument("--source", type=str, required=True, help="Path to Papers-Literature-ML-DL-RL-AI clone.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    args = parser.parse_args()
    ingest_research_corpus(Path(args.source), Path(args.output))

