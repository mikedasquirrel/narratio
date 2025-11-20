#!/usr/bin/env python3
"""
Ingest WikiPlots corpus into narrative form.

Expected input: the `plots` and `titles` text files extracted from the WikiPlots repo.
Each story plot is separated by <EOS> lines.

Usage:
    python scripts/ingest_wikiplots.py \
        --plots /path/to/plots \
        --titles /path/to/titles \
        --output data/literary_corpus/wikiplots_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literary_corpus" / "wikiplots_corpus.json"


def _read_titles(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Titles file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ingest_wikiplots(plots_path: Path, titles_path: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    if not plots_path.exists():
        raise FileNotFoundError(f"Plots file not found: {plots_path}")
    titles = _read_titles(titles_path)
    plots_raw = plots_path.read_text(encoding="utf-8").split("<EOS>")

    entries = []
    for idx, plot in enumerate(plots_raw):
        cleaned = " ".join(line.strip() for line in plot.splitlines() if line.strip())
        if len(cleaned) < 200:
            continue
        title = titles[idx] if idx < len(titles) else f"Story-{idx}"
        entries.append(
            {
                "id": f"wikiplot_{idx}",
                "title": title,
                "narrative": cleaned,
                "impact_score": len(cleaned.split()) / 300,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"✓ Saved {len(entries)} wiki plots to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest WikiPlots dataset.")
    parser.add_argument("--plots", type=str, required=True, help="Path to plots file.")
    parser.add_argument("--titles", type=str, required=True, help="Path to titles file.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output JSON path.")
    args = parser.parse_args()
    ingest_wikiplots(Path(args.plots), Path(args.titles), Path(args.output))

