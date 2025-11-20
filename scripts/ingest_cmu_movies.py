#!/usr/bin/env python3
"""
Ingest CMU Movie Summary Corpus
================================

Converts CMU's plot_summaries.txt + movie.metadata.tsv into unified JSON.

Usage:
    python scripts/ingest_cmu_movies.py \
        --source data/MovieSummaries \
        --output data/literary_corpus/cmu_movies_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literary_corpus" / "cmu_movies_corpus.json"


def ingest_cmu_movies(source: Path, output: Path = DEFAULT_OUTPUT) -> Path:
    """
    Parse CMU Movie Summary Corpus into unified JSON format.
    
    Files:
    - plot_summaries.txt: tab-separated (movie_id, plot_text)
    - movie.metadata.tsv: tab-separated metadata (id, name, date, revenue, runtime, etc.)
    """
    if not source.exists():
        raise FileNotFoundError(f"CMU source directory not found: {source}")

    plots_path = source / "plot_summaries.txt"
    metadata_path = source / "movie.metadata.tsv"

    if not plots_path.exists():
        raise FileNotFoundError(f"plot_summaries.txt not found in {source}")

    # Load plots
    plots: Dict[str, str] = {}
    with open(plots_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                movie_id, plot = parts
                plots[movie_id] = plot

    # Load metadata if available
    # Format: movie_id, freebase_id, name, release_date, revenue, runtime, languages, countries, genres
    metadata: Dict[str, Dict] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    movie_id = parts[0]
                    
                    # Safe float conversion
                    def safe_float(val):
                        try:
                            return float(val) if val else 0.0
                        except ValueError:
                            return 0.0
                    
                    metadata[movie_id] = {
                        "freebase_id": parts[1] if len(parts) > 1 else "",
                        "name": parts[2] if len(parts) > 2 else "",
                        "release_date": parts[3] if len(parts) > 3 else "",
                        "revenue": safe_float(parts[4]) if len(parts) > 4 else 0.0,
                        "runtime": safe_float(parts[5]) if len(parts) > 5 else 0.0,
                        "languages": parts[6] if len(parts) > 6 else "",
                        "countries": parts[7] if len(parts) > 7 else "",
                        "genres": parts[8] if len(parts) > 8 else "",
                    }

    # Combine into unified corpus
    entries: List[Dict] = []
    for movie_id, plot in plots.items():
        meta = metadata.get(movie_id, {})
        
        # Estimate impact from revenue + plot length
        revenue = meta.get("revenue", 0.0)
        runtime = meta.get("runtime", 0.0)
        impact_score = (
            (revenue / 1_000_000 if revenue > 0 else 1.0) * 
            (1 + len(plot) / 1000) * 
            (1 + runtime / 100)
        )
        
        entries.append({
            "id": movie_id,
            "title": meta.get("name", f"Movie_{movie_id}"),
            "narrative": plot,
            "impact_score": float(impact_score),
            "metadata": meta,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    
    print(f"✓ Saved {len(entries)} CMU movie plots to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CMU Movie Summary Corpus.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(PROJECT_ROOT / "data" / "MovieSummaries"),
        help="Path to MovieSummaries directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path",
    )
    args = parser.parse_args()
    ingest_cmu_movies(Path(args.source), Path(args.output))

