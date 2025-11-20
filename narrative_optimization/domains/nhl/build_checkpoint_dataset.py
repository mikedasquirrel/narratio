#!/usr/bin/env python3
"""
NHL Checkpoint Dataset Builder
==============================

Creates per-period snapshot datasets (P1, P2, Final) by invoking the domain
registry checkpoint pipeline. Outputs both Parquet + JSON metadata artifacts so
models/analytics can ingest checkpoint-aware narratives immediately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "narrative_optimization" / "domains" / "nhl"


def _load_checkpoint_snapshots(
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    from narrative_optimization.domain_registry import load_domain_checkpoints

    snapshots, config = load_domain_checkpoints("nhl", checkpoint=checkpoint, limit=limit)

    if not snapshots:
        raise RuntimeError("No NHL checkpoint snapshots generated.")

    df = pd.DataFrame(snapshots)
    df_scores = pd.json_normalize(df.pop("score"))
    df_metrics = pd.json_normalize(df.pop("metrics"))
    df_metadata = pd.json_normalize(df.pop("metadata"))

    combined = pd.concat(
        [
            df.reset_index(drop=True),
            df_scores.add_prefix("score."),
            df_metrics.add_prefix("metric."),
            df_metadata.add_prefix("meta."),
        ],
        axis=1,
    )

    info = {
        "checkpoint": checkpoint or "all",
        "records": len(combined),
        "schema": config.checkpoint_schema,
        "fields": {
            "score": list(df_scores.columns),
            "metrics": list(df_metrics.columns),
            "metadata": list(df_metadata.columns),
        },
    }

    return combined, info


def build_checkpoint_dataset(
    checkpoint: Optional[str] = None,
    limit: Optional[int] = None,
    output_basename: Optional[str] = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, info = _load_checkpoint_snapshots(checkpoint=checkpoint, limit=limit)

    base = output_basename or f"nhl_checkpoint_snapshots_{checkpoint or 'all'}"
    parquet_path = OUTPUT_DIR / f"{base}.parquet"
    json_path = OUTPUT_DIR / f"{base}_metadata.json"

    df.to_parquet(parquet_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            {
                **info,
                "parquet_path": str(parquet_path.relative_to(PROJECT_ROOT)),
                "columns": df.columns.tolist(),
            },
            f,
            indent=2,
        )

    return parquet_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build NHL checkpoint dataset.")
    parser.add_argument(
        "--checkpoint",
        choices=["P1", "P2", "FINAL"],
        default=None,
        help="Optional checkpoint filter (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit snapshot count for quick experiments.",
    )
    parser.add_argument(
        "--output-basename",
        type=str,
        default=None,
        help="Custom basename for output files (without extension).",
    )
    args = parser.parse_args()

    path = build_checkpoint_dataset(
        checkpoint=args.checkpoint,
        limit=args.limit,
        output_basename=args.output_basename,
    )

    print(f"✓ Saved checkpoint dataset -> {path}")


if __name__ == "__main__":
    main()


