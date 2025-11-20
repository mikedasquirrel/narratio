"""
Utility for inspecting the transformer catalog.

Example usages:
    python -m narrative_optimization.tools.list_transformers
    python -m narrative_optimization.tools.list_transformers --filter nominative
    python -m narrative_optimization.tools.list_transformers --check NarrativeMass foo_bar
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, List

from narrative_optimization.src.transformers.registry import get_transformer_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the available narrative transformers.")
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output style. Default: table.",
    )
    parser.add_argument(
        "--filter",
        dest="text_filter",
        help="Case-insensitive substring filter applied to class name, slug, or docstring.",
    )
    parser.add_argument(
        "--category",
        help="Limit results to a specific category (semantic, temporal, meta, etc.).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of rows to display after filtering.",
    )
    parser.add_argument(
        "--check",
        nargs="+",
        help="List of transformer names to validate. Returns non-zero if any are missing.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a per-category summary after the listing.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Include the filesystem path column in tabular output.",
    )
    return parser


def filter_metadata(metadata, text_filter: str | None) -> List:
    if not text_filter:
        return list(metadata)
    needle = text_filter.lower()
    return [
        meta
        for meta in metadata
        if needle in meta.class_name.lower()
        or needle in meta.slug.lower()
        or needle in meta.docstring.lower()
        or needle in meta.category.lower()
    ]


def render_table(metadata, *, show_paths: bool) -> str:
    if not metadata:
        return "No transformers found."
    headers = ["Class", "Slug", "Category", "Module"]
    if show_paths:
        headers.append("File")
    rows: List[List[str]] = []
    for meta in metadata:
        row = [meta.class_name, meta.slug, meta.category, meta.module_path]
        if show_paths:
            row.append(str(meta.file_path))
        rows.append(row)
    widths = [max(len(str(row[idx])) for row in ([headers] + rows)) for idx in range(len(headers))]
    lines = [
        " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def render_markdown(metadata) -> str:
    if not metadata:
        return "_No transformers found._"
    headers = ["Class", "Slug", "Category", "Module"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for meta in metadata:
        lines.append(
            "| "
            + " | ".join(
                [
                    meta.class_name,
                    meta.slug,
                    meta.category,
                    f"`{meta.module_path}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_json(metadata) -> str:
    payload = [meta.to_dict() for meta in metadata]
    return json.dumps(payload, indent=2)


def print_summary(summary: dict) -> None:
    if not summary:
        return
    print("\nCategory summary:")
    width = max(len(category) for category in summary)
    for category, count in summary.items():
        print(f"  {category.ljust(width)} : {count}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = get_transformer_registry()
    metadata = registry.list_metadata(category=args.category)
    metadata = filter_metadata(metadata, args.text_filter)
    if args.limit:
        metadata = metadata[: args.limit]

    if args.format == "json":
        print(render_json(metadata))
    elif args.format == "markdown":
        print(render_markdown(metadata))
    else:
        print(render_table(metadata, show_paths=args.show_paths))

    if args.summary:
        print_summary(registry.summary_by_category())

    exit_code = 0
    if args.check:
        missing = registry.describe_missing(args.check)
        if missing:
            print("\nMissing transformers detected:")
            for name, suggestions in missing.items():
                hint = f" Suggestions: {', '.join(suggestions)}" if suggestions else ""
                print(f"  - {name}{hint}")
            exit_code = 1
        else:
            print("\nAll requested transformers are available.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())


