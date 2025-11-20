#!/usr/bin/env python3
"""
Domain data utilization audit.

This script cross-references the canonical domain registry with the scraped
datasets under ``data/domains`` and the processed artifacts under
``narrative_optimization/results``.  It highlights three categories of issues:

1. Registered domains whose source data is missing or whose analysis results
   have not been generated (or are stale).
2. Scraped domain directories/files that are not referenced by any domain
   configuration (likely completely un-analysed datasets).
3. A machine-readable JSON report (optional) that downstream automation or
   dashboards can ingest.

Example:

    python3 scripts/audit_domain_data_usage.py \
        --json-output logs/domain_audit.json \
        --stale-days 14 \
        --exit-on-issues
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "domains"
RESULTS_ROOT = PROJECT_ROOT / "narrative_optimization" / "results" / "domains"


@dataclass
class DomainAuditStatus:
    """Status row for a registered domain."""

    name: str
    pi: Optional[float]
    data_path: str
    data_exists: bool
    data_size_bytes: Optional[int]
    results: List[str]
    last_result_at: Optional[str]
    status: str
    notes: List[str] = field(default_factory=list)


@dataclass
class OrphanPath:
    """Represents a scraped directory/file that is not mapped to any domain."""

    path: str
    kind: str  # "directory" or file extension
    size_bytes: Optional[int]
    last_modified_at: Optional[str]


def _import_domain_registry() -> Dict[str, object]:
    """Dynamically import the domain registry without needing installation."""

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from narrative_optimization.domain_registry import DOMAINS  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        sys.exit(f"Unable to import domain registry: {exc}")
    finally:
        # Avoid polluting sys.path for downstream tooling.
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:
            pass
    return DOMAINS


def _resolve_path(path_obj: Path) -> Path:
    """Convert relative paths (from DomainConfig) into absolute paths."""

    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()


def _format_bytes(size: Optional[int]) -> str:
    """Human-readable byte formatter."""

    if size is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    value = float(size)
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f}{units[idx]}"


def _format_dt(timestamp: Optional[float]) -> Optional[str]:
    """Convert epoch seconds into ISO 8601 strings."""

    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _find_result_paths(domain_name: str) -> List[Path]:
    """List analysis artifacts for a given domain."""

    paths: Set[Path] = set()

    domain_dir = RESULTS_ROOT / domain_name
    if domain_dir.exists() and domain_dir.is_dir():
        paths.update(sorted(domain_dir.glob("*.json")))

    globbed = RESULTS_ROOT.glob(f"{domain_name}_*.json")
    paths.update(sorted(globbed))

    # Some legacy outputs live directly under narrative_optimization/results.
    legacy_dir = RESULTS_ROOT.parent
    legacy_glob = legacy_dir.glob(f"{domain_name}_*.json")
    paths.update(sorted(legacy_glob))

    return sorted(paths)


def gather_domain_statuses(stale_days: int) -> List[DomainAuditStatus]:
    """Build audit rows for every registered domain."""

    try:
        raw_domains = _import_domain_registry()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        sys.exit(f"Failed to load domain registry: {exc}")

    statuses: List[DomainAuditStatus] = []

    for name, config in sorted(raw_domains.items(), key=lambda item: item[0]):
        data_path = _resolve_path(config.data_path)  # type: ignore[attr-defined]
        data_exists = data_path.exists()
        data_size = data_path.stat().st_size if data_exists and data_path.is_file() else None

        result_paths = _find_result_paths(name)
        last_result_ts: Optional[float] = None
        if result_paths:
            last_result_ts = max(path.stat().st_mtime for path in result_paths)

        status_label = "OK"
        notes: List[str] = []

        if not data_exists:
            status_label = "NO_DATA"
            notes.append("Data path missing")
        elif not result_paths:
            status_label = "UNANALYZED"
            notes.append("No results found")
        elif stale_days > 0 and last_result_ts is not None:
            age_days = (datetime.now(timezone.utc) - datetime.fromtimestamp(last_result_ts, tz=timezone.utc)).days
            if age_days > stale_days:
                status_label = "STALE"
                notes.append(f"Results {age_days} days old")

        statuses.append(
            DomainAuditStatus(
                name=name,
                pi=getattr(config, "estimated_pi", None),
                data_path=str(data_path),
                data_exists=data_exists,
                data_size_bytes=data_size,
                results=[str(path) for path in result_paths],
                last_result_at=_format_dt(last_result_ts),
                status=status_label,
                notes=notes,
            )
        )

    return statuses


def discover_orphans(
    registered_paths: Iterable[str], domain_names: Sequence[str]
) -> List[OrphanPath]:
    """Identify scraped directories/files not referenced by the registry."""

    registered_resolved: Set[Path] = {Path(path).resolve() for path in registered_paths}
    referenced_roots: Set[Path] = set()
    referenced_files: Set[Path] = set()

    for path in registered_resolved:
        try:
            rel = path.relative_to(DATA_ROOT)
        except ValueError:
            continue
        if len(rel.parts) > 1:
            referenced_roots.add((DATA_ROOT / rel.parts[0]).resolve())
        else:
            referenced_files.add(path)

    orphans: List[OrphanPath] = []
    if not DATA_ROOT.exists():
        return orphans

    domain_prefixes = [name.lower() for name in domain_names]

    for child in sorted(DATA_ROOT.iterdir()):
        if child.is_dir():
            if child.resolve() not in referenced_roots:
                stat = child.stat()
                orphans.append(
                    OrphanPath(
                        path=str(child.resolve()),
                        kind="directory",
                        size_bytes=stat.st_size if hasattr(stat, "st_size") else None,
                        last_modified_at=_format_dt(stat.st_mtime),
                    )
                )
        elif child.is_file() and child.suffix.lower() in {".json", ".csv", ".jsonl", ".parquet"}:
            stem = child.stem.lower()
            if any(stem.startswith(prefix) for prefix in domain_prefixes):
                continue
            if child.resolve() not in referenced_files:
                stat = child.stat()
                orphans.append(
                    OrphanPath(
                        path=str(child.resolve()),
                        kind=child.suffix.lower(),
                        size_bytes=stat.st_size,
                        last_modified_at=_format_dt(stat.st_mtime),
                    )
                )

    return orphans


def render_table(rows: Sequence[DomainAuditStatus]) -> str:
    """Render a compact table summarizing domain coverage."""

    headers = ["Domain", "π", "Data", "Results", "Status", "Notes"]
    table_rows: List[List[str]] = []

    for row in rows:
        data_str = "missing"
        if row.data_exists:
            data_str = _format_bytes(row.data_size_bytes)
        results_str = "-"
        if row.results:
            results_str = f"{len(row.results)} file(s)"
            if row.last_result_at:
                results_str += f" • {row.last_result_at}"
        note_str = "; ".join(row.notes) if row.notes else ""
        table_rows.append(
            [
                row.name,
                f"{row.pi:.2f}" if row.pi is not None else "-",
                data_str,
                results_str,
                row.status,
                note_str,
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(values: Sequence[str]) -> str:
        return " | ".join(val.ljust(col_widths[idx]) for idx, val in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in col_widths])]
    lines.extend(fmt_row(row) for row in table_rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scraped domain data utilization.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write a JSON summary (directories will be created).",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=30,
        help="Threshold (days) after which results are flagged as stale (default: 30).",
    )
    parser.add_argument(
        "--exit-on-issues",
        action="store_true",
        help="Return exit code 1 when missing/stale/unprocessed domains or orphan data are detected.",
    )

    args = parser.parse_args()

    statuses = gather_domain_statuses(stale_days=args.stale_days)
    orphan_paths = discover_orphans(
        (row.data_path for row in statuses), [row.name for row in statuses]
    )

    print("\n=== Registered Domain Coverage ===\n")
    print(render_table(statuses))

    if orphan_paths:
        print("\n=== Orphaned Scraped Data (not referenced by any domain) ===\n")
        for orphan in orphan_paths:
            size_str = _format_bytes(orphan.size_bytes)
            ts = orphan.last_modified_at or "-"
            print(f"- {orphan.path} [{orphan.kind}, {size_str}, updated {ts}]")
    else:
        print("\nNo orphaned scraped directories/files detected under data/domains.")

    if args.json_output:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "statuses": [asdict(row) for row in statuses],
            "orphans": [asdict(orphan) for orphan in orphan_paths],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON report written to {args.json_output}")

    issues_present = any(
        row.status in {"NO_DATA", "UNANALYZED", "STALE"} for row in statuses
    ) or bool(orphan_paths)

    if args.exit_on_issues and issues_present:
        sys.exit(1)


if __name__ == "__main__":
    main()

