"""
Transformer discovery and registry utilities.

The registry is responsible for:
- Discovering every transformer class that lives inside this package
- Exposing rich metadata (category, file path, documentation) for tooling
- Providing fuzzy lookups so typos produce actionable feedback

This allows bots and humans alike to answer, in one place, which transformers
ship with the system and which ones still need to be implemented.
"""

from __future__ import annotations

import ast
import difflib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_PACKAGE_NAME = __package__  # e.g. "narrative_optimization.src.transformers"


def _camel_to_snake(name: str) -> str:
    """Convert CamelCaseTransformer to snake_case identifier without suffix."""
    chars = []
    for idx, char in enumerate(name):
        if idx and char.isupper() and (
            not name[idx - 1].isupper() or (idx + 1 < len(name) and name[idx + 1].islower())
        ):
            chars.append("_")
        chars.append(char.lower())
    snake = "".join(chars)
    if snake.endswith("_transformer"):
        snake = snake[: -len("_transformer")]
    return snake


def _extract_base_name(base: ast.expr) -> Optional[str]:
    """Best-effort extraction of the textual base-class name."""
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        parts = []
        current = base
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        if parts:
            return ".".join(reversed(parts))
    return None


def _infer_category(relative_parts: Tuple[str, ...]) -> str:
    """Infer a category string from the path (first component)."""
    if not relative_parts:
        return "general"
    top = relative_parts[0]
    # Normalize friendly categories for known subpackages
    mapping = {
        "semantic": "semantic",
        "temporal": "temporal",
        "narrative": "narrative",
        "meta": "meta",
        "cultural": "cultural",
        "cognitive": "cognitive",
        "ritual": "ritual",
        "relational": "relational",
        "sports": "sports",
        "geographic": "geographic",
        "legal": "legal",
        "media": "media",
        "archetypes": "archetype",
        "ships": "ships",
        "mental_health": "mental_health",
        "cultural_context": "contextual",
    }
    if top in mapping:
        return mapping[top]
    if top == "__pycache__":
        return "internal"
    if "/" in top:
        top = top.split("/")[0]
    if top in {"utils", "caching"}:
        return "infrastructure"
    return "core" if len(relative_parts) == 1 else top.replace("_", " ")


@dataclass(frozen=True)
class TransformerMetadata:
    """Read-only metadata for a transformer class."""

    class_name: str
    slug: str
    module_path: str
    file_path: Path
    category: str
    docstring: str
    bases: Tuple[str, ...]

    def to_dict(self) -> Dict[str, str]:
        return {
            "class_name": self.class_name,
            "slug": self.slug,
            "module_path": self.module_path,
            "file_path": str(self.file_path),
            "category": self.category,
            "docstring": self.docstring,
            "bases": list(self.bases),
        }


class TransformerRegistry:
    """
    Discovers transformer classes and exposes them via helpful query APIs.

    Discovery is done once, lazily, using Python's AST so that heavy optional
    dependencies are never imported just to enumerate classes.
    """

    _EXCLUDE_FILES = {
        "__init__.py",
        "base.py",
        "base_transformer.py",
        "domain_adaptive_base.py",
    }
    _EXCLUDE_CLASS_NAMES = {
        "NarrativeTransformer",
        "TextNarrativeTransformer",
        "FeatureNarrativeTransformer",
        "MixedNarrativeTransformer",
    }

    def __init__(self, root_dir: Optional[Path] = None, package_name: str = _PACKAGE_NAME):
        self.root_dir = (root_dir or Path(__file__).resolve().parent)
        self.package_name = package_name
        self._lock = threading.Lock()
        self._metadata_by_class: Dict[str, TransformerMetadata] = {}
        self._metadata_by_slug: Dict[str, TransformerMetadata] = {}
        self._loaded = False

    # ------------------------------------------------------------------ public
    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if not self._loaded:
                self._discover()
                self._loaded = True

    def list_metadata(self, *, category: Optional[str] = None) -> List[TransformerMetadata]:
        self.ensure_loaded()
        items = list(self._metadata_by_class.values())
        if category:
            items = [meta for meta in items if meta.category == category]
        return sorted(items, key=lambda meta: (meta.category, meta.class_name))

    def class_names(self) -> List[str]:
        self.ensure_loaded()
        return sorted(self._metadata_by_class.keys())

    def get(self, class_name: str) -> Optional[TransformerMetadata]:
        self.ensure_loaded()
        return self._metadata_by_class.get(class_name)

    def resolve(self, name: str) -> Optional[TransformerMetadata]:
        """
        Resolve any of the following to metadata:
        - Exact class name (NarrativePotentialTransformer)
        - Class name without suffix (NarrativePotential)
        - snake_case identifier (narrative_potential)
        """
        if not name:
            return None
        self.ensure_loaded()
        candidates = self._candidate_keys(name)
        for candidate in candidates:
            if candidate in self._metadata_by_class:
                return self._metadata_by_class[candidate]
            if candidate in self._metadata_by_slug:
                return self._metadata_by_slug[candidate]
        return None

    def module_for(self, class_name: str) -> str:
        metadata = self.get(class_name)
        if not metadata:
            raise KeyError(f"Unknown transformer: {class_name}")
        return metadata.module_path

    def suggest(self, name: str, *, limit: int = 5) -> List[str]:
        """Return fuzzy suggestions for an unknown transformer name."""
        self.ensure_loaded()
        pool = self.class_names() + list(self._metadata_by_slug.keys())
        matches = difflib.get_close_matches(name, pool, n=limit, cutoff=0.35)
        # Remove duplicates while preserving order
        seen = set()
        ordered: List[str] = []
        for match in matches:
            if match not in seen:
                ordered.append(match)
                seen.add(match)
        return ordered

    def describe_missing(self, names: Sequence[str]) -> Dict[str, List[str]]:
        """Return mapping of missing names to best-effort suggestions."""
        report: Dict[str, List[str]] = {}
        for name in names:
            if self.resolve(name):
                continue
            report[name] = self.suggest(name)
        return report

    def summary_by_category(self) -> Dict[str, int]:
        self.ensure_loaded()
        summary: Dict[str, int] = {}
        for meta in self._metadata_by_class.values():
            summary[meta.category] = summary.get(meta.category, 0) + 1
        return dict(sorted(summary.items(), key=lambda item: (-item[1], item[0])))

    # ----------------------------------------------------------------- helpers
    def _discover(self) -> None:
        for file_path in sorted(self.root_dir.rglob("*.py")):
            if file_path.name in self._EXCLUDE_FILES:
                continue
            relative_parts = file_path.relative_to(self.root_dir).parts
            if "__pycache__" in relative_parts:
                continue
            module_parts = file_path.relative_to(self.root_dir).with_suffix("").parts
            module_path = ".".join((self.package_name, *module_parts))
            try:
                source = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                raise RuntimeError(f"Failed to parse {file_path}: {exc}") from exc
            for node in (n for n in tree.body if isinstance(n, ast.ClassDef)):
                if not node.name.endswith("Transformer"):
                    continue
                if node.name in self._EXCLUDE_CLASS_NAMES:
                    continue
                slug = _camel_to_snake(node.name)
                bases = tuple(
                    base_name
                    for base in node.bases
                    if (base_name := _extract_base_name(base)) is not None
                )
                docstring = (ast.get_docstring(node) or "").strip()
                metadata = TransformerMetadata(
                    class_name=node.name,
                    slug=slug,
                    module_path=module_path,
                    file_path=file_path,
                    category=_infer_category(relative_parts),
                    docstring=docstring,
                    bases=bases,
                )
                self._metadata_by_class[node.name] = metadata
                self._metadata_by_slug[slug] = metadata

    @staticmethod
    def _candidate_keys(name: str) -> List[str]:
        cleaned = name.strip()
        if not cleaned:
            return []
        candidates = [cleaned]
        if not cleaned.endswith("Transformer"):
            candidates.append(f"{cleaned}Transformer")
        snake = cleaned.lower().replace(" ", "_")
        if snake not in candidates:
            candidates.append(snake)
        return candidates


_REGISTRY: Optional[TransformerRegistry] = None


def get_transformer_registry() -> TransformerRegistry:
    """Return the singleton registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = TransformerRegistry()
    return _REGISTRY


