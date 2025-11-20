from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_optimization.src.transformers.registry import get_transformer_registry


def test_registry_resolves_multiple_name_styles():
    registry = get_transformer_registry()
    meta_direct = registry.resolve("NarrativePotentialTransformer")
    meta_short = registry.resolve("NarrativePotential")
    meta_slug = registry.resolve("narrative_potential")
    assert meta_direct is not None
    assert meta_short is not None
    assert meta_slug is not None
    assert meta_direct.class_name == "NarrativePotentialTransformer"


def test_registry_suggests_for_unknown_names():
    registry = get_transformer_registry()
    missing = registry.describe_missing(["NarrativePotental", "totally_unknown"])
    assert "NarrativePotental" in missing
    assert any("NarrativePotential" in suggestion for suggestion in missing["NarrativePotental"])
    assert "totally_unknown" in missing


def test_category_summary_has_content():
    registry = get_transformer_registry()
    summary = registry.summary_by_category()
    assert summary  # not empty
    assert sum(summary.values()) >= len(registry.class_names())

