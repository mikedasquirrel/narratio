"""
Custom sklearn transformers for narrative feature engineering.

Updated: November 2025 – now uses **lazy imports** so TensorFlow/Torch are only
loaded if a specific transformer demands them. This prevents macOS mutex deadlocks.
"""

from importlib import import_module
from datetime import datetime
import os

from .base import NarrativeTransformer
from .registry import get_transformer_registry

_IMPORT_LOGGING_ENABLED = os.environ.get("TRANSFORMER_IMPORT_LOGS", "0").lower() not in {"", "0", "false", "no"}
_REGISTRY = get_transformer_registry()
AVAILABLE_TRANSFORMERS = tuple(_REGISTRY.class_names())

__all__ = ["NarrativeTransformer", "AVAILABLE_TRANSFORMERS"] + list(AVAILABLE_TRANSFORMERS)


def _log_import(message: str) -> None:
    if _IMPORT_LOGGING_ENABLED:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[transformers][{timestamp}] {message}")


def _raise_not_found(name: str) -> None:
    suggestions = _REGISTRY.suggest(name)
    suggestion_text = ""
    if suggestions:
        suggestion_text = f" Did you mean: {', '.join(suggestions)}?"
    raise AttributeError(
        f"module '{__name__}' has no transformer '{name}'."
        f"{suggestion_text} Run "
        "'python -m narrative_optimization.tools.list_transformers' to inspect the catalog."
    )


def _load_transformer_class(name: str):
    metadata = _REGISTRY.resolve(name)
    if not metadata:
        _raise_not_found(name)
    module = import_module(metadata.module_path)
    cls = getattr(module, metadata.class_name)
    globals()[metadata.class_name] = cls
    globals()[name] = cls
    _log_import(f"Loaded {metadata.class_name} from {metadata.module_path}")
    return cls


def __getattr__(name: str):
    return _load_transformer_class(name)


def __dir__():
    return sorted(__all__)
