"""
Relational narrative transformer namespace.

Exports both the legacy opponent context helpers (module-local) and the
general-purpose RelationalValueTransformer defined in the sibling module
`relational.py`.  This keeps backwards compatibility for imports such as
`from transformers.relational import RelationalValueTransformer`.
"""

from .opponent_context import OpponentContextTransformer
from ..relational_value import RelationalValueTransformer

__all__ = ['OpponentContextTransformer', 'RelationalValueTransformer']
