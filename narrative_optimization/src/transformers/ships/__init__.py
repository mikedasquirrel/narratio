"""
Naval Ships Transformers

Transforms ship names into features that predict historical significance:
- Name gravitas (weight/importance)
- Purpose alignment (name fits mission)
- Temporal patterns

Expected effect: r ~ 0.18 (visibility=50%)
"""

from .gravitas_transformer import GravitasTransformer
from .purpose_alignment_transformer import PurposeAlignmentTransformer

__all__ = [
    'GravitasTransformer',
    'PurposeAlignmentTransformer'
]

