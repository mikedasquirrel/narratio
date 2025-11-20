"""
Semantic Transformers

Embedding-based transformers that use semantic similarity instead of hardcoded word lists.

Core principle: Learn meaning from embeddings, not dictionaries.
"""

from .emotional_semantic import EmotionalSemanticTransformer

__all__ = [
    'EmotionalSemanticTransformer',
]

