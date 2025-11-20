"""
Transformer Utilities

Shared intelligent mechanisms for all transformers:
- Embedding manager (sentence-transformers)
- LLM client wrapper (Claude/GPT)
- Zero-shot classifier
- Semantic similarity utilities
- Feature caching
- Shared model registry (90% RAM reduction)
"""

from .embeddings import EmbeddingManager
from .semantic_similarity import SemanticSimilarity
from .feature_cache import FeatureCache
from .shared_models import SharedModelRegistry, use_shared_models
from .feature_cache import get_default_cache, use_feature_cache

__all__ = [
    'EmbeddingManager',
    'SemanticSimilarity',
    'FeatureCache',
    'SharedModelRegistry',
    'use_shared_models',
    'get_default_cache',
    'use_feature_cache'
]

