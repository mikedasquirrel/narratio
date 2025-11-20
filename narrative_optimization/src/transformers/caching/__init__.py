"""
Caching infrastructure for narrative transformers.

Provides disk-based caching with version tracking and automatic invalidation.
"""

from .feature_cache import FeatureCache

__all__ = ['FeatureCache']

