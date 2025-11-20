"""
Taxonomy System for Cross-Domain Narrative Analysis

Organizes domains along multiple dimensions:
- Visibility level (0-100%)
- Narrative importance (low/medium/high)
- Domain type (sports, natural, social, medical, historical)
- Effect size
- Study status

Enables cross-domain comparisons and meta-analysis.
"""

from .domain_registry import DomainRegistry
from .taxonomy_builder import TaxonomyBuilder
from .visibility_classifier import VisibilityClassifier
from .cross_domain_comparator import CrossDomainComparator

__all__ = [
    'DomainRegistry',
    'TaxonomyBuilder',
    'VisibilityClassifier',
    'CrossDomainComparator'
]
