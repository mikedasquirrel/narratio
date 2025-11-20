"""
Registry Systems

Domain registry and version management.

Author: Narrative Integration System
Date: November 2025
"""

from .domain_registry import (
    DomainRegistry,
    DomainEntry,
    get_domain_registry,
    register_domain,
    list_all_domains
)

__all__ = [
    'DomainRegistry',
    'DomainEntry',
    'get_domain_registry',
    'register_domain',
    'list_all_domains'
]

