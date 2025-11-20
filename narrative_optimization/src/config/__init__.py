"""
Configuration module for domain-specific narrative analysis.

Includes:
- Domain archetypal patterns and configuration
- Archetype discovery and learning tools
- Complete genome (ж) structure with historial and uniquity
"""

from .domain_archetypes import DOMAIN_ARCHETYPES, get_generic_archetype
from .domain_config import DomainConfig
from .archetype_discovery import (
    ArchetypeDiscovery,
    ArchetypeRegistry,
    ArchetypeValidator
)
from .genome_structure import (
    GenomeStructure,
    HistorialCalculator,
    UniquityCalculator,
    CompleteGenomeExtractor
)
from .advanced_archetype_discovery import (
    SemanticArchetypeDiscovery,
    HierarchicalArchetypeDiscovery,
    TemporalArchetypeEvolution,
    CrossDomainPatternTransfer
)
from .temporal_decay import (
    TemporalDecay,
    DecayType
)

__all__ = [
    'DOMAIN_ARCHETYPES',
    'get_generic_archetype',
    'DomainConfig',
    'ArchetypeDiscovery',
    'ArchetypeRegistry',
    'ArchetypeValidator',
    'GenomeStructure',
    'HistorialCalculator',
    'UniquityCalculator',
    'CompleteGenomeExtractor',
    'SemanticArchetypeDiscovery',
    'HierarchicalArchetypeDiscovery',
    'TemporalArchetypeEvolution',
    'CrossDomainPatternTransfer',
    'TemporalDecay',
    'DecayType'
]

