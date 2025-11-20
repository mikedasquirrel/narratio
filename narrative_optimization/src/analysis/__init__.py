"""
Core Analysis Modules

Implements complete variable system: ж, ю, ❊, Д, п, μ, ф, ة, Ξ

Following formal theoretical framework exactly.
"""

from .story_quality import StoryQualityCalculator
from .bridge_calculator import BridgeCalculator
from .gravitational_forces import GravitationalCalculator
# from .golden_narratio import GoldenNarratio  # TODO: Implement or remove
from .universal_analyzer import UniversalDomainAnalyzer

# New renovation modules
try:
    from .multi_stream_narrative_processor import MultiStreamNarrativeProcessor
    from .nested_temporal_scales import NestedTemporalAnalyzer
    from .sequential_narrative_processor import SequentialNarrativeProcessor
    from .unsupervised_narrative_discovery import UnsupervisedNarrativeDiscovery
except ImportError:
    pass  # Optional imports

__all__ = [
    'StoryQualityCalculator',
    'BridgeCalculator',
    'GravitationalCalculator',
    # 'GoldenNarratio',
    'UniversalDomainAnalyzer'
]
