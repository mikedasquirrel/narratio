"""Pipeline modules for narrative optimization."""

from .feature_extraction_pipeline import FeatureExtractionPipeline
from .feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

# Legacy pipeline imports (may not exist)
try:
    from .weighted_integration import WeightedFeatureUnion, StackedNarrativeModel
    __all__ = [
        'FeatureExtractionPipeline',
        'SupervisedFeatureExtractionPipeline',
        'WeightedFeatureUnion',
        'StackedNarrativeModel'
    ]
except ImportError:
    __all__ = ['FeatureExtractionPipeline', 'SupervisedFeatureExtractionPipeline']

try:
    from .cached_pipeline import CachedTransformerPipeline
    __all__.append('CachedTransformerPipeline')
except ImportError:
    pass
