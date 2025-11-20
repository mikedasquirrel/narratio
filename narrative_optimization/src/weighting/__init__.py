"""
Narrative Context Weighting

"Each game is a story. Story weight is by context. 
Better stories win over time, better ones over longer periods."

Not all narratives have equal predictive power. Context matters.
"""

from .narrative_context import (
    NarrativeContextWeighter,
    MultiHorizonNarrativeTester,
    StoryOutcomeAnalyzer
)

__all__ = [
    'NarrativeContextWeighter',
    'MultiHorizonNarrativeTester', 
    'StoryOutcomeAnalyzer'
]

