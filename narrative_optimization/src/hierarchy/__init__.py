"""
Hierarchical Narrative Analysis Module

Tracks narratives across nested temporal scales:
- Moment → Game → Series → Season → Era

Stories emerge at each level, with lower stories feeding into higher.
"""

from .nested_narrative_tracker import NestedNarrativeTracker
from .story_accumulator import StoryAccumulator
from .emergence_detector import EmergenceDetector

__all__ = ['NestedNarrativeTracker', 'StoryAccumulator', 'EmergenceDetector']

