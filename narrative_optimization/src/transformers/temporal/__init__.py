"""
Temporal Dynamics Transformers

Phase 1 implementation of temporal dynamics analysis:
- τ (tau): Duration ratios
- ς (sigma): Compression ratios  
- ρ (rho): Temporal rhythm

Theory formalized in TEMPORAL_DYNAMICS_THEORY.md

Transformers:
1. TemporalCompressionTransformer: Extracts τ, ς, ρ (60 features)
2. DurationEffectsTransformer: Duration accessibility and constraints (45 features)
3. PacingRhythmTransformer: Optimal pacing and beat placement (50 features)
4. CrossTemporalIsomorphismTransformer: Structural equivalence across timescales (40 features)
5. TemporalNarrativeCyclesTransformer: Revenge, milestone, and momentum patterns (40 features)
6. CalendarRhythmTransformer: Calendar positioning and day/month/holiday patterns (30 features)
7. SeasonSeriesNarrativeTransformer: Season series evolution and rivalry dynamics (20 features)

Total: 285 temporal dynamics features

Author: Narrative Optimization Framework
Date: November 2025
"""

from .temporal_compression import TemporalCompressionTransformer
from .duration_effects import DurationEffectsTransformer
from .pacing_rhythm import PacingRhythmTransformer
from .cross_temporal_isomorphism import CrossTemporalIsomorphismTransformer
from .narrative_cycles import TemporalNarrativeCyclesTransformer
from .calendar_rhythm import CalendarRhythmTransformer
from .season_series import SeasonSeriesNarrativeTransformer

__all__ = [
    'TemporalCompressionTransformer',
    'DurationEffectsTransformer',
    'PacingRhythmTransformer',
    'CrossTemporalIsomorphismTransformer',
    'TemporalNarrativeCyclesTransformer',
    'CalendarRhythmTransformer',
    'SeasonSeriesNarrativeTransformer',
]

