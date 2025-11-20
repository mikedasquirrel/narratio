"""
Classical Archetype Transformers

Theory-guided empirical discovery of narrative archetypes.

Available Transformers:
- HeroJourneyTransformer: Campbell's Hero's Journey
- CharacterArchetypeTransformer: Jung, Vogler, Propp archetypes
- PlotArchetypeTransformer: Booker's plots + Polti's situations
- StructuralBeatTransformer: 3-act, 5-act, Save the Cat
- ThematicArchetypeTransformer: Frye's mythoi
- MythologicalPatternTransformer: Creation myths, divine intervention
- ScriptureParableTransformer: Parable structure, moral teaching
- FilmNarrativeTransformer: Visual storytelling patterns
- MusicNarrativeTransformer: Lyrical narrative structure
- LiteraryDeviceTransformer: Symbolism, metaphor, foreshadowing
"""

from .hero_journey import HeroJourneyTransformer, analyze_hero_journey, discover_journey_patterns
from .character_archetype import CharacterArchetypeTransformer, discover_archetype_patterns
from .plot_archetype import PlotArchetypeTransformer, discover_plot_patterns
from .structural_beat import StructuralBeatTransformer
from .thematic_archetype import ThematicArchetypeTransformer

__all__ = [
    'HeroJourneyTransformer',
    'CharacterArchetypeTransformer',
    'PlotArchetypeTransformer',
    'StructuralBeatTransformer',
    'ThematicArchetypeTransformer',
    'analyze_hero_journey',
    'discover_journey_patterns',
    'discover_archetype_patterns',
    'discover_plot_patterns',
]
