"""
Legal Narrative Transformers

Specialized transformers for legal narratives (Supreme Court opinions, briefs, arguments).

These transformers capture legal-specific narrative patterns:
- Argumentative structure (claim-evidence-warrant)
- Precedential narrative (citation patterns, authority)
- Persuasive framing (emotional/logical appeals, rights framing)
- Judicial rhetoric (formality, certainty, voice)

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

from .argumentative_structure import ArgumentativeStructureTransformer
from .precedential_narrative import PrecedentialNarrativeTransformer
from .persuasive_framing import PersuasiveFramingTransformer
from .judicial_rhetoric import JudicialRhetoricTransformer

__all__ = [
    'ArgumentativeStructureTransformer',
    'PrecedentialNarrativeTransformer',
    'PersuasiveFramingTransformer',
    'JudicialRhetoricTransformer'
]

