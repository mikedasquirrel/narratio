"""
Mental Health Nomenclature Transformers

Transforms disorder names into features that predict:
- Stigma levels
- Treatment seeking behavior
- Clinical outcomes

Key hypothesis: Harsh-sounding names → increased stigma → reduced help-seeking
"""

from .phonetic_severity_transformer import PhoneticSeverityTransformer
from .clinical_framing_transformer import ClinicalFramingTransformer
from .treatment_seeking_transformer import TreatmentSeekingTransformer

__all__ = [
    'PhoneticSeverityTransformer',
    'ClinicalFramingTransformer',
    'TreatmentSeekingTransformer'
]

