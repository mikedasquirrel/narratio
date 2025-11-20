"""
Mental Health Nomenclature Domain

Tests whether mental health disorder NAMES predict clinical outcomes, funding, and stigma.

Key Hypothesis: Harsh-sounding disorder names → worse stigma → less treatment seeking → worse outcomes

Research Foundation:
- 510 disorders with DSM/ICD codes
- Phonetic analysis (harshness, plosives, sibilants)
- Clinical framing (medical vs colloquial terminology)
- Social impact (stigma, discrimination, help-seeking)

Expected Effect: r ~ 0.29 (visibility=25%, narrative importance=high)
"""

from .data_loader import MentalHealthDataLoader
from .stigma_analyzer import StigmaAnalyzer

__all__ = [
    'MentalHealthDataLoader',
    'StigmaAnalyzer'
]

