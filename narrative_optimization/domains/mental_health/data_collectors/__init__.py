"""
Data collectors for mental health domain.

Automated collection from:
- NIH RePORTER (funding)
- PubMed (article counts)
- CDC WONDER (mortality)
- ClinicalTrials.gov (active trials)
"""

from .nih_collector import NIHFundingCollector
from .pubmed_collector import PubMedCollector
from .cdc_collector import CDCMortalityCollector

__all__ = [
    'NIHFundingCollector',
    'PubMedCollector',
    'CDCMortalityCollector'
]

