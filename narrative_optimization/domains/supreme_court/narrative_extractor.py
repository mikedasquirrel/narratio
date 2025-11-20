"""
Supreme Court narrative extractor.

Ensures we can analyze Supreme Court opinions with the universal processor by:
- Pulling the majority opinion (or full text fallback)
- Enforcing minimum narrative length
- Extracting citation counts from nested metadata/outcome blocks
- Returning clean arrays for downstream analysis
"""

from typing import Iterable, List, Tuple, Dict, Any
import numpy as np

MIN_NARRATIVE_LENGTH = 500


def _flatten_cases(data: Any) -> List[Dict]:
    """Normalize supreme court dataset into a list of case dicts."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        cases: List[Dict] = []
        for value in data.values():
            if isinstance(value, list):
                cases.extend([item for item in value if isinstance(item, dict)])
            elif isinstance(value, dict):
                cases.append(value)
        return cases
    raise ValueError(f"Unsupported Supreme Court data format: {type(data)}")


def _extract_narrative(case: Dict) -> str:
    """Return the best available narrative text for a case."""
    narrative_fields = [
        'majority_opinion',
        'opinion_full_text',
        'concurring_opinion',
        'dissenting_opinion',
        'plurality_opinion',
    ]
    for field in narrative_fields:
        text = case.get(field) or ''
        if text and len(text) >= MIN_NARRATIVE_LENGTH:
            return text.strip()
    return ''


def _extract_citation_count(case: Dict) -> float:
    """Pull citation count from any known location within the case payload."""
    direct = case.get('citation_count')
    if direct not in (None, ''):
        return float(direct)
    
    outcome_block = case.get('outcome') or {}
    if isinstance(outcome_block, dict):
        citation = outcome_block.get('citation_count')
        if citation not in (None, ''):
            return float(citation)
    
    metadata_block = case.get('metadata') or {}
    if isinstance(metadata_block, dict):
        citation = metadata_block.get('citation_count')
        if citation not in (None, ''):
            return float(citation)
    
    return float('nan')


def extract_supreme_court_narratives(data: Iterable[Dict]) -> Tuple[List[str], np.ndarray, int]:
    """
    Custom extractor for Supreme Court cases.
    
    Parameters
    ----------
    data : iterable or dict
        Raw dataset loaded from data/domains/supreme_court_complete.json
    
    Returns
    -------
    narratives : list of str
        Majority or full opinion texts
    outcomes : np.ndarray
        Citation counts (continuous outcome)
    total_count : int
        Total number of raw records examined
    """
    cases = _flatten_cases(data)
    total_count = len(cases)
    
    narratives: List[str] = []
    outcomes: List[float] = []
    
    for case in cases:
        narrative = _extract_narrative(case)
        if not narrative:
            continue
        
        citation_count = _extract_citation_count(case)
        if np.isnan(citation_count):
            continue
        
        narratives.append(narrative)
        outcomes.append(float(citation_count))
    
    return narratives, np.asarray(outcomes, dtype=float), total_count


