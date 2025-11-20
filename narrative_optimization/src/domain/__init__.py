"""
Domain Expertise and Detection System

Tracks what domains the system has been trained on and
assesses confidence based on familiarity with domain collages.
"""

from .expertise_tracker import DomainExpertiseTracker
from .domain_detector import MultiDomainDetector
from .confidence_calibrator import ConfidenceCalibrator

__all__ = [
    'DomainExpertiseTracker',
    'MultiDomainDetector',
    'ConfidenceCalibrator'
]

