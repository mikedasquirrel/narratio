"""
NBA domain module for narrative-driven game prediction.

Tests whether narrative patterns in team descriptions, media coverage,
and historical framing can predict game outcomes and create betting value.
"""

from .data_collector import NBADataCollector
from .narrative_extractor import NBANarrativeExtractor
from .game_predictor import NBAGamePredictor
from .betting_strategy import NBABettingStrategy
from .backtester import NBABacktester

__all__ = [
    'NBADataCollector',
    'NBANarrativeExtractor',
    'NBAGamePredictor',
    'NBABettingStrategy',
    'NBABacktester'
]

