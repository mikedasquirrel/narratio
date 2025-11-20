"""
Betting Utilities
==================

Helper functions for betting calculations, odds conversion, and EV analysis.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import numpy as np
from typing import Dict, Tuple, Optional


def american_to_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.
    
    Parameters
    ----------
    odds : float
        American odds (e.g., -150, +200)
        
    Returns
    -------
    probability : float
        Implied probability (0-1)
    """
    if odds < 0:
        # Favorite
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog
        return 100 / (odds + 100)


def odds_to_probability(odds: float, format: str = 'american') -> float:
    """
    Convert odds to probability.
    
    Parameters
    ----------
    odds : float
        Odds value
    format : str
        'american', 'decimal', or 'fractional'
        
    Returns
    -------
    probability : float
    """
    if format == 'american':
        return american_to_probability(odds)
    elif format == 'decimal':
        return 1 / odds
    elif format == 'fractional':
        return 1 / (odds + 1)
    else:
        raise ValueError(f"Unknown format: {format}")


def calculate_ev(
    win_prob: float,
    odds: float,
    stake: float = 1.0,
    odds_format: str = 'american'
) -> float:
    """
    Calculate expected value of a bet.
    
    Parameters
    ----------
    win_prob : float
        Model's win probability (0-1)
    odds : float
        Market odds
    stake : float
        Bet size (default 1 unit)
    odds_format : str
        Odds format
        
    Returns
    -------
    ev : float
        Expected value in units
    """
    if odds_format == 'american':
        if odds < 0:
            win_amount = stake * (100 / abs(odds))
        else:
            win_amount = stake * (odds / 100)
    elif odds_format == 'decimal':
        win_amount = stake * (odds - 1)
    else:
        raise ValueError(f"Unknown format: {odds_format}")
    
    ev = (win_prob * win_amount) - ((1 - win_prob) * stake)
    return ev


def calculate_edge(
    model_prob: float,
    market_odds: float,
    odds_format: str = 'american'
) -> float:
    """
    Calculate edge: difference between model probability and market probability.
    
    Parameters
    ----------
    model_prob : float
        Model's win probability
    market_odds : float
        Market odds
    odds_format : str
        Odds format
        
    Returns
    -------
    edge : float
        Edge in probability units (positive = favorable)
    """
    market_prob = odds_to_probability(market_odds, odds_format)
    return model_prob - market_prob


def calculate_kelly_size(
    edge: float,
    win_prob: float,
    odds: float,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    odds_format: str = 'american'
) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.
    
    Parameters
    ----------
    edge : float
        Edge in probability
    win_prob : float
        Win probability
    odds : float
        Market odds
    bankroll : float
        Total bankroll
    kelly_fraction : float
        Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
    odds_format : str
        Odds format
        
    Returns
    -------
    bet_size : float
        Optimal bet size
    """
    if odds_format == 'american':
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
        else:
            decimal_odds = 1 + (odds / 100)
    elif odds_format == 'decimal':
        decimal_odds = odds
    else:
        raise ValueError(f"Unknown format: {odds_format}")
    
    # Kelly formula: f = (bp - q) / b
    # where b = decimal_odds - 1, p = win_prob, q = 1 - win_prob
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fraction and ensure positive
    kelly = max(0, kelly * kelly_fraction)
    
    # Convert to bet size
    bet_size = kelly * bankroll
    
    # Cap at reasonable maximum (10% of bankroll)
    bet_size = min(bet_size, bankroll * 0.10)
    
    return bet_size


def should_bet(
    model_prob: float,
    market_odds: float,
    min_edge: float = 0.05,
    min_confidence: float = 0.60,
    odds_format: str = 'american'
) -> Tuple[bool, str]:
    """
    Determine if a bet should be placed.
    
    Parameters
    ----------
    model_prob : float
        Model's win probability
    market_odds : float
        Market odds
    min_edge : float
        Minimum edge required (0.05 = 5%)
    min_confidence : float
        Minimum confidence required (0.60 = 60%)
    odds_format : str
        Odds format
        
    Returns
    -------
    should_bet : bool
        Whether to place bet
    reason : str
        Reasoning
    """
    # Check confidence
    if model_prob < min_confidence:
        return False, f"Confidence too low ({model_prob:.1%} < {min_confidence:.1%})"
    
    # Check edge
    edge = calculate_edge(model_prob, market_odds, odds_format)
    if edge < min_edge:
        return False, f"Edge too small ({edge:+.1%} < {min_edge:+.1%})"
    
    # Check EV
    ev = calculate_ev(model_prob, market_odds, stake=1.0, odds_format=odds_format)
    if ev <= 0:
        return False, f"Negative EV ({ev:+.2f})"
    
    return True, f"PASS: {model_prob:.1%} prob, {edge:+.1%} edge, {ev:+.2f} EV"


def categorize_confidence(prob: float) -> str:
    """Categorize confidence level"""
    if prob >= 0.70:
        return "MAXIMUM"
    elif prob >= 0.65:
        return "STRONG"
    elif prob >= 0.60:
        return "STANDARD"
    else:
        return "SKIP"


def calculate_bet_units(prob: float, base_unit: float = 1.0) -> float:
    """
    Calculate bet size in units based on confidence.
    
    Parameters
    ----------
    prob : float
        Win probability
    base_unit : float
        Base bet size
        
    Returns
    -------
    units : float
        Bet size in units
    """
    if prob >= 0.70:
        return base_unit * 2.0  # Maximum conviction
    elif prob >= 0.65:
        return base_unit * 1.5  # Strong
    elif prob >= 0.60:
        return base_unit * 1.0  # Standard
    else:
        return 0.0  # No bet


def format_odds_display(odds: float, format: str = 'american') -> str:
    """Format odds for display"""
    if format == 'american':
        if odds > 0:
            return f"+{int(odds)}"
        else:
            return str(int(odds))
    elif format == 'decimal':
        return f"{odds:.2f}"
    else:
        return str(odds)

