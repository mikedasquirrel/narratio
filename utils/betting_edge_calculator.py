"""
Betting Edge Calculator

Converts betting odds to implied probability and calculates edge.
CRITICAL for real betting - we need to beat the market, not just predict winners.

Functions:
- odds_to_probability: Convert American odds to implied probability
- calculate_edge: Model probability - market probability
- calculate_expected_value: Expected profit/loss per bet
- kelly_criterion: Optimal bet sizing

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

def american_odds_to_probability(odds):
    """
    Convert American odds to implied probability.
    
    Examples:
    - Favorite odds (-150): Implies 60% win probability
    - Underdog odds (+200): Implies 33.3% win probability
    
    Parameters
    ----------
    odds : int
        American odds (e.g., -150, +200)
    
    Returns
    -------
    probability : float
        Implied win probability (0-1)
    """
    if odds < 0:
        # Favorite: probability = |odds| / (|odds| + 100)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog: probability = 100 / (odds + 100)
        return 100 / (odds + 100)


def calculate_edge(model_probability, odds):
    """
    Calculate betting edge: model probability - market implied probability.
    
    Positive edge = model thinks team/player is better than market does → BET
    Negative edge = market thinks team/player is better than model does → SKIP
    
    Parameters
    ----------
    model_probability : float
        Model's predicted win probability (0-1)
    odds : int
        American odds
    
    Returns
    -------
    edge : float
        Edge in percentage points (e.g., 0.13 = +13% edge)
    """
    market_prob = american_odds_to_probability(odds)
    edge = model_probability - market_prob
    return edge


def calculate_expected_value(model_probability, odds, stake=100):
    """
    Calculate expected value of a bet.
    
    EV = (probability_win × payout) - (probability_lose × stake)
    
    Positive EV → profitable long-term
    Negative EV → losing proposition
    
    Parameters
    ----------
    model_probability : float
        Model's predicted win probability
    odds : int
        American odds
    stake : float
        Bet amount in dollars
    
    Returns
    -------
    expected_value : float
        Expected profit/loss per bet
    """
    if odds < 0:
        # Favorite: payout = stake / (|odds| / 100)
        payout = stake * (100 / abs(odds))
    else:
        # Underdog: payout = stake * (odds / 100)
        payout = stake * (odds / 100)
    
    prob_win = model_probability
    prob_lose = 1 - model_probability
    
    ev = (prob_win * payout) - (prob_lose * stake)
    return ev


def kelly_criterion(model_probability, odds):
    """
    Calculate optimal bet size using Kelly Criterion.
    
    f* = (bp - q) / b
    
    Where:
    - f* = fraction of bankroll to bet
    - b = decimal odds - 1
    - p = model probability of winning
    - q = 1 - p
    
    Parameters
    ----------
    model_probability : float
        Model's predicted win probability
    odds : int
        American odds
    
    Returns
    -------
    kelly_fraction : float
        Optimal fraction of bankroll to bet (0-1)
        Return 0 if no edge (don't bet)
    """
    # Convert to decimal odds
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)
    
    b = decimal_odds - 1
    p = model_probability
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Don't bet if Kelly is negative
    return max(0, kelly)


def betting_recommendation(model_probability, odds, min_edge=0.05, stake=100):
    """
    Generate complete betting recommendation with edge analysis.
    
    Parameters
    ----------
    model_probability : float
        Model's predicted win probability
    odds : int
        American odds
    min_edge : float
        Minimum edge to recommend betting (default 5%)
    stake : float
        Base bet amount
    
    Returns
    -------
    recommendation : dict
        Complete betting analysis
    """
    market_prob = american_odds_to_probability(odds)
    edge = calculate_edge(model_probability, odds)
    ev = calculate_expected_value(model_probability, odds, stake)
    kelly = kelly_criterion(model_probability, odds)
    
    # Betting decision
    if edge >= min_edge:
        action = "BET"
        confidence = "HIGH" if edge >= 0.10 else "MEDIUM"
    elif edge > 0:
        action = "MARGINAL"
        confidence = "LOW"
    else:
        action = "SKIP"
        confidence = "NONE"
    
    # Kelly sizing (use fractional Kelly for safety)
    fractional_kelly = kelly * 0.25  # Quarter Kelly (conservative)
    recommended_stake = stake * fractional_kelly if edge > 0 else 0
    
    return {
        'model_probability': round(model_probability, 3),
        'market_implied_probability': round(market_prob, 3),
        'edge_percentage': round(edge * 100, 2),
        'expected_value': round(ev, 2),
        'kelly_full': round(kelly, 3),
        'kelly_quarter': round(fractional_kelly, 3),
        'recommended_stake': round(recommended_stake, 2),
        'action': action,
        'confidence': confidence,
        'reasoning': f"Model: {model_probability:.1%}, Market: {market_prob:.1%}, Edge: {edge:+.1%}"
    }


# Example usage:
if __name__ == '__main__':
    # Example: Model thinks Lakers have 73% chance to win
    # Market odds are -150 (implies 60%)
    
    rec = betting_recommendation(
        model_probability=0.73,
        odds=-150,
        stake=100
    )
    
    print("Betting Analysis:")
    print(f"  Model Probability: {rec['model_probability']:.1%}")
    print(f"  Market Probability: {rec['market_implied_probability']:.1%}")
    print(f"  EDGE: {rec['edge_percentage']:+.1f}%")
    print(f"  Expected Value: ${rec['expected_value']:+.2f}")
    print(f"  Kelly (1/4): {rec['kelly_quarter']:.1%} of bankroll")
    print(f"  Recommended Stake: ${rec['recommended_stake']:.2f}")
    print(f"  ACTION: {rec['action']}")

