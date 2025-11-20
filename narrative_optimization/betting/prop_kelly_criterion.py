"""
Prop Betting Kelly Criterion

Specialized Kelly sizing for player prop bets with:
- Lower base Kelly fractions (props are higher variance)
- Correlation adjustments (same-game props)
- Book limit awareness
- Dynamic sizing based on confidence

Props require more conservative sizing than game bets due to:
1. Higher variance outcomes (player performance)
2. Lower book limits
3. Sharper line movements
4. Correlation between props in same game

Author: Prop Risk Management System
Date: November 20, 2024
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PropBet:
    """Player prop bet details"""
    player_name: str
    game_id: str
    prop_type: str  # goals, assists, shots, saves
    line: float
    side: str  # over/under
    odds: int  # American odds
    probability: float  # Model probability
    edge: float  # probability - implied_prob
    confidence: float  # Model confidence (0-1)
    correlation_group: Optional[str] = None  # For same-game correlations
    book_limit: float = 500  # Default prop limit


@dataclass
class PropKellyResult:
    """Kelly calculation results for prop bet"""
    bet: PropBet
    kelly_fraction: float  # Full Kelly
    recommended_fraction: float  # Adjusted fraction
    bet_amount: float
    expected_value: float
    risk_score: float  # 0-1, higher = riskier
    adjustments: Dict[str, float]  # Applied adjustments
    reasoning: str


class PropKellyCriterion:
    """
    Kelly Criterion calculator specialized for NHL player props.
    
    Key differences from game Kelly:
    - Quarter Kelly as base (vs half Kelly for games)
    - Correlation penalties for same-game props
    - Book limit considerations
    - Confidence-based adjustments
    """
    
    def __init__(
        self,
        base_fraction: float = 0.25,  # Quarter Kelly for props
        max_prop_exposure: float = 0.10,  # Max 10% bankroll on props
        max_single_prop: float = 0.02,  # Max 2% per prop
        correlation_penalty: float = 0.5  # Reduce size by 50% for correlated props
    ):
        """
        Parameters
        ----------
        base_fraction : float
            Base Kelly fraction (0.25 = quarter Kelly)
        max_prop_exposure : float
            Maximum total prop exposure as fraction of bankroll
        max_single_prop : float
            Maximum bet on single prop as fraction of bankroll
        correlation_penalty : float
            Reduction factor for correlated props
        """
        self.base_fraction = base_fraction
        self.max_prop_exposure = max_prop_exposure
        self.max_single_prop = max_single_prop
        self.correlation_penalty = correlation_penalty
        
        # Prop-specific limits
        self.prop_limits = {
            'goals': {'min_edge': 0.04, 'max_bet_pct': 0.02},
            'assists': {'min_edge': 0.04, 'max_bet_pct': 0.015},
            'shots': {'min_edge': 0.03, 'max_bet_pct': 0.02},
            'points': {'min_edge': 0.04, 'max_bet_pct': 0.02},
            'saves': {'min_edge': 0.05, 'max_bet_pct': 0.015},
        }
        
    def calculate_prop_kelly(
        self,
        bet: PropBet,
        bankroll: float,
        existing_props: Optional[List[PropBet]] = None
    ) -> PropKellyResult:
        """
        Calculate optimal bet size for a prop bet.
        
        Parameters
        ----------
        bet : PropBet
            The prop bet to size
        bankroll : float
            Current bankroll
        existing_props : List[PropBet], optional
            Already placed prop bets (for correlation)
            
        Returns
        -------
        result : PropKellyResult
            Sizing recommendation with reasoning
        """
        # Get decimal odds
        decimal_odds = self._american_to_decimal(bet.odds)
        
        # Calculate full Kelly fraction
        # f = (p * b - q) / b
        # where p = win prob, q = lose prob, b = net odds
        p = bet.probability
        q = 1 - p
        b = decimal_odds - 1  # Net odds (profit on $1 bet)
        
        full_kelly = (p * b - q) / b if b > 0 else 0
        
        # Start with base fraction
        recommended_fraction = full_kelly * self.base_fraction
        adjustments = {'base_reduction': self.base_fraction}
        
        # Apply prop-type specific limits
        prop_config = self.prop_limits.get(bet.prop_type, {})
        min_edge = prop_config.get('min_edge', 0.04)
        max_bet_pct = prop_config.get('max_bet_pct', self.max_single_prop)
        
        if bet.edge < min_edge:
            recommended_fraction = 0
            adjustments['edge_threshold'] = 0
            
        # Confidence adjustment
        confidence_mult = self._confidence_multiplier(bet.confidence)
        recommended_fraction *= confidence_mult
        adjustments['confidence'] = confidence_mult
        
        # Correlation adjustment
        if existing_props:
            correlation_mult = self._correlation_adjustment(bet, existing_props)
            recommended_fraction *= correlation_mult
            adjustments['correlation'] = correlation_mult
            
        # Apply maximums
        recommended_fraction = min(recommended_fraction, max_bet_pct)
        
        # Check total prop exposure
        if existing_props:
            current_exposure = sum(
                p.probability * self._american_to_decimal(p.odds) - 1
                for p in existing_props
            ) / bankroll
            
            if current_exposure + recommended_fraction > self.max_prop_exposure:
                recommended_fraction = max(0, self.max_prop_exposure - current_exposure)
                adjustments['exposure_limit'] = recommended_fraction
                
        # Calculate bet amount
        bet_amount = bankroll * recommended_fraction
        
        # Apply book limits
        if bet_amount > bet.book_limit:
            bet_amount = bet.book_limit
            recommended_fraction = bet_amount / bankroll
            adjustments['book_limit'] = bet.book_limit
            
        # Calculate expected value
        ev = bet_amount * bet.edge
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(bet, recommended_fraction, existing_props)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(bet, adjustments, risk_score)
        
        return PropKellyResult(
            bet=bet,
            kelly_fraction=full_kelly,
            recommended_fraction=recommended_fraction,
            bet_amount=round(bet_amount, 2),
            expected_value=round(ev, 2),
            risk_score=risk_score,
            adjustments=adjustments,
            reasoning=reasoning
        )
        
    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust Kelly fraction based on model confidence.
        
        Low confidence = smaller bets
        """
        if confidence >= 0.7:
            return 1.0
        elif confidence >= 0.6:
            return 0.8
        elif confidence >= 0.5:
            return 0.6
        else:
            return 0.4
            
    def _correlation_adjustment(self, bet: PropBet, existing_props: List[PropBet]) -> float:
        """
        Reduce bet size for correlated props.
        
        Same game = high correlation
        Same player = very high correlation
        """
        multiplier = 1.0
        
        for existing in existing_props:
            # Same game correlation
            if bet.game_id == existing.game_id:
                multiplier *= (1 - self.correlation_penalty * 0.5)
                
                # Same player is even more correlated
                if bet.player_name == existing.player_name:
                    multiplier *= (1 - self.correlation_penalty)
                    
        return max(multiplier, 0.2)  # Don't reduce more than 80%
        
    def _calculate_risk_score(self, bet: PropBet, fraction: float, 
                            existing_props: Optional[List[PropBet]]) -> float:
        """Calculate risk score (0-1, higher = riskier)"""
        risk = 0.0
        
        # Edge risk (smaller edge = higher risk)
        if bet.edge < 0.05:
            risk += 0.3
        elif bet.edge < 0.08:
            risk += 0.2
        elif bet.edge < 0.10:
            risk += 0.1
            
        # Confidence risk
        if bet.confidence < 0.6:
            risk += 0.2
        elif bet.confidence < 0.65:
            risk += 0.1
            
        # Prop type risk
        risky_props = {'goals': 0.15, 'saves': 0.1}  # Higher variance
        risk += risky_props.get(bet.prop_type, 0)
        
        # Correlation risk
        if existing_props:
            same_game = sum(1 for p in existing_props if p.game_id == bet.game_id)
            risk += same_game * 0.1
            
        return min(risk, 1.0)
        
    def _generate_reasoning(self, bet: PropBet, adjustments: Dict, risk_score: float) -> str:
        """Generate explanation for bet sizing"""
        reasons = []
        
        # Base fraction
        reasons.append(f"Quarter Kelly base ({self.base_fraction:.0%})")
        
        # Confidence
        if adjustments.get('confidence', 1.0) < 1.0:
            reasons.append(f"Confidence adjustment ({adjustments['confidence']:.0%})")
            
        # Correlation
        if adjustments.get('correlation', 1.0) < 1.0:
            reasons.append(f"Correlation penalty ({adjustments['correlation']:.0%})")
            
        # Limits
        if 'book_limit' in adjustments:
            reasons.append(f"Book limit (${adjustments['book_limit']})")
            
        # Risk
        if risk_score >= 0.5:
            reasons.append(f"High risk score ({risk_score:.2f})")
        elif risk_score >= 0.3:
            reasons.append(f"Moderate risk ({risk_score:.2f})")
            
        return " | ".join(reasons)
        
    def _american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal"""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
            
    def optimize_prop_portfolio(
        self,
        props: List[PropBet],
        bankroll: float,
        max_props: int = 10
    ) -> List[PropKellyResult]:
        """
        Optimize a portfolio of props considering correlations.
        
        Parameters
        ----------
        props : List[PropBet]
            All available prop bets
        bankroll : float
            Current bankroll
        max_props : int
            Maximum props to include
            
        Returns
        -------
        portfolio : List[PropKellyResult]
            Optimized prop portfolio
        """
        # Sort by expected value
        props.sort(key=lambda p: p.edge * p.confidence, reverse=True)
        
        portfolio = []
        total_exposure = 0.0
        
        for prop in props:
            if len(portfolio) >= max_props:
                break
                
            # Calculate sizing
            result = self.calculate_prop_kelly(prop, bankroll, portfolio)
            
            # Skip if no bet recommended
            if result.recommended_fraction == 0:
                continue
                
            # Check if adding would exceed limits
            new_exposure = total_exposure + result.recommended_fraction
            if new_exposure > self.max_prop_exposure:
                continue
                
            portfolio.append(result)
            total_exposure = new_exposure
            
        return portfolio


def calculate_prop_kelly_sizing(
    props: List[Dict],
    bankroll: float = 10000
) -> List[Dict]:
    """
    Convenience function to calculate Kelly sizing for props.
    
    Parameters
    ----------
    props : List[Dict]
        Prop predictions with edge calculations
    bankroll : float
        Current bankroll
        
    Returns
    -------
    sized_props : List[Dict]
        Props with Kelly sizing recommendations
    """
    kelly = PropKellyCriterion()
    
    # Convert to PropBet objects
    prop_bets = []
    for prop in props:
        bet = PropBet(
            player_name=prop['player_name'],
            game_id=prop['game_id'],
            prop_type=prop['prop_type'],
            line=prop['line'],
            side=prop['side'],
            odds=prop['odds'],
            probability=prop.get('prob_over' if prop['side'] == 'over' else 'prob_under'),
            edge=prop['edge'],
            confidence=prop.get('confidence', 0.6)
        )
        prop_bets.append(bet)
        
    # Optimize portfolio
    portfolio = kelly.optimize_prop_portfolio(prop_bets, bankroll)
    
    # Convert back to dicts
    sized_props = []
    for result in portfolio:
        sized_prop = {
            **prop,
            'kelly_fraction': result.kelly_fraction,
            'recommended_fraction': result.recommended_fraction,
            'bet_amount': result.bet_amount,
            'expected_value': result.expected_value,
            'risk_score': result.risk_score,
            'sizing_reasoning': result.reasoning
        }
        sized_props.append(sized_prop)
        
    return sized_props


def test_prop_kelly():
    """Test prop Kelly calculations"""
    print("PROP KELLY CRITERION TEST")
    print("=" * 80)
    
    # Test props
    test_props = [
        PropBet(
            player_name="Auston Matthews",
            game_id="20241120-BOS-TOR",
            prop_type="goals",
            line=0.5,
            side="over",
            odds=-115,
            probability=0.58,
            edge=0.045,
            confidence=0.68
        ),
        PropBet(
            player_name="Mitch Marner",
            game_id="20241120-BOS-TOR",
            prop_type="assists",
            line=0.5,
            side="over",
            odds=+105,
            probability=0.52,
            edge=0.032,
            confidence=0.61
        ),
        PropBet(
            player_name="Auston Matthews",
            game_id="20241120-BOS-TOR",
            prop_type="shots",
            line=3.5,
            side="over",
            odds=-120,
            probability=0.57,
            edge=0.025,
            confidence=0.59
        ),
    ]
    
    kelly = PropKellyCriterion()
    bankroll = 10000
    
    print(f"\nBankroll: ${bankroll:,}")
    print(f"Base Kelly: {kelly.base_fraction:.0%} (Quarter Kelly)")
    print(f"Max prop exposure: {kelly.max_prop_exposure:.0%}")
    
    # Calculate individual sizings
    print("\n" + "-" * 80)
    print("INDIVIDUAL PROP SIZING")
    print("-" * 80)
    
    for prop in test_props:
        result = kelly.calculate_prop_kelly(prop, bankroll)
        
        print(f"\n{prop.player_name} - {prop.prop_type} {prop.side} {prop.line}")
        print(f"  Odds: {prop.odds:+d} | Our prob: {prop.probability:.1%} | Edge: {prop.edge:.1%}")
        print(f"  Full Kelly: {result.kelly_fraction:.3%}")
        print(f"  Recommended: {result.recommended_fraction:.3%} = ${result.bet_amount:.2f}")
        print(f"  Risk score: {result.risk_score:.2f}")
        print(f"  Reasoning: {result.reasoning}")
        
    # Portfolio optimization
    print("\n" + "-" * 80)
    print("PORTFOLIO OPTIMIZATION")
    print("-" * 80)
    
    portfolio = kelly.optimize_prop_portfolio(test_props, bankroll, max_props=5)
    
    total_bet = sum(r.bet_amount for r in portfolio)
    total_ev = sum(r.expected_value for r in portfolio)
    
    print(f"\nOptimal portfolio: {len(portfolio)} props")
    print(f"Total wagered: ${total_bet:.2f} ({total_bet/bankroll:.1%} of bankroll)")
    print(f"Expected value: ${total_ev:.2f}")
    
    for i, result in enumerate(portfolio):
        print(f"\n{i+1}. {result.bet.player_name} - {result.bet.prop_type} "
              f"{result.bet.side} {result.bet.line}: ${result.bet_amount:.2f}")
        

if __name__ == "__main__":
    test_prop_kelly()
