"""
MLB Betting Strategy - Kelly Criterion Bet Sizing
Determines when and how much to bet based on model predictions vs market odds

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
from typing import Dict, Optional, List


class MLBBettingStrategy:
    """Kelly criterion betting strategy for MLB"""
    
    def __init__(self, bankroll: float = 1000.0, kelly_fraction: float = 0.25,
                 min_edge: float = 0.05, max_bet_pct: float = 0.05):
        """
        Initialize betting strategy
        
        Args:
            bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly to bet (0.25 = quarter Kelly)
            min_edge: Minimum edge required to place bet (5%)
            max_bet_pct: Maximum bet as % of bankroll (5%)
        """
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        
        # Tracking
        self.bets_placed = []
        self.total_wagered = 0
        self.total_won = 0
        self.total_lost = 0
    
    def calculate_edge(self, model_prob: float, market_odds: float, side: str = 'home') -> float:
        """
        Calculate expected value edge
        
        Args:
            model_prob: Model's probability for outcome
            market_odds: American odds (e.g., -110, +150)
            side: 'home' or 'away'
            
        Returns:
            Edge as decimal (e.g., 0.10 = 10% edge)
        """
        # Convert American odds to implied probability
        if market_odds < 0:
            implied_prob = abs(market_odds) / (abs(market_odds) + 100)
        else:
            implied_prob = 100 / (market_odds + 100)
        
        # Edge = model probability - market probability
        edge = model_prob - implied_prob
        
        return edge
    
    def kelly_bet_size(self, model_prob: float, market_odds: float) -> float:
        """
        Calculate Kelly criterion bet size
        
        Args:
            model_prob: Model's win probability
            market_odds: American odds
            
        Returns:
            Bet size as fraction of bankroll
        """
        # Convert odds to decimal
        if market_odds < 0:
            decimal_odds = 1 + (100 / abs(market_odds))
        else:
            decimal_odds = 1 + (market_odds / 100)
        
        # Kelly formula: f = (bp - q) / b
        # where b = decimal odds - 1, p = win prob, q = 1 - p
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly = max(0, kelly * self.kelly_fraction)
        
        # Cap at max bet percentage
        kelly = min(kelly, self.max_bet_pct)
        
        return kelly
    
    def get_bet_recommendation(self, game_prediction: Dict, home_odds: float, 
                              away_odds: float, game_context: Dict = None) -> Optional[Dict]:
        """
        Get bet recommendation for a game (JOURNEY-AWARE)
        
        Args:
            game_prediction: Model prediction dictionary
            home_odds: American odds for home team
            away_odds: American odds for away team
            game_context: Journey completion score and context
            
        Returns:
            Bet recommendation dict or None if no bet
        """
        home_prob = game_prediction['home_win_probability']
        away_prob = game_prediction['away_win_probability']
        
        # TRANSFORMER-GUIDED: Adjust thresholds for high-journey games
        # MLB mean journey = 13.5%, high-journey = 20%+
        journey_score = game_context.get('journey_completion_score', 0) if game_context else 0
        
        min_edge = self.min_edge
        kelly_frac = self.kelly_fraction
        
        # High-journey games (above MLB mean of 0.135)
        if journey_score > 0.15:
            min_edge = 0.04  # Lower threshold (more bets on high-journey)
            kelly_frac = 0.30  # Increase bet size
            
        # Extreme journey games (top quartile - 20%+)
        if journey_score > 0.20:
            min_edge = 0.03  # Even lower threshold
            kelly_frac = 0.35  # Max aggression on peak narrative
        
        # Calculate edges for both sides
        home_edge = self.calculate_edge(home_prob, home_odds, 'home')
        away_edge = self.calculate_edge(away_prob, away_odds, 'away')
        
        # Find best edge
        if home_edge > away_edge and home_edge > min_edge:
            # Bet on home
            bet_amount = kelly_frac * self.current_bankroll
            
            return {
                'side': 'home',
                'probability': home_prob,
                'edge': home_edge,
                'odds': home_odds,
                'bet_amount': bet_amount,
                'kelly_fraction': kelly_frac,
                'expected_value': bet_amount * home_edge,
                'journey_score': journey_score,
                'journey_boosted': journey_score > 0.15
            }
        
        elif away_edge > min_edge:
            # Bet on away
            bet_amount = kelly_frac * self.current_bankroll
            
            return {
                'side': 'away',
                'probability': away_prob,
                'edge': away_edge,
                'odds': away_odds,
                'bet_amount': bet_amount,
                'kelly_fraction': kelly_frac,
                'expected_value': bet_amount * away_edge,
                'journey_score': journey_score,
                'journey_boosted': journey_score > 0.15
            }
        
        # No bet if edge below threshold
        return None
    
    def place_bet(self, recommendation: Dict, game_id: str, game_info: Dict = None) -> Dict:
        """
        Record a placed bet
        
        Args:
            recommendation: Bet recommendation dict
            game_id: Unique game identifier
            game_info: Optional game context
            
        Returns:
            Bet record
        """
        bet = {
            'game_id': game_id,
            'side': recommendation['side'],
            'amount': recommendation['bet_amount'],
            'odds': recommendation['odds'],
            'probability': recommendation['probability'],
            'edge': recommendation['edge'],
            'expected_value': recommendation['expected_value'],
            'bankroll_before': self.current_bankroll,
            'status': 'pending'
        }
        
        if game_info:
            bet.update(game_info)
        
        self.bets_placed.append(bet)
        self.total_wagered += bet['amount']
        
        return bet
    
    def settle_bet(self, game_id: str, actual_winner: str) -> Dict:
        """
        Settle a bet after game completes
        
        Args:
            game_id: Game identifier
            actual_winner: 'home' or 'away'
            
        Returns:
            Settlement details
        """
        # Find bet
        bet = next((b for b in self.bets_placed if b['game_id'] == game_id and b['status'] == 'pending'), None)
        
        if not bet:
            return {'error': 'Bet not found'}
        
        # Determine outcome
        won = (bet['side'] == actual_winner)
        
        if won:
            # Calculate payout
            odds = bet['odds']
            if odds < 0:
                payout = bet['amount'] * (100 / abs(odds))
            else:
                payout = bet['amount'] * (odds / 100)
            
            profit = payout
            self.current_bankroll += profit
            self.total_won += 1
        else:
            profit = -bet['amount']
            self.current_bankroll += profit  # Will be negative
            self.total_lost += 1
        
        # Update bet record
        bet['status'] = 'settled'
        bet['won'] = won
        bet['profit'] = profit
        bet['bankroll_after'] = self.current_bankroll
        
        return {
            'game_id': game_id,
            'won': won,
            'profit': profit,
            'new_bankroll': self.current_bankroll
        }
    
    def get_performance_stats(self) -> Dict:
        """Get overall performance statistics"""
        settled_bets = [b for b in self.bets_placed if b['status'] == 'settled']
        
        if not settled_bets:
            return {'error': 'No settled bets'}
        
        total_profit = sum(b['profit'] for b in settled_bets)
        roi = (total_profit / self.total_wagered) * 100 if self.total_wagered > 0 else 0
        
        win_rate = self.total_won / len(settled_bets) if settled_bets else 0
        
        # Calculate max drawdown
        bankroll_history = [self.initial_bankroll] + [b['bankroll_after'] for b in settled_bets]
        peak = bankroll_history[0]
        max_drawdown = 0
        
        for balance in bankroll_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_bets': len(settled_bets),
            'wins': self.total_won,
            'losses': self.total_lost,
            'win_rate': win_rate,
            'total_wagered': self.total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.current_bankroll,
            'return': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100,
            'max_drawdown': max_drawdown * 100,
            'avg_bet_size': self.total_wagered / len(settled_bets) if settled_bets else 0
        }


if __name__ == '__main__':
    # Example usage
    print("MLB Betting Strategy - Kelly Criterion Example")
    print("=" * 80)
    
    strategy = MLBBettingStrategy(bankroll=1000, kelly_fraction=0.25, min_edge=0.05)
    
    # Example game prediction
    prediction = {
        'predicted_winner': 'home',
        'home_win_probability': 0.62,
        'away_win_probability': 0.38,
        'confidence': 0.62,
        'edge': 0.12
    }
    
    # Market odds
    home_odds = -110  # Slight favorite
    away_odds = +100  # Even money
    
    # Get recommendation
    rec = strategy.get_bet_recommendation(prediction, home_odds, away_odds)
    
    if rec:
        print("\nBet Recommendation:")
        print(f"  Side: {rec['side']}")
        print(f"  Model probability: {rec['probability']:.3f}")
        print(f"  Edge: {rec['edge']:.3f} ({rec['edge']*100:.1f}%)")
        print(f"  Odds: {rec['odds']}")
        print(f"  Kelly fraction: {rec['kelly_fraction']:.4f}")
        print(f"  Bet amount: ${rec['bet_amount']:.2f}")
        print(f"  Expected value: ${rec['expected_value']:.2f}")
        
        # Place bet
        bet = strategy.place_bet(rec, 'game_123')
        print(f"\nBet placed - Bankroll: ${strategy.current_bankroll:.2f}")
        
        # Simulate outcome - home wins
        settlement = strategy.settle_bet('game_123', 'home')
        print(f"\nBet settled - {'WON' if settlement['won'] else 'LOST'}")
        print(f"  Profit: ${settlement['profit']:.2f}")
        print(f"  New bankroll: ${settlement['new_bankroll']:.2f}")
    else:
        print("\nNo bet recommended - insufficient edge")
    
    print("=" * 80)

