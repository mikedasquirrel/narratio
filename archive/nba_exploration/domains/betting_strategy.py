"""
NBA Betting Strategy System

Implements multiple betting strategies based on narrative predictions
and tests whether narrative modeling creates betting value.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class NBABettingStrategy:
    """
    Base class for NBA betting strategies.
    
    Converts model predictions into betting decisions with
    bankroll management and risk control.
    """
    
    def __init__(self, initial_bankroll: float = 1000.0, unit_size: float = 10.0):
        """
        Initialize betting strategy.
        
        Parameters
        ----------
        initial_bankroll : float
            Starting bankroll in dollars
        unit_size : float
            Base bet size as percentage of bankroll
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.unit_size = unit_size
        self.bet_history = []
    
    def recommend_bet(self, prediction: Dict, game_context: Dict) -> Dict[str, Any]:
        """
        Generate betting recommendation.
        
        Parameters
        ----------
        prediction : dict
            Model prediction with probabilities
        game_context : dict
            Game context including Vegas line
        
        Returns
        -------
        recommendation : dict
            Bet decision, size, and reasoning
        """
        raise NotImplementedError("Subclass must implement recommend_bet()")
    
    def place_bet(self, recommendation: Dict, actual_outcome: str) -> Dict[str, Any]:
        """
        Execute bet and update bankroll.
        
        Parameters
        ----------
        recommendation : dict
            Betting recommendation
        actual_outcome : str
            'home' or 'away'
        
        Returns
        -------
        result : dict
            Bet outcome and profit/loss
        """
        if recommendation['action'] == 'pass':
            return {'bet': False, 'profit': 0, 'bankroll': self.current_bankroll}
        
        bet_amount = recommendation['bet_size']
        bet_team = recommendation['team']
        
        # Determine if bet won
        won = (bet_team == actual_outcome)
        
        # Calculate profit (assuming -110 American odds)
        if won:
            profit = bet_amount * 0.909  # Win $90.90 per $100 bet
        else:
            profit = -bet_amount
        
        # Update bankroll
        self.current_bankroll += profit
        
        # Record bet
        bet_record = {
            'bet': True,
            'team': bet_team,
            'amount': bet_amount,
            'won': won,
            'profit': profit,
            'bankroll': self.current_bankroll,
            'roi': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        }
        
        self.bet_history.append(bet_record)
        
        return bet_record
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_bet_size': 0,
                'final_bankroll': self.current_bankroll,
                'bankroll_growth': 0
            }
        
        bets_made = [b for b in self.bet_history if b['bet']]
        
        if not bets_made:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_bet_size': 0,
                'final_bankroll': self.current_bankroll,
                'bankroll_growth': 0
            }
        
        total_bets = len(bets_made)
        wins = sum(1 for b in bets_made if b['won'])
        total_wagered = sum(b['amount'] for b in bets_made)
        total_profit = sum(b['profit'] for b in bets_made)
        
        return {
            'total_bets': total_bets,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'avg_bet_size': total_wagered / total_bets if total_bets > 0 else 0,
            'final_bankroll': self.current_bankroll,
            'bankroll_growth': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        }


class NarrativeEdgeStrategy(NBABettingStrategy):
    """
    Betting strategy that exploits narrative-stats disconnects.
    
    Bets when narrative model disagrees significantly with Vegas line,
    suggesting market inefficiency.
    """
    
    def __init__(self, edge_threshold: float = 0.10, **kwargs):
        """
        Initialize narrative edge strategy.
        
        Parameters
        ----------
        edge_threshold : float
            Minimum probability edge to trigger bet (default: 10%)
        """
        super().__init__(**kwargs)
        self.edge_threshold = edge_threshold
    
    def recommend_bet(self, prediction: Dict, game_context: Dict) -> Dict[str, Any]:
        """
        Recommend bet based on narrative edge.
        
        Bets when model probability significantly differs from implied Vegas probability.
        """
        # Convert Vegas line to implied probability
        vegas_line = game_context.get('betting_line', 0)
        implied_prob = self._line_to_probability(vegas_line)
        
        # Get model probability
        model_home_prob = prediction['home_win_probability']
        model_away_prob = prediction['away_win_probability']
        
        # Calculate edge
        home_edge = model_home_prob - implied_prob
        away_edge = model_away_prob - (1 - implied_prob)
        
        # Determine if there's value
        if abs(home_edge) > self.edge_threshold:
            if home_edge > 0:
                # Bet home
                bet_size = self._kelly_criterion(model_home_prob, implied_prob)
                return {
                    'action': 'bet',
                    'team': 'home',
                    'bet_size': bet_size,
                    'edge': home_edge,
                    'reasoning': f"Narrative model sees {home_edge*100:.1f}% edge for home team"
                }
            else:
                # Bet away
                bet_size = self._kelly_criterion(model_away_prob, 1 - implied_prob)
                return {
                    'action': 'bet',
                    'team': 'away',
                    'bet_size': bet_size,
                    'edge': abs(home_edge),
                    'reasoning': f"Narrative model sees {abs(home_edge)*100:.1f}% edge for away team"
                }
        
        return {
            'action': 'pass',
            'reasoning': f"No significant edge detected (edge: {max(abs(home_edge), abs(away_edge))*100:.1f}%)"
        }
    
    def _line_to_probability(self, line: float) -> float:
        """
        Convert betting line to implied win probability for home team.
        
        Parameters
        ----------
        line : float
            Point spread (positive = home favored)
        
        Returns
        -------
        probability : float
            Implied home win probability
        """
        # Rough conversion: each point ≈ 3-4% probability
        # Line of +7 means home favored by 7, ~65% win probability
        # Line of -7 means home underdog, ~35% win probability
        
        base_prob = 0.5
        line_impact = line * 0.03  # 3% per point
        
        prob = base_prob + line_impact
        return max(0.05, min(0.95, prob))  # Clamp to reasonable range
    
    def _kelly_criterion(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.
        
        Kelly = (bp - q) / b
        where b = odds, p = win prob, q = 1-p
        
        Parameters
        ----------
        model_prob : float
            Model's win probability
        market_prob : float
            Market implied probability
        
        Returns
        -------
        bet_size : float
            Optimal bet size as dollars
        """
        # Convert probabilities to odds
        if model_prob >= 0.95:
            model_prob = 0.95
        if market_prob >= 0.95:
            market_prob = 0.95
        
        # Kelly fraction
        edge = model_prob - market_prob
        odds = (1 / market_prob) - 1
        
        kelly_fraction = edge / odds if odds > 0 else 0
        
        # Use fractional Kelly (1/4 Kelly for safety)
        kelly_fraction *= 0.25
        
        # Calculate bet size
        bet_size = self.current_bankroll * max(0, min(kelly_fraction, 0.05))  # Max 5% of bankroll
        
        # Minimum bet or pass
        return max(self.unit_size, bet_size) if bet_size >= self.unit_size else 0


class MomentumStrategy(NBABettingStrategy):
    """
    Strategy that follows strong narrative momentum signals.
    
    Bets on teams showing high momentum language and future orientation.
    """
    
    def __init__(self, momentum_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.momentum_threshold = momentum_threshold
    
    def recommend_bet(self, prediction: Dict, game_context: Dict) -> Dict[str, Any]:
        """Recommend bet based on momentum indicators."""
        home_prob = prediction['home_win_probability']
        
        # Extract momentum from context if available
        home_momentum = game_context.get('home_momentum_score', 0)
        away_momentum = game_context.get('away_momentum_score', 0)
        
        momentum_diff = home_momentum - away_momentum
        
        # Bet if strong momentum AND model agrees
        if abs(momentum_diff) > self.momentum_threshold:
            if momentum_diff > 0 and home_prob > 0.52:
                return {
                    'action': 'bet',
                    'team': 'home',
                    'bet_size': self.unit_size * (1 + momentum_diff),
                    'reasoning': f"Strong home momentum (+{momentum_diff:.3f})"
                }
            elif momentum_diff < 0 and home_prob < 0.48:
                return {
                    'action': 'bet',
                    'team': 'away',
                    'bet_size': self.unit_size * (1 + abs(momentum_diff)),
                    'reasoning': f"Strong away momentum (+{abs(momentum_diff):.3f})"
                }
        
        return {'action': 'pass', 'reasoning': 'No clear momentum advantage'}


class ContrarianStrategy(NBABettingStrategy):
    """
    Contrarian strategy that fades overconfident narratives.
    
    Bets against teams with excessive confidence markers
    when stats don't support the narrative.
    """
    
    def __init__(self, confidence_threshold: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
    
    def recommend_bet(self, prediction: Dict, game_context: Dict) -> Dict[str, Any]:
        """Fade overconfident teams."""
        # Extract confidence from narrative features if available
        home_confidence = game_context.get('home_confidence_score', 0.5)
        away_confidence = game_context.get('away_confidence_score', 0.5)
        
        # Check for overconfidence (high narrative confidence but <60% model prob)
        home_prob = prediction['home_win_probability']
        
        if home_confidence > self.confidence_threshold and home_prob < 0.55:
            # Home team overconfident - fade them
            return {
                'action': 'bet',
                'team': 'away',
                'bet_size': self.unit_size,
                'reasoning': f"Fading overconfident home narrative (conf:{home_confidence:.2f}, prob:{home_prob:.2f})"
            }
        
        if away_confidence > self.confidence_threshold and home_prob > 0.45:
            # Away team overconfident - fade them
            return {
                'action': 'bet',
                'team': 'home',
                'bet_size': self.unit_size,
                'reasoning': f"Fading overconfident away narrative (conf:{away_confidence:.2f})"
            }
        
        return {'action': 'pass', 'reasoning': 'No overconfidence detected'}

