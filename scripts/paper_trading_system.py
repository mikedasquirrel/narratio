"""
Paper Trading System
====================

Risk-free validation system that:
- Tracks simulated bets without real money
- Monitors performance vs expectations
- Generates performance reports
- Validates all system components

Run for 2-4 weeks before live deployment.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_optimization.betting.kelly_criterion import KellyCriterion


class PaperTradingSystem:
    """Simulates betting without real money."""
    
    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize paper trading system.
        
        Args:
            initial_bankroll: Starting paper bankroll
            data_dir: Directory for trade records
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data' / 'paper_trading'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.kelly = KellyCriterion()
        self.trades = []
        self.daily_bankroll = []
        
    def place_paper_bet(
        self,
        game_id: str,
        bet_type: str,
        side: str,
        odds: float,
        win_probability: float,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Place a simulated bet.
        
        Args:
            game_id: Game identifier
            bet_type: Type of bet
            side: Side of bet
            odds: American odds
            win_probability: Model's win probability
            timestamp: When bet was placed
            
        Returns:
            Dict with bet details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate Kelly sizing
        kelly_bet = self.kelly.calculate_bet(
            game_id=game_id,
            bet_type=bet_type,
            side=side,
            american_odds=odds,
            win_probability=win_probability,
            bankroll=self.current_bankroll
        )
        
        # Don't bet if no edge
        if kelly_bet.recommended_units == 0:
            return {
                'status': 'skipped',
                'reason': kelly_bet.reasoning
            }
        
        # Calculate bet amount
        bet_amount = (kelly_bet.recommended_fraction * self.current_bankroll)
        
        # Record bet
        trade = {
            'trade_id': f"paper_{len(self.trades)}",
            'timestamp': timestamp.isoformat(),
            'game_id': game_id,
            'bet_type': bet_type,
            'side': side,
            'odds': odds,
            'amount': bet_amount,
            'units': kelly_bet.recommended_units,
            'win_probability': win_probability,
            'edge': kelly_bet.edge,
            'expected_value': kelly_bet.expected_value,
            'bankroll_before': self.current_bankroll,
            'status': 'pending',
            'outcome': None,
            'profit': None
        }
        
        self.trades.append(trade)
        
        return {
            'status': 'placed',
            'trade_id': trade['trade_id'],
            'amount': bet_amount,
            'units': kelly_bet.recommended_units,
            'expected_value': kelly_bet.expected_value
        }
    
    def settle_bet(
        self,
        trade_id: str,
        outcome: int,  # 1 = win, 0 = loss
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Settle a paper bet.
        
        Args:
            trade_id: Trade identifier
            outcome: 1 for win, 0 for loss
            timestamp: When bet was settled
            
        Returns:
            Dict with settlement details
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Find trade
        trade = next((t for t in self.trades if t['trade_id'] == trade_id), None)
        
        if not trade:
            return {'error': 'Trade not found'}
        
        if trade['status'] != 'pending':
            return {'error': 'Trade already settled'}
        
        # Calculate profit/loss
        decimal_odds = self.kelly.american_to_decimal(trade['odds'])
        
        if outcome == 1:  # Win
            profit = trade['amount'] * (decimal_odds - 1.0)
        else:  # Loss
            profit = -trade['amount']
        
        # Update bankroll
        self.current_bankroll += profit
        
        # Update trade
        trade['outcome'] = outcome
        trade['profit'] = profit
        trade['status'] = 'settled'
        trade['settled_timestamp'] = timestamp.isoformat()
        trade['bankroll_after'] = self.current_bankroll
        
        # Record daily bankroll
        self.daily_bankroll.append({
            'date': timestamp.date().isoformat(),
            'bankroll': self.current_bankroll
        })
        
        return {
            'status': 'settled',
            'trade_id': trade_id,
            'outcome': 'win' if outcome == 1 else 'loss',
            'profit': profit,
            'new_bankroll': self.current_bankroll
        }
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        settled_trades = [t for t in self.trades if t['status'] == 'settled']
        
        if not settled_trades:
            return {'error': 'No settled trades'}
        
        # Calculate metrics
        n_trades = len(settled_trades)
        wins = sum(1 for t in settled_trades if t['outcome'] == 1)
        losses = n_trades - wins
        win_rate = wins / n_trades
        
        total_profit = sum(t['profit'] for t in settled_trades)
        total_wagered = sum(t['amount'] for t in settled_trades)
        roi = (total_profit / total_wagered) if total_wagered > 0 else 0
        
        total_return = ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll)
        
        # Calculate Sharpe-like ratio
        profits = [t['profit'] for t in settled_trades]
        if len(profits) > 1:
            sharpe = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
        else:
            sharpe = 0
        
        # Max drawdown
        bankrolls = [t['bankroll_after'] for t in settled_trades]
        max_bankroll = max(bankrolls)
        min_after_max = min([b for b in bankrolls if bankrolls.index(b) >= bankrolls.index(max_bankroll)])
        max_drawdown = (max_bankroll - min_after_max) / max_bankroll if max_bankroll > 0 else 0
        
        # Expected vs actual
        expected_wins = sum(t['win_probability'] for t in settled_trades)
        actual_wins = wins
        accuracy = (actual_wins / expected_wins) if expected_wins > 0 else 0
        
        return {
            'period': {
                'start': settled_trades[0]['timestamp'],
                'end': settled_trades[-1]['settled_timestamp'],
                'n_days': (datetime.fromisoformat(settled_trades[-1]['settled_timestamp']) - 
                          datetime.fromisoformat(settled_trades[0]['timestamp'])).days
            },
            'trades': {
                'total': n_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate
            },
            'financial': {
                'initial_bankroll': self.initial_bankroll,
                'final_bankroll': self.current_bankroll,
                'total_profit': total_profit,
                'total_wagered': total_wagered,
                'roi': roi,
                'total_return': total_return
            },
            'risk': {
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'avg_bet_size': total_wagered / n_trades
            },
            'model_accuracy': {
                'expected_wins': expected_wins,
                'actual_wins': actual_wins,
                'accuracy_ratio': accuracy
            }
        }
    
    def save_state(self):
        """Save paper trading state."""
        filepath = self.data_dir / 'paper_trading_state.json'
        
        state = {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'trades': self.trades,
            'daily_bankroll': self.daily_bankroll,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ State saved to {filepath}")


def test_paper_trading():
    """Test paper trading system."""
    print("=" * 80)
    print("PAPER TRADING SYSTEM TEST")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Create system
    system = PaperTradingSystem(initial_bankroll=10000.0)
    
    print(f"Initial bankroll: ${system.initial_bankroll:,.2f}\n")
    
    # Simulate 50 bets
    print("Simulating 50 paper trades...")
    print("-" * 80)
    
    for i in range(50):
        # Simulate a betting opportunity
        win_prob = 0.55 + np.random.uniform(-0.1, 0.15)
        odds = -110
        
        # Place bet
        result = system.place_paper_bet(
            game_id=f"game_{i}",
            bet_type='moneyline',
            side='home',
            odds=odds,
            win_probability=win_prob
        )
        
        if result['status'] == 'placed':
            # Simulate outcome
            actual_outcome = 1 if np.random.random() < win_prob else 0
            
            # Settle bet
            settlement = system.settle_bet(result['trade_id'], actual_outcome)
            
            if (i + 1) % 10 == 0:
                print(f"  Trade {i+1}: {settlement['outcome'].upper()} | " +
                      f"Bankroll: ${settlement['new_bankroll']:,.2f}")
    
    # Get report
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    
    report = system.get_performance_report()
    
    print(f"\nTrading Period:")
    print(f"  Total Trades: {report['trades']['total']}")
    print(f"  Wins: {report['trades']['wins']}")
    print(f"  Losses: {report['trades']['losses']}")
    print(f"  Win Rate: {report['trades']['win_rate']:.1%}")
    
    print(f"\nFinancial:")
    print(f"  Initial Bankroll: ${report['financial']['initial_bankroll']:,.2f}")
    print(f"  Final Bankroll: ${report['financial']['final_bankroll']:,.2f}")
    print(f"  Total Profit: ${report['financial']['total_profit']:+,.2f}")
    print(f"  ROI: {report['financial']['roi']:+.1%}")
    print(f"  Total Return: {report['financial']['total_return']:+.1%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown: {report['risk']['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio: {report['risk']['sharpe_ratio']:.3f}")
    print(f"  Avg Bet Size: ${report['risk']['avg_bet_size']:,.2f}")
    
    print(f"\nModel Accuracy:")
    print(f"  Expected Wins: {report['model_accuracy']['expected_wins']:.1f}")
    print(f"  Actual Wins: {report['model_accuracy']['actual_wins']}")
    print(f"  Accuracy Ratio: {report['model_accuracy']['accuracy_ratio']:.2f}")
    
    # Save state
    system.save_state()
    
    print("\n" + "=" * 80)
    print("PAPER TRADING TEST COMPLETE")
    print("=" * 80)
    
    print("\nRecommendation:")
    if report['financial']['roi'] > 0.10:
        print("  ✓ System showing positive ROI - continue paper trading")
    elif report['financial']['roi'] > 0:
        print("  ⚠️  Positive but low ROI - monitor closely")
    else:
        print("  ⚠️  Negative ROI - review system before live deployment")


if __name__ == '__main__':
    test_paper_trading()
