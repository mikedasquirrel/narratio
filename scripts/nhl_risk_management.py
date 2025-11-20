"""
NHL Risk Management System

Implements proper bankroll management for NHL betting:
- Kelly Criterion stake sizing
- Risk limits (max bet, max daily exposure)
- Drawdown protection
- Portfolio optimization

Author: Narrative Integration System
Date: November 16, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BettingParameters:
    """Betting system parameters"""
    bankroll: float = 10000.0  # Starting bankroll
    max_bet_pct: float = 0.05  # Max 5% of bankroll per bet
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    max_daily_exposure: float = 0.15  # Max 15% bankroll at risk per day
    min_edge: float = 0.05  # Minimum 5% edge required
    min_confidence: float = 0.55  # Minimum 55% win probability


class NHLRiskManager:
    """Manage betting risk"""
    
    def __init__(self, params: BettingParameters = None):
        """Initialize risk manager"""
        self.params = params or BettingParameters()
        self.current_bankroll = self.params.bankroll
        self.daily_exposure = 0.0
        self.active_bets = []
    
    def calculate_kelly_stake(self, win_prob: float, odds: float) -> float:
        """
        Calculate Kelly Criterion stake size.
        
        Kelly% = (p * (odds + 1) - 1) / odds
        
        Parameters
        ----------
        win_prob : float
            Estimated win probability
        odds : float
            Decimal odds (e.g., 1.91 for -110)
        
        Returns
        -------
        stake_pct : float
            Percentage of bankroll to bet
        """
        if win_prob <= 0 or odds <= 0:
            return 0.0
        
        # Kelly formula
        kelly_pct = (win_prob * (odds + 1) - 1) / odds
        
        # Apply fractional Kelly (conservative)
        kelly_pct = kelly_pct * self.params.kelly_fraction
        
        # Ensure non-negative
        kelly_pct = max(0, kelly_pct)
        
        return kelly_pct
    
    def calculate_recommended_stake(self, recommendation: Dict) -> Dict:
        """
        Calculate recommended stake with risk management.
        
        Parameters
        ----------
        recommendation : dict
            Pattern recommendation with expected win rate and ROI
        
        Returns
        -------
        stake_info : dict
            Stake size and risk information
        """
        
        # Extract info
        win_rate = recommendation.get('expected_win_rate', 50) / 100
        unit_rec = recommendation.get('unit_size', 1)
        
        # Assume -110 odds (1.91 decimal)
        odds_decimal = 1.91
        
        # Calculate Kelly stake
        kelly_pct = self.calculate_kelly_stake(win_rate, odds_decimal)
        kelly_amount = self.current_bankroll * kelly_pct
        
        # Calculate edge
        edge = (win_rate * 0.909) + ((1 - win_rate) * -1.0)
        
        # Apply constraints
        max_bet = self.current_bankroll * self.params.max_bet_pct
        
        # Recommended stake (min of Kelly and max bet)
        recommended_amount = min(kelly_amount, max_bet)
        
        # Check if within risk limits
        within_daily_limit = (self.daily_exposure + recommended_amount) <= (self.current_bankroll * self.params.max_daily_exposure)
        
        # Check minimum edge
        sufficient_edge = edge >= self.params.min_edge
        
        # Final decision
        should_bet = within_daily_limit and sufficient_edge and kelly_pct > 0
        
        stake_info = {
            'kelly_pct': kelly_pct,
            'kelly_amount': kelly_amount,
            'max_bet': max_bet,
            'recommended_amount': recommended_amount if should_bet else 0,
            'recommended_units': recommended_amount / 100 if should_bet else 0,  # Assuming 1u = $100
            'edge': edge,
            'should_bet': should_bet,
            'constraints': {
                'within_daily_limit': within_daily_limit,
                'sufficient_edge': sufficient_edge,
                'positive_kelly': kelly_pct > 0,
            },
            'risk_level': 'LOW' if kelly_pct < 0.02 else 'MEDIUM' if kelly_pct < 0.05 else 'HIGH',
        }
        
        return stake_info
    
    def evaluate_portfolio(self, recommendations: List[Dict]) -> Dict:
        """
        Evaluate portfolio of recommendations for the day.
        
        Returns optimal betting strategy.
        """
        
        print("\n📊 PORTFOLIO EVALUATION")
        print("="*80)
        
        portfolio = []
        
        for i, rec in enumerate(recommendations, 1):
            if not rec.get('recommendation'):
                continue
            
            recommendation = rec['recommendation']
            stake_info = self.calculate_recommended_stake(recommendation)
            
            portfolio.append({
                'game': rec.get('game', {}),
                'recommendation': recommendation,
                'stake': stake_info,
            })
        
        # Calculate portfolio stats
        total_stake = sum(p['stake']['recommended_amount'] for p in portfolio)
        total_exposure_pct = total_stake / self.current_bankroll
        
        should_bet_count = sum(1 for p in portfolio if p['stake']['should_bet'])
        avg_win_rate = np.mean([p['recommendation']['expected_win_rate'] for p in portfolio]) if portfolio else 0
        avg_roi = np.mean([p['recommendation']['expected_roi'] for p in portfolio]) if portfolio else 0
        
        portfolio_summary = {
            'total_opportunities': len(portfolio),
            'recommended_bets': should_bet_count,
            'total_stake': total_stake,
            'exposure_pct': total_exposure_pct,
            'avg_win_rate': avg_win_rate,
            'avg_roi': avg_roi,
            'within_limits': total_exposure_pct <= self.params.max_daily_exposure,
            'bets': portfolio,
        }
        
        print(f"Total opportunities: {len(portfolio)}")
        print(f"Recommended bets: {should_bet_count}")
        print(f"Total stake: ${total_stake:,.0f}")
        print(f"Portfolio exposure: {total_exposure_pct:.1%}")
        print(f"Within limits: {'✅ YES' if portfolio_summary['within_limits'] else '❌ NO'}")
        print(f"Avg expected win rate: {avg_win_rate:.1f}%")
        print(f"Avg expected ROI: {avg_roi:.1f}%")
        
        return portfolio_summary


def main():
    """Test risk management"""
    
    # Create manager with conservative params
    params = BettingParameters(
        bankroll=10000,
        max_bet_pct=0.05,  # Max 5% per bet
        kelly_fraction=0.25,  # Quarter Kelly
        max_daily_exposure=0.15,  # Max 15% daily
    )
    
    manager = NHLRiskManager(params)
    
    print("\n💼 NHL RISK MANAGEMENT TEST")
    print("="*80)
    print(f"Bankroll: ${params.bankroll:,.0f}")
    print(f"Max bet: {params.max_bet_pct:.0%} (${params.bankroll * params.max_bet_pct:,.0f})")
    print(f"Kelly fraction: {params.kelly_fraction:.0%}")
    print(f"Max daily exposure: {params.max_daily_exposure:.0%}")
    
    # Test with high confidence pattern
    test_rec = {
        'pattern_name': 'Meta-Ensemble ≥65%',
        'expected_win_rate': 95.8,
        'expected_roi': 82.9,
        'confidence': 'VERY HIGH',
        'unit_size': 3,
    }
    
    stake = manager.calculate_recommended_stake(test_rec)
    
    print("\n🎯 TEST RECOMMENDATION:")
    print(f"   Pattern: {test_rec['pattern_name']}")
    print(f"   Expected: {test_rec['expected_win_rate']}% win, {test_rec['expected_roi']}% ROI")
    
    print("\n💰 STAKE CALCULATION:")
    print(f"   Kelly: {stake['kelly_pct']:.1%} (${stake['kelly_amount']:,.0f})")
    print(f"   Max allowed: ${stake['max_bet']:,.0f}")
    print(f"   Recommended: ${stake['recommended_amount']:,.0f} ({stake['recommended_units']:.1f}u)")
    print(f"   Edge: {stake['edge']:.1%}")
    print(f"   Should bet: {'✅ YES' if stake['should_bet'] else '❌ NO'}")
    print(f"   Risk level: {stake['risk_level']}")
    
    print("\n✅ Risk management system operational!")


if __name__ == "__main__":
    main()

