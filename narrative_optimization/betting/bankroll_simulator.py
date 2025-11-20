"""
Monte Carlo Bankroll Simulator
===============================

Simulates betting performance over 10,000 seasons to:
- Analyze drawdown risk
- Determine optimal Kelly fraction
- Calculate risk of ruin
- Visualize growth vs volatility tradeoff

Author: AI Coding Assistant
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from kelly_criterion import KellyCriterion


class BankrollSimulator:
    """Monte Carlo simulator for bankroll management analysis."""
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        n_simulations: int = 10000,
        n_bets_per_sim: int = 250,  # ~1 season of bets
        random_seed: Optional[int] = 42
    ):
        """
        Initialize bankroll simulator.
        
        Args:
            initial_bankroll: Starting bankroll
            n_simulations: Number of Monte Carlo simulations
            n_bets_per_sim: Number of bets per simulation
            random_seed: Random seed for reproducibility
        """
        self.initial_bankroll = initial_bankroll
        self.n_simulations = n_simulations
        self.n_bets_per_sim = n_bets_per_sim
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.kelly = KellyCriterion()
        
    def simulate_season(
        self,
        win_rate: float = 0.55,
        avg_odds: float = -110,
        kelly_fraction: float = 0.5,
        edge: float = 0.03
    ) -> Dict[str, float]:
        """
        Simulate one season of betting.
        
        Args:
            win_rate: Average win rate (e.g., 0.55 = 55%)
            avg_odds: Average American odds
            kelly_fraction: Kelly fraction to use (0.5 = half Kelly)
            edge: Average edge per bet
            
        Returns:
            Dict with season results
        """
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        max_bankroll = bankroll
        min_bankroll = bankroll
        
        # Calculate bet size using Kelly
        kelly_full = self.kelly.calculate_full_kelly(win_rate, avg_odds)
        bet_fraction = kelly_full * kelly_fraction
        bet_fraction = min(bet_fraction, 0.02)  # Cap at 2%
        
        decimal_odds = self.kelly.american_to_decimal(avg_odds)
        
        for _ in range(self.n_bets_per_sim):
            # Bet size as fraction of current bankroll
            bet_size = bankroll * bet_fraction
            
            # Simulate outcome
            win = np.random.random() < win_rate
            
            if win:
                profit = bet_size * (decimal_odds - 1.0)
                bankroll += profit
            else:
                bankroll -= bet_size
            
            # Track history
            bankroll_history.append(bankroll)
            max_bankroll = max(max_bankroll, bankroll)
            min_bankroll = min(min_bankroll, bankroll)
            
            # Check for ruin
            if bankroll <= 0:
                break
        
        # Calculate metrics
        final_bankroll = max(bankroll, 0)
        total_return = (final_bankroll - self.initial_bankroll) / self.initial_bankroll
        max_drawdown = (max_bankroll - min_bankroll) / max_bankroll if max_bankroll > 0 else 1.0
        
        # Sharpe-like ratio
        returns = np.diff(bankroll_history) / bankroll_history[:-1]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        return {
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'bankroll_history': bankroll_history,
            'ruined': bankroll <= 0
        }
    
    def run_simulations(
        self,
        win_rate: float = 0.55,
        avg_odds: float = -110,
        kelly_fractions: List[float] = [0.25, 0.5, 0.75, 1.0],
        edge: float = 0.03
    ) -> Dict[float, List[Dict]]:
        """
        Run Monte Carlo simulations for different Kelly fractions.
        
        Args:
            win_rate: Average win rate
            avg_odds: Average odds
            kelly_fractions: List of Kelly fractions to test
            edge: Average edge
            
        Returns:
            Dict mapping Kelly fraction to list of simulation results
        """
        print("=" * 80)
        print("MONTE CARLO BANKROLL SIMULATION")
        print("=" * 80)
        print(f"Simulations: {self.n_simulations}")
        print(f"Bets per season: {self.n_bets_per_sim}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Average odds: {avg_odds:+.0f}")
        print(f"Edge: {edge:+.1%}")
        print(f"Kelly fractions: {kelly_fractions}")
        
        results = {}
        
        for fraction in kelly_fractions:
            print(f"\nSimulating {fraction:.0%} Kelly...")
            
            sim_results = []
            for i in range(self.n_simulations):
                result = self.simulate_season(win_rate, avg_odds, fraction, edge)
                sim_results.append(result)
                
                if (i + 1) % 2000 == 0:
                    print(f"  Completed {i+1}/{self.n_simulations} simulations")
            
            results[fraction] = sim_results
        
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        
        return results
    
    def analyze_results(
        self,
        results: Dict[float, List[Dict]]
    ) -> pd.DataFrame:
        """
        Analyze simulation results.
        
        Args:
            results: Results from run_simulations
            
        Returns:
            DataFrame with summary statistics
        """
        summary = []
        
        for fraction, sims in results.items():
            final_bankrolls = [s['final_bankroll'] for s in sims]
            total_returns = [s['total_return'] for s in sims]
            max_drawdowns = [s['max_drawdown'] for s in sims]
            sharpe_ratios = [s['sharpe_ratio'] for s in sims if not np.isnan(s['sharpe_ratio'])]
            ruin_rate = sum(1 for s in sims if s['ruined']) / len(sims)
            
            summary.append({
                'kelly_fraction': fraction,
                'avg_final_bankroll': np.mean(final_bankrolls),
                'median_final_bankroll': np.median(final_bankrolls),
                'p10_final_bankroll': np.percentile(final_bankrolls, 10),
                'p90_final_bankroll': np.percentile(final_bankrolls, 90),
                'avg_return': np.mean(total_returns),
                'median_return': np.median(total_returns),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'max_max_drawdown': np.max(max_drawdowns),
                'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'ruin_rate': ruin_rate
            })
        
        df = pd.DataFrame(summary)
        df = df.sort_values('kelly_fraction')
        
        return df
    
    def plot_results(
        self,
        summary: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Final bankroll distribution
        ax = axes[0, 0]
        kelly_fractions = summary['kelly_fraction'].values
        medians = summary['median_final_bankroll'].values
        p10s = summary['p10_final_bankroll'].values
        p90s = summary['p90_final_bankroll'].values
        
        ax.plot(kelly_fractions, medians, 'o-', label='Median', linewidth=2)
        ax.fill_between(kelly_fractions, p10s, p90s, alpha=0.3, label='10th-90th percentile')
        ax.axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial bankroll')
        ax.set_xlabel('Kelly Fraction')
        ax.set_ylabel('Final Bankroll ($)')
        ax.set_title('Final Bankroll Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Return vs Drawdown
        ax = axes[0, 1]
        returns = summary['median_return'].values * 100
        drawdowns = summary['avg_max_drawdown'].values * 100
        
        for i, fraction in enumerate(kelly_fractions):
            ax.scatter(drawdowns[i], returns[i], s=200, label=f'{fraction:.0%} Kelly')
            ax.annotate(f'{fraction:.0%}', (drawdowns[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Average Max Drawdown (%)')
        ax.set_ylabel('Median Return (%)')
        ax.set_title('Return vs Drawdown Tradeoff')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Sharpe Ratio
        ax = axes[1, 0]
        sharpes = summary['avg_sharpe'].values
        ax.bar([f'{f:.0%}' for f in kelly_fractions], sharpes, alpha=0.7)
        ax.set_xlabel('Kelly Fraction')
        ax.set_ylabel('Average Sharpe Ratio')
        ax.set_title('Risk-Adjusted Returns')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Risk of Ruin
        ax = axes[1, 1]
        ruin_rates = summary['ruin_rate'].values * 100
        ax.bar([f'{f:.0%}' for f in kelly_fractions], ruin_rates, alpha=0.7, color='red')
        ax.set_xlabel('Kelly Fraction')
        ax.set_ylabel('Risk of Ruin (%)')
        ax.set_title('Probability of Bankroll Ruin')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def recommend_kelly_fraction(self, summary: pd.DataFrame) -> float:
        """Recommend optimal Kelly fraction based on risk tolerance."""
        # Calculate risk-adjusted score
        summary['score'] = (
            summary['median_return'] * 0.4 +
            (1 - summary['avg_max_drawdown']) * 0.3 +
            summary['avg_sharpe'] * 0.2 -
            summary['ruin_rate'] * 0.1
        )
        
        best_idx = summary['score'].idxmax()
        best_fraction = summary.loc[best_idx, 'kelly_fraction']
        
        return best_fraction


def test_bankroll_simulator():
    """Test the bankroll simulator."""
    simulator = BankrollSimulator(
        initial_bankroll=1000.0,
        n_simulations=1000,  # Reduced for faster testing
        n_bets_per_sim=250
    )
    
    # Run simulations
    results = simulator.run_simulations(
        win_rate=0.55,
        avg_odds=-110,
        kelly_fractions=[0.25, 0.5, 0.75, 1.0],
        edge=0.03
    )
    
    # Analyze
    summary = simulator.analyze_results(results)
    
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(summary.to_string(index=False))
    
    # Recommend
    recommended = simulator.recommend_kelly_fraction(summary)
    print(f"\n" + "=" * 80)
    print(f"RECOMMENDED KELLY FRACTION: {recommended:.0%}")
    print("=" * 80)
    
    # Plot
    save_path = Path(__file__).parent / 'bankroll_simulation.png'
    simulator.plot_results(summary, save_path=str(save_path))
    
    print("\nTest complete!")


if __name__ == '__main__':
    test_bankroll_simulator()

