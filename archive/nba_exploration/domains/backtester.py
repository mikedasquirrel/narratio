"""
NBA Betting Backtest Framework

Simulates betting strategies across historical test seasons
to evaluate performance and compare to baselines.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class NBABacktester:
    """
    Backtest NBA betting strategies on historical data.
    
    Simulates real betting scenarios with:
    - Temporal validation (no look-ahead bias)
    - Bankroll management
    - Multiple strategy comparison
    - Performance metrics calculation
    """
    
    def __init__(self, initial_bankroll: float = 1000.0):
        """
        Initialize backtester.
        
        Parameters
        ----------
        initial_bankroll : float
            Starting bankroll for simulation
        """
        self.initial_bankroll = initial_bankroll
        self.results = []
    
    def run_backtest(self, games: List[Dict], strategy, predictor) -> Dict[str, Any]:
        """
        Run complete backtest on test games.
        
        Parameters
        ----------
        games : list of dict
            Test games (every 10th season)
        strategy : NBABettingStrategy
            Betting strategy to test
        predictor : NBAGamePredictor
            Trained prediction model
        
        Returns
        -------
        backtest_results : dict
            Complete performance analysis
        """
        print(f"\n{'='*60}")
        print(f"RUNNING BACKTEST: {strategy.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Test games: {len(games)}")
        print(f"Initial bankroll: ${self.initial_bankroll:.2f}")
        print(f"{'='*60}\n")
        
        # Reset strategy bankroll
        strategy.current_bankroll = self.initial_bankroll
        strategy.bet_history = []
        
        bet_results = []
        correct_predictions = 0
        total_predictions = 0
        
        for i, game in enumerate(games):
            # Extract features (assuming they're in game data)
            if 'differential' not in game:
                # If no differential, compute it
                if 'home_features' not in game or 'away_features' not in game:
                    continue  # Skip if features not available
                
                home_features = np.array(game['home_features'])
                away_features = np.array(game['away_features'])
                differential = home_features - away_features
            else:
                differential = np.array(game['differential'])
                # Still need home/away for interpretation
                home_features = np.array(game.get('home_features', differential))
                away_features = np.array(game.get('away_features', -differential))
            
            # Get prediction (only use differential for model)
            prediction = predictor.predict_game(differential, differential, differential)
            
            # Add context for strategy
            game_context = {
                'betting_line': game.get('betting_line', 0),
                'home_momentum_score': game.get('home_momentum', 0),
                'away_momentum_score': game.get('away_momentum', 0),
                'home_confidence_score': game.get('home_confidence', 0.5),
                'away_confidence_score': game.get('away_confidence', 0.5)
            }
            
            # Get betting recommendation
            recommendation = strategy.recommend_bet(prediction, game_context)
            
            # Determine actual outcome
            actual_outcome = 'home' if game['home_wins'] else 'away'
            
            # Place bet
            bet_result = strategy.place_bet(recommendation, actual_outcome)
            bet_result['game_id'] = game['game_id']
            bet_result['prediction_correct'] = (prediction['predicted_winner'] == actual_outcome)
            bet_results.append(bet_result)
            
            # Track prediction accuracy
            if prediction['predicted_winner'] == actual_outcome:
                correct_predictions += 1
            total_predictions += 1
            
            # Progress update every 100 games
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(games)} games... Bankroll: ${strategy.current_bankroll:.2f}")
        
        # Calculate comprehensive metrics
        performance = strategy.get_performance_metrics()
        performance['prediction_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0
        performance['games_analyzed'] = len(games)
        
        # Calculate additional metrics
        if len(bet_results) > 0:
            profits = [b['profit'] for b in bet_results if b['bet']]
            if profits:
                performance['sharpe_ratio'] = self._calculate_sharpe(profits)
                performance['max_drawdown'] = self._calculate_max_drawdown(bet_results)
                performance['win_streak_max'] = self._calculate_max_streak(bet_results, True)
                performance['loss_streak_max'] = self._calculate_max_streak(bet_results, False)
        
        # Store results
        self.results = {
            'performance': performance,
            'bet_history': bet_results,
            'strategy_name': strategy.__class__.__name__,
            'predictor_type': predictor.model_type
        }
        
        # Print summary
        self._print_summary(performance)
        
        return self.results
    
    def compare_strategies(self, games: List[Dict], strategies: List, predictor) -> pd.DataFrame:
        """
        Compare multiple strategies on same games.
        
        Parameters
        ----------
        games : list
            Test games
        strategies : list
            List of strategy instances to compare
        predictor : NBAGamePredictor
            Prediction model
        
        Returns
        -------
        comparison : pd.DataFrame
            Strategy comparison table
        """
        comparison_data = []
        
        for strategy in strategies:
            results = self.run_backtest(games, strategy, predictor)
            comparison_data.append({
                'Strategy': results['strategy_name'],
                'Accuracy': f"{results['performance']['prediction_accuracy']:.1%}",
                'Bets Made': results['performance']['total_bets'],
                'Win Rate': f"{results['performance']['win_rate']:.1%}",
                'Total Profit': f"${results['performance']['total_profit']:.2f}",
                'ROI': f"{results['performance']['roi']:.1f}%",
                'Final Bankroll': f"${results['performance']['final_bankroll']:.2f}",
                'Sharpe Ratio': f"{results['performance'].get('sharpe_ratio', 0):.2f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def _calculate_sharpe(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        if not profits or len(profits) < 2:
            return 0.0
        
        mean_return = np.mean(profits)
        std_return = np.std(profits)
        
        if std_return == 0:
            return 0.0
        
        # Annualize assuming ~250 betting days per year
        sharpe = (mean_return / std_return) * np.sqrt(250)
        return float(sharpe)
    
    def _calculate_max_drawdown(self, bet_results: List[Dict]) -> float:
        """Calculate maximum drawdown from peak."""
        if not bet_results:
            return 0.0
        
        bankrolls = [b['bankroll'] for b in bet_results if 'bankroll' in b]
        if not bankrolls:
            return 0.0
        
        peak = bankrolls[0]
        max_dd = 0
        
        for bankroll in bankrolls:
            if bankroll > peak:
                peak = bankroll
            
            drawdown = (peak - bankroll) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100  # As percentage
    
    def _calculate_max_streak(self, bet_results: List[Dict], winning: bool) -> int:
        """Calculate maximum winning or losing streak."""
        bets_made = [b for b in bet_results if b['bet']]
        if not bets_made:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for bet in bets_made:
            if bet['won'] == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _print_summary(self, performance: Dict):
        """Print backtest summary."""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Prediction Accuracy:  {performance['prediction_accuracy']:.1%}")
        print(f"Total Bets Made:      {performance['total_bets']}")
        print(f"Betting Win Rate:     {performance['win_rate']:.1%}")
        print(f"Total Profit:         ${performance['total_profit']:.2f}")
        print(f"ROI:                  {performance['roi']:.1f}%")
        print(f"Final Bankroll:       ${performance['final_bankroll']:.2f}")
        print(f"Bankroll Growth:      {performance['bankroll_growth']:.1f}%")
        
        if 'sharpe_ratio' in performance:
            print(f"Sharpe Ratio:         {performance['sharpe_ratio']:.2f}")
        if 'max_drawdown' in performance:
            print(f"Max Drawdown:         {performance['max_drawdown']:.1f}%")
        
        print(f"{'='*60}\n")
    
    def plot_performance(self) -> Dict[str, Any]:
        """
        Generate performance visualization data.
        
        Returns
        -------
        plot_data : dict
            Data for plotting cumulative profit, win rate, etc.
        """
        if not self.results or not self.results.get('bet_history'):
            return {}
        
        bet_history = self.results['bet_history']
        bets_made = [b for b in bet_history if b['bet']]
        
        if not bets_made:
            return {}
        
        # Cumulative profit
        cumulative_profit = []
        running_total = 0
        for bet in bets_made:
            running_total += bet['profit']
            cumulative_profit.append(running_total)
        
        # Bankroll over time
        bankroll_series = [b['bankroll'] for b in bets_made]
        
        # Rolling win rate (last 20 bets)
        rolling_win_rate = []
        window = 20
        for i in range(len(bets_made)):
            window_bets = bets_made[max(0, i-window+1):i+1]
            wins = sum(1 for b in window_bets if b['won'])
            rolling_win_rate.append(wins / len(window_bets))
        
        return {
            'bet_numbers': list(range(1, len(bets_made) + 1)),
            'cumulative_profit': cumulative_profit,
            'bankroll_series': bankroll_series,
            'rolling_win_rate': rolling_win_rate,
            'total_bets': len(bets_made),
            'final_profit': cumulative_profit[-1] if cumulative_profit else 0
        }

