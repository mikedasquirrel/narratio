"""
MLB Historical Backtester
Test betting strategy on historical games with real outcomes

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import json

from mlb_betting_model import MLBBettingModel
from mlb_betting_strategy import MLBBettingStrategy
from mlb_feature_pipeline import MLBFeaturePipeline


class MLBBacktester:
    """Backtest MLB betting strategy on historical games"""
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.results = []
    
    def run_backtest(self, games: List[Dict], model: MLBBettingModel,
                    feature_pipeline: MLBFeaturePipeline,
                    stats_dict: Dict, roster_dict: Dict,
                    odds_dict: Dict = None) -> Dict:
        """
        Run complete backtest on historical games
        
        Args:
            games: List of games with outcomes
            model: Trained betting model
            feature_pipeline: Feature extraction pipeline
            stats_dict: Team stats by team code
            roster_dict: Team rosters by team code
            odds_dict: Historical odds by game_id (if available)
            
        Returns:
            Backtest results dictionary
        """
        strategy = MLBBettingStrategy(
            bankroll=self.initial_bankroll,
            kelly_fraction=0.25,
            min_edge=0.05
        )
        
        bets_summary = []
        bankroll_history = [self.initial_bankroll]
        
        for game in games:
            game_id = str(game.get('game_id'))
            
            # Skip if no outcome
            if 'home_wins' not in game:
                continue
            
            # Extract features
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            home_stats = stats_dict.get(home_team, {})
            away_stats = stats_dict.get(away_team, {})
            home_roster = roster_dict.get(home_team, [])
            away_roster = roster_dict.get(away_team, [])
            
            features = feature_pipeline.extract_all_features(
                game, home_stats, away_stats, home_roster, away_roster
            )
            
            # Get model prediction
            prediction = model.predict_game(features)
            
            # Get odds (use defaults if not available)
            if odds_dict and game_id in odds_dict:
                home_odds = odds_dict[game_id].get('home_odds', -110)
                away_odds = odds_dict[game_id].get('away_odds', -110)
            else:
                # Default odds (slight home favorite)
                home_odds = -115
                away_odds = -105
            
            # Get bet recommendation (with journey context)
            game_context = {
                'journey_completion_score': features.get('journey_completion_score', 0),
                'quest_intensity': features.get('quest_intensity', 0),
                'high_journey_game': features.get('high_journey_game', 0)
            }
            rec = strategy.get_bet_recommendation(prediction, home_odds, away_odds, game_context)
            
            if rec:
                # Place bet
                bet_info = {
                    'home_team': game.get('home_team_name', home_team),
                    'away_team': game.get('away_team_name', away_team),
                    'date': game.get('game_date', 'unknown')
                }
                bet = strategy.place_bet(rec, game_id, bet_info)
                
                # Settle bet
                actual_winner = 'home' if game['home_wins'] else 'away'
                settlement = strategy.settle_bet(game_id, actual_winner)
                
                # Track
                bets_summary.append({
                    'game_id': game_id,
                    'date': bet_info['date'],
                    'matchup': f"{bet_info['away_team']} @ {bet_info['home_team']}",
                    'bet_side': bet['side'],
                    'amount': bet['amount'],
                    'odds': bet['odds'],
                    'predicted_prob': bet['probability'],
                    'edge': bet['edge'],
                    'won': settlement['won'],
                    'profit': settlement['profit'],
                    'bankroll': settlement['new_bankroll']
                })
                
                bankroll_history.append(settlement['new_bankroll'])
        
        # Get final stats
        performance = strategy.get_performance_stats()
        
        # Calculate additional metrics
        if bets_summary:
            df = pd.DataFrame(bets_summary)
            
            # Win rate by edge bucket
            edge_buckets = pd.cut(df['edge'], bins=[0, 0.05, 0.10, 0.15, 1.0], 
                                 labels=['5-10%', '10-15%', '15-20%', '20%+'])
            edge_analysis = df.groupby(edge_buckets)['won'].agg(['count', 'sum', 'mean'])
            
            performance['edge_analysis'] = edge_analysis.to_dict('index')
            performance['total_games_analyzed'] = len(games)
            performance['bet_frequency'] = len(bets_summary) / len(games)
        
        # Store results
        self.results = {
            'performance': performance,
            'bets': bets_summary,
            'bankroll_history': bankroll_history
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate text report of backtest results"""
        if not self.results:
            return "No backtest results available"
        
        perf = self.results['performance']
        
        report = []
        report.append("=" * 80)
        report.append("MLB BETTING BACKTEST RESULTS")
        report.append("=" * 80)
        report.append("")
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Total Bets: {perf['total_bets']}")
        report.append(f"  Win Rate: {perf['win_rate']:.1%}")
        report.append(f"  Total Wagered: ${perf['total_wagered']:.2f}")
        report.append(f"  Total Profit: ${perf['total_profit']:.2f}")
        report.append(f"  ROI: {perf['roi']:.1f}%")
        report.append("")
        report.append("BANKROLL:")
        report.append(f"  Initial: ${perf['initial_bankroll']:.2f}")
        report.append(f"  Final: ${perf['final_bankroll']:.2f}")
        report.append(f"  Return: {perf['return']:.1f}%")
        report.append(f"  Max Drawdown: {perf['max_drawdown']:.1f}%")
        report.append("")
        report.append("BETTING DISCIPLINE:")
        report.append(f"  Avg Bet Size: ${perf['avg_bet_size']:.2f}")
        report.append(f"  Bet Frequency: {perf.get('bet_frequency', 0):.1%}")
        report.append("")
        
        if 'edge_analysis' in perf:
            report.append("PERFORMANCE BY EDGE:")
            for edge_range, stats in perf['edge_analysis'].items():
                report.append(f"  {edge_range}: {stats['sum']}/{stats['count']} ({stats['mean']:.1%})")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save backtest results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def plot_bankroll_curve(self):
        """Generate bankroll curve visualization data"""
        if not self.results:
            return None
        
        bankroll_history = self.results['bankroll_history']
        
        return {
            'x': list(range(len(bankroll_history))),
            'y': bankroll_history,
            'title': 'Bankroll Over Time',
            'xlabel': 'Bets Placed',
            'ylabel': 'Bankroll ($)'
        }


if __name__ == '__main__':
    print("MLB Backtester - Example")
    print("=" * 80)
    print("\nThis module requires trained model and historical data.")
    print("Use train_mlb_complete.py to run full backtest.")
    print("=" * 80)

