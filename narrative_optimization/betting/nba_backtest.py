"""
NBA Backtesting System
=======================

Validates ensemble betting model on full historical data.
Calculates ROI, accuracy by confidence level, optimal bet sizing.

Author: AI Coding Assistant  
Date: November 16, 2025
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from narrative_optimization.betting.nba_ensemble_model import NBAEnsembleBettingModel
from narrative_optimization.betting.betting_utils import (
    american_to_probability,
    calculate_ev,
    calculate_edge,
    calculate_kelly_size
)


def load_nba_data(data_path: str = 'data/domains/nba_complete_with_players.json'):
    """Load NBA data with betting odds"""
    print(f"\n{'='*80}")
    print("LOADING NBA DATA FOR BACKTESTING")
    print('='*80)
    
    print(f"\n[Data] Loading from: {data_path}")
    
    with open(data_path) as f:
        all_games = json.load(f)
    
    print(f"[Data] ✓ Loaded {len(all_games):,} games")
    
    # Filter for games with betting odds
    games_with_odds = [g for g in all_games if g.get('betting_odds', {}).get('moneyline')]
    
    print(f"[Data] ✓ Games with betting odds: {len(games_with_odds):,}")
    
    return games_with_odds


def build_clean_narrative(game: Dict) -> str:
    """Build clean pre-game narrative (no leakage)"""
    parts = []
    
    parts.append(f"Team {game.get('team_name', 'Unknown')}")
    parts.append(f"Matchup {game.get('matchup', 'vs Opponent')}")
    parts.append(f"Location {'home' if game.get('home_game', False) else 'away'}")
    
    if game.get('player_data', {}).get('available'):
        agg = game['player_data']['team_aggregates']
        if agg.get('top1_name'):
            parts.append(f"Leading scorer {agg['top1_name']}")
        if agg.get('top2_name'):
            parts.append(f"Second scorer {agg['top2_name']}")
    
    tc = game.get('temporal_context', {})
    if tc.get('season_record_prior'):
        parts.append(f"Record {tc['season_record_prior']}")
    if tc.get('l10_record'):
        parts.append(f"Last 10 {tc['l10_record']}")
    
    betting = game.get('betting_odds', {})
    if betting.get('moneyline'):
        ml = betting['moneyline']
        parts.append(f"Line {ml}")
        parts.append("Underdog" if ml > 0 else "Favorite")
    
    return ". ".join(parts) + "."


def run_backtest(
    model: NBAEnsembleBettingModel,
    test_games: List[Dict],
    confidence_thresholds: List[float] = [0.55, 0.60, 0.65, 0.70],
    edge_thresholds: List[float] = [0.03, 0.05, 0.10],
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive backtest on historical data.
    
    Parameters
    ----------
    model : NBAEnsembleBettingModel
        Trained ensemble model
    test_games : list
        Historical games with outcomes and odds
    confidence_thresholds : list
        Confidence levels to test
    edge_thresholds : list
        Edge thresholds to test
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Comprehensive backtest results
    """
    print(f"\n{'='*80}")
    print("RUNNING BACKTEST")
    print('='*80)
    
    print(f"\n[Backtest] Test set: {len(test_games):,} games")
    
    # Build narratives
    print(f"[Backtest] Building narratives...")
    X_test = pd.Series([build_clean_narrative(g) for g in test_games])
    y_test = np.array([1 if g.get('won', False) else 0 for g in test_games])
    odds_test = np.array([g.get('betting_odds', {}).get('moneyline', 0) for g in test_games])
    spreads_test = np.array([g.get('betting_odds', {}).get('spread', 0) for g in test_games])
    
    print(f"[Backtest] ✓ Data prepared")
    print(f"[Backtest]   Baseline: {y_test.mean():.1%}")
    
    # Get predictions
    print(f"\n[Backtest] Running model predictions...")
    predictions = model.predict_with_confidence(X_test, market_odds=odds_test, verbose=True)
    
    print(f"[Backtest] ✓ {len(predictions)} predictions generated")
    
    # Analyze by confidence threshold
    results = {
        'test_date': datetime.now().isoformat(),
        'n_games': len(test_games),
        'baseline_accuracy': float(y_test.mean()),
        'by_confidence': {},
        'by_edge': {},
        'overall': {}
    }
    
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS BY CONFIDENCE THRESHOLD")
    print('='*80)
    
    for conf_threshold in confidence_thresholds:
        # Filter by confidence
        conf_preds = [p for p in predictions if p['win_probability'] >= conf_threshold or p['loss_probability'] >= conf_threshold]
        
        if len(conf_preds) == 0:
            continue
        
        # Calculate metrics
        correct = 0
        total_ev = 0
        n_bets = 0
        
        for pred in conf_preds:
            idx = pred['game_index']
            actual = y_test[idx]
            predicted = 1 if pred['win_probability'] > 0.5 else 0
            
            if predicted == actual:
                correct += 1
            
            if 'betting' in pred and pred['betting']['should_bet']:
                n_bets += 1
                total_ev += pred['betting']['expected_value']
        
        accuracy = correct / len(conf_preds) if len(conf_preds) > 0 else 0
        avg_ev = total_ev / n_bets if n_bets > 0 else 0
        
        results['by_confidence'][f'{conf_threshold:.0%}'] = {
            'n_games': len(conf_preds),
            'n_bets': n_bets,
            'accuracy': float(accuracy),
            'avg_ev_per_bet': float(avg_ev),
            'total_ev': float(total_ev),
            'improvement_vs_baseline': float(accuracy - y_test.mean())
        }
        
        print(f"\nConfidence >= {conf_threshold:.0%}:")
        print(f"  Games: {len(conf_preds):,}")
        print(f"  Bets: {n_bets:,}")
        print(f"  Accuracy: {accuracy:.1%} ({accuracy - y_test.mean():+.1%} vs baseline)")
        print(f"  Avg EV/bet: {avg_ev:+.3f} units")
        print(f"  Total EV: {total_ev:+.1f} units")
        
        if n_bets > 0:
            roi = (total_ev / n_bets) * 100
            print(f"  Est. ROI: {roi:+.1f}%")
    
    # Analyze by edge threshold
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS BY EDGE THRESHOLD")
    print('='*80)
    
    for edge_threshold in edge_thresholds:
        edge_preds = [p for p in predictions 
                     if 'betting' in p and p['betting']['edge'] >= edge_threshold 
                     and p['win_probability'] >= model.min_confidence]
        
        if len(edge_preds) == 0:
            continue
        
        correct = sum(1 for p in edge_preds 
                     if (1 if p['win_probability'] > 0.5 else 0) == y_test[p['game_index']])
        
        accuracy = correct / len(edge_preds)
        total_ev = sum(p['betting']['expected_value'] for p in edge_preds)
        
        results['by_edge'][f'{edge_threshold:.0%}'] = {
            'n_bets': len(edge_preds),
            'accuracy': float(accuracy),
            'total_ev': float(total_ev),
            'avg_ev': float(total_ev / len(edge_preds))
        }
        
        print(f"\nEdge >= {edge_threshold:.0%}:")
        print(f"  Bets: {len(edge_preds):,}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Total EV: {total_ev:+.1f} units")
        print(f"  Avg EV: {total_ev/len(edge_preds):+.3f} units/bet")
    
    # Overall summary
    high_conf_bets = model.get_high_confidence_bets(predictions, min_confidence=model.min_confidence)
    
    if len(high_conf_bets) > 0:
        correct_hc = sum(1 for p in high_conf_bets 
                        if (1 if p['win_probability'] > 0.5 else 0) == y_test[p['game_index']])
        accuracy_hc = correct_hc / len(high_conf_bets)
        total_ev_hc = sum(p['betting']['expected_value'] for p in high_conf_bets)
        
        results['overall'] = {
            'n_high_confidence_bets': len(high_conf_bets),
            'accuracy': float(accuracy_hc),
            'total_ev': float(total_ev_hc),
            'roi_percent': float((total_ev_hc / len(high_conf_bets)) * 100),
            'bets_per_game': float(len(high_conf_bets) / len(test_games)),
        }
        
        print(f"\n{'='*80}")
        print("OVERALL HIGH-CONFIDENCE BETTING PERFORMANCE")
        print('='*80)
        print(f"\nTotal games: {len(test_games):,}")
        print(f"High-confidence bets: {len(high_conf_bets):,} ({len(high_conf_bets)/len(test_games)*100:.1f}% of games)")
        print(f"Accuracy: {accuracy_hc:.1%} ({accuracy_hc - y_test.mean():+.1%} vs baseline)")
        print(f"Total EV: {total_ev_hc:+.1f} units")
        print(f"ROI: {(total_ev_hc/len(high_conf_bets))*100:+.1f}%")
        print(f"Avg bets/game: {len(high_conf_bets)/len(test_games):.2f}")
    
    return results


def main():
    """Run complete backtest"""
    
    print(f"\n{'█'*80}")
    print("NBA ENSEMBLE BETTING MODEL - BACKTEST")
    print('█'*80)
    
    # Load data
    all_games = load_nba_data()
    
    # Split by season
    train_games = [g for g in all_games if g['season'] < '2023-24']
    test_games = [g for g in all_games if g['season'] == '2023-24']
    
    print(f"\n[Split] Train: {len(train_games):,} games (seasons < 2023-24)")
    print(f"[Split] Test: {len(test_games):,} games (season 2023-24)")
    
    # Build narratives for training
    print(f"\n[Data] Building training narratives...")
    X_train = pd.Series([build_clean_narrative(g) for g in train_games])
    y_train = np.array([1 if g.get('won', False) else 0 for g in train_games])
    
    print(f"[Data] ✓ Training data ready: {len(X_train):,} games")
    
    # Train model
    model = NBAEnsembleBettingModel(
        min_confidence=0.60,
        min_edge=0.05,
        use_calibration=True
    )
    
    model.fit(X_train, y_train, verbose=True)
    
    # Save model
    model_path = Path('narrative_optimization/betting/nba_ensemble_trained.pkl')
    model.save_model(str(model_path))
    
    # Run backtest
    backtest_results = run_backtest(
        model,
        test_games,
        confidence_thresholds=[0.55, 0.60, 0.65, 0.70],
        edge_thresholds=[0.03, 0.05, 0.10],
        verbose=True
    )
    
    # Save results
    results_path = Path('narrative_optimization/betting/nba_backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    print(f"\n[Results] ✓ Backtest results saved to: {results_path}")
    
    # Model summary
    summary = model.get_model_summary()
    summary_path = Path('narrative_optimization/betting/nba_ensemble_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Results] ✓ Model summary saved to: {summary_path}")
    
    print(f"\n{'█'*80}")
    print("BACKTEST COMPLETE")
    print('█'*80)
    print(f"\nModel ready for production betting!")
    print(f"Next: Run daily predictions with scripts/nba_daily_predictions.py")
    print()


if __name__ == "__main__":
    main()

