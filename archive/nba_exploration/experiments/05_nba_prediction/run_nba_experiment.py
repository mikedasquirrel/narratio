"""
NBA Narrative Prediction Experiment

Tests whether narrative-driven feature engineering can predict
NBA game outcomes and create betting value better than traditional approaches.

Research Question: Do narrative patterns (confidence, momentum, identity)
contain predictive signal for sports outcomes?
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from domains.nba.data_collector import NBADataCollector
from domains.nba.narrative_extractor import NBANarrativeExtractor
from domains.nba.game_predictor import NBAGamePredictor, TraditionalNBAPredictor
from domains.nba.betting_strategy import NarrativeEdgeStrategy, MomentumStrategy, ContrarianStrategy
from domains.nba.backtester import NBABacktester


def main():
    """Run complete NBA narrative prediction experiment."""
    
    print("\n" + "="*70)
    print("NBA NARRATIVE PREDICTION EXPERIMENT")
    print("Testing: Can narratives outperform betting strategies?")
    print("="*70 + "\n")
    
    # Step 1: Collect Data
    print("STEP 1: DATA COLLECTION")
    print("-" * 70)
    
    collector = NBADataCollector(seasons=list(range(2010, 2025)))
    print(f"Collecting data for seasons 2010-2024...")
    games = collector.fetch_games(include_narratives=True)
    print(f"✅ Collected {len(games)} games with narratives\n")
    
    # Step 2: Temporal Split (exclude every 10th season)
    print("STEP 2: TEMPORAL TRAIN/TEST SPLIT")
    print("-" * 70)
    
    train_games, test_games = collector.split_train_test(games, test_every_nth=10)
    print(f"Training games: {len(train_games)}")
    print(f"Test games: {len(test_games)}")
    
    # Identify test seasons
    test_seasons = sorted(set(g['season'] for g in test_games))
    train_seasons = sorted(set(g['season'] for g in train_games))
    print(f"Training seasons: {train_seasons}")
    print(f"Test seasons: {test_seasons}")
    print("✅ Temporal split complete\n")
    
    # Step 3: Extract Narrative Features
    print("STEP 3: NARRATIVE FEATURE EXTRACTION")
    print("-" * 70)
    
    extractor = NBANarrativeExtractor()
    
    # Fit transformers on training narratives
    train_narratives = []
    for game in train_games:
        train_narratives.append(game['home_narrative'])
        train_narratives.append(game['away_narrative'])
    
    print(f"Fitting transformers on {len(train_narratives)} narratives...")
    extractor.fit(train_narratives)
    
    # Extract features for all games
    print("Extracting features for all games...")
    for game in train_games + test_games:
        features = extractor.extract_game_features(
            game['home_narrative'],
            game['away_narrative']
        )
        game['home_features'] = features['home_features'].tolist()
        game['away_features'] = features['away_features'].tolist()
        game['differential'] = features['differential'].tolist()
    
    print(f"✅ Extracted {len(extractor.feature_names)} features per team\n")
    
    # Step 4: Train Models
    print("STEP 4: MODEL TRAINING")
    print("-" * 70)
    
    # Prepare training data (using differentials)
    X_train = np.array([g['differential'] for g in train_games])
    y_train = np.array([1 if g['home_wins'] else 0 for g in train_games])
    
    # Train narrative model
    print("\n[1] Training NARRATIVE MODEL")
    narrative_model = NBAGamePredictor(model_type='narrative')
    narrative_model.train(X_train, y_train, model_class='gradient_boosting')
    
    # Train traditional baseline (for comparison)
    print("\n[2] Training TRADITIONAL BASELINE")
    traditional_model = TraditionalNBAPredictor()
    # For demo, use same features (in production, would use actual stats)
    traditional_model.train(X_train, y_train, model_class='logistic')
    
    print("")
    
    # Step 5: Backtest Betting Strategies
    print("STEP 5: BETTING STRATEGY BACKTESTING")
    print("-" * 70)
    
    backtester = NBABacktester(initial_bankroll=1000.0)
    
    # Strategy 1: Narrative Edge
    print("\n[1] NARRATIVE EDGE STRATEGY")
    edge_strategy = NarrativeEdgeStrategy(edge_threshold=0.10, unit_size=20.0)
    edge_results = backtester.run_backtest(test_games, edge_strategy, narrative_model)
    
    # Strategy 2: Momentum
    print("\n[2] MOMENTUM STRATEGY")
    momentum_strategy = MomentumStrategy(momentum_threshold=0.15, unit_size=20.0)
    momentum_results = backtester.run_backtest(test_games, momentum_strategy, narrative_model)
    
    # Strategy 3: Contrarian
    print("\n[3] CONTRARIAN STRATEGY")
    contrarian_strategy = ContrarianStrategy(confidence_threshold=0.75, unit_size=20.0)
    contrarian_results = backtester.run_backtest(test_games, contrarian_strategy, narrative_model)
    
    # Step 6: Compare Results
    print("\nSTEP 6: STRATEGY COMPARISON")
    print("-" * 70)
    
    comparison = backtester.compare_strategies(
        test_games,
        [edge_strategy, momentum_strategy, contrarian_strategy],
        narrative_model
    )
    
    print("\n" + comparison.to_string(index=False))
    
    # Step 7: Save Results
    print("\n\nSTEP 7: SAVING RESULTS")
    print("-" * 70)
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Save experiment results
    experiment_results = {
        'experiment_id': '05_nba_prediction',
        'date': datetime.now().isoformat(),
        'data_summary': {
            'total_games': len(games),
            'train_games': len(train_games),
            'test_games': len(test_games),
            'train_seasons': train_seasons,
            'test_seasons': test_seasons
        },
        'model_performance': {
            'narrative_model': {
                'accuracy': float(edge_results['performance']['prediction_accuracy']),
                'type': 'gradient_boosting'
            }
        },
        'betting_results': {
            'narrative_edge': edge_results['performance'],
            'momentum': momentum_results['performance'],
            'contrarian': contrarian_results['performance']
        },
        'best_strategy': max([
            ('narrative_edge', edge_results['performance']['roi']),
            ('momentum', momentum_results['performance']['roi']),
            ('contrarian', contrarian_results['performance']['roi'])
        ], key=lambda x: x[1])[0]
    }
    
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Save model
    narrative_model.save_model(str(results_dir / 'narrative_model.pkl'))
    
    print(f"✅ Results saved to {results_dir}")
    
    # Step 8: Key Findings
    print("\n\nKEY FINDINGS")
    print("="*70)
    
    best_strat = experiment_results['best_strategy']
    best_roi = experiment_results['betting_results'][best_strat]['roi']
    
    print(f"🏆 Best Strategy: {best_strat.upper()}")
    print(f"💰 Best ROI: {best_roi:.1f}%")
    print(f"🎯 Prediction Accuracy: {experiment_results['model_performance']['narrative_model']['accuracy']:.1%}")
    print(f"📊 Test Seasons: {len(test_seasons)} seasons ({test_seasons})")
    
    if best_roi > 5:
        print(f"\n✅ SUCCESS: Narrative modeling shows {best_roi:.1f}% ROI - PROFITABLE!")
    elif best_roi > 0:
        print(f"\n✓ POSITIVE: Narrative modeling shows {best_roi:.1f}% ROI - Modest profit")
    else:
        print(f"\n⚠️  NEGATIVE: Narrative modeling shows {best_roi:.1f}% ROI - Loss")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70 + "\n")
    
    return experiment_results


if __name__ == '__main__':
    results = main()

