"""
Complete MLB Betting System Training Pipeline
Ties together: data collection → feature extraction → model training → backtesting → deployment

Author: Narrative Optimization Framework
Date: November 2024
"""

import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mlb_game_collector import MLBGameCollector
from mlb_feature_pipeline import MLBFeaturePipeline
from mlb_betting_model import MLBBettingModel
from mlb_betting_strategy import MLBBettingStrategy
from mlb_backtester import MLBBacktester


class MLBTrainingPipeline:
    """Complete training pipeline for MLB betting system"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / 'trained_models'
        self.output_dir.mkdir(exist_ok=True)
        
        self.collector = MLBGameCollector()
        self.pipeline = MLBFeaturePipeline()
        self.model = None
        
    def step1_collect_data(self, seasons: list = [2020, 2021, 2022, 2023, 2024]) -> dict:
        """
        Step 1: Collect game data and rosters
        
        Returns:
            Dictionary with games, stats, rosters
        """
        print("=" * 80)
        print("STEP 1: COLLECTING MLB DATA")
        print("=" * 80)
        
        all_games = []
        stats_dict = {}
        roster_dict = {}
        
        # For demo purposes, create synthetic data
        # In production, this would call real APIs
        print("\nGenerating synthetic data for demonstration...")
        
        # Generate games
        team_codes = list(self.collector.teams.keys())
        n_games = 2000  # Smaller dataset for demo
        
        for i in range(n_games):
            home_team = np.random.choice(team_codes)
            away_team = np.random.choice([t for t in team_codes if t != home_team])
            
            # Check if rivalry
            is_rivalry = self.collector._is_rivalry(home_team, away_team)
            
            # Journey features: game number and month (correlated)
            game_number = int(np.random.randint(1, 163))
            if game_number <= 40:
                month = int(np.random.choice([4, 5]))
            elif game_number <= 100:
                month = int(np.random.choice([6, 7]))
            elif game_number <= 130:
                month = 8
            else:
                month = 9  # Highest journey completion
            
            game = {
                'game_id': 2024000000 + i,
                'game_date': f'2024-{month:02d}-{np.random.randint(1,28):02d}',
                'game_number': game_number,  # KEY for journey features
                'home_team': home_team,
                'away_team': away_team,
                'home_team_name': self.collector.teams[home_team]['name'],
                'away_team_name': self.collector.teams[away_team]['name'],
                'venue': 'Ballpark',
                'is_rivalry': is_rivalry,
                'is_historic_stadium': np.random.random() < 0.1,
                'month': month,
                'home_pitcher': f'Pitcher_{home_team}_{i}',
                'away_pitcher': f'Pitcher_{away_team}_{i}',
                'home_wins': np.random.random() < 0.54  # Slight home advantage
            }
            
            all_games.append(game)
        
        print(f"  ✓ Generated {len(all_games)} games")
        
        # Generate team stats
        for team_code in team_codes:
            wins = np.random.randint(60, 100)
            losses = 162 - wins
            stats_dict[team_code] = {
                'wins': wins,
                'losses': losses,
                'win_pct': wins / 162
            }
        
        print(f"  ✓ Generated stats for {len(stats_dict)} teams")
        
        # Generate rosters (nominative features!)
        first_names = ['Aaron', 'Juan', 'Mike', 'Chris', 'Jose', 'Carlos', 'Alex', 'Tyler', 
                      'Jacob', 'Ryan', 'Matt', 'Brandon', 'Kyle', 'Nick', 'Josh']
        last_names = ['Smith', 'Rodriguez', 'Martinez', 'Johnson', 'Garcia', 'Williams',
                     'Brown', 'Jones', 'Davis', 'Miller', 'Wilson', 'Anderson', 'Taylor']
        
        for team_code in team_codes:
            roster = []
            for j in range(25):  # 25 players per roster
                full_name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                roster.append({
                    'full_name': full_name,
                    'position_code': 'P' if j < 10 else np.random.choice(['1B', '2B', '3B', 'SS', 'OF', 'C'])
                })
            roster_dict[team_code] = roster
        
        print(f"  ✓ Generated rosters for {len(roster_dict)} teams (25 players each)")
        
        data = {
            'games': all_games,
            'stats_dict': stats_dict,
            'roster_dict': roster_dict
        }
        
        # Save raw data
        data_file = self.output_dir / 'mlb_training_data.json'
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n  ✓ Saved data to {data_file}")
        return data
    
    def step2_extract_features(self, data: dict) -> tuple:
        """
        Step 2: Extract nominative + statistical features
        
        Returns:
            (X, y, feature_names, game_ids)
        """
        print("\n" + "=" * 80)
        print("STEP 2: EXTRACTING FEATURES (Nominative + Statistical)")
        print("=" * 80)
        
        games = data['games']
        stats_dict = data['stats_dict']
        roster_dict = data['roster_dict']
        
        print(f"\nProcessing {len(games)} games...")
        
        # Extract features
        X, game_ids, feature_names = self.pipeline.extract_batch(games, stats_dict, roster_dict)
        
        # Extract outcomes
        y = np.array([1 if g['home_wins'] else 0 for g in games])
        
        print(f"\n  ✓ Feature matrix shape: {X.shape}")
        print(f"  ✓ Total features: {len(feature_names)}")
        print(f"  ✓ Games with outcomes: {len(y)}")
        print(f"  ✓ Home win rate: {y.mean():.3f}")
        
        # Show feature breakdown
        nom_features = [f for f in feature_names if any(x in f for x in ['name', 'player', 'pitcher', 'international', 'complexity'])]
        stat_features = [f for f in feature_names if any(x in f for x in ['win', 'loss', 'pct', 'diff'])]
        context_features = [f for f in feature_names if any(x in f for x in ['rivalry', 'stadium', 'venue', 'month'])]
        
        print(f"\n  Feature Breakdown:")
        print(f"    Nominative features: {len(nom_features)}")
        print(f"    Statistical features: {len(stat_features)}")
        print(f"    Context features: {len(context_features)}")
        print(f"    Interaction features: {len(feature_names) - len(nom_features) - len(stat_features) - len(context_features)}")
        
        # Save features
        np.savez(
            self.output_dir / 'mlb_features.npz',
            X=X,
            y=y,
            feature_names=feature_names,
            game_ids=game_ids
        )
        
        print(f"\n  ✓ Saved features to mlb_features.npz")
        
        return X, y, feature_names, game_ids
    
    def step3_train_model(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> MLBBettingModel:
        """
        Step 3: Train ensemble betting model
        
        Returns:
            Trained model
        """
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING BETTING MODEL")
        print("=" * 80)
        
        print(f"\nTraining ensemble model on {len(X)} games...")
        
        self.model = MLBBettingModel()
        metrics = self.model.train(X, y, feature_names)
        
        print("\n  Model Performance:")
        print(f"    Logistic Regression - Val Acc: {metrics['logistic']['val_accuracy']:.4f}, AUC: {metrics['logistic']['val_auc']:.4f}")
        print(f"    Random Forest - Val Acc: {metrics['random_forest']['val_accuracy']:.4f}, AUC: {metrics['random_forest']['val_auc']:.4f}")
        print(f"    Gradient Boosting - Val Acc: {metrics['gradient_boost']['val_accuracy']:.4f}, AUC: {metrics['gradient_boost']['val_auc']:.4f}")
        print(f"    Ensemble - Val Acc: {metrics['ensemble']['val_accuracy']:.4f}, AUC: {metrics['ensemble']['val_auc']:.4f}")
        
        # Save model
        model_file = self.output_dir / 'mlb_betting_model.pkl'
        self.model.save(str(model_file))
        
        print(f"\n  ✓ Model saved to {model_file}")
        
        return self.model
    
    def step4_backtest(self, data: dict, model: MLBBettingModel) -> dict:
        """
        Step 4: Backtest on historical games
        
        Returns:
            Backtest results
        """
        print("\n" + "=" * 80)
        print("STEP 4: BACKTESTING STRATEGY")
        print("=" * 80)
        
        games = data['games']
        stats_dict = data['stats_dict']
        roster_dict = data['roster_dict']
        
        # Use last 20% for backtesting
        test_size = int(len(games) * 0.2)
        test_games = games[-test_size:]
        
        print(f"\nBacktesting on {len(test_games)} games...")
        
        backtester = MLBBacktester(initial_bankroll=1000.0)
        results = backtester.run_backtest(
            test_games,
            model,
            self.pipeline,
            stats_dict,
            roster_dict
        )
        
        # Print report
        print("\n" + backtester.generate_report())
        
        # Save results
        results_file = self.output_dir / 'mlb_backtest_results.json'
        backtester.save_results(str(results_file))
        
        return results
    
    def step5_deploy_config(self, model_performance: dict, backtest_results: dict):
        """
        Step 5: Create deployment configuration
        """
        print("\n" + "=" * 80)
        print("STEP 5: CREATING DEPLOYMENT CONFIG")
        print("=" * 80)
        
        perf = backtest_results['performance']
        
        config = {
            'model_version': '1.0.0',
            'trained_date': datetime.now().isoformat(),
            'model_path': 'trained_models/mlb_betting_model.pkl',
            'feature_count': len(self.pipeline.feature_names),
            'training_games': model_performance.get('training_games', 'N/A'),
            'validation_accuracy': model_performance.get('ensemble', {}).get('val_accuracy', 0),
            'backtest_performance': {
                'win_rate': perf.get('win_rate', 0),
                'roi': perf.get('roi', 0),
                'total_bets': perf.get('total_bets', 0),
                'return': perf.get('return', 0)
            },
            'strategy_params': {
                'kelly_fraction': 0.25,
                'min_edge': 0.05,
                'max_bet_pct': 0.05
            },
            'status': 'READY',
            'notes': [
                'Nominative features (player names) + statistics = core approach',
                '32 players per game provides rich nominative context',
                'Ensemble model: Logistic + Random Forest + Gradient Boosting',
                'Kelly Criterion bet sizing with fractional (0.25)'
            ]
        }
        
        config_file = self.output_dir / 'deployment_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n  ✓ Deployment config saved to {config_file}")
        print(f"\n  Model Status: {config['status']}")
        print(f"  Win Rate: {config['backtest_performance']['win_rate']:.1%}")
        print(f"  ROI: {config['backtest_performance']['roi']:.1f}%")
        print(f"  Total Bets: {config['backtest_performance']['total_bets']}")
        
        return config
    
    def run_complete_pipeline(self):
        """Run all steps in sequence"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "MLB BETTING SYSTEM TRAINING PIPELINE" + " " * 22 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\nNominative Features + Statistics → 55-60% Accuracy, 35-45% ROI Target\n")
        
        # Step 1: Collect data
        data = self.step1_collect_data()
        
        # Step 2: Extract features
        X, y, feature_names, game_ids = self.step2_extract_features(data)
        
        # Step 3: Train model
        model = self.step3_train_model(X, y, feature_names)
        
        # Step 4: Backtest
        backtest_results = self.step4_backtest(data, model)
        
        # Step 5: Deploy config
        config = self.step5_deploy_config({'training_games': len(X)}, backtest_results)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        print(f"\nAll files saved to: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Review backtest results")
        print("  2. Test predictions via web interface (/mlb)")
        print("  3. Deploy for live betting (connect to real odds)")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    # Run complete pipeline
    pipeline = MLBTrainingPipeline()
    pipeline.run_complete_pipeline()

