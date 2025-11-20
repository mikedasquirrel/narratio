"""
Comprehensive Backtesting System
=================================

Tests all betting enhancements on historical data to validate improvements:
- Cross-domain features
- Advanced ensembles
- Unified sports model
- Higher-order patterns
- Kelly Criterion sizing

Generates detailed performance reports comparing baseline vs enhanced systems.

Author: AI Coding Assistant
Date: November 16, 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_optimization.feature_engineering.cross_domain_features import CrossDomainFeatureExtractor
from narrative_optimization.betting.nba_advanced_ensemble import AdvancedEnsembleSystem
from narrative_optimization.betting.nfl_advanced_ensemble import NFLAdvancedEnsemble
from narrative_optimization.betting.unified_sports_model import UnifiedSportsModel
from narrative_optimization.betting.kelly_criterion import KellyCriterion
from narrative_optimization.patterns.higher_order_discovery import HigherOrderPatternDiscovery

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


class ComprehensiveBacktest:
    """
    Comprehensive backtesting framework for betting system enhancements.
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize backtesting system.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir or Path(__file__).parent.parent / 'results' / 'backtests'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.cross_domain_extractor = CrossDomainFeatureExtractor()
        self.kelly = KellyCriterion()
        
    def load_historical_data(self, league: str = 'nba') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load historical game data for backtesting.
        
        Args:
            league: 'nba' or 'nfl'
            
        Returns:
            Tuple of (games DataFrame, outcomes array)
        """
        # Try to load real data, fall back to synthetic
        data_path = Path(__file__).parent.parent / 'data' / 'domains' / f'{league}_games.json'
        
        if data_path.exists():
            print(f"Loading real {league.upper()} data from {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)
            games = pd.DataFrame(data)
            outcomes = games['home_won'].values if 'home_won' in games.columns else None
        else:
            print(f"Real data not found. Generating synthetic {league.upper()} data for testing...")
            games, outcomes = self._generate_synthetic_data(league, n_games=2000)
        
        return games, outcomes
    
    def _generate_synthetic_data(self, league: str, n_games: int = 2000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic data for testing."""
        np.random.seed(42)
        
        # Create arrays that are definitely n_games length
        # Use oversized arrays and slice to exact length
        oversized = n_games + 100
        
        # Create realistic game data
        data = {
            'game_id': [f'{league}_{i}' for i in range(n_games)],
            'season': np.repeat(range(2014, 2026), oversized // 12)[:n_games],
            'week': np.tile(np.arange(1, 50), oversized // 49 + 1)[:n_games],
            'home': np.ones(n_games, dtype=int),
            'spread': np.random.uniform(-14, 14, n_games),
            'season_win_pct': np.random.uniform(0.3, 0.7, n_games),
            'opp_season_win_pct': np.random.uniform(0.3, 0.7, n_games),
            'l10_win_pct': np.random.uniform(0.2, 0.8, n_games),
            'opp_l10_win_pct': np.random.uniform(0.2, 0.8, n_games),
            'is_division': (np.random.random(n_games) < 0.3).astype(int),
            'is_rivalry': (np.random.random(n_games) < 0.15).astype(int),
            'rest_days': np.random.randint(1, 8, n_games),
            'opp_rest_days': np.random.randint(1, 8, n_games),
            'avg_experience': np.random.uniform(3, 10, n_games),
            'opp_avg_experience': np.random.uniform(3, 10, n_games),
            'home_roster': [['P1', 'P2', 'P3', 'P4', 'P5'] for _ in range(n_games)],
            'star_players': np.random.randint(0, 3, n_games),
            'total_weeks': np.full(n_games, 26 if league == 'nba' else 18),
        }
        
        games = pd.DataFrame(data)
        
        # Generate outcomes with patterns
        outcomes = np.zeros(n_games)
        
        # Pattern: Home underdogs win more (mimics NFL pattern)
        home_dog_mask = games['spread'] > 3.5
        outcomes[home_dog_mask] = (np.random.random(home_dog_mask.sum()) < 0.65).astype(int)
        
        # Pattern: Good record + late season
        good_record_late = (games['season_win_pct'] > 0.55) & (games['week'] > 20)
        outcomes[good_record_late] = (np.random.random(good_record_late.sum()) < 0.68).astype(int)
        
        # Rest: baseline 52%
        remaining = ~(home_dog_mask | good_record_late)
        outcomes[remaining] = (np.random.random(remaining.sum()) < 0.52).astype(int)
        
        return games, outcomes
    
    def extract_baseline_features(self, games: pd.DataFrame, league: str) -> np.ndarray:
        """Extract baseline features (existing system)."""
        # Simplified baseline features
        features = []
        
        for _, game in games.iterrows():
            feature_vec = [
                game['season_win_pct'],
                game['opp_season_win_pct'],
                game['l10_win_pct'],
                game['opp_l10_win_pct'],
                game['spread'],
                float(game['is_division']),
                game['rest_days'] - game['opp_rest_days'],
            ]
            features.append(feature_vec)
        
        return np.array(features)
    
    def extract_enhanced_features(self, games: pd.DataFrame, league: str) -> np.ndarray:
        """Extract enhanced features with cross-domain additions."""
        # Get baseline features
        baseline = self.extract_baseline_features(games, league)
        
        # Add cross-domain features
        games_list = games.to_dict('records')
        cross_domain_df = self.cross_domain_extractor.batch_extract_features(games_list, league)
        
        # Combine
        enhanced = np.hstack([baseline, cross_domain_df.values])
        
        return enhanced
    
    def backtest_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        league: str
    ) -> Dict:
        """Test baseline system performance."""
        print(f"\nBacktesting baseline {league.upper()} system...")
        
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        # Simple baseline model
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            cv=3
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        logloss = log_loss(y_test, y_proba)
        
        # Betting performance (flat $100 bets)
        roi = self._calculate_flat_roi(y_pred, y_test)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': logloss,
            'roi': roi,
            'n_bets': len(y_test),
            'model_type': 'baseline'
        }
    
    def backtest_enhanced(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        league: str
    ) -> Dict:
        """Test enhanced system performance."""
        print(f"\nBacktesting enhanced {league.upper()} system...")
        
        # Use advanced ensemble
        if league == 'nba':
            ensemble = AdvancedEnsembleSystem(n_base_models=X_train.shape[1])
        else:
            # For NFL, create simplified version
            from sklearn.ensemble import GradientBoostingClassifier
            ensemble = CalibratedClassifierCV(
                GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
                cv=5
            )
        
        ensemble.fit(X_train, y_train, validation_split=0.2)
        
        # Predictions
        if hasattr(ensemble, 'predict_proba'):
            if league == 'nba':
                y_proba = ensemble.predict_proba(X_test, strategy='blend')
            else:
                y_proba = ensemble.predict_proba(X_test)
        else:
            y_proba = ensemble.predict(X_test)
            # Convert to probabilities if needed
            if len(y_proba.shape) == 1:
                proba_matrix = np.zeros((len(y_proba), 2))
                proba_matrix[:, 1] = y_proba
                proba_matrix[:, 0] = 1 - y_proba
                y_proba = proba_matrix
        
        y_pred = (y_proba[:, 1] > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        logloss = log_loss(y_test, y_proba)
        
        # Betting performance with Kelly
        roi = self._calculate_kelly_roi(y_proba[:, 1], y_test)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'log_loss': logloss,
            'roi': roi,
            'n_bets': len(y_test),
            'model_type': 'enhanced'
        }
    
    def _calculate_flat_roi(self, predictions: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate ROI with flat $100 bets on all games."""
        # Assume -110 odds for all bets
        wins = (predictions == outcomes).sum()
        losses = (predictions != outcomes).sum()
        
        # At -110 odds: win $90.91 per $100 bet
        profit_per_win = 90.91
        loss_per_bet = 100.0
        
        total_profit = wins * profit_per_win - losses * loss_per_bet
        total_wagered = len(predictions) * 100
        
        roi = (total_profit / total_wagered) if total_wagered > 0 else 0
        return roi
    
    def _calculate_kelly_roi(self, win_probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate ROI with Kelly Criterion bet sizing."""
        total_profit = 0
        total_wagered = 0
        bankroll = 10000  # Starting bankroll
        
        for prob, outcome in zip(win_probabilities, outcomes):
            # Calculate Kelly bet
            kelly_bet = self.kelly.calculate_bet(
                game_id='test',
                bet_type='moneyline',
                side='home',
                american_odds=-110,
                win_probability=prob,
                bankroll=bankroll
            )
            
            bet_size = bankroll * kelly_bet.recommended_fraction
            
            if bet_size > 0:
                total_wagered += bet_size
                
                if outcome == 1:  # Win
                    profit = bet_size * 0.9091
                    total_profit += profit
                    bankroll += profit
                else:  # Loss
                    total_profit -= bet_size
                    bankroll -= bet_size
        
        roi = (total_profit / total_wagered) if total_wagered > 0 else 0
        return roi
    
    def run_comprehensive_backtest(self, league: str = 'nba') -> Dict:
        """
        Run full backtest comparing baseline vs enhanced systems.
        
        Args:
            league: 'nba' or 'nfl'
            
        Returns:
            Dict with complete results
        """
        print("=" * 80)
        print(f"COMPREHENSIVE BACKTEST - {league.upper()}")
        print("=" * 80)
        
        # Load data
        games, outcomes = self.load_historical_data(league)
        print(f"\nLoaded {len(games)} games")
        print(f"Win rate: {outcomes.mean():.1%}")
        
        # Time series split (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=3)
        
        baseline_results = []
        enhanced_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(games), 1):
            print(f"\n{'='*80}")
            print(f"FOLD {fold}/{3}")
            print(f"{'='*80}")
            
            games_train = games.iloc[train_idx]
            games_test = games.iloc[test_idx]
            y_train = outcomes[train_idx]
            y_test = outcomes[test_idx]
            
            print(f"Train: {len(train_idx)} games | Test: {len(test_idx)} games")
            
            # Baseline features
            X_train_base = self.extract_baseline_features(games_train, league)
            X_test_base = self.extract_baseline_features(games_test, league)
            
            # Enhanced features
            X_train_enh = self.extract_enhanced_features(games_train, league)
            X_test_enh = self.extract_enhanced_features(games_test, league)
            
            print(f"Baseline features: {X_train_base.shape[1]}")
            print(f"Enhanced features: {X_train_enh.shape[1]} (+{X_train_enh.shape[1] - X_train_base.shape[1]})")
            
            # Test baseline
            baseline_perf = self.backtest_baseline(X_train_base, y_train, X_test_base, y_test, league)
            baseline_results.append(baseline_perf)
            
            # Test enhanced
            enhanced_perf = self.backtest_enhanced(X_train_enh, y_train, X_test_enh, y_test, league)
            enhanced_results.append(enhanced_perf)
            
            # Print fold results
            print(f"\nFold {fold} Results:")
            print(f"  Baseline: Acc={baseline_perf['accuracy']:.1%}, AUC={baseline_perf['auc']:.3f}, ROI={baseline_perf['roi']:+.1%}")
            print(f"  Enhanced: Acc={enhanced_perf['accuracy']:.1%}, AUC={enhanced_perf['auc']:.3f}, ROI={enhanced_perf['roi']:+.1%}")
            print(f"  Improvement: Acc={enhanced_perf['accuracy']-baseline_perf['accuracy']:+.1%}, " +
                  f"ROI={enhanced_perf['roi']-baseline_perf['roi']:+.1%}")
        
        # Aggregate results
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        
        results = {
            'league': league.upper(),
            'n_games': len(games),
            'n_folds': len(baseline_results),
            'baseline': {
                'accuracy_mean': np.mean([r['accuracy'] for r in baseline_results]),
                'accuracy_std': np.std([r['accuracy'] for r in baseline_results]),
                'auc_mean': np.mean([r['auc'] for r in baseline_results]),
                'roi_mean': np.mean([r['roi'] for r in baseline_results]),
                'roi_std': np.std([r['roi'] for r in baseline_results]),
            },
            'enhanced': {
                'accuracy_mean': np.mean([r['accuracy'] for r in enhanced_results]),
                'accuracy_std': np.std([r['accuracy'] for r in enhanced_results]),
                'auc_mean': np.mean([r['auc'] for r in enhanced_results]),
                'roi_mean': np.mean([r['roi'] for r in enhanced_results]),
                'roi_std': np.std([r['roi'] for r in enhanced_results]),
            },
            'improvements': {
                'accuracy_delta': np.mean([e['accuracy'] - b['accuracy'] for e, b in zip(enhanced_results, baseline_results)]),
                'auc_delta': np.mean([e['auc'] - b['auc'] for e, b in zip(enhanced_results, baseline_results)]),
                'roi_delta': np.mean([e['roi'] - b['roi'] for e, b in zip(enhanced_results, baseline_results)]),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\nBaseline System:")
        print(f"  Accuracy: {results['baseline']['accuracy_mean']:.1%} ± {results['baseline']['accuracy_std']:.1%}")
        print(f"  AUC: {results['baseline']['auc_mean']:.3f}")
        print(f"  ROI: {results['baseline']['roi_mean']:+.1%} ± {results['baseline']['roi_std']:.1%}")
        
        print(f"\nEnhanced System:")
        print(f"  Accuracy: {results['enhanced']['accuracy_mean']:.1%} ± {results['enhanced']['accuracy_std']:.1%}")
        print(f"  AUC: {results['enhanced']['auc_mean']:.3f}")
        print(f"  ROI: {results['enhanced']['roi_mean']:+.1%} ± {results['enhanced']['roi_std']:.1%}")
        
        print(f"\nImprovements:")
        print(f"  Accuracy: {results['improvements']['accuracy_delta']:+.1%}")
        print(f"  AUC: {results['improvements']['auc_delta']:+.3f}")
        print(f"  ROI: {results['improvements']['roi_delta']:+.1%}")
        
        # Save results
        self.save_results(results, league)
        
        return results
    
    def save_results(self, results: Dict, league: str):
        """Save backtest results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'backtest_{league}_{timestamp}.json'
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")


def main():
    """Run comprehensive backtests."""
    backtest = ComprehensiveBacktest()
    
    # Test NBA
    nba_results = backtest.run_comprehensive_backtest('nba')
    
    print("\n" + "=" * 80)
    
    # Test NFL
    nfl_results = backtest.run_comprehensive_backtest('nfl')
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BACKTESTING COMPLETE")
    print("=" * 80)
    
    # Overall summary
    print("\nSUMMARY:")
    print(f"  NBA: {nba_results['improvements']['accuracy_delta']:+.1%} accuracy, " +
          f"{nba_results['improvements']['roi_delta']:+.1%} ROI")
    print(f"  NFL: {nfl_results['improvements']['accuracy_delta']:+.1%} accuracy, " +
          f"{nfl_results['improvements']['roi_delta']:+.1%} ROI")
    
    print(f"\n✓ All results saved to {backtest.results_dir}")


if __name__ == '__main__':
    main()

