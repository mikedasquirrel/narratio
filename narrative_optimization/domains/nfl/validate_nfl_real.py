"""
NFL Full Validation with Proper Outcome Extraction

Extracts home game outcomes correctly from NFL data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from narrative_optimization.utils.validation_pipeline import ValidationPipeline

def main():
    print(f"\n{'='*60}")
    print("NFL COMPLETE VALIDATION")
    print(f"{'='*60}\n")
    
    # 1. Load features and games
    features_path = Path(__file__).parent.parent.parent / 'data' / 'features' / 'nfl_all_features.npz'
    games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    
    print(f"Loading features from: {features_path}")
    data = np.load(features_path, allow_pickle=True)
    X = data['features']
    
    print(f"Loading games from: {games_path}")
    with open(games_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} NFL games")
    print(f"✓ Feature matrix: {X.shape}\n")
    
    # 2. Extract outcomes
    print("Extracting outcomes...")
    y = np.array([1 if game['home_won'] else 0 for game in games])
    
    print(f"✓ Extracted outcomes")
    print(f"  Home win rate: {y.mean():.1%}\n")
    
    # 3. Create model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # 4. Run overall validation
    print("="*60)
    print("OVERALL VALIDATION")
    print("="*60)
    
    pipeline = ValidationPipeline('NFL-Overall')
    overall_results = pipeline.run_full_validation(
        X, y, model, temporal_split=True
    )
    
    pipeline.print_summary()
    
    # 5. Compile results
    final_results = {
        'overall': overall_results,
        'contexts': {},
        'data_summary': {
            'total_games': len(X),
            'total_features': X.shape[1],
            'home_win_rate': float(y.mean()),
            'validation_date': '2025-11-14'
        }
    }
    
    # 6. Save results
    results_path = Path(__file__).parent / 'nfl_betting_validated_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # 7. Print summary
    print(f"\n{'='*80}")
    print("NFL BETTING VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Test Accuracy: {overall_results['test_accuracy']:.1%}")
    print(f"95% CI: [{overall_results['confidence_interval_95']['lower']:.1%}, "
          f"{overall_results['confidence_interval_95']['upper']:.1%}]")
    print(f"P-value: {overall_results['p_value']:.6f}")
    print(f"Significant: {'YES' if overall_results['significant'] else 'NO'}")
    
    if 'backtesting' in overall_results and 'error' not in overall_results['backtesting']:
        bt = overall_results['backtesting']
        print(f"\nBACKTESTING:")
        print(f"  Bets: {bt['total_bets']}")
        print(f"  Accuracy: {bt['accuracy']:.1%}")
        print(f"  ROI: {bt['roi_pct']:.1%}")
        print(f"  Net Return: ${bt['net_return']:,.2f}")
    
    print(f"\n{'='*80}\n")
    print("✓ NFL VALIDATION COMPLETE!\n")
    
    return final_results

if __name__ == '__main__':
    results = main()

