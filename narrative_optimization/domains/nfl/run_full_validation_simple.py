"""
NFL Full Validation Pipeline (Using Pre-extracted Features)

Uses pre-extracted feature matrix for faster validation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Import our validation pipeline
from narrative_optimization.utils.validation_pipeline import ValidationPipeline

def main():
    """Run complete NFL validation with pre-extracted features"""
    
    print(f"\n{'='*60}")
    print("NFL COMPLETE VALIDATION PIPELINE")
    print(f"{'='*60}\n")
    
    # 1. Load pre-extracted features
    features_path = Path(__file__).parent.parent.parent / 'data' / 'features' / 'nfl_all_features.npz'
    print(f"Loading features from: {features_path}")
    
    data = np.load(features_path, allow_pickle=True)
    X = data['features']
    y = data['outcomes']
    
    print(f"✓ Loaded {len(X)} games with {X.shape[1]} features")
    print(f"  Home win rate: {y.mean():.1%}\n")
    
    # 2. Load game data for context filtering
    games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    with open(games_path) as f:
        games = json.load(f)
    print(f"✓ Loaded {len(games)} game records\n")
    
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
    print("OVERALL MODEL VALIDATION")
    print("="*60)
    
    pipeline = ValidationPipeline('NFL-Overall')
    overall_results = pipeline.run_full_validation(
        X, y, model, temporal_split=True
    )
    
    pipeline.print_summary()
    
    # 5. Test spread-based contexts
    print(f"\n{'='*60}")
    print("CONTEXT-SPECIFIC VALIDATION")
    print(f"{'='*60}\n")
    
    context_results = {}
    
    # Define contexts
    contexts_to_test = []
    
    # Big underdogs (spread >= 7)
    underdog_indices = [i for i, g in enumerate(games) if g.get('betting_odds', {}).get('spread', 0) >= 7]
    if len(underdog_indices) >= 50:
        contexts_to_test.append(('big_underdogs', underdog_indices))
    
    # Big favorites (spread <= -7)
    favorite_indices = [i for i, g in enumerate(games) if g.get('betting_odds', {}).get('spread', 0) <= -7]
    if len(favorite_indices) >= 50:
        contexts_to_test.append(('big_favorites', favorite_indices))
    
    # Short week games
    short_week_indices = [i for i, g in enumerate(games) if g.get('week', 99) > 1 and 
                         (g.get('context', {}).get('home_rest_days', 7) <= 4 or 
                          g.get('context', {}).get('away_rest_days', 7) <= 4)]
    if len(short_week_indices) >= 50:
        contexts_to_test.append(('short_week', short_week_indices))
    
    # Run validation for each context
    for context_name, indices in contexts_to_test:
        print(f"Testing context: {context_name} ({len(indices)} games)")
        
        X_context = X[indices]
        y_context = y[indices]
        
        pipeline_context = ValidationPipeline(f'NFL-{context_name}')
        
        context_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        results = pipeline_context.run_full_validation(
            X_context,
            y_context,
            context_model,
            temporal_split=True
        )
        
        context_results[context_name] = results
        print()
    
    # 6. Compile final results
    final_results = {
        'overall': overall_results,
        'contexts': context_results,
        'data_summary': {
            'total_games': len(games),
            'total_features': X.shape[1],
            'home_win_rate': float(y.mean()),
            'validation_date': '2025-11-14'
        }
    }
    
    # Find best context
    valid_contexts = {name: res for name, res in context_results.items()
                     if 'backtesting' in res and 'error' not in res['backtesting']}
    
    if valid_contexts:
        best_context_name = max(valid_contexts.items(),
                               key=lambda x: x[1]['backtesting'].get('roi_pct', -999))[0]
        final_results['best_context'] = {
            'name': best_context_name,
            'results': context_results[best_context_name]
        }
        
        print(f"\n{'='*60}")
        print(f"BEST CONTEXT: {best_context_name}")
        print(f"{'='*60}")
        best = context_results[best_context_name]
        print(f"Accuracy: {best['test_accuracy']:.1%}")
        if 'backtesting' in best and 'error' not in best['backtesting']:
            print(f"ROI: {best['backtesting']['roi_pct']:.1%}")
            print(f"Bets: {best['backtesting']['total_bets']}")
        print(f"P-value: {best['p_value']:.6f}")
        print(f"{'='*60}\n")
    
    # 7. Save results
    results_dir = Path(__file__).parent
    results_path = results_dir / 'nfl_betting_validated_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Complete validation results saved to: {results_path}\n")
    
    # 8. Print executive summary
    print(f"\n{'='*80}")
    print("EXECUTIVE SUMMARY: NFL BETTING VALIDATION")
    print(f"{'='*80}\n")
    
    print("OVERALL PERFORMANCE:")
    print(f"  Test Accuracy: {overall_results['test_accuracy']:.1%}")
    print(f"  95% CI: [{overall_results['confidence_interval_95']['lower']:.1%}, "
          f"{overall_results['confidence_interval_95']['upper']:.1%}]")
    print(f"  P-value: {overall_results['p_value']:.6f}")
    print(f"  Significant: {'✓ YES' if overall_results['significant'] else '✗ NO'}")
    
    if 'backtesting' in overall_results and 'error' not in overall_results['backtesting']:
        bt = overall_results['backtesting']
        print(f"\nBACKTESTING (70% Confidence):")
        print(f"  Total Bets: {bt['total_bets']}")
        print(f"  Accuracy: {bt['accuracy']:.1%}")
        print(f"  ROI: {bt['roi_pct']:.1%}")
        print(f"  Net Return: ${bt['net_return']:,.2f}")
    
    if 'best_context' in final_results:
        best_name = final_results['best_context']['name']
        best = final_results['best_context']['results']
        print(f"\nBEST CONTEXT: {best_name}")
        print(f"  Test Accuracy: {best['test_accuracy']:.1%}")
        if 'backtesting' in best and 'error' not in best['backtesting']:
            print(f"  ROI: {best['backtesting']['roi_pct']:.1%}")
            print(f"  Bets: {best['backtesting']['total_bets']}")
        print(f"  P-value: {best['p_value']:.6f}")
    
    print(f"\n{'='*80}\n")
    
    print("✓ NFL VALIDATION COMPLETE!")
    
    return final_results

if __name__ == '__main__':
    results = main()

