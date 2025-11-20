"""
NBA Full Validation Pipeline

Runs complete statistical validation including:
- Train/test split
- P-values and confidence intervals
- Cross-validation
- ROI backtesting
- Context-specific optimization
- Confidence threshold testing
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Import our validation pipeline
from narrative_optimization.utils.validation_pipeline import ValidationPipeline

def load_nba_data():
    """Load NBA game data with temporal context"""
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'
    
    print(f"Loading NBA data from: {data_path}")
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} NBA games\n")
    return games

def extract_features_and_outcomes(games):
    """Extract feature matrix and outcomes from games"""
    X_list = []
    y_list = []
    
    print("Extracting features and outcomes...")
    for game in games:
        # Get differential features (this is what the model uses)
        if 'differential' in game:
            differential = np.array(game['differential'])
        elif 'home_features' in game and 'away_features' in game:
            home_features = np.array(game['home_features'])
            away_features = np.array(game['away_features'])
            differential = home_features - away_features
        else:
            continue  # Skip games without features
        
        X_list.append(differential)
        y_list.append(1 if game['home_wins'] else 0)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✓ Extracted {len(X)} games with {X.shape[1]} features")
    print(f"  Home win rate: {y.mean():.1%}\n")
    
    return X, y

def test_context_strategies(games, model, scaler):
    """Test model performance in different contexts"""
    
    print(f"\n{'='*60}")
    print("CONTEXT-SPECIFIC VALIDATION")
    print(f"{'='*60}\n")
    
    context_results = {}
    
    # Define contexts to test
    contexts = {
        'overall': lambda g: True,  # All games
        'record_gap_large': lambda g: abs(g.get('home_win_pct', 0.5) - g.get('away_win_pct', 0.5)) > 0.30,
        'late_season': lambda g: g.get('game_number', 0) > 60,
        'home_streak': lambda g: g.get('home_streak', 0) >= 3,
        'away_b2b': lambda g: g.get('away_rest_days', 3) <= 1,
        'record_gap_late_season': lambda g: (abs(g.get('home_win_pct', 0.5) - g.get('away_win_pct', 0.5)) > 0.30 and g.get('game_number', 0) > 60)
    }
    
    for context_name, context_filter in contexts.items():
        print(f"Testing context: {context_name}")
        
        # Filter games for this context
        context_games = [g for g in games if context_filter(g)]
        
        if len(context_games) < 100:
            print(f"  ⚠️  Only {len(context_games)} games, skipping\n")
            continue
        
        # Extract features
        X_context = []
        y_context = []
        
        for game in context_games:
            if 'differential' in game:
                differential = np.array(game['differential'])
            elif 'home_features' in game and 'away_features' in game:
                home_features = np.array(game['home_features'])
                away_features = np.array(game['away_features'])
                differential = home_features - away_features
            else:
                continue
            
            X_context.append(differential)
            y_context.append(1 if game['home_wins'] else 0)
        
        if len(X_context) < 100:
            print(f"  ⚠️  Only {len(X_context)} valid games, skipping\n")
            continue
        
        X_context = np.array(X_context)
        y_context = np.array(y_context)
        
        # Run validation on this context
        pipeline = ValidationPipeline(f'NBA-{context_name}')
        
        # Use a fresh model for this context
        context_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        results = pipeline.run_full_validation(
            X_context, 
            y_context, 
            context_model,
            temporal_split=True
        )
        
        context_results[context_name] = results
        print()
    
    return context_results

def main():
    """Run complete NBA validation"""
    
    print(f"\n{'='*60}")
    print("NBA COMPLETE VALIDATION PIPELINE")
    print(f"{'='*60}\n")
    
    # 1. Load data
    games = load_nba_data()
    
    # 2. Extract features
    X, y = extract_features_and_outcomes(games)
    
    # 3. Load existing model (if available) or create new one
    model_path = Path(__file__).parent.parent.parent / 'experiments' / 'nba_complete' / 'results' / 'nba_quick_model.pkl'
    
    if model_path.exists():
        print(f"Loading existing model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        print(f"✓ Model loaded\n")
    else:
        print("No existing model found, creating new one")
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        scaler = None
    
    # 4. Run overall validation
    print("="*60)
    print("OVERALL MODEL VALIDATION")
    print("="*60)
    
    pipeline = ValidationPipeline('NBA-Overall')
    overall_results = pipeline.run_full_validation(
        X, y, model, temporal_split=True
    )
    
    pipeline.print_summary()
    
    # 5. Test context-specific strategies
    context_results = test_context_strategies(games, model, scaler)
    
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
    results_path = results_dir / 'nba_betting_validated_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Complete validation results saved to: {results_path}\n")
    
    # 8. Print executive summary
    print(f"\n{'='*80}")
    print("EXECUTIVE SUMMARY: NBA BETTING VALIDATION")
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
    
    print("✓ NBA VALIDATION COMPLETE!")
    
    return final_results

if __name__ == '__main__':
    results = main()

