"""
NFL Full Validation Pipeline

Runs complete statistical validation including:
- Train/test split
- P-values and confidence intervals
- Cross-validation
- ROI backtesting with real spreads
- Context-specific optimization (big underdogs, favorites, short week)
- Confidence threshold testing
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `narrative_optimization` package imports work
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import our validation pipeline
from narrative_optimization.utils.validation_pipeline import ValidationPipeline


def load_precomputed_features():
    """Load precomputed genome features if available."""
    genome_path = Path(__file__).parent / 'nfl_genome_data.npz'
    
    if not genome_path.exists():
        return None, None
    
    data = np.load(genome_path, allow_pickle=True)
    return data['genome'], data.get('outcomes')

def load_nfl_data():
    """Load NFL game data"""
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    
    print(f"Loading NFL data from: {data_path}")
    with open(data_path) as f:
        games = json.load(f)
    
    print(f"✓ Loaded {len(games)} NFL games\n")
    return games

def extract_features_and_outcomes(games, fallback_features=None, fallback_outcomes=None):
    """Extract feature matrix and outcomes from games"""
    X_list = []
    y_list = []
    spread_list = []
    
    print("Extracting features and outcomes...")
    for game in games:
        # Get features
        if 'combined_features' in game:
            features = np.array(game['combined_features'])
        elif 'narrative_features' in game and 'statistical_features' in game:
            narrative = np.array(game['narrative_features'])
            statistical = np.array(game['statistical_features'])
            features = np.concatenate([narrative, statistical])
        elif 'home_features' in game and 'away_features' in game:
            home = np.array(game['home_features'])
            away = np.array(game['away_features'])
            features = home - away  # Differential
        else:
            continue
        
        X_list.append(features)
        
        # Outcome: home wins
        if 'home_score' in game and 'away_score' in game:
            y_list.append(1 if game['home_score'] > game['away_score'] else 0)
        elif 'home_wins' in game:
            y_list.append(1 if game['home_wins'] else 0)
        else:
            continue
        
        # Spread (if available)
        spread = game.get('spread', game.get('betting_line', 0))
        spread_list.append(spread)
    
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        spreads = np.array(spread_list) if spread_list else None
        
        print(f"✓ Extracted {len(X)} games with {X.shape[1]} features")
        print(f"  Home win rate: {y.mean():.1%}\n")
        return X, y, spreads
    
    # Fall back to precomputed genome features
    if fallback_features is not None:
        X = fallback_features
        
        if fallback_outcomes is not None and len(fallback_outcomes) == len(X):
            y = fallback_outcomes
        else:
            y = np.array([
                1 if g.get('home_score', 0) > g.get('away_score', 0) else 0
                for g in games
            ])
        
        if len(X) != len(y):
            raise ValueError("Fallback feature matrix length does not match outcomes length")
        
        print("✓ Using precomputed genome features (no embedded feature vectors in dataset)")
        print(f"  Games: {len(X)}, Features: {X.shape[1]}")
        print(f"  Home win rate: {y.mean():.1%}\n")
        
        return X, y, None
    
    raise ValueError("No feature vectors available in dataset and no fallback features supplied.")
def test_spread_contexts(games, model, scaler, X_full, y_full):
    """Test model performance with different spread scenarios"""
    
    print(f"\n{'='*60}")
    print("SPREAD-BASED CONTEXT VALIDATION")
    print(f"{'='*60}\n")
    
    context_results = {}
    
    # Define spread contexts
    contexts = {
        'overall': lambda g: True,
        'big_underdogs': lambda g: g.get('spread', 0) >= 7,  # Home team is 7+ point underdog
        'big_favorites': lambda g: g.get('spread', 0) <= -7,  # Home team is 7+ point favorite
        'short_week': lambda g: (g.get('home_rest_days', 7) <= 4 or g.get('away_rest_days', 7) <= 4),
        'close_spread': lambda g: abs(g.get('spread', 0)) <= 3,
        'division_games': lambda g: g.get('division_game', False),
    }
    
    for context_name, context_filter in contexts.items():
        print(f"Testing context: {context_name}")
        
        context_indices = [
            idx for idx, g in enumerate(games)
            if context_filter(g)
        ]
        
        if len(context_indices) < 50:
            print(f"  ⚠️  Only {len(context_indices)} games, skipping\n")
            continue
        
        X_context = X_full[context_indices]
        y_context = y_full[context_indices]
        
        # Run validation
        pipeline = ValidationPipeline(f'NFL-{context_name}')
        
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

def test_ats_prediction(games, model, scaler, X_full):
    """Test Against The Spread (ATS) predictions"""
    
    print(f"\n{'='*60}")
    print("AGAINST THE SPREAD (ATS) VALIDATION")
    print(f"{'='*60}\n")
    
    # Filter games with spread data
    ats_indices = []
    y_ats_list = []
    
    for idx, game in enumerate(games):
        home_score = game.get('home_score')
        away_score = game.get('away_score')
        odds = game.get('betting_odds', {})
        spread = odds.get('spread', game.get('betting_line'))
        
        if home_score is None or away_score is None or spread is None:
            continue
        
        actual_margin = home_score - away_score
        ats_outcome = 1 if (actual_margin + spread) > 0 else 0
        
        ats_indices.append(idx)
        y_ats_list.append(ats_outcome)
    
    if len(ats_indices) < 100:
        print("⚠️  Insufficient games with spread data\n")
        return None
    
    X = X_full[ats_indices]
    y_ats = np.array(y_ats_list)
    
    print(f"✓ {len(X)} games for ATS validation")
    print(f"  Home covers rate: {y_ats.mean():.1%}\n")
    
    # Run ATS validation
    pipeline = ValidationPipeline('NFL-ATS')
    
    ats_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    ats_results = pipeline.run_full_validation(
        X, y_ats, ats_model, temporal_split=True
    )
    
    return ats_results

def main():
    """Run complete NFL validation"""
    
    print(f"\n{'='*60}")
    print("NFL COMPLETE VALIDATION PIPELINE")
    print(f"{'='*60}\n")
    
    # 1. Load data
    games = load_nfl_data()
    fallback_X, fallback_y = load_precomputed_features()
    
    # 2. Extract features
    X, y, spreads = extract_features_and_outcomes(
        games,
        fallback_features=fallback_X,
        fallback_outcomes=fallback_y
    )
    
    # 3. Create model
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    scaler = StandardScaler()
    
    # 4. Run overall validation
    print("="*60)
    print("OVERALL MODEL VALIDATION")
    print("="*60)
    
    pipeline = ValidationPipeline('NFL-Overall')
    overall_results = pipeline.run_full_validation(
        X, y, model, temporal_split=True
    )
    
    pipeline.print_summary()
    
    # 5. Test spread contexts
    context_results = test_spread_contexts(games, model, scaler, X, y)
    
    # 6. Test ATS prediction
    ats_results = test_ats_prediction(games, model, scaler, X)
    
    # 7. Compile final results
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
    
    if ats_results:
        final_results['ats_prediction'] = ats_results
    
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
    
    # 8. Save results
    results_dir = Path(__file__).parent
    results_path = results_dir / 'nfl_betting_validated_results.json'
    
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Complete validation results saved to: {results_path}\n")
    
    # 9. Print executive summary
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
    
    if 'ats_prediction' in final_results:
        ats = final_results['ats_prediction']
        print(f"\nAGAINST THE SPREAD:")
        print(f"  Test Accuracy: {ats['test_accuracy']:.1%}")
        if 'backtesting' in ats and 'error' not in ats['backtesting']:
            print(f"  ROI: {ats['backtesting']['roi_pct']:.1%}")
    
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

