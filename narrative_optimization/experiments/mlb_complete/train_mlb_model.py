"""
MLB Complete Model Training and Validation

Trains MLB prediction model using archetype features and validates with:
- Train/test split
- Statistical significance testing
- Cross-validation
- ROI backtesting
- Confidence threshold optimization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Import our validation pipeline
from narrative_optimization.utils.validation_pipeline import ValidationPipeline

def load_mlb_archetype_features():
    """Load all MLB archetype features"""
    
    base_path = Path(__file__).parent.parent.parent / 'data' / 'archetype_features'
    
    print("Loading MLB archetype features...")
    
    feature_types = ['hero_journey', 'character', 'plot', 'thematic', 'structural']
    all_features = []
    
    for feat_type in feature_types:
        feat_path = base_path / f'mlb_{feat_type}_features.npz'
        data = np.load(feat_path)
        features = data['features']
        all_features.append(features)
        print(f"  {feat_type}: {features.shape}")
    
    # Combine all archetype features
    X_archetype = np.hstack(all_features)
    
    print(f"\n✓ Combined archetype features: {X_archetype.shape}")
    print(f"  Total games: {X_archetype.shape[0]}")
    print(f"  Total features: {X_archetype.shape[1]}\n")
    
    return X_archetype

def load_mlb_outcomes():
    """Load MLB game outcomes"""
    
    games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'mlb_complete_dataset.json'
    
    print(f"Loading MLB game outcomes from: {games_path}")
    
    with open(games_path) as f:
        games = json.load(f)
    
    # Extract outcomes
    outcomes = []
    
    for game in games:
        outcome = game.get('outcome', {})
        
        # Home wins if home score > away score
        if 'home_score' in outcome and 'away_score' in outcome:
            home_wins = outcome['home_score'] > outcome['away_score']
        elif 'winner' in outcome:
            home_wins = outcome['winner'] == game['home_team']
        else:
            # Skip games without clear outcome
            home_wins = None
        
        outcomes.append(1 if home_wins else 0 if home_wins is not None else -1)
    
    # Filter out games with no outcome
    valid_indices = [i for i, o in enumerate(outcomes) if o != -1]
    outcomes_array = np.array([outcomes[i] for i in valid_indices])
    
    print(f"✓ Loaded {len(outcomes_array)} games with valid outcomes")
    print(f"  Home win rate: {outcomes_array.mean():.1%}\n")
    
    return outcomes_array, valid_indices, games

def test_mlb_contexts(X, y, games, valid_indices):
    """Test model performance in different MLB contexts"""
    
    print(f"\n{'='*60}")
    print("CONTEXT-SPECIFIC VALIDATION")
    print(f"{'='*60}\n")
    
    context_results = {}
    
    # Get valid games
    valid_games = [games[i] for i in valid_indices]
    
    # Define contexts
    contexts = {}
    
    # Pitcher contexts (if available)
    ace_pitcher_indices = []
    division_game_indices = []
    
    for idx, game in enumerate(valid_games):
        # Ace pitcher games (placeholder - would need pitcher stats)
        pitchers = game.get('pitchers', {})
        
        # Division games (placeholder - would need division info)
        if 'division_game' in game and game['division_game']:
            division_game_indices.append(idx)
    
    if len(division_game_indices) >= 100:
        contexts['division_games'] = division_game_indices
    
    # Always test overall
    contexts['overall'] = list(range(len(X)))
    
    # Run validation for each context
    for context_name, indices in contexts.items():
        if len(indices) < 100:
            print(f"Skipping {context_name}: only {len(indices)} games\n")
            continue
        
        print(f"Testing context: {context_name} ({len(indices)} games)")
        
        X_context = X[indices]
        y_context = y[indices]
        
        pipeline = ValidationPipeline(f'MLB-{context_name}')
        
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
    """Run complete MLB model training and validation"""
    
    print(f"\n{'='*60}")
    print("MLB COMPLETE MODEL TRAINING & VALIDATION")
    print(f"{'='*60}\n")
    
    # 1. Load archetype features
    X_archetype = load_mlb_archetype_features()
    
    # 2. Load outcomes
    y, valid_indices, games = load_mlb_outcomes()
    
    # 3. Filter features to match valid outcomes
    X = X_archetype[valid_indices]
    
    print(f"Final dataset:")
    print(f"  Games: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Home win rate: {y.mean():.1%}\n")
    
    # 4. Create and train model
    print("="*60)
    print("TRAINING MLB MODEL")
    print("="*60)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    scaler = StandardScaler()
    
    # 5. Run overall validation
    pipeline = ValidationPipeline('MLB-Overall')
    overall_results = pipeline.run_full_validation(
        X, y, model, temporal_split=True
    )
    
    pipeline.print_summary()
    
    # 6. Test contexts
    context_results = test_mlb_contexts(X, y, games, valid_indices)
    
    # 7. Compile final results
    final_results = {
        'overall': overall_results,
        'contexts': context_results,
        'data_summary': {
            'total_games': len(X),
            'total_features': X.shape[1],
            'archetype_features': X.shape[1],
            'home_win_rate': float(y.mean()),
            'validation_date': '2025-11-14'
        },
        'archetype_contribution': {
            'archetype_only': True,
            'feature_breakdown': {
                'hero_journey': 58,
                'character': 51,
                'plot': 40,
                'thematic': 21,
                'structural': 29,
                'total': 199
            }
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
    
    # 8. Save model
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    model_path = results_dir / 'mlb_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_count': X.shape[1]
        }, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    
    # 9. Save results
    results_path = results_dir / 'mlb_complete_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    
    # Also save to domains directory for web interface
    domains_path = Path(__file__).parent.parent.parent / 'domains' / 'mlb' / 'mlb_betting_validated_results.json'
    domains_path.parent.mkdir(parents=True, exist_ok=True)
    with open(domains_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Results also saved to: {domains_path}\n")
    
    # 10. Print executive summary
    print(f"\n{'='*80}")
    print("EXECUTIVE SUMMARY: MLB BETTING VALIDATION")
    print(f"{'='*80}\n")
    
    print("ARCHETYPE FEATURES ONLY:")
    print(f"  Total Features: {X.shape[1]}")
    print(f"  Hero Journey: 58")
    print(f"  Character: 51")
    print(f"  Plot: 40")
    print(f"  Thematic: 21")
    print(f"  Structural: 29")
    
    print("\nOVERALL PERFORMANCE:")
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
    
    print("✓ MLB TRAINING & VALIDATION COMPLETE!")
    
    return final_results

if __name__ == '__main__':
    results = main()

