"""
NFL Narrative Formula Optimization

Find optimal feature weights to maximize narrative signal.
Similar to movie domain achieving 59.7% R².

Strategy:
1. Use strong contexts (|r| > 0.3) as training data
2. Optimize feature weights via regularized regression
3. Discover golden ratio of feature types
4. Test on held-out contexts
5. Compare to basic formula
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from collections import defaultdict


def load_data():
    """Load genome, story quality, outcomes, and game metadata."""
    # Load games
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
    with open(dataset_path) as f:
        games = json.load(f)
    
    # Load genome data
    genome_path = Path(__file__).parent / 'nfl_genome_data.npz'
    genome_data = np.load(genome_path, allow_pickle=True)
    
    ж = genome_data['genome']
    ю_basic = genome_data['story_quality']
    outcomes = genome_data['outcomes']
    feature_names = genome_data['feature_names'].tolist()
    
    return games, ж, ю_basic, outcomes, feature_names


def identify_strong_contexts(games, ж, outcomes, min_abs_r=0.30, min_n=10):
    """
    Identify contexts with strong narrative signal.
    Use these as training data for optimization.
    """
    print("\n" + "="*80)
    print("IDENTIFYING STRONG NARRATIVE CONTEXTS")
    print("="*80)
    
    strong_contexts = []
    
    # Group by coach-season (strongest contexts from discovery)
    coach_season_groups = defaultdict(list)
    for i, game in enumerate(games):
        coach = game['home_coaches']['head_coach']
        season = game['season']
        key = f"{coach}_{season}"
        coach_season_groups[key].append(i)
    
    print(f"\nSearching {len(coach_season_groups)} coach-season combinations...")
    
    for context_key, indices in coach_season_groups.items():
        if len(indices) < min_n:
            continue
        
        # Calculate story quality for this context
        ж_subset = ж[indices]
        outcomes_subset = outcomes[indices]
        
        if len(np.unique(outcomes_subset)) < 2:
            continue
        
        # Basic story quality
        ю_subset = ж_subset.mean(axis=1)
        ю_subset = (ю_subset - ю_subset.min()) / (ю_subset.max() - ю_subset.min() + 1e-10)
        
        r = np.corrcoef(ю_subset, outcomes_subset)[0, 1]
        abs_r = abs(r)
        
        if abs_r >= min_abs_r:
            strong_contexts.append({
                'context': context_key,
                'indices': indices,
                'abs_r': abs_r,
                'r': r,
                'n': len(indices)
            })
    
    # Sort by |r|
    strong_contexts.sort(key=lambda x: x['abs_r'], reverse=True)
    
    print(f"\n✓ Found {len(strong_contexts)} strong contexts (|r| >= {min_abs_r})")
    print(f"\nTop 10 Strong Contexts:")
    for i, ctx in enumerate(strong_contexts[:10], 1):
        print(f"  {i}. {ctx['context']}: |r| = {ctx['abs_r']:.4f} (n={ctx['n']})")
    
    return strong_contexts


def create_training_data(strong_contexts, ж, outcomes, test_size=0.3):
    """
    Create training dataset from strong contexts.
    Reserve some contexts for testing.
    """
    print("\n" + "="*80)
    print("CREATING TRAINING DATA")
    print("="*80)
    
    # Split contexts into train/test
    n_contexts = len(strong_contexts)
    n_test = max(2, int(n_contexts * test_size))
    
    test_contexts = strong_contexts[:n_test]  # Strongest for testing
    train_contexts = strong_contexts[n_test:]  # Rest for training
    
    print(f"\nTrain contexts: {len(train_contexts)}")
    print(f"Test contexts: {len(test_contexts)}")
    
    # Gather all indices
    train_indices = []
    test_indices = []
    
    for ctx in train_contexts:
        train_indices.extend(ctx['indices'])
    
    for ctx in test_contexts:
        test_indices.extend(ctx['indices'])
    
    # Extract data
    X_train = ж[train_indices]
    y_train = outcomes[train_indices]
    
    X_test = ж[test_indices]
    y_test = outcomes[test_indices]
    
    print(f"\nTraining samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def optimize_formula(X_train, y_train, feature_names):
    """
    Find optimal feature weights to maximize narrative signal.
    """
    print("\n" + "="*80)
    print("OPTIMIZING NARRATIVE FORMULA")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ========================================================================
    # METHOD 1: Feature Selection (Find Most Predictive Features)
    # ========================================================================
    
    print("\n[Method 1] Feature Selection...")
    
    # Select top K features by mutual information
    k_values = [50, 100, 200, 300, 500]
    best_k = None
    best_score = -np.inf
    
    for k in k_values:
        if k > X_train.shape[1]:
            continue
        
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X_train_scaled, y_train)
        
        # Score selected features
        X_selected = selector.transform(X_train_scaled)
        
        # Simple correlation
        ю_selected = X_selected.mean(axis=1)
        ю_selected = (ю_selected - ю_selected.min()) / (ю_selected.max() - ю_selected.min() + 1e-10)
        
        r = np.corrcoef(ю_selected, y_train)[0, 1]
        abs_r = abs(r)
        
        print(f"  k={k:3d}: |r| = {abs_r:.4f}")
        
        if abs_r > best_score:
            best_score = abs_r
            best_k = k
            best_selector = selector
    
    print(f"\n✓ Best k: {best_k} features, |r| = {best_score:.4f}")
    
    # Get selected feature indices and names
    selected_features = best_selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_features if i < len(feature_names)]
    
    print(f"\nTop 20 Selected Features:")
    feature_scores = best_selector.scores_
    top_indices = np.argsort(feature_scores)[-20:][::-1]
    for i, idx in enumerate(top_indices, 1):
        if idx < len(feature_names):
            print(f"  {i}. {feature_names[idx]}: {feature_scores[idx]:.4f}")
    
    # ========================================================================
    # METHOD 2: Regularized Regression (Find Optimal Weights)
    # ========================================================================
    
    print("\n[Method 2] Regularized Regression...")
    
    # Try different regularization methods
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=5000),
        'ElasticNet': ElasticNet(max_iter=5000)
    }
    
    best_model = None
    best_model_name = None
    best_model_score = -np.inf
    
    for name, model in models.items():
        # Grid search for best alpha
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        
        if name == 'ElasticNet':
            param_grid['l1_ratio'] = [0.1, 0.5, 0.9]
        
        grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid.fit(X_train_scaled, y_train)
        
        # Get predictions
        y_pred = grid.predict(X_train_scaled)
        r = np.corrcoef(y_pred, y_train)[0, 1]
        abs_r = abs(r)
        r2 = grid.best_score_
        
        print(f"  {name:12s}: |r| = {abs_r:.4f}, R² = {r2:.4f}, alpha = {grid.best_params_['alpha']:.3f}")
        
        if abs_r > best_model_score:
            best_model_score = abs_r
            best_model = grid.best_estimator_
            best_model_name = name
    
    print(f"\n✓ Best model: {best_model_name}, |r| = {best_model_score:.4f}")
    
    # ========================================================================
    # METHOD 3: Combined Approach (Feature Selection + Regression)
    # ========================================================================
    
    print("\n[Method 3] Combined (Selection + Regression)...")
    
    # Use selected features with best model
    X_selected = best_selector.transform(X_train_scaled)
    
    combined_model = Ridge(alpha=1.0)
    combined_model.fit(X_selected, y_train)
    
    y_pred_combined = combined_model.predict(X_selected)
    r_combined = np.corrcoef(y_pred_combined, y_train)[0, 1]
    abs_r_combined = abs(r_combined)
    
    print(f"  Combined: |r| = {abs_r_combined:.4f}")
    
    # Choose best approach
    if abs_r_combined > max(best_model_score, best_score):
        print(f"\n✓ BEST: Combined approach, |r| = {abs_r_combined:.4f}")
        final_method = 'combined'
        final_score = abs_r_combined
    elif best_model_score > best_score:
        print(f"\n✓ BEST: {best_model_name} regression, |r| = {best_model_score:.4f}")
        final_method = 'regression'
        final_score = best_model_score
    else:
        print(f"\n✓ BEST: Feature selection (k={best_k}), |r| = {best_score:.4f}")
        final_method = 'selection'
        final_score = best_score
    
    return {
        'method': final_method,
        'score': final_score,
        'scaler': scaler,
        'selector': best_selector,
        'selected_k': best_k,
        'selected_features': selected_features,
        'selected_feature_names': selected_feature_names,
        'regression_model': best_model if final_method == 'regression' else combined_model,
        'model_name': best_model_name
    }


def test_optimized_formula(optimization_results, X_test, y_test, X_train, y_train):
    """Test optimized formula on held-out contexts."""
    print("\n" + "="*80)
    print("TESTING OPTIMIZED FORMULA")
    print("="*80)
    
    scaler = optimization_results['scaler']
    selector = optimization_results['selector']
    model = optimization_results['regression_model']
    method = optimization_results['method']
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Apply formula
    if method == 'selection':
        X_test_selected = selector.transform(X_test_scaled)
        ю_test = X_test_selected.mean(axis=1)
        ю_test = (ю_test - ю_test.min()) / (ю_test.max() - ю_test.min() + 1e-10)
    elif method == 'regression':
        # Model was trained on full features
        ю_test = model.predict(X_test_scaled)
    else:  # combined
        X_test_selected = selector.transform(X_test_scaled)
        ю_test = model.predict(X_test_selected)
    
    # Measure correlation
    r_test = np.corrcoef(ю_test, y_test)[0, 1]
    abs_r_test = abs(r_test)
    
    print(f"\nTest Set Performance:")
    print(f"  |r| = {abs_r_test:.4f}")
    print(f"  r = {r_test:.4f}")
    print(f"  Pattern: {'Inverse' if r_test < 0 else 'Positive'}")
    
    # Compare to basic formula
    X_test_basic_ю = X_test.mean(axis=1)
    X_test_basic_ю = (X_test_basic_ю - X_test_basic_ю.min()) / (X_test_basic_ю.max() - X_test_basic_ю.min() + 1e-10)
    r_basic = np.corrcoef(X_test_basic_ю, y_test)[0, 1]
    abs_r_basic = abs(r_basic)
    
    print(f"\nBasic Formula (for comparison):")
    print(f"  |r| = {abs_r_basic:.4f}")
    
    improvement = (abs_r_test - abs_r_basic) / abs_r_basic * 100 if abs_r_basic > 0 else 0
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Test on training set for comparison
    X_train_scaled = scaler.transform(X_train)
    X_train_selected = selector.transform(X_train_scaled)
    
    if method == 'selection':
        ю_train = X_train_selected.mean(axis=1)
        ю_train = (ю_train - ю_train.min()) / (ю_train.max() - ю_train.min() + 1e-10)
    elif method == 'regression':
        ю_train = model.predict(X_train_scaled)
    else:  # combined
        ю_train = model.predict(X_train_selected)
    
    r_train = np.corrcoef(ю_train, y_train)[0, 1]
    abs_r_train = abs(r_train)
    
    print(f"\nTrain Set Performance:")
    print(f"  |r| = {abs_r_train:.4f}")
    
    # Calculate R² equivalent
    r2_test = r_test ** 2
    r2_train = r_train ** 2
    
    print(f"\nR² (variance explained):")
    print(f"  Train: {r2_train:.4f} ({r2_train*100:.2f}%)")
    print(f"  Test:  {r2_test:.4f} ({r2_test*100:.2f}%)")
    
    return {
        'test_abs_r': float(abs_r_test),
        'test_r': float(r_test),
        'test_r2': float(r2_test),
        'train_abs_r': float(abs_r_train),
        'train_r': float(r_train),
        'train_r2': float(r2_train),
        'basic_abs_r': float(abs_r_basic),
        'improvement_pct': float(improvement)
    }


def analyze_feature_types(selected_feature_names):
    """Analyze what types of features are most important."""
    print("\n" + "="*80)
    print("FEATURE TYPE ANALYSIS")
    print("="*80)
    
    # Categorize features
    categories = {
        'nominative': ['nominative', 'name', 'phonetic', 'universal_nominative', 
                      'hierarchical_nominative', 'pure_nominative', 'nominative_interaction'],
        'ensemble': ['ensemble'],
        'conflict': ['conflict', 'suspense', 'tension'],
        'emotional': ['emotional', 'authenticity'],
        'linguistic': ['linguistic', 'cognitive'],
        'statistical': ['statistical'],
        'temporal': ['temporal'],
        'cultural': ['cultural', 'social'],
        'expertise': ['expertise', 'authority'],
        'other': []
    }
    
    category_counts = defaultdict(int)
    
    for fname in selected_feature_names:
        categorized = False
        for category, keywords in categories.items():
            if category == 'other':
                continue
            if any(kw in fname.lower() for kw in keywords):
                category_counts[category] += 1
                categorized = True
                break
        if not categorized:
            category_counts['other'] += 1
    
    total = len(selected_feature_names)
    
    print(f"\nFeature Type Distribution (n={total}):")
    print(f"{'Category':<20} {'Count':<8} {'Percentage':<12}")
    print("-" * 40)
    
    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        count = category_counts[category]
        pct = count / total * 100
        print(f"{category:<20} {count:<8} {pct:>6.1f}%")
    
    print(f"\nGolden Ratio Discovery:")
    top_3 = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    for category, count in top_3:
        pct = count / total * 100
        print(f"  {category}: {pct:.1f}%")
    
    return dict(category_counts)


def main():
    """Optimize NFL narrative formula."""
    print("="*80)
    print("NFL NARRATIVE FORMULA OPTIMIZATION")
    print("="*80)
    print("\nGoal: Find optimal feature weights to maximize |r|")
    print("Strategy: Use strong contexts as training data")
    
    # Load data
    print("\nLoading data...")
    games, ж, ю_basic, outcomes, feature_names = load_data()
    print(f"✓ Loaded {len(games)} games, {ж.shape[1]} features")
    
    # Identify strong contexts
    strong_contexts = identify_strong_contexts(games, ж, outcomes, min_abs_r=0.30)
    
    if len(strong_contexts) < 3:
        print("\n⚠ Warning: Only found {len(strong_contexts)} strong contexts")
        print("  Lowering threshold to find more training data...")
        strong_contexts = identify_strong_contexts(games, ж, outcomes, min_abs_r=0.20)
    
    # Create training data
    X_train, X_test, y_train, y_test, train_idx, test_idx = create_training_data(
        strong_contexts, ж, outcomes
    )
    
    # Optimize formula
    optimization_results = optimize_formula(X_train, y_train, feature_names)
    
    # Test optimized formula
    test_results = test_optimized_formula(
        optimization_results, X_test, y_test, X_train, y_train
    )
    
    # Analyze feature types
    feature_distribution = analyze_feature_types(
        optimization_results['selected_feature_names']
    )
    
    # ========================================================================
    # SAVE OPTIMIZED FORMULA
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING OPTIMIZED FORMULA")
    print("="*80)
    
    results = {
        'optimization_method': optimization_results['method'],
        'selected_features': int(optimization_results['selected_k']),
        'total_features': int(ж.shape[1]),
        'feature_reduction': f"{(1 - optimization_results['selected_k']/ж.shape[1])*100:.1f}%",
        'performance': {
            'train': {
                'abs_r': test_results['train_abs_r'],
                'r': test_results['train_r'],
                'r2': test_results['train_r2'],
                'variance_explained': f"{test_results['train_r2']*100:.2f}%"
            },
            'test': {
                'abs_r': test_results['test_abs_r'],
                'r': test_results['test_r'],
                'r2': test_results['test_r2'],
                'variance_explained': f"{test_results['test_r2']*100:.2f}%"
            },
            'basic_formula': {
                'abs_r': test_results['basic_abs_r']
            },
            'improvement': f"{test_results['improvement_pct']:.1f}%"
        },
        'golden_ratio': feature_distribution,
        'top_features': optimization_results['selected_feature_names'][:50],
        'strong_contexts_used': len(strong_contexts),
        'training_samples': len(train_idx),
        'test_samples': len(test_idx)
    }
    
    output_path = Path(__file__).parent / 'nfl_optimized_formula.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to: {output_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\nOptimized Formula Performance:")
    print(f"  Test |r|: {test_results['test_abs_r']:.4f}")
    print(f"  Test R²: {test_results['test_r2']:.4f} ({test_results['test_r2']*100:.2f}%)")
    print(f"  Train |r|: {test_results['train_abs_r']:.4f}")
    print(f"  Train R²: {test_results['train_r2']:.4f} ({test_results['train_r2']*100:.2f}%)")
    
    print(f"\nComparison:")
    print(f"  Basic formula |r|: {test_results['basic_abs_r']:.4f}")
    print(f"  Optimized formula |r|: {test_results['test_abs_r']:.4f}")
    print(f"  Improvement: {test_results['improvement_pct']:.1f}%")
    
    print(f"\nFormula Composition:")
    print(f"  Features used: {optimization_results['selected_k']} / {ж.shape[1]}")
    print(f"  Feature reduction: {(1 - optimization_results['selected_k']/ж.shape[1])*100:.1f}%")
    
    print(f"\nGolden Ratio (Top 3 feature types):")
    top_3 = sorted(feature_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    for category, count in top_3:
        pct = count / optimization_results['selected_k'] * 100
        print(f"  {category}: {pct:.1f}%")
    
    print(f"\n{'='*80}")
    
    # Compare to movie domain
    print(f"\nNFL vs Movie Domain:")
    print(f"  Movie R²: 59.7%")
    print(f"  NFL R²: {test_results['test_r2']*100:.2f}%")
    
    if test_results['test_r2'] > 0.30:
        print(f"  ✓ Strong narrative signal extracted!")
    elif test_results['test_r2'] > 0.10:
        print(f"  ✓ Moderate narrative signal extracted")
    else:
        print(f"  → Weak signal (performance domain)")


if __name__ == '__main__':
    main()

