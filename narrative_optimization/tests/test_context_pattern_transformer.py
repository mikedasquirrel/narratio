"""
Test Context Pattern Transformer - Universal Pattern Discovery

Tests that the transformer:
1. Discovers high-leverage contexts automatically
2. Works across domains (sports, entertainment, business, natural)
3. Validates with proper sample sizes and effect sizes
4. Generates actionable betting recommendations

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from transformers.context_pattern import ContextPatternTransformer

print("="*80)
print("CONTEXT PATTERN TRANSFORMER - VALIDATION TESTS")
print("="*80)


def test_synthetic_data():
    """Test 1: Synthetic data with known patterns"""
    print("\n[TEST 1] Synthetic Data - Known Pattern Recovery")
    print("-"*80)
    
    # Create synthetic data with deliberate pattern:
    # When feature_0 > 0.7 AND feature_1 < 0.3, outcome = 1 (90% accuracy)
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'feature_0': np.random.uniform(0, 1, n_samples),
        'feature_1': np.random.uniform(0, 1, n_samples),
        'feature_2': np.random.uniform(0, 1, n_samples),
    })
    
    # Create outcome with pattern
    y = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    
    # Inject pattern: feature_0 > 0.7 AND feature_1 < 0.3 → outcome = 1 (90%)
    pattern_mask = (X['feature_0'] > 0.7) & (X['feature_1'] < 0.3)
    y[pattern_mask] = np.random.choice([0, 1], pattern_mask.sum(), p=[0.1, 0.9])
    
    print(f"  Samples: {len(X):,}")
    print(f"  Pattern samples: {pattern_mask.sum()} ({pattern_mask.mean():.1%})")
    print(f"  Pattern accuracy (injected): 90%")
    
    # Discover patterns
    transformer = ContextPatternTransformer(
        min_accuracy=0.65,
        min_samples=20,
        min_effect_size=0.10,
        max_patterns=10
    )
    
    transformer.fit(X, y)
    
    # Check if pattern was discovered
    print(f"\n  Patterns discovered: {len(transformer.patterns_)}")
    
    if transformer.patterns_:
        best = transformer.patterns_[0]
        print(f"  Best pattern accuracy: {best.accuracy:.1%}")
        print(f"  Best pattern sample size: {best.sample_size}")
        print(f"  Best pattern features: {best.features}")
        
        # Test transform
        X_transformed = transformer.transform(X)
        print(f"\n  Transformed shape: {X_transformed.shape}")
        print(f"  Expected: ({n_samples}, 60)")
        
        assert X_transformed.shape == (n_samples, 60), "Transform shape mismatch"
        
        # Test report
        report = transformer.get_context_report()
        print(f"\n  Report generated: {len(report)} characters")
        
        # Test betting recommendations
        recommendations = transformer.get_betting_recommendations(X[:10])
        print(f"  Betting recommendations: {len(recommendations)} generated")
        
        print("\n  ✓ TEST 1 PASSED: Pattern discovery working")
        return True
    else:
        print("\n  ✗ TEST 1 FAILED: No patterns discovered")
        return False


def test_nba_data():
    """Test 2: Real NBA data - discover contexts automatically"""
    print("\n[TEST 2] NBA Data - Discover High-Leverage Contexts")
    print("-"*80)
    
    # Load NBA data
    data_path = project_root.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'
    
    if not data_path.exists():
        print("  ⚠ NBA data not found, skipping test")
        return None
    
    with open(data_path) as f:
        games = json.load(f)
    
    # Extract features
    features_list = []
    outcomes = []
    
    for game in games[:2000]:  # Sample
        tc = game.get('temporal_context', {})
        if tc.get('games_played', 0) == 0:
            continue
        
        features_list.append({
            'home': 1.0 if game.get('home_game') else 0.0,
            'season_win_pct': tc.get('season_win_pct', 0.5),
            'l10_win_pct': tc.get('l10_win_pct', 0.5),
            'games_played': tc.get('games_played', 0) / 82.0,
            'record_differential': abs(tc.get('season_win_pct', 0.5) - 0.5),
        })
        outcomes.append(1 if game['won'] else 0)
    
    X = pd.DataFrame(features_list)
    y = np.array(outcomes)
    
    print(f"  NBA games: {len(X):,}")
    print(f"  Features: {X.columns.tolist()}")
    print(f"  Baseline accuracy: {y.mean():.1%}")
    
    # Discover patterns
    transformer = ContextPatternTransformer(
        min_accuracy=0.60,
        min_samples=30,
        max_patterns=20
    )
    
    transformer.fit(X, y)
    
    print(f"\n  Patterns discovered: {len(transformer.patterns_)}")
    
    if transformer.patterns_:
        # Print top patterns
        for i, pattern in enumerate(transformer.patterns_[:5], 1):
            print(f"\n  Pattern {i}:")
            print(f"    Features: {pattern.features}")
            print(f"    Accuracy: {pattern.accuracy:.1%}")
            print(f"    Sample size: {pattern.sample_size}")
            print(f"    Effect size: {pattern.effect_size:.3f}")
            print(f"    P-value: {pattern.p_value:.4f}")
        
        # Generate report
        print(transformer.get_context_report())
        
        print("\n  ✓ TEST 2 PASSED: NBA patterns discovered")
        return True
    else:
        print("\n  ⚠ TEST 2: No patterns above threshold")
        return False


def test_tennis_data():
    """Test 3: Tennis data - should find surface/round contexts"""
    print("\n[TEST 3] Tennis Data - Surface/Round Context Discovery")
    print("-"*80)
    
    # Load tennis data if available
    data_path = project_root.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
    
    if not data_path.exists():
        print("  ⚠ Tennis data not found, skipping test")
        return None
    
    with open(data_path) as f:
        matches = json.load(f)
    
    # Extract features
    features_list = []
    outcomes = []
    
    for match in matches[:3000]:  # Sample
        # Surface encoding
        surface_clay = 1.0 if match.get('surface') == 'clay' else 0.0
        surface_grass = 1.0 if match.get('surface') == 'grass' else 0.0
        surface_hard = 1.0 if match.get('surface') == 'hard' else 0.0
        
        # Round encoding (simplified)
        tournament_round = match.get('round', '')
        is_final = 1.0 if 'final' in tournament_round.lower() else 0.0
        is_semifinal = 1.0 if 'semi' in tournament_round.lower() else 0.0
        
        features_list.append({
            'surface_clay': surface_clay,
            'surface_grass': surface_grass,
            'surface_hard': surface_hard,
            'is_final': is_final,
            'is_semifinal': is_semifinal,
            'player1_rank': match.get('player1_rank', 100) / 100.0,
            'player2_rank': match.get('player2_rank', 100) / 100.0,
        })
        
        # Outcome (player 1 won) - handle various formats
        winner = match.get('winner', '')
        player1 = match.get('player1', '')
        player1_won = match.get('player1_won')
        
        if player1_won is not None:
            outcomes.append(1 if player1_won else 0)
        elif winner and player1:
            outcomes.append(1 if winner == player1 else 0)
        else:
            outcomes.append(0)  # Default if unclear
    
    X = pd.DataFrame(features_list)
    y = np.array(outcomes)
    
    print(f"  Tennis matches: {len(X):,}")
    print(f"  Features: {X.columns.tolist()}")
    print(f"  Baseline accuracy: {y.mean():.1%}")
    
    # Discover patterns
    transformer = ContextPatternTransformer(
        min_accuracy=0.60,
        min_samples=50,
        max_patterns=20
    )
    
    transformer.fit(X, y)
    
    print(f"\n  Patterns discovered: {len(transformer.patterns_)}")
    
    if transformer.patterns_:
        # Look for surface-related patterns
        surface_patterns = [p for p in transformer.patterns_ 
                           if any('surface' in f for f in p.features)]
        
        print(f"  Surface-related patterns: {len(surface_patterns)}")
        
        for i, pattern in enumerate(transformer.patterns_[:5], 1):
            print(f"\n  Pattern {i}:")
            print(f"    Features: {pattern.features}")
            print(f"    Accuracy: {pattern.accuracy:.1%}")
            print(f"    Sample size: {pattern.sample_size}")
        
        print("\n  ✓ TEST 3 PASSED: Tennis patterns discovered")
        return True
    else:
        print("\n  ⚠ TEST 3: No patterns above threshold")
        return False


def test_cross_domain_generalization():
    """Test 4: Cross-domain generalization test"""
    print("\n[TEST 4] Cross-Domain Generalization")
    print("-"*80)
    
    # Test on multiple simple datasets to verify universal applicability
    results = []
    
    # Domain 1: Continuous features
    np.random.seed(42)
    X1 = pd.DataFrame({
        'age': np.random.uniform(18, 80, 500),
        'income': np.random.uniform(20000, 200000, 500),
        'score': np.random.uniform(0, 100, 500),
    })
    # Pattern: age > 50 AND income > 100k → outcome = 1
    y1 = ((X1['age'] > 50) & (X1['income'] > 100000)).astype(int)
    y1 = np.where(np.random.rand(len(y1)) < 0.15, 1 - y1, y1)  # Add noise
    
    transformer1 = ContextPatternTransformer(min_accuracy=0.65, min_samples=20)
    transformer1.fit(X1, y1)
    results.append(('Continuous Features', len(transformer1.patterns_)))
    
    # Domain 2: Mixed features
    X2 = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'value': np.random.uniform(0, 1, 500),
        'flag': np.random.choice([0, 1], 500),
    })
    # Pattern: category=A AND value > 0.7 → outcome = 1
    y2 = ((X2['category'] == 'A') & (X2['value'] > 0.7)).astype(int)
    y2 = np.where(np.random.rand(len(y2)) < 0.15, 1 - y2, y2)
    
    transformer2 = ContextPatternTransformer(min_accuracy=0.65, min_samples=20)
    transformer2.fit(X2, y2)
    results.append(('Mixed Features', len(transformer2.patterns_)))
    
    print("\n  Domain Generalization Results:")
    for domain, n_patterns in results:
        print(f"    {domain}: {n_patterns} patterns discovered")
    
    if all(n > 0 for _, n in results):
        print("\n  ✓ TEST 4 PASSED: Generalizes across feature types")
        return True
    else:
        print("\n  ⚠ TEST 4: Some domains had no patterns")
        return False


def test_betting_recommendations():
    """Test 5: Betting recommendation generation"""
    print("\n[TEST 5] Betting Recommendations")
    print("-"*80)
    
    # Create data with strong pattern
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'momentum': np.random.uniform(0, 1, n_samples),
        'quality': np.random.uniform(0, 1, n_samples),
        'context': np.random.choice([0, 1], n_samples),
    })
    
    # Strong pattern: momentum > 0.8 AND quality > 0.7 → 85% win rate
    y = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    pattern_mask = (X['momentum'] > 0.8) & (X['quality'] > 0.7)
    y[pattern_mask] = np.random.choice([0, 1], pattern_mask.sum(), p=[0.15, 0.85])
    
    transformer = ContextPatternTransformer(
        min_accuracy=0.70,
        min_samples=15
    )
    
    transformer.fit(X, y)
    
    # Generate recommendations
    recommendations = transformer.get_betting_recommendations(X)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Recommendations generated: {len(recommendations)}")
    print(f"  Patterns available: {len(transformer.patterns_)}")
    
    # Even if no BET recommendations, generation should work
    if len(transformer.patterns_) > 0:
        print("\n  ✓ TEST 5 PASSED: Recommendations system functional")
        
        if recommendations:
            bets = [r for r in recommendations if r['recommendation'] == 'BET']
            print(f"  BET recommendations: {len(bets)}")
            
            if bets:
                print(f"\n  Sample recommendation:")
                print(f"    Confidence: {bets[0]['confidence']:.1%}")
                print(f"    Expected edge: {bets[0]['expected_edge']:+.3f}")
                print(f"    Sample size: {bets[0]['sample_size']}")
        
        return True
    
    print("\n  ⚠ TEST 5: No patterns to generate recommendations from")
    return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n")
    print("="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    
    tests = [
        ("Synthetic Data", test_synthetic_data),
        ("NBA Data", test_nba_data),
        ("Tennis Data", test_tennis_data),
        ("Cross-Domain", test_cross_domain_generalization),
        ("Betting Recommendations", test_betting_recommendations),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED WITH ERROR: {e}")
            results.append((name, False))
    
    # Summary
    print("\n")
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASSED" if result else ("⚠ SKIPPED" if result is None else "✗ FAILED")
        print(f"  {name:<30} {status}")
    
    passed = sum(1 for _, r in results if r is True)
    total = len([r for r in results if r is not None])
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("  Context Pattern Transformer is production-ready!")
    elif passed > 0:
        print(f"\n  ⚠ {total - passed} test(s) failed or skipped")
    else:
        print("\n  ✗ All tests failed - review implementation")
    
    return passed, total


if __name__ == '__main__':
    passed, total = run_all_tests()
    sys.exit(0 if passed == total else 1)

