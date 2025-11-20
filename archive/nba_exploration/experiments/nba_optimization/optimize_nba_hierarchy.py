"""
NBA Hierarchical Narrative Optimization

Discovers optimal formulas at each narrative level (game, series, season, era).
Finds α parameters, feature weights, context multipliers, and mathematical constants.

Usage:
    python experiments/nba_optimization/optimize_nba_hierarchy.py
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hierarchy.nested_narrative_tracker import NestedNarrativeTracker
from src.hierarchy.story_accumulator import StoryAccumulator
from src.hierarchy.emergence_detector import EmergenceDetector
from src.optimizers.hierarchical_optimizer import HierarchicalOptimizer
from src.optimizers.constant_detector import ConstantDetector

# Import transformers
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.statistical import StatisticalTransformer


def load_nba_data():
    """Load NBA game data."""
    print("="*70)
    print("LOADING NBA DATA")
    print("="*70)
    
    # Load ALL seasons for proper validation
    base_path = Path(__file__).parent.parent.parent.parent
    
    # Try all-seasons file first
    all_seasons_path = base_path / 'data/domains/nba_all_seasons_real.json'
    
    if all_seasons_path.exists():
        print(f"Loading complete dataset: {all_seasons_path}")
        try:
            with open(all_seasons_path, 'r') as f:
                games = json.load(f)
                
                if isinstance(games, dict) and 'samples' in games:
                    games = games['samples']
                
                print(f"✅ Loaded {len(games)} games (10 seasons)")
                
                # Add required fields
                for game in games:
                    if 'text' not in game:
                        game['text'] = game.get('narrative', '')
                    if 'label' not in game:
                        game['label'] = 1 if game.get('won', False) else 0
                
                return games
        except Exception as e:
            print(f"Error loading all seasons: {e}")
    
    print("❌ No NBA data found. Please run data collection first:")
    print("   python collect_all_nba_seasons.py")
    return None


def extract_narrative_features(games: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Extract narrative features from games using transformers."""
    print("\n" + "="*70)
    print("EXTRACTING NARRATIVE FEATURES")
    print("="*70)
    
    # Get texts
    texts = [game['text'] for game in games]
    
    # Initialize transformers
    print("\nInitializing 6 narrative transformers...")
    nom = NominativeAnalysisTransformer(track_proper_nouns=True)
    sp = SelfPerceptionTransformer(track_attribution=True, track_growth=True)
    np_trans = NarrativePotentialTransformer(track_modality=True, track_flexibility=True)
    ling = LinguisticPatternsTransformer(track_evolution=True, n_segments=3)
    rel = RelationalValueTransformer(n_features=50)
    ens = EnsembleNarrativeTransformer(n_top_terms=30)
    
    # Fit transformers
    print("Fitting transformers to corpus...")
    nom.fit(texts)
    sp.fit(texts)
    np_trans.fit(texts)
    ling.fit(texts)
    rel.fit(texts)
    ens.fit(texts)
    
    # Transform
    print("Transforming texts to features...")
    nom_features = nom.transform(texts)
    sp_features = sp.transform(texts)
    np_features = np_trans.transform(texts)
    ling_features = ling.transform(texts)
    rel_features = rel.transform(texts)
    ens_features = ens.transform(texts)
    
    # Concatenate
    narrative_features = np.hstack([
        nom_features,
        sp_features,
        np_features,
        ling_features,
        rel_features,
        ens_features
    ])
    
    # Feature names
    feature_names = (
        [f"nom_{i}" for i in range(nom_features.shape[1])] +
        [f"sp_{i}" for i in range(sp_features.shape[1])] +
        [f"np_{i}" for i in range(np_features.shape[1])] +
        [f"ling_{i}" for i in range(ling_features.shape[1])] +
        [f"rel_{i}" for i in range(rel_features.shape[1])] +
        [f"ens_{i}" for i in range(ens_features.shape[1])]
    )
    
    print(f"✅ Extracted {narrative_features.shape[1]} narrative features")
    
    return narrative_features, feature_names


def extract_empirical_features(games: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Extract empirical/statistical features."""
    print("\n" + "="*70)
    print("EXTRACTING EMPIRICAL FEATURES")
    print("="*70)
    
    texts = [game['text'] for game in games]
    
    # Statistical transformer
    stat = StatisticalTransformer(max_features=50)
    stat.fit(texts)
    stat_features = stat.transform(texts)
    
    # Convert sparse to dense if needed
    if hasattr(stat_features, 'toarray'):
        stat_features = stat_features.toarray()
    
    # Ensure numpy array and 2D
    stat_features = np.asarray(stat_features)
    if stat_features.ndim == 1:
        stat_features = stat_features.reshape(-1, 1)
    
    feature_names = [f"stat_{i}" for i in range(stat_features.shape[1])]
    
    print(f"✅ Extracted {stat_features.shape[1]} empirical features")
    print(f"   Shape: {stat_features.shape}")
    print(f"   Type: {type(stat_features)}")
    
    return stat_features, feature_names


def organize_by_hierarchy(tracker: NestedNarrativeTracker, games: List[Dict]):
    """Organize games into hierarchical structure."""
    print("\n" + "="*70)
    print("ORGANIZING INTO HIERARCHICAL NARRATIVES")
    print("="*70)
    
    for game in games:
        # Parse date
        try:
            date = datetime.fromisoformat(game.get('date', datetime.now().isoformat()))
        except:
            date = datetime.now()
        
        # Add to tracker
        tracker.add_game(
            game_id=game['game_id'],
            team=game['team_name'],
            opponent=game['matchup'].split(' vs. ')[1] if ' vs. ' in game['matchup'] else 'Unknown',
            narrative_features=game.get('features', {}),
            context=game.get('context', {}),
            date=date,
            outcome=game.get('label', None)
        )
    
    # Get statistics
    stats = tracker.get_statistics()
    
    print(f"\n✅ Organized into hierarchy:")
    for level, level_stats in stats.items():
        if isinstance(level_stats, dict) and 'count' in level_stats:
            print(f"   {level.capitalize():<10} {level_stats['count']:>4} narratives")
    
    return tracker


def main():
    """Main optimization execution."""
    print("\n" + "="*70)
    print("NBA HIERARCHICAL NARRATIVE OPTIMIZATION")
    print("Discovering optimal formulas at each narrative scale")
    print("="*70)
    
    # 1. Load data
    games = load_nba_data()
    if not games:
        return
    
    print(f"\n📊 Working with {len(games)} NBA games")
    
    # 2. Extract features
    X_narrative, narrative_feature_names = extract_narrative_features(games)
    X_empirical, empirical_feature_names = extract_empirical_features(games)
    y = np.array([game['label'] for game in games])
    
    # Ensure both are 2D
    if X_empirical.ndim == 1:
        X_empirical = X_empirical.reshape(-1, 1)
    if X_narrative.ndim == 1:
        X_narrative = X_narrative.reshape(-1, 1)
    
    print(f"\n📊 Feature dimensions:")
    print(f"   Narrative: {X_narrative.shape}")
    print(f"   Empirical: {X_empirical.shape}")
    print(f"   Outcomes: {y.shape}")
    
    # 3. Initialize hierarchy
    tracker = NestedNarrativeTracker()
    story_accumulator = StoryAccumulator()
    emergence_detector = EmergenceDetector()
    
    # 4. Organize into hierarchy
    # (For now, work at game level - full hierarchy needs temporal data)
    print("\n" + "="*70)
    print("GAME-LEVEL OPTIMIZATION (Foundation)")
    print("="*70)
    
    # 5. Initialize optimizer
    optimizer = HierarchicalOptimizer()
    constant_detector = ConstantDetector()
    
    # Save copies BEFORE optimization (optimizer modifies arrays)
    X_narrative_orig = X_narrative.copy()
    X_empirical_orig = X_empirical.copy()
    
    # 6. Optimize game-level α
    print("\n🎯 PHASE 1: Discover optimal α_game")
    alpha_result = optimizer.optimize_alpha_at_level(
        X_nominative=X_narrative.copy(),  # Pass copies
        X_empirical=X_empirical.copy(),   # Pass copies
        y=y.copy(),
        level='game'
    )
    
    # 7. Discover feature weights (use original saved copies)
    print("\n🎯 PHASE 2: Discover feature weights")
    
    X_combined = np.concatenate([X_narrative_orig, X_empirical_orig], axis=1)
    combined_names = narrative_feature_names + empirical_feature_names
    
    feature_result = optimizer.discover_feature_weights(
        X=X_combined,
        y=y,
        feature_names=combined_names,
        level='game'
    )
    
    # 8. Search for constants
    print("\n🎯 PHASE 3: Search for mathematical constants")
    
    # Create feature dictionary
    feature_dict = {}
    for i, name in enumerate(combined_names):
        feature_dict[name] = X_combined[:, i]
    
    constants_result = constant_detector.search_for_ratios(feature_dict)
    golden_ratios = constant_detector.detect_golden_ratios(feature_dict)
    
    # 9. Generate formula report
    print("\n" + "="*70)
    print("GENERATING FORMULA REPORT")
    print("="*70)
    
    formula_report = optimizer.generate_formula_report()
    print("\n" + formula_report)
    
    # 10. Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_games': len(games),
        'alpha_optimization': alpha_result,
        'feature_weights': feature_result,
        'constants_found': len(constants_result),
        'top_constants': constants_result[:10] if constants_result else [],
        'golden_ratios': golden_ratios,
        'formula_report': formula_report
    }
    
    output_path = 'results/nba_hierarchical_optimization.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")
    except:
        print(f"\n✅ Results generated (save manually if needed)")
    
    # 11. Summary
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\n✅ Discovered optimal α_game = {alpha_result['optimal_alpha']:.3f}")
    print(f"✅ Found {feature_result['n_features_selected']} predictive features")
    print(f"✅ Detected {len(constants_result)} consistent ratios")
    if golden_ratios:
        print(f"🌟 Found {len(golden_ratios)} special constant matches!")
    
    print(f"\n📊 Best game-level accuracy: {alpha_result['best_accuracy']:.1%}")
    print(f"📊 Nominative contribution: {alpha_result['nominative_weight']*100:.0f}%")
    print(f"📊 Empirical contribution: {alpha_result['empirical_weight']*100:.0f}%")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review results in results/nba_hierarchical_optimization.json")
    print("2. Collect more NBA data for series/season optimization")
    print("3. Apply discovered formula to new predictions")
    print("4. Document in NBA_FORMULA_DISCOVERED.md")
    
    return results


if __name__ == '__main__':
    results = main()

