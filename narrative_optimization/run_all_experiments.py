#!/usr/bin/env python3
"""
Run all advanced transformer experiments with progress reporting.

Tests all 6 advanced transformers individually and in combinations.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from src.transformers.statistical import StatisticalTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.nominative import NominativeAnalysisTransformer
from src.pipelines.narrative_pipeline import NarrativePipeline
from src.experiments.experiment import NarrativeExperiment
from src.utils.toy_data import quick_load_toy_data


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_quick_test(transformer_name, transformer_class, transformer_kwargs=None):
    """Quick test of a single transformer vs baseline."""
    if transformer_kwargs is None:
        transformer_kwargs = {}
    
    print_header(f"Testing {transformer_name}")
    
    # Load data
    data = quick_load_toy_data()
    X_train, y_train = data['X_train'], data['y_train']
    
    # Build pipeline
    pipeline = NarrativePipeline(transformer_name)
    pipeline.add_step('features', transformer_class(**transformer_kwargs), "Extract features")
    pipeline.add_step('scaler', StandardScaler(), "Normalize")
    pipeline.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), "Classify")
    
    # Quick CV
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline.build(), X_train, y_train, cv=3, scoring='accuracy')
    
    mean_score = scores.mean()
    print(f"✓ {transformer_name}: {mean_score:.4f} accuracy (±{scores.std():.4f})")
    
    return mean_score


def run_combination_test(name, transformers_dict):
    """Test combination of transformers."""
    print_header(f"Testing {name}")
    
    data = quick_load_toy_data()
    X_train, y_train = data['X_train'], data['y_train']
    
    # Build with FeatureUnion
    pipeline = NarrativePipeline(name)
    pipeline.add_parallel_features('features', transformers_dict, "Combined features")
    pipeline.add_step('scaler', StandardScaler(with_mean=False), "Normalize (sparse-safe)")
    pipeline.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), "Classify")
    
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline.build(), X_train, y_train, cv=3, scoring='accuracy')
    
    mean_score = scores.mean()
    print(f"✓ {name}: {mean_score:.4f} accuracy (±{scores.std():.4f})")
    
    return mean_score


def main():
    print("\n" + "🚀" * 40)
    print("\n  COMPLETE EXPERIMENT SUITE - ADVANCED TRANSFORMER VALIDATION")
    print("  Baseline to beat: 69.0% accuracy (Statistical TF-IDF)")
    print("\n" + "🚀" * 40)
    
    results = {}
    
    # PHASE 1: Individual Advanced Transformers (Quick 3-fold CV)
    print("\n\n📊 PHASE 1: INDIVIDUAL TRANSFORMER TESTS (3-fold CV for speed)")
    print("-" * 80)
    
    print("\n[1/6] Testing Ensemble...")
    results['ensemble'] = run_quick_test("Ensemble", EnsembleNarrativeTransformer, {'n_top_terms': 30})
    
    print("\n[2/6] Testing Linguistic...")
    results['linguistic'] = run_quick_test("Linguistic", LinguisticPatternsTransformer)
    
    print("\n[3/6] Testing Self-Perception...")
    results['self_perception'] = run_quick_test("Self-Perception", SelfPerceptionTransformer)
    
    print("\n[4/6] Testing Narrative Potential...")
    results['potential'] = run_quick_test("Narrative Potential", NarrativePotentialTransformer)
    
    print("\n[5/6] Testing Relational...")
    results['relational'] = run_quick_test("Relational", RelationalValueTransformer, {'n_features': 50})
    
    print("\n[6/6] Testing Nominative...")
    results['nominative'] = run_quick_test("Nominative", NominativeAnalysisTransformer)
    
    # PHASE 2: Combinations with Statistical Baseline
    print("\n\n📊 PHASE 2: COMBINATIONS WITH TF-IDF BASELINE")
    print("-" * 80)
    print("Testing if advanced transformers ADD VALUE to statistical baseline...")
    
    print("\n[1/3] Statistical + Ensemble...")
    results['stat+ensemble'] = run_combination_test(
        "Statistical + Ensemble",
        {
            'statistical': (StatisticalTransformer(max_features=500), "TF-IDF"),
            'ensemble': (EnsembleNarrativeTransformer(n_top_terms=30), "Network")
        }
    )
    
    print("\n[2/3] Statistical + Linguistic...")
    results['stat+linguistic'] = run_combination_test(
        "Statistical + Linguistic",
        {
            'statistical': (StatisticalTransformer(max_features=500), "TF-IDF"),
            'linguistic': (LinguisticPatternsTransformer(), "Patterns")
        }
    )
    
    print("\n[3/3] Statistical + Self-Perception...")
    results['stat+selfperc'] = run_combination_test(
        "Statistical + Self-Perception",
        {
            'statistical': (StatisticalTransformer(max_features=500), "TF-IDF"),
            'self_perception': (SelfPerceptionTransformer(), "Identity")
        }
    )
    
    # PHASE 3: Multi-Modal Supreme Test
    print("\n\n📊 PHASE 3: MULTI-MODAL SUPREME TEST")
    print("-" * 80)
    print("Testing ALL transformers combined...")
    
    print("\n[Supreme Test] All 6 Advanced + Statistical...")
    results['multimodal_supreme'] = run_combination_test(
        "Multi-Modal Supreme",
        {
            'statistical': (StatisticalTransformer(max_features=300), "TF-IDF"),
            'ensemble': (EnsembleNarrativeTransformer(n_top_terms=20), "Network"),
            'linguistic': (LinguisticPatternsTransformer(), "Linguistic"),
            'self_perc': (SelfPerceptionTransformer(), "Identity")
        }
    )
    
    # FINAL SUMMARY
    print("\n\n" + "=" * 80)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    baseline = 0.69
    
    print(f"\n📊 BASELINE: {baseline:.1%}")
    print("\n📈 INDIVIDUAL ADVANCED TRANSFORMERS:")
    for name in ['ensemble', 'linguistic', 'self_perception', 'potential', 'relational', 'nominative']:
        if name in results:
            score = results[name]
            diff = score - baseline
            symbol = "✓" if score > baseline else "→"
            print(f"  {symbol} {name.title():20s}: {score:.4f} ({diff:+.1%})")
    
    print("\n🔗 COMBINATIONS WITH STATISTICAL:")
    for name in ['stat+ensemble', 'stat+linguistic', 'stat+selfperc']:
        if name in results:
            score = results[name]
            diff = score - baseline
            symbol = "✓" if score > baseline else "→"
            print(f"  {symbol} {name:30s}: {score:.4f} ({diff:+.1%})")
    
    print("\n🌟 SUPREME MULTI-MODAL:")
    if 'multimodal_supreme' in results:
        score = results['multimodal_supreme']
        diff = score - baseline
        symbol = "✓✓✓" if score > baseline else "→"
        print(f"  {symbol} Multi-Modal (All Combined): {score:.4f} ({diff:+.1%})")
    
    # Key insights
    print("\n\n💡 KEY INSIGHTS:")
    
    best_individual = max((results.get(k, 0), k) for k in ['ensemble', 'linguistic', 'self_perception', 'potential', 'relational', 'nominative'])[1]
    best_combo = max((results.get(k, 0), k) for k in ['stat+ensemble', 'stat+linguistic', 'stat+selfperc'])[1]
    
    print(f"1. Best individual advanced transformer: {best_individual.title()} ({results.get(best_individual, 0):.1%})")
    print(f"2. Best combination: {best_combo} ({results.get(best_combo, 0):.1%})")
    
    if results.get('multimodal_supreme', 0) > baseline:
        print(f"3. ✓ BREAKTHROUGH: Multi-modal beats baseline! ({results['multimodal_supreme']:.1%} vs {baseline:.1%})")
        print("   → Narrative dimensions capture additional signal")
    elif results.get('multimodal_supreme', 0) > 0.65:
        print(f"3. Multi-modal competitive ({results.get('multimodal_supreme', 0):.1%}), needs refinement")
    else:
        print("3. Current features better suited for domain-specific tasks than generic classification")
        print("   → Recommendation: Test on datasets where narrative structure matters more")
    
    # Save summary
    output_file = Path(__file__).parent / "experiments" / "EXPERIMENT_SUMMARY.md"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Complete Experiment Suite Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Results\n\n")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{name}**: {score:.4f} ({(score-baseline)*100:+.1f}%)\n")
    
    print(f"\n✓ Complete summary saved to {output_file}")
    print("\n" + "🎉" * 40)
    print("\n  EXPERIMENT SUITE COMPLETE")
    print("\n" + "🎉" * 40 + "\n")


if __name__ == '__main__':
    main()

