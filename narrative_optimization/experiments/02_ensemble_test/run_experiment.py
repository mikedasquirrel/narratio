"""
Experiment 02: Ensemble Effects Test

Tests H4: Network relationships capture signals that word frequencies miss.
Compares Ensemble transformer (co-occurrence, network centrality) against baseline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.transformers.statistical import StatisticalTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.pipelines.narrative_pipeline import NarrativePipeline
from src.experiments.experiment import NarrativeExperiment
from src.utils.toy_data import quick_load_toy_data


def build_ensemble_pipeline():
    """Build ensemble network analysis pipeline."""
    pipeline = NarrativePipeline(
        narrative_name="Ensemble Network",
        hypothesis="Network relationships and co-occurrence patterns capture meaning beyond word frequencies",
        expected_outcome="Beat 69% baseline by capturing ensemble effects",
        domain_assumptions=["Elements gain meaning through relationships", "Network structure matters"]
    )
    
    pipeline.add_step(
        'features',
        EnsembleNarrativeTransformer(n_top_terms=50, network_metrics=True),
        rationale="Network analysis captures relational meaning"
    )
    
    pipeline.add_step('scaler', StandardScaler(), rationale="Normalize")
    pipeline.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), rationale="Classify")
    
    return pipeline.build()


def run_ensemble_test():
    print("=" * 80)
    print("EXPERIMENT 02: ENSEMBLE EFFECTS TEST (H4)")
    print("=" * 80)
    print("Baseline to beat: 69.0% accuracy, 68.8% F1-macro")
    print()
    
    # Load data
    print("📊 Loading data...")
    data = quick_load_toy_data(data_dir=str(project_root / "data" / "toy"))
    X_train, y_train = data['X_train'], data['y_train']
    
    print(f"✓ Loaded {len(X_train)} training samples")
    print()
    
    # Build pipelines
    print("🔧 Building pipelines...")
    statistical_pipeline = NarrativePipeline("Statistical Baseline")
    statistical_pipeline.add_step('features', StatisticalTransformer(max_features=1000), "TF-IDF")
    statistical_pipeline.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), "Classify")
    
    ensemble_pipeline = build_ensemble_pipeline()
    
    print("✓ Statistical Baseline pipeline")
    print("✓ Ensemble Network pipeline")
    print()
    
    # Create experiment
    output_dir = Path(__file__).parent
    experiment = NarrativeExperiment(
        experiment_id="02_ensemble_test",
        description="Test if ensemble network effects beat statistical baseline",
        output_dir=str(output_dir)
    )
    
    experiment.add_narrative(statistical_pipeline.build(), "Statistical is sufficient", "Statistical")
    experiment.add_narrative(ensemble_pipeline, "Network relationships add value", "Ensemble")
    
    experiment.define_evaluation(
        metrics=['accuracy', 'f1_macro'],
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Run
    print("🚀 Running 5-fold cross-validation...")
    print()
    results = experiment.run(X_train, y_train, verbose=True)
    
    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    stat_acc = results['narratives']['Statistical']['cv_scores']['accuracy']['test_mean']
    ensemble_acc = results['narratives']['Ensemble']['cv_scores']['accuracy']['test_mean']
    stat_f1 = results['narratives']['Statistical']['cv_scores']['f1_macro']['test_mean']
    ensemble_f1 = results['narratives']['Ensemble']['cv_scores']['f1_macro']['test_mean']
    
    print(f"\nStatistical Baseline: {stat_acc:.4f} accuracy, {stat_f1:.4f} F1")
    print(f"Ensemble Network:     {ensemble_acc:.4f} accuracy, {ensemble_f1:.4f} F1")
    print()
    
    improvement = ((ensemble_acc - stat_acc) / stat_acc) * 100
    if ensemble_acc > stat_acc:
        print(f"✓ ENSEMBLE WINS! +{improvement:.1f}% improvement")
        print("→ Network relationships capture additional signal beyond word frequencies")
    elif ensemble_acc > 0.65:
        print(f"→ Ensemble competitive ({improvement:.1f}%), shows promise")
    else:
        print(f"→ Ensemble underperformed ({improvement:.1f}%), needs refinement or different domain")
    
    # Generate report
    experiment.generate_report()
    print(f"\n✓ Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    run_ensemble_test()

