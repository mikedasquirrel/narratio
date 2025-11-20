"""
Experiment 03: Linguistic Patterns Test

Tests: How stories are told matters as much as what is said.
Linguistic patterns (voice, agency, temporality) vs baseline.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.transformers.statistical import StatisticalTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.pipelines.narrative_pipeline import NarrativePipeline
from src.experiments.experiment import NarrativeExperiment
from src.utils.toy_data import quick_load_toy_data


def run_linguistic_test():
    print("=" * 80)
    print("EXPERIMENT 03: LINGUISTIC PATTERNS TEST")
    print("=" * 80)
    print("Baseline to beat: 69.0% accuracy")
    print()
    
    data = quick_load_toy_data(data_dir=str(project_root / "data" / "toy"))
    X_train, y_train = data['X_train'], data['y_train']
    
    print(f"✓ Data loaded: {len(X_train)} samples\n")
    
    # Build pipelines
    statistical = NarrativePipeline("Statistical")
    statistical.add_step('features', StatisticalTransformer(), "TF-IDF")
    statistical.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), "Classify")
    
    linguistic = NarrativePipeline(
        "Linguistic Patterns",
        hypothesis="Voice, agency, temporality capture predictive signals"
    )
    linguistic.add_step('features', LinguisticPatternsTransformer(), "Extract linguistic patterns")
    linguistic.add_step('scaler', StandardScaler(), "Normalize")
    linguistic.add_step('classifier', LogisticRegression(max_iter=1000, random_state=42), "Classify")
    
    print("✓ Pipelines built\n")
    
    experiment = NarrativeExperiment("03_linguistic_test", "Linguistic patterns vs baseline", str(Path(__file__).parent))
    experiment.add_narrative(statistical.build(), "Statistical sufficient", "Statistical")
    experiment.add_narrative(linguistic.build(), "Linguistic adds value", "Linguistic")
    experiment.define_evaluation(metrics=['accuracy', 'f1_macro'], cv_strategy=StratifiedKFold(5, shuffle=True, random_state=42))
    
    print("🚀 Running experiment...\n")
    results = experiment.run(X_train, y_train, verbose=True)
    
    # Analysis
    stat_acc = results['narratives']['Statistical']['cv_scores']['accuracy']['test_mean']
    ling_acc = results['narratives']['Linguistic']['cv_scores']['accuracy']['test_mean']
    
    print(f"\n{'='*80}")
    print(f"Statistical: {stat_acc:.4f} | Linguistic: {ling_acc:.4f}")
    
    if ling_acc > stat_acc:
        print(f"✓ LINGUISTIC WINS! +{((ling_acc-stat_acc)/stat_acc)*100:.1f}%")
    else:
        print(f"→ Linguistic: {((ling_acc-stat_acc)/stat_acc)*100:.1f}% vs baseline")
    
    experiment.generate_report()
    return results


if __name__ == '__main__':
    run_linguistic_test()

