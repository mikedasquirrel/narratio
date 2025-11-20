"""
Experiment 01: Baseline Comparison - Do Stories Matter?

Tests H1: Narrative-driven features predict outcomes better than statistical baselines.

Compares three competing narratives:
1. Statistical Baseline (TF-IDF)
2. Semantic Narrative (embeddings + clustering)
3. Domain Narrative (style + structure + topics)
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.transformers.statistical import StatisticalTransformer
from src.transformers.semantic import SemanticNarrativeTransformer
from src.transformers.domain_text import DomainTextNarrativeTransformer
from src.pipelines.narrative_pipeline import NarrativePipeline
from src.experiments.experiment import NarrativeExperiment
from src.evaluation.evaluator import NarrativeEvaluator
from src.utils.toy_data import quick_load_toy_data
from src.visualization.narrative_plots import quick_plot_results


def build_statistical_pipeline():
    """Build the statistical baseline pipeline."""
    pipeline = NarrativePipeline(
        narrative_name="Statistical Baseline",
        hypothesis="Pure statistical word frequencies (TF-IDF) are sufficient for classification",
        expected_outcome="Baseline performance - other narratives should beat this if they add value",
        domain_assumptions=["Text is just bags of words", "Context doesn't matter"]
    )
    
    pipeline.add_step(
        'features',
        StatisticalTransformer(max_features=1000),
        rationale="TF-IDF captures statistical word importance without domain knowledge"
    )
    
    pipeline.add_step(
        'classifier',
        LogisticRegression(max_iter=1000, random_state=42),
        rationale="Simple linear classifier on statistical features"
    )
    
    return pipeline.build()


def build_semantic_pipeline():
    """Build the semantic narrative pipeline."""
    pipeline = NarrativePipeline(
        narrative_name="Semantic Narrative",
        hypothesis="Understanding semantic meaning through embeddings improves predictions",
        expected_outcome="Better than statistical baseline by capturing deeper meaning",
        domain_assumptions=[
            "Documents have semantic structure",
            "Similar meanings cluster together",
            "Dense representations capture more than sparse TF-IDF"
        ]
    )
    
    pipeline.add_step(
        'features',
        SemanticNarrativeTransformer(n_components=50, n_clusters=10),
        rationale="Semantic embeddings and clustering capture meaning beyond word counts"
    )
    
    pipeline.add_step(
        'scaler',
        StandardScaler(),
        rationale="Normalize features for consistent scale"
    )
    
    pipeline.add_step(
        'classifier',
        LogisticRegression(max_iter=1000, random_state=42),
        rationale="Linear classifier on semantic features"
    )
    
    return pipeline.build()


def build_domain_pipeline():
    """Build the domain narrative pipeline."""
    pipeline = NarrativePipeline(
        narrative_name="Domain Narrative",
        hypothesis="Expert-crafted domain features (style + structure + topics) outperform generic approaches",
        expected_outcome="Best performance by encoding domain expertise about text quality",
        domain_assumptions=[
            "Good writing has consistent style",
            "Clear structure matters",
            "Topical coherence is important",
            "Domain experts know what features matter"
        ]
    )
    
    pipeline.add_step(
        'features',
        DomainTextNarrativeTransformer(
            n_topics=20,
            style_features=True,
            structure_features=True
        ),
        rationale="Domain-specific features capture text quality through expert knowledge"
    )
    
    pipeline.add_step(
        'scaler',
        StandardScaler(),
        rationale="Normalize features for consistent scale"
    )
    
    pipeline.add_step(
        'classifier',
        LogisticRegression(max_iter=1000, random_state=42),
        rationale="Linear classifier on domain features"
    )
    
    return pipeline.build()


def run_baseline_comparison():
    """
    Run the complete baseline comparison experiment.
    """
    print("=" * 80)
    print("EXPERIMENT 01: BASELINE COMPARISON - DO STORIES MATTER?")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading toy dataset...")
    data = quick_load_toy_data(data_dir=str(project_root / "narrative_optimization" / "data" / "toy"))
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    target_names = data['target_names']
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Categories: {target_names}")
    print()
    
    # Build pipelines
    print("Building narrative pipelines...")
    statistical_pipeline = build_statistical_pipeline()
    semantic_pipeline = build_semantic_pipeline()
    domain_pipeline = build_domain_pipeline()
    print("  ✓ Statistical Baseline")
    print("  ✓ Semantic Narrative")
    print("  ✓ Domain Narrative")
    print()
    
    # Create experiment
    output_dir = Path(__file__).parent
    experiment = NarrativeExperiment(
        experiment_id="01_baseline_comparison",
        description="Test whether narrative-driven features outperform statistical baselines",
        output_dir=str(output_dir)
    )
    
    # Add narratives
    experiment.add_narrative(
        statistical_pipeline,
        hypothesis="Statistical baseline: TF-IDF is sufficient",
        name="Statistical Baseline"
    )
    
    experiment.add_narrative(
        semantic_pipeline,
        hypothesis="Semantic narrative: embeddings + clustering add value",
        name="Semantic Narrative"
    )
    
    experiment.add_narrative(
        domain_pipeline,
        hypothesis="Domain narrative: expert features are best",
        name="Domain Narrative"
    )
    
    # Define evaluation
    experiment.define_evaluation(
        metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Run experiment
    print("Running cross-validation (this may take a few minutes)...")
    print()
    results = experiment.run(X_train, y_train, verbose=True)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    analysis = experiment.analyze()
    
    print("\nBest Narratives by Metric:")
    for metric, info in analysis['best_narratives'].items():
        print(f"  {metric}: {info['name']} ({info['score']:.4f})")
    
    print("\nKey Insights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Generate report
    print("\nGenerating report...")
    report = experiment.generate_report()
    print(f"  ✓ Report saved to {output_dir / 'report.md'}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    quick_plot_results(results, output_dir=str(output_dir))
    print(f"  ✓ Plots saved to {output_dir}")
    
    # Comprehensive evaluation on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    evaluator = NarrativeEvaluator(random_state=42)
    
    for narrative_name, narrative_results in results['narratives'].items():
        print(f"\n{narrative_name}:")
        
        # Get best estimator from CV
        best_estimator = narrative_results['fitted_estimators'][0]
        
        # Evaluate
        eval_results = evaluator.comprehensive_evaluation(
            best_estimator,
            X_test,
            y_test,
            X_train,
            y_train,
            include_robustness=False  # Skip for speed
        )
        
        print(f"  Test Accuracy: {eval_results['performance']['accuracy']:.4f}")
        print(f"  Test F1 (macro): {eval_results['performance']['f1_macro']:.4f}")
        print(f"  Coherence Score: {eval_results['coherence']['coherence_score']:.3f}")
        print(f"  Interpretability Score: {eval_results['interpretability']['interpretability_score']:.3f}")
        print(f"  Overall Quality: {eval_results['overall_quality']:.3f}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("Key files:")
    print(f"  - {output_dir / 'report.md'} (experiment report)")
    print(f"  - {output_dir / 'results.json'} (detailed results)")
    print(f"  - {output_dir / 'experiment_summary.png'} (visualization)")
    print()
    
    return results, analysis


if __name__ == '__main__':
    results, analysis = run_baseline_comparison()

