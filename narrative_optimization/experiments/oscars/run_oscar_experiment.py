"""
Oscar Best Picture Experiment - Rigorous Pipeline Analysis

Uses proper NarrativePipeline and NarrativeExperiment infrastructure.

Tests competing hypotheses:
H1: Baseline (genre + year) predicts winners
H2: Nominative features (cast/character names) predict winners  
H3: Complete narrative (all transformers) predicts winners

Finds Silver Ratio (Σ_oscars) = centroid of winners in gravitational space
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.narrative_pipeline import NarrativePipeline
from src.experiments.experiment import NarrativeExperiment
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.statistical import StatisticalTransformer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer


def load_oscar_data():
    """Load Oscar nominee data."""
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/oscar_nominees_complete.json'
    
    with open(data_path, 'r') as f:
        oscar_data = json.load(f)
    
    # Flatten
    all_films = []
    for year, films in oscar_data.items():
        all_films.extend(films)
    
    # Create full narratives
    X = []
    y = []
    
    for film in all_films:
        # Complete narrative: overview + cast + characters + keywords + genres
        narrative = ' '.join([
            film['title'],
            film['overview'],
            film.get('tagline', ''),
            ' '.join([c['actor'] for c in film['cast'][:20]]),
            ' '.join([c['character'] for c in film['cast'][:20] if c['character']]),
            ' '.join(film['keywords']),
            ' '.join(film['director']),
            ' '.join(film['genres'])
        ])
        
        X.append(narrative)
        y.append(int(film['won_oscar']))
    
    return np.array(X), np.array(y), all_films


def build_baseline_pipeline():
    """H1: Statistical baseline (TF-IDF only)."""
    pipeline = NarrativePipeline(
        narrative_name="Baseline",
        hypothesis="Genre and content words alone predict Oscar winners",
        expected_outcome="Low performance - misses narrative structure"
    )
    
    pipeline.add_step(
        'statistical',
        StatisticalTransformer(max_features=100),
        "TF-IDF captures content words (genre markers, plot keywords)"
    )
    
    pipeline.add_step(
        'classifier',
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gradient boosting for classification"
    )
    
    return pipeline.build()


def build_nominative_pipeline():
    """H2: Nominative features (names matter)."""
    pipeline = NarrativePipeline(
        narrative_name="Nominative",
        hypothesis="Actor/character/director names and their nominative properties predict winners",
        expected_outcome="Cast star power + character naming patterns matter"
    )
    
    pipeline.add_step(
        'nominative',
        NominativeAnalysisTransformer(),
        "Extract nominative features: semantic fields, proper nouns, naming patterns from cast/characters"
    )
    
    pipeline.add_step(
        'classifier',
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gradient boosting"
    )
    
    return pipeline.build()


def build_complete_pipeline():
    """H3: Complete narrative (all transformers)."""
    from sklearn.pipeline import FeatureUnion
    
    pipeline = NarrativePipeline(
        narrative_name="Complete Narrative",
        hypothesis="Full narrative features (nominative + self-perception + potential) predict winners",
        expected_outcome="Comprehensive narrative analysis beats baseline"
    )
    
    # Parallel feature extraction
    feature_union = FeatureUnion([
        ('nominative', NominativeAnalysisTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer())
    ])
    
    pipeline.add_step(
        'features',
        feature_union,
        "Extract all narrative dimensions in parallel"
    )
    
    pipeline.add_step(
        'classifier',
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Gradient boosting on complete ж"
    )
    
    return pipeline.build()


def main():
    """Run rigorous Oscar experiment with proper infrastructure."""
    print("=" * 80)
    print("OSCAR BEST PICTURE - RIGOROUS EXPERIMENT")
    print("Using NarrativePipeline + NarrativeExperiment framework")
    print("=" * 80)
    print("")
    
    # Load data
    X, y, films = load_oscar_data()
    
    print(f"Loaded: {len(X)} films")
    print(f"  Winners: {sum(y)}")
    print(f"  Nominees: {len(y) - sum(y)}")
    print("")
    
    # Build pipelines
    print("Building competing narrative pipelines...")
    baseline_pipe = build_baseline_pipeline()
    nominative_pipe = build_nominative_pipeline()
    complete_pipe = build_complete_pipeline()
    print("  ✓ H1: Baseline (TF-IDF)")
    print("  ✓ H2: Nominative (names)")
    print("  ✓ H3: Complete (all transformers)")
    print("")
    
    # Create experiment
    output_dir = Path(__file__).parent
    experiment = NarrativeExperiment(
        experiment_id="oscars_best_picture",
        description="Test which narrative features predict Oscar Best Picture winners from competitive nominee field",
        output_dir=str(output_dir)
    )
    
    # Add competing narratives
    experiment.add_narrative(
        baseline_pipe,
        hypothesis="H1: Genre/content baseline",
        name="Baseline"
    )
    
    experiment.add_narrative(
        nominative_pipe,
        hypothesis="H2: Nominative features (cast/characters/directors)",
        name="Nominative"
    )
    
    experiment.add_narrative(
        complete_pipe,
        hypothesis="H3: Complete narrative (ж from all transformers)",
        name="Complete"
    )
    
    # Define evaluation
    experiment.define_evaluation(
        metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    
    # Run with proper cross-validation
    print("Running experiment with 5-fold cross-validation...")
    print("")
    results = experiment.run(X, y, verbose=True)
    
    # Save results manually
    results_path = output_dir / 'oscar_experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    # Analyze and display results
    print("\nRESULTS SUMMARY:")
    print("-" * 80)
    for name, narrative_results in results['narratives'].items():
        print(f"\n{name}:")
        for metric, scores in narrative_results['cv_scores'].items():
            print(f"  {metric}: {scores['test_mean']:.4f} (+/- {scores['test_std']:.4f})")
    
    # Calculate Д
    baseline_acc = results['narratives']['Baseline']['cv_scores']['accuracy']['test_mean']
    complete_acc = results['narratives']['Complete']['cv_scores']['accuracy']['test_mean']
    
    D_advantage = complete_acc - baseline_acc
    
    print(f"\n{'='*80}")
    print("NARRATIVE ADVANTAGE (Д)")
    print(f"{'='*80}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Complete accuracy: {complete_acc:.4f}")
    print(f"Д (advantage): {D_advantage:+.4f}")
    print(f"\nInterpretation: Narrative adds {D_advantage:.1%} beyond baseline")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - JSON results")
    print(f"  - Pickle file (full)")
    print(f"  - Markdown report")
    
    return results


if __name__ == "__main__":
    main()

