"""
Hurricane Names Baseline Experiment

Validates the Jung et al. (2014) finding and tests extended hypotheses
about hurricane name effects on perception and behavior.

Hypotheses:
H1: Gender predicts evacuation rates (controlling for severity)
H2: Syllables negatively correlate with threat perception
H3: Memorability improves emergency preparedness
H4: Name effects are stronger for moderate-intensity storms
H5: Combined nominative + severity beats either alone

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from domains.hurricanes.data_collector import HurricaneDataCollector
from domains.hurricanes.name_analyzer import HurricaneNameAnalyzer
from src.transformers.hurricanes.nominative_hurricane import HurricaneNominativeTransformer
from src.transformers.hurricanes.hurricane_ensemble import HurricaneEnsembleTransformer


def collect_hurricane_data(n_samples: int = 100):
    """Collect hurricane dataset."""
    print("📊 Collecting hurricane data...")
    collector = HurricaneDataCollector(use_real_data=False)
    dataset = collector.collect_dataset(start_year=1950, end_year=2024, min_category=1)
    
    # Ensure we have enough samples
    if len(dataset) < n_samples:
        print(f"⚠️  Generated {len(dataset)} hurricanes, target was {n_samples}")
    else:
        # Subsample if we have too many
        np.random.seed(42)
        dataset = np.random.choice(dataset, size=n_samples, replace=False).tolist()
    
    print(f"✅ Collected {len(dataset)} hurricane records\n")
    return dataset


def test_h1_gender_effect(dataset):
    """
    H1: Gender predicts evacuation rates (controlling for severity)
    
    Expected: Feminine names → lower evacuation, d = 0.38, p < 0.05
    """
    print("=" * 70)
    print("HYPOTHESIS 1: Gender Effect on Evacuation")
    print("=" * 70)
    
    # Extract features
    names = [h['name'] for h in dataset]
    gender_ratings = np.array([h['gender_rating'] for h in dataset])
    evacuation_rates = np.array([h['outcomes']['evacuation_rate'] for h in dataset])
    categories = np.array([h['actual_severity']['category'] for h in dataset])
    
    # Test 1: Simple correlation (without severity control)
    r_simple, p_simple = stats.pearsonr(gender_ratings, evacuation_rates)
    print(f"\n1. Simple correlation (no controls):")
    print(f"   r = {r_simple:.3f}, p = {p_simple:.4f}")
    
    # Test 2: Controlling for severity
    # Regression: evacuation ~ gender + category
    X = np.column_stack([gender_ratings, categories])
    y = evacuation_rates
    
    model = LinearRegression()
    model.fit(X, y)
    
    gender_coef = model.coef_[0]
    r2 = r2_score(y, model.predict(X))
    
    print(f"\n2. Controlling for severity:")
    print(f"   Gender coefficient: {gender_coef:.4f}")
    print(f"   Model R² = {r2:.3f}")
    
    # Calculate effect size (Cohen's d)
    # Split by gender (median split)
    median_gender = np.median(gender_ratings)
    masculine_evac = evacuation_rates[gender_ratings <= median_gender]
    feminine_evac = evacuation_rates[gender_ratings > median_gender]
    
    mean_diff = np.mean(masculine_evac) - np.mean(feminine_evac)
    pooled_std = np.sqrt((np.std(masculine_evac)**2 + np.std(feminine_evac)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    # T-test
    t_stat, p_val = stats.ttest_ind(masculine_evac, feminine_evac)
    
    print(f"\n3. Effect size:")
    print(f"   Masculine names: μ = {np.mean(masculine_evac):.3f}, σ = {np.std(masculine_evac):.3f}")
    print(f"   Feminine names: μ = {np.mean(feminine_evac):.3f}, σ = {np.std(feminine_evac):.3f}")
    print(f"   Mean difference: {mean_diff:.3f} ({mean_diff*100:.1f}%)")
    print(f"   Cohen's d = {cohens_d:.3f}")
    print(f"   t = {t_stat:.3f}, p = {p_val:.4f}")
    
    # Expected from research: d = 0.38, p = 0.004
    print(f"\n4. Comparison to research:")
    print(f"   Expected: d = 0.38, p < 0.05")
    print(f"   Observed: d = {cohens_d:.3f}, p = {p_val:.4f}")
    print(f"   Status: {'✅ VALIDATED' if abs(cohens_d) > 0.25 and p_val < 0.05 else '⚠️ WEAK/NOT SIGNIFICANT'}")
    
    return {
        'hypothesis': 'H1_gender_effect',
        'r_simple': r_simple,
        'p_simple': p_simple,
        'gender_coef': gender_coef,
        'r2_controlled': r2,
        'cohens_d': cohens_d,
        'p_value': p_val,
        'validated': abs(cohens_d) > 0.25 and p_val < 0.05
    }


def test_h2_syllable_effect(dataset):
    """
    H2: Syllables negatively correlate with threat perception
    
    Expected: r = -0.18, p < 0.10 (marginal)
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Syllable Effect")
    print("=" * 70)
    
    syllables = np.array([h['syllables'] for h in dataset])
    evacuation_rates = np.array([h['outcomes']['evacuation_rate'] for h in dataset])
    
    r, p = stats.pearsonr(syllables, evacuation_rates)
    
    print(f"\n1. Syllables vs Evacuation Rate:")
    print(f"   r = {r:.3f}, p = {p:.4f}")
    print(f"\n2. Comparison to research:")
    print(f"   Expected: r = -0.18, p < 0.10")
    print(f"   Observed: r = {r:.3f}, p = {p:.4f}")
    print(f"   Status: {'✅ VALIDATED' if p < 0.10 else '⚠️ NOT SIGNIFICANT'}")
    
    return {
        'hypothesis': 'H2_syllable_effect',
        'correlation': r,
        'p_value': p,
        'validated': p < 0.10
    }


def test_h3_memorability_effect(dataset):
    """
    H3: Memorability improves emergency preparedness
    
    Expected: r = 0.22, p < 0.05
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Memorability Effect")
    print("=" * 70)
    
    memorability = np.array([h['memorability'] for h in dataset])
    evacuation_rates = np.array([h['outcomes']['evacuation_rate'] for h in dataset])
    
    r, p = stats.pearsonr(memorability, evacuation_rates)
    
    print(f"\n1. Memorability vs Evacuation Rate:")
    print(f"   r = {r:.3f}, p = {p:.4f}")
    print(f"\n2. Comparison to research:")
    print(f"   Expected: r = 0.22, p < 0.05")
    print(f"   Observed: r = {r:.3f}, p = {p:.4f}")
    print(f"   Status: {'✅ VALIDATED' if r > 0 and p < 0.05 else '⚠️ WEAK/NOT SIGNIFICANT'}")
    
    return {
        'hypothesis': 'H3_memorability_effect',
        'correlation': r,
        'p_value': p,
        'validated': r > 0 and p < 0.05
    }


def test_h4_interaction_effect(dataset):
    """
    H4: Name effects are stronger for moderate-intensity storms
    
    Tests gender × severity interaction
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Gender × Severity Interaction")
    print("=" * 70)
    
    gender_ratings = np.array([h['gender_rating'] for h in dataset])
    categories = np.array([h['actual_severity']['category'] for h in dataset])
    evacuation_rates = np.array([h['outcomes']['evacuation_rate'] for h in dataset])
    
    # Model with interaction term
    X_no_interact = np.column_stack([gender_ratings, categories])
    X_interact = np.column_stack([gender_ratings, categories, gender_ratings * categories])
    y = evacuation_rates
    
    # Fit both models
    model_simple = LinearRegression()
    model_simple.fit(X_no_interact, y)
    r2_simple = r2_score(y, model_simple.predict(X_no_interact))
    
    model_interact = LinearRegression()
    model_interact.fit(X_interact, y)
    r2_interact = r2_score(y, model_interact.predict(X_interact))
    
    interaction_coef = model_interact.coef_[2]
    
    # F-test for interaction significance
    n = len(y)
    p = 2  # parameters in simple model
    q = 3  # parameters in interaction model
    
    f_stat = ((r2_interact - r2_simple) / (q - p)) / ((1 - r2_interact) / (n - q - 1))
    p_val = 1 - stats.f.cdf(f_stat, q - p, n - q - 1)
    
    print(f"\n1. Model comparison:")
    print(f"   Without interaction: R² = {r2_simple:.3f}")
    print(f"   With interaction: R² = {r2_interact:.3f}")
    print(f"   Improvement: ΔR² = {r2_interact - r2_simple:.3f}")
    
    print(f"\n2. Interaction term:")
    print(f"   Coefficient: {interaction_coef:.4f}")
    print(f"   F-statistic: {f_stat:.3f}")
    print(f"   p-value: {p_val:.4f}")
    print(f"   Status: {'✅ SIGNIFICANT' if p_val < 0.05 else '⚠️ NOT SIGNIFICANT'}")
    
    return {
        'hypothesis': 'H4_interaction_effect',
        'r2_simple': r2_simple,
        'r2_interact': r2_interact,
        'delta_r2': r2_interact - r2_simple,
        'interaction_coef': interaction_coef,
        'f_stat': f_stat,
        'p_value': p_val,
        'validated': p_val < 0.05
    }


def test_h5_ensemble_superiority(dataset):
    """
    H5: Combined nominative + severity beats either alone
    
    Compares:
    - Severity only
    - Nominative only
    - Combined
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Ensemble Model Superiority")
    print("=" * 70)
    
    # Prepare data
    names = [h['name'] for h in dataset]
    evacuation_rates = np.array([h['outcomes']['evacuation_rate'] for h in dataset])
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(dataset)), test_size=0.3, random_state=42
    )
    
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]
    train_names = [names[i] for i in train_idx]
    test_names = [names[i] for i in test_idx]
    y_train = evacuation_rates[train_idx]
    y_test = evacuation_rates[test_idx]
    
    # Model 1: Severity only
    X_train_sev = np.array([[d['actual_severity']['category']] for d in train_data])
    X_test_sev = np.array([[d['actual_severity']['category']] for d in test_data])
    
    model_sev = Ridge(alpha=1.0)
    model_sev.fit(X_train_sev, y_train)
    y_pred_sev = model_sev.predict(X_test_sev)
    r2_sev = r2_score(y_test, y_pred_sev)
    
    # Model 2: Nominative only
    nom_transformer = HurricaneNominativeTransformer(
        include_interactions=True, normalize_features=True
    )
    X_train_nom = nom_transformer.fit_transform(train_names, y_train)
    X_test_nom = nom_transformer.transform(test_names)
    
    model_nom = Ridge(alpha=1.0)
    model_nom.fit(X_train_nom, y_train)
    y_pred_nom = model_nom.predict(X_test_nom)
    r2_nom = r2_score(y_test, y_pred_nom)
    
    # Model 3: Ensemble (Combined)
    ensemble_transformer = HurricaneEnsembleTransformer(
        include_interactions=True, include_severity=True
    )
    X_train_ens = ensemble_transformer.fit_transform(train_data, y_train)
    X_test_ens = ensemble_transformer.transform(test_data)
    
    model_ens = Ridge(alpha=1.0)
    model_ens.fit(X_train_ens, y_train)
    y_pred_ens = model_ens.predict(X_test_ens)
    r2_ens = r2_score(y_test, y_pred_ens)
    
    print(f"\n1. Model Performance (R² on test set):")
    print(f"   Severity only: R² = {r2_sev:.3f}")
    print(f"   Nominative only: R² = {r2_nom:.3f}")
    print(f"   Ensemble (combined): R² = {r2_ens:.3f}")
    
    print(f"\n2. Improvements:")
    print(f"   Ensemble vs Severity: ΔR² = {r2_ens - r2_sev:+.3f}")
    print(f"   Ensemble vs Nominative: ΔR² = {r2_ens - r2_nom:+.3f}")
    
    improvement = r2_ens > r2_sev and r2_ens > r2_nom
    print(f"\n3. Status: {'✅ ENSEMBLE SUPERIOR' if improvement else '⚠️ NO CLEAR WINNER'}")
    
    return {
        'hypothesis': 'H5_ensemble_superiority',
        'r2_severity': r2_sev,
        'r2_nominative': r2_nom,
        'r2_ensemble': r2_ens,
        'improvement_vs_severity': r2_ens - r2_sev,
        'improvement_vs_nominative': r2_ens - r2_nom,
        'validated': improvement
    }


def run_full_experiment():
    """Run complete hurricane names experiment suite."""
    print("\n" + "🌀" * 35)
    print("HURRICANE NAMES ANALYSIS: BASELINE EXPERIMENT")
    print("🌀" * 35)
    print("\nResearch Question:")
    print("Do hurricane names predict perceived threat and evacuation behavior?")
    print("\nBased on: Jung et al. (2014)")
    print("Key Finding: Feminine names → 8.2% lower evacuation")
    print("Effect: R² = 0.11, p = 0.008, d = 0.38")
    print("\n")
    
    # Collect data
    dataset = collect_hurricane_data(n_samples=100)
    
    # Run all hypothesis tests
    results = {}
    
    results['h1'] = test_h1_gender_effect(dataset)
    results['h2'] = test_h2_syllable_effect(dataset)
    results['h3'] = test_h3_memorability_effect(dataset)
    results['h4'] = test_h4_interaction_effect(dataset)
    results['h5'] = test_h5_ensemble_superiority(dataset)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    validated_count = sum(1 for r in results.values() if r.get('validated', False))
    
    print(f"\nHypotheses Tested: 5")
    print(f"Validated: {validated_count}")
    print(f"Success Rate: {validated_count/5*100:.0f}%")
    
    print("\n" + "─" * 70)
    for key, result in results.items():
        status = "✅ VALIDATED" if result.get('validated', False) else "⚠️ NOT VALIDATED"
        print(f"{result['hypothesis']}: {status}")
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'baseline_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'baseline_results.json'}")
    
    # Save dataset
    with open(output_dir / 'hurricane_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Dataset saved to: {output_dir / 'hurricane_dataset.json'}")
    print("\n" + "🌀" * 35 + "\n")
    
    return results, dataset


if __name__ == '__main__':
    results, dataset = run_full_experiment()

