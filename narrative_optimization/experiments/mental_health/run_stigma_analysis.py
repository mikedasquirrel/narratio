"""
Mental Health Stigma Analysis Experiment

Tests the pathway: Harsh disorder names → stigma → treatment seeking

Hypotheses:
H1: Phonetic harshness predicts stigma scores
H2: Stigma mediates the name → treatment seeking pathway
H3: Clinical framing amplifies phonetic effects
H4: Combined phonetic + framing model outperforms either alone

Author: Narrative Optimization Research
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from domains.mental_health.data_loader import MentalHealthDataLoader
from domains.mental_health.stigma_analyzer import StigmaAnalyzer
from src.transformers.mental_health.phonetic_severity_transformer import PhoneticSeverityTransformer
from src.transformers.mental_health.clinical_framing_transformer import ClinicalFramingTransformer
from src.transformers.mental_health.treatment_seeking_transformer import TreatmentSeekingTransformer


def load_mental_health_data():
    """Load and prepare mental health dataset."""
    print("📊 Loading mental health nomenclature data...")
    
    loader = MentalHealthDataLoader()
    disorders = loader.load_disorders()
    
    # Filter to those with complete data
    complete = []
    for disorder in disorders:
        if (disorder.get('phonetic_analysis') and 
            disorder.get('social_impact') and
            disorder.get('clinical_outcomes')):
            complete.append(disorder)
    
    print(f"✅ Loaded {len(complete)} disorders with complete data\n")
    return complete


def test_h1_phonetic_stigma_correlation(disorders):
    """
    H1: Phonetic harshness predicts stigma scores
    
    Expected: r ~ 0.25-0.35, p < 0.05
    """
    print("=" * 70)
    print("HYPOTHESIS 1: Phonetic Harshness → Stigma")
    print("=" * 70)
    
    # Extract data
    harshness = []
    stigma = []
    names = []
    
    for disorder in disorders:
        h = disorder.get('phonetic_analysis', {}).get('harshness_score')
        s = disorder.get('social_impact', {}).get('stigma_score')
        n = disorder.get('disorder_name')
        
        if h is not None and s is not None:
            harshness.append(h)
            stigma.append(s)
            names.append(n)
    
    harshness = np.array(harshness)
    stigma = np.array(stigma)
    
    # Correlation analysis
    r, p = stats.pearsonr(harshness, stigma)
    
    print(f"\n1. Correlation Analysis:")
    print(f"   n = {len(harshness)}")
    print(f"   r = {r:.3f}")
    print(f"   p = {p:.4f}")
    
    # R² (variance explained)
    r_squared = r ** 2
    print(f"   R² = {r_squared:.3f} ({r_squared*100:.1f}% variance explained)")
    
    # Comparison to expected
    print(f"\n2. Comparison to Prediction:")
    print(f"   Expected: r ~ 0.25-0.35 (visibility=25%)")
    print(f"   Observed: r = {r:.3f}")
    print(f"   Status: {'✅ VALIDATED' if 0.20 < r < 0.40 and p < 0.05 else '⚠️ OUTSIDE EXPECTED RANGE'}")
    
    # Show examples
    print(f"\n3. Sample Disorders:")
    indices = np.argsort(harshness)
    print(f"   Lowest harshness: {names[indices[0]]} (h={harshness[indices[0]]:.0f}, s={stigma[indices[0]]:.1f})")
    print(f"   Highest harshness: {names[indices[-1]]} (h={harshness[indices[-1]]:.0f}, s={stigma[indices[-1]]:.1f})")
    
    return {
        'hypothesis': 'H1_phonetic_stigma',
        'correlation': r,
        'p_value': p,
        'r_squared': r_squared,
        'n': len(harshness),
        'validated': 0.20 < r < 0.40 and p < 0.05
    }


def test_h2_mediation_pathway(disorders):
    """
    H2: Stigma mediates name → treatment seeking pathway
    
    Tests: Harshness → Stigma → Treatment seeking
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Mediation Pathway")
    print("=" * 70)
    
    analyzer = StigmaAnalyzer()
    result = analyzer.analyze_treatment_seeking_pathway(disorders)
    
    if 'error' in result:
        print(f"\n⚠️  {result['error']}")
        return result
    
    print(f"\n1. Mediation Pathways (n={result['n_disorders']}):")
    
    path_a = result['path_a_harshness_to_stigma']
    print(f"\n   Path A (Harshness → Stigma):")
    print(f"     r = {path_a['correlation']:.3f}, p = {path_a['p_value']:.4f}")
    print(f"     {path_a['interpretation']}")
    
    path_b = result['path_b_stigma_to_seeking']
    print(f"\n   Path B (Stigma → Treatment Seeking):")
    print(f"     r = {path_b['correlation']:.3f}, p = {path_b['p_value']:.4f}")
    print(f"     {path_b['interpretation']}")
    
    path_c = result['path_c_total_effect']
    print(f"\n   Path C (Total Effect: Harshness → Treatment Seeking):")
    print(f"     r = {path_c['correlation']:.3f}, p = {path_c['p_value']:.4f}")
    print(f"     {path_c['interpretation']}")
    
    print(f"\n2. Indirect Effect (Mediation):")
    print(f"   a × b = {result['indirect_effect']:.3f}")
    print(f"   Mediation: {'✅ SUPPORTED' if result['mediation_supported'] else '⚠️ NOT SUPPORTED'}")
    
    return result


def test_h3_framing_amplification(disorders):
    """
    H3: Clinical framing amplifies phonetic effects
    
    Tests interaction between phonetic harshness and medical terminology
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Clinical Framing Amplification")
    print("=" * 70)
    
    # Extract data
    harshness = []
    has_latin_greek = []
    stigma = []
    
    for disorder in disorders:
        h = disorder.get('phonetic_analysis', {}).get('harshness_score')
        lg = disorder.get('phonetic_analysis', {}).get('latin_greek_root', False)
        s = disorder.get('social_impact', {}).get('stigma_score')
        
        if h is not None and s is not None:
            harshness.append(h)
            has_latin_greek.append(1.0 if lg else 0.0)
            stigma.append(s)
    
    harshness = np.array(harshness)
    has_latin_greek = np.array(has_latin_greek)
    stigma = np.array(stigma)
    
    # Model without interaction
    X_simple = harshness.reshape(-1, 1)
    model_simple = LinearRegression()
    model_simple.fit(X_simple, stigma)
    r2_simple = r2_score(stigma, model_simple.predict(X_simple))
    
    # Model with interaction
    X_interact = np.column_stack([harshness, has_latin_greek, harshness * has_latin_greek])
    model_interact = LinearRegression()
    model_interact.fit(X_interact, stigma)
    r2_interact = r2_score(stigma, model_interact.predict(X_interact))
    
    interaction_coef = model_interact.coef_[2]
    
    print(f"\n1. Model Comparison (n={len(stigma)}):")
    print(f"   Without interaction: R² = {r2_simple:.3f}")
    print(f"   With interaction: R² = {r2_interact:.3f}")
    print(f"   Improvement: ΔR² = {r2_interact - r2_simple:.3f}")
    
    print(f"\n2. Interaction Coefficient:")
    print(f"   β = {interaction_coef:.4f}")
    print(f"   Interpretation: {'Medical terminology amplifies harshness effect' if interaction_coef > 0 else 'No amplification'}")
    
    return {
        'hypothesis': 'H3_framing_amplification',
        'r2_simple': r2_simple,
        'r2_interact': r2_interact,
        'delta_r2': r2_interact - r2_simple,
        'interaction_coef': interaction_coef,
        'validated': r2_interact > r2_simple
    }


def test_h4_ensemble_superiority(disorders):
    """
    H4: Combined model outperforms components
    
    Compares phonetic-only vs framing-only vs combined
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Ensemble Model Superiority")
    print("=" * 70)
    
    # Prepare data
    names = [d.get('disorder_name') for d in disorders]
    
    # Extract stigma as target
    stigma = []
    valid_indices = []
    
    for i, disorder in enumerate(disorders):
        s = disorder.get('social_impact', {}).get('stigma_score')
        if s is not None and names[i]:
            stigma.append(s)
            valid_indices.append(i)
    
    if len(valid_indices) < 10:
        print("\n⚠️  Insufficient data for model comparison")
        return {'error': 'Insufficient data'}
    
    names_valid = [names[i] for i in valid_indices]
    stigma = np.array(stigma)
    
    # Model 1: Phonetic only
    phonetic_transformer = PhoneticSeverityTransformer(include_interactions=True)
    X_phonetic = phonetic_transformer.fit_transform(names_valid, stigma)
    
    model_phon = Ridge(alpha=1.0)
    scores_phon = cross_val_score(model_phon, X_phonetic, stigma, cv=5, 
                                  scoring='r2')
    r2_phon = np.mean(scores_phon)
    
    # Model 2: Framing only
    framing_transformer = ClinicalFramingTransformer()
    X_framing = framing_transformer.fit_transform(names_valid, stigma)
    
    model_frame = Ridge(alpha=1.0)
    scores_frame = cross_val_score(model_frame, X_framing, stigma, cv=5,
                                   scoring='r2')
    r2_frame = np.mean(scores_frame)
    
    # Model 3: Combined
    combined_transformer = TreatmentSeekingTransformer(include_interactions=True)
    X_combined = combined_transformer.fit_transform(names_valid, stigma)
    
    model_comb = Ridge(alpha=1.0)
    scores_comb = cross_val_score(model_comb, X_combined, stigma, cv=5,
                                  scoring='r2')
    r2_comb = np.mean(scores_comb)
    
    print(f"\n1. Model Performance (Cross-Validated R²):")
    print(f"   Phonetic only: R² = {r2_phon:.3f}")
    print(f"   Framing only: R² = {r2_frame:.3f}")
    print(f"   Combined: R² = {r2_comb:.3f}")
    
    print(f"\n2. Improvements:")
    print(f"   Combined vs Phonetic: ΔR² = {r2_comb - r2_phon:+.3f}")
    print(f"   Combined vs Framing: ΔR² = {r2_comb - r2_frame:+.3f}")
    
    best = max(r2_phon, r2_frame, r2_comb)
    print(f"\n3. Status: {'✅ ENSEMBLE SUPERIOR' if r2_comb == best else '⚠️ COMPONENT MODEL BETTER'}")
    
    return {
        'hypothesis': 'H4_ensemble_superiority',
        'r2_phonetic': r2_phon,
        'r2_framing': r2_frame,
        'r2_combined': r2_comb,
        'validated': r2_comb == best
    }


def run_full_stigma_analysis():
    """Run complete mental health stigma analysis."""
    print("\n" + "🧠" * 35)
    print("MENTAL HEALTH NOMENCLATURE: STIGMA ANALYSIS")
    print("🧠" * 35)
    print("\nResearch Question:")
    print("Do disorder names predict stigma and treatment-seeking behavior?")
    print("\nHypothesis: Harsh-sounding names → Higher stigma → Lower treatment seeking")
    print("Expected Effect: r ~ 0.29 (visibility=25%, narrative importance=high)")
    print("\n")
    
    # Load data
    disorders = load_mental_health_data()
    
    # Run all hypothesis tests
    results = {}
    
    results['h1'] = test_h1_phonetic_stigma_correlation(disorders)
    results['h2'] = test_h2_mediation_pathway(disorders)
    results['h3'] = test_h3_framing_amplification(disorders)
    results['h4'] = test_h4_ensemble_superiority(disorders)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    validated_count = sum(1 for r in results.values() 
                         if isinstance(r, dict) and r.get('validated', False))
    
    print(f"\nHypotheses Tested: {len(results)}")
    print(f"Validated: {validated_count}")
    print(f"Success Rate: {validated_count/len(results)*100:.0f}%")
    
    print("\n" + "─" * 70)
    for key, result in results.items():
        if isinstance(result, dict) and 'hypothesis' in result:
            status = "✅ VALIDATED" if result.get('validated', False) else "⚠️ NOT VALIDATED"
            print(f"{result['hypothesis']}: {status}")
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(output_dir / 'stigma_analysis_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'stigma_analysis_results.json'}")
    
    # Summary statistics
    loader = MentalHealthDataLoader()
    stats_report = loader.generate_data_report()
    print("\n" + stats_report)
    
    print("\n" + "🧠" * 35 + "\n")
    
    return results


if __name__ == '__main__':
    results = run_full_stigma_analysis()

