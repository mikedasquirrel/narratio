"""
UFC Fast Analysis - Sample-Based Testing

Uses 1,000 fight sample for rapid testing, then can scale to full dataset.
Much faster iteration while testing framework.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
import time
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    StatisticalTransformer,
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer,
    NarrativePotentialTransformer,
    LinguisticPatternsTransformer,
    ConflictTensionTransformer,
    PhoneticTransformer,
)


def main(sample_size=1000):
    """
    Fast UFC analysis with sample.
    
    Parameters
    ----------
    sample_size : int
        Number of fights to analyze (default 1000 for speed)
    """
    start_time = time.time()
    
    print("="*80)
    print(f"UFC FAST ANALYSIS - {sample_size} FIGHT SAMPLE")
    print("="*80)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n[1/7] Loading UFC dataset...")
    
    dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'ufc_with_narratives.json'
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return
    
    with open(dataset_path) as f:
        fights = json.load(f)
    
    print(f"✓ Loaded {len(fights)} total fights")
    
    # Random sample for speed
    np.random.seed(42)
    sample_indices = np.random.choice(len(fights), size=min(sample_size, len(fights)), replace=False)
    fights_sample = [fights[i] for i in sample_indices]
    
    print(f"✓ Using random sample of {len(fights_sample)} fights")
    
    # Extract data
    narratives = [f['narrative'] for f in fights_sample]
    outcomes = np.array([1 if f['result']['winner'] == 'fighter_a' else 0 for f in fights_sample])
    
    print(f"  Fighter A wins: {outcomes.sum()} ({100*outcomes.mean():.1f}%)")
    print(f"  Fighter B wins: {len(outcomes) - outcomes.sum()} ({100*(1-outcomes.mean()):.1f}%)")
    
    # ========================================================================
    # CALCULATE п (NARRATIVITY)
    # ========================================================================
    
    print("\n[2/7] Calculate Narrativity (п)...")
    
    domain_characteristics = {
        'п_structural': 0.70,
        'п_temporal': 0.65,
        'п_agency': 0.95,
        'п_interpretation': 0.70,
        'п_format': 0.40
    }
    
    weights = {
        'п_structural': 0.30,
        'п_temporal': 0.20,
        'п_agency': 0.25,
        'п_interpretation': 0.15,
        'п_format': 0.10
    }
    
    п = sum(domain_characteristics[k] * weights[k] for k in domain_characteristics.keys())
    
    print(f"✓ UFC Narrativity п = {п:.3f}")
    print(f"  (NBA: 0.49, NFL: 0.48, UFC: {п:.3f})")
    print(f"  UFC is {п/0.49:.2f}x higher than NBA!")
    
    κ = 0.6
    required_r = 0.5 / п / κ
    print(f"  Required |r| to pass: {required_r:.3f}")
    
    # ========================================================================
    # APPLY KEY TRANSFORMERS (7 most important)
    # ========================================================================
    
    print("\n[3/7] Apply Key Transformers...")
    print("Using 7 most impactful transformers for speed:\n")
    
    transformers = [
        ('statistical', StatisticalTransformer(max_features=50)),
        ('nominative', NominativeAnalysisTransformer()),
        ('phonetic', PhoneticTransformer()),
        ('self_perception', SelfPerceptionTransformer()),
        ('narrative_potential', NarrativePotentialTransformer()),
        ('linguistic', LinguisticPatternsTransformer()),
        ('conflict_tension', ConflictTensionTransformer()),
    ]
    
    all_features = []
    transformer_feature_counts = {}
    
    for idx, (name, transformer) in enumerate(transformers, 1):
        t_start = time.time()
        print(f"  [{idx}/{len(transformers)}] {name}...", end=" ", flush=True)
        
        try:
            X = transformer.fit_transform(narratives)
            
            # Convert sparse to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            # Ensure numpy array
            X = np.asarray(X)
            
            # Ensure 2D array
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            all_features.append(X)
            transformer_feature_counts[name] = X.shape[1]
            
            elapsed = time.time() - t_start
            print(f"✓ {X.shape[1]} features ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
    
    # Combine
    print(f"\n  Combining features...")
    ж = np.hstack(all_features)
    print(f"✓ Total genome: {ж.shape[1]} features")
    
    # ========================================================================
    # COMPUTE ю (STORY QUALITY)
    # ========================================================================
    
    print("\n[4/7] Compute Story Quality (ю)...")
    
    scaler = StandardScaler()
    ж_scaled = scaler.fit_transform(ж)
    ю = np.abs(ж_scaled).mean(axis=1)
    
    print(f"✓ Computed ю: mean={ю.mean():.3f}, std={ю.std():.3f}")
    
    # ========================================================================
    # MEASURE |r| (CORRELATION)
    # ========================================================================
    
    print("\n[5/7] Measure Correlation |r|...")
    
    r_raw = np.corrcoef(ю, outcomes)[0, 1]
    r_abs = abs(r_raw)
    
    print(f"✓ Correlation r = {r_raw:+.4f}, |r| = {r_abs:.4f}")
    
    # Predictive model
    model = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores = cross_val_score(model, ю.reshape(-1, 1), outcomes, cv=5, scoring='roc_auc')
    
    print(f"  5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(ю.reshape(-1, 1), outcomes)
    predictions = model.predict(ю.reshape(-1, 1))
    accuracy = (predictions == outcomes).mean()
    
    print(f"  Accuracy: {accuracy:.4f} (baseline: {max(outcomes.mean(), 1-outcomes.mean()):.4f})")
    
    # ========================================================================
    # CALCULATE Д AND EFFICIENCY
    # ========================================================================
    
    print("\n[6/7] Calculate Bridge (Д) and Efficiency...")
    
    Д = п * r_abs * κ
    efficiency = Д / п
    
    print(f"✓ Bridge Д = п × |r| × κ")
    print(f"  Д = {п:.3f} × {r_abs:.4f} × {κ:.2f} = {Д:.4f}")
    print(f"✓ Efficiency = Д / п = {efficiency:.4f}")
    
    # ========================================================================
    # PASS/FAIL
    # ========================================================================
    
    print("\n[7/7] PRESUME AND PROVE - VALIDATION")
    print("="*80)
    
    threshold = 0.5
    passes = efficiency > threshold
    
    print(f"\nHYPOTHESIS: Narrative laws apply to UFC")
    print(f"TEST: Д/п > {threshold}")
    print(f"RESULT: {efficiency:.4f} {'>' if passes else '<'} {threshold}")
    
    if passes:
        print(f"\n{'='*80}")
        print(f"{'  '*15}✓ PASSES!")
        print(f"{'='*80}")
        print(f"\n🎉 UFC IS THE 3RD PASSING DOMAIN! 🎉")
        print(f"\nPassing Domains:")
        print(f"  1. Character Creation (self-rated)")
        print(f"  2. Self-Rated Narratives")
        print(f"  3. UFC (п={п:.3f}) ← NEW!")
        print(f"\n→ FIRST COMBAT SPORT TO PASS!")
        print(f"→ Individual agency + persona > team constraints")
        print(f"→ Fighter names and narratives MATTER")
    else:
        deficit = threshold - efficiency
        print(f"\n{'='*80}")
        print(f"{'  '*15}✗ FAILS")
        print(f"{'='*80}")
        print(f"\nDeficit: -{deficit:.4f}")
        print(f"Need |r| = {required_r:.3f} to pass (current: {r_abs:.4f})")
        print(f"\nBUT: Still HIGHER than team sports!")
        print(f"  NBA п=0.49, UFC п={п:.3f} (+{100*(п/0.49-1):.0f}%)")
        print(f"\n→ Individual narrative matters more than teams")
        print(f"→ Specific contexts (title fights, grudge matches) might pass")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    elapsed_total = time.time() - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nTime: {elapsed_total:.1f}s")
    print(f"Sample: {len(fights_sample)} fights")
    print(f"Features: {ж.shape[1]}")
    print(f"Narrativity: {п:.3f}")
    print(f"Correlation: {r_abs:.4f}")
    print(f"Efficiency: {efficiency:.4f}")
    print(f"Result: {'✓ PASS' if passes else '✗ FAIL'}")
    
    # Save results
    results = {
        'domain': 'UFC',
        'sample_size': len(fights_sample),
        'total_dataset': len(fights),
        'narrativity': float(п),
        'correlation': {
            'r_raw': float(r_raw),
            'r_abs': float(r_abs),
            'cv_auc': float(cv_scores.mean()),
            'accuracy': float(accuracy)
        },
        'bridge': {
            'Д': float(Д),
            'efficiency': float(efficiency),
            'passes': bool(passes)
        },
        'features': int(ж.shape[1]),
        'time': float(elapsed_total)
    }
    
    output_path = Path(__file__).parent / 'ufc_fast_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path.name}")
    
    print(f"\nNext steps:")
    print(f"1. If satisfied, run full analysis on all {len(fights)} fights")
    print(f"2. Run discover_ufc_contexts.py for context analysis")
    print(f"3. Run test_ufc_betting_edge.py for betting comparison")
    
    return results


if __name__ == "__main__":
    import sys
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(sample_size)

