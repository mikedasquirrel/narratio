"""
Tennis Formula Optimization - Surface-Specific

Find optimal feature weights for each surface (clay/grass/hard).
Expected: Higher R² than NFL due to cleaner individual dynamics.

Strategy:
1. Surface-specific optimization (clay/grass/hard)
2. Tournament-level optimization (Grand Slams vs regular)
3. Player-tier optimization (Top 10, Top 50, etc.)
4. Test betting ROI (target 3-8%)
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from collections import defaultdict

print("="*80)
print("TENNIS FORMULA OPTIMIZATION - SURFACE-SPECIFIC")
print("="*80)
print("\nLoading data...")

# Load full dataset
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
with open(dataset_path) as f:
    all_matches = json.load(f)

print(f"✓ Loaded {len(all_matches)} total matches")

# Load genome
genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ж_sample = genome_data['genome']
ю_sample = genome_data['story_quality']
outcomes_sample = genome_data['outcomes']
feature_names = genome_data['feature_names'].tolist()

print(f"✓ Loaded genome: {ж_sample.shape[1]} features, {ж_sample.shape[0]} matches (sample)")

# ============================================================================
# SURFACE-SPECIFIC OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("SURFACE-SPECIFIC OPTIMIZATION")
print("="*80)

# Get sample matches metadata
sample_matches = all_matches[:5000]  # Same sample as analysis

surface_results = {}

for surface in ['clay', 'grass', 'hard']:
    print(f"\n[Optimizing {surface.upper()}]")
    
    # Filter matches by surface
    surface_indices = [i for i, m in enumerate(sample_matches) if m['surface'] == surface]
    
    if len(surface_indices) < 100:
        print(f"  ⚠ Only {len(surface_indices)} matches, skipping")
        continue
    
    print(f"  Matches: {len(surface_indices)}")
    
    # Extract features for this surface
    ж_surf = ж_sample[surface_indices]
    outcomes_surf = outcomes_sample[surface_indices]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        ж_surf, outcomes_surf, test_size=0.3, random_state=42
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Feature selection
    print(f"  Selecting features...", end=" ", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different k values
    best_k = 200
    selector = SelectKBest(mutual_info_regression, k=best_k)
    selector.fit(X_train_scaled, y_train)
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = scaler.transform(X_test)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"✓ {best_k} features")
    
    # Ridge regression
    print(f"  Training model...", end=" ", flush=True)
    model = Ridge(alpha=10.0)
    model.fit(X_train_selected, y_train)
    print(f"✓ trained")
    
    # Test
    print(f"  Testing...", end=" ", flush=True)
    y_pred_train = model.predict(X_train_selected)
    y_pred_test = model.predict(X_test_selected)
    
    r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
    r_test = np.corrcoef(y_pred_test, y_test)[0, 1]
    
    abs_r_train = abs(r_train)
    abs_r_test = abs(r_test)
    
    r2_train = r_train ** 2
    r2_test = r_test ** 2
    
    print(f"✓ done")
    
    print(f"\n  Results:")
    print(f"    Train |r|: {abs_r_train:.4f}, R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
    print(f"    Test |r|: {abs_r_test:.4f}, R²: {r2_test:.4f} ({r2_test*100:.1f}%)")
    
    surface_results[surface] = {
        'n_matches': len(surface_indices),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features_selected': best_k,
        'train_abs_r': float(abs_r_train),
        'train_r2': float(r2_train),
        'test_abs_r': float(abs_r_test),
        'test_r2': float(r2_test),
        'pattern': 'inverse' if r_test < 0 else 'positive'
    }

# ============================================================================
# OVERALL OPTIMIZATION (ALL SURFACES)
# ============================================================================

print("\n" + "="*80)
print("OVERALL OPTIMIZATION (All Surfaces Combined)")
print("="*80)

print(f"\n  Dataset: {ж_sample.shape}")
print(f"  Splitting...", end=" ", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    ж_sample, outcomes_sample, test_size=0.3, random_state=42
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

print(f"  Scaling...", end=" ", flush=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓")

print(f"  Selecting features...", end=" ", flush=True)
selector = SelectKBest(mutual_info_regression, k=300)
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
print(f"✓ 300 features")

print(f"  Training Ridge model...", end=" ", flush=True)
model = Ridge(alpha=10.0)
model.fit(X_train_selected, y_train)
print(f"✓")

print(f"  Testing...", end=" ", flush=True)
y_pred_train = model.predict(X_train_selected)
y_pred_test = model.predict(X_test_selected)

r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
r_test = np.corrcoef(y_pred_test, y_test)[0, 1]

abs_r_train = abs(r_train)
abs_r_test = abs(r_test)

r2_train = r_train ** 2
r2_test = r_test ** 2

print(f"✓")

print(f"\nOverall Results:")
print(f"  Train |r|: {abs_r_train:.4f}, R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
print(f"  Test |r|: {abs_r_test:.4f}, R²: {r2_test:.4f} ({r2_test*100:.1f}%)")

# ============================================================================
# GOLDEN RATIO ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("GOLDEN RATIO DISCOVERY")
print("="*80)

selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices if i < len(feature_names)]

# Categorize
categories = defaultdict(int)
nominative_kw = ['nominative', 'phonetic', 'name', 'hierarchical', 'pure_nominative', 'universal_nominative']
mental_kw = ['emotional', 'conflict', 'tension', 'suspense', 'authenticity', 'self_perception']
surface_kw = ['temporal', 'expertise', 'cultural']

for fname in selected_features:
    fname_lower = fname.lower()
    if any(kw in fname_lower for kw in nominative_kw):
        categories['nominative'] += 1
    elif any(kw in fname_lower for kw in mental_kw):
        categories['mental_game'] += 1
    elif any(kw in fname_lower for kw in surface_kw):
        categories['context'] += 1
    else:
        categories['other'] += 1

print(f"\nFeature Distribution ({len(selected_features)} features):")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(selected_features) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING OPTIMIZATION RESULTS")
print("="*80)

results = {
    'optimization_method': 'Ridge regression with feature selection',
    'overall': {
        'features_selected': 300,
        'train_abs_r': float(abs_r_train),
        'train_r2': float(r2_train),
        'test_abs_r': float(abs_r_test),
        'test_r2': float(r2_test)
    },
    'by_surface': surface_results,
    'golden_ratio': dict(categories),
    'top_features': selected_features[:50]
}

output_path = Path(__file__).parent / 'tennis_optimized_formula.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE")
print("="*80)
print(f"\nOverall Performance:")
print(f"  Test R²: {r2_test*100:.1f}%")
print(f"  Train R²: {r2_train*100:.1f}%")

print(f"\nSurface-Specific Results:")
for surf, res in surface_results.items():
    print(f"  {surf.upper()}: Test R² = {res['test_r2']*100:.1f}%, Train R² = {res['train_r2']*100:.1f}%")

print(f"\nGolden Ratio:")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]:
    pct = count / len(selected_features) * 100
    print(f"  {cat}: {pct:.1f}%")

