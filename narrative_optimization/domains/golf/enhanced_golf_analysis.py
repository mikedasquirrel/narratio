"""
Enhanced Golf Analysis - Test Nominative Enrichment Hypothesis

Compare:
- BASELINE: 40% R² with sparse nominatives (~5 proper nouns)
- ENHANCED: ??? R² with rich nominatives (~15-20 proper nouns)

Hypothesis: Nominative enrichment closes gap toward tennis-level (93% R²)

If Enhanced R² > 60%: Confirms nominative hypothesis
If Enhanced R² = 40-50%: Golf has fundamental differences
If Enhanced R² > 80%: Field dynamics were the missing piece
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    StatisticalTransformer, NominativeAnalysisTransformer, SelfPerceptionTransformer,
    NarrativePotentialTransformer, LinguisticPatternsTransformer, EnsembleNarrativeTransformer,
    RelationalValueTransformer, OpticsTransformer, FramingTransformer, PhoneticTransformer,
    TemporalEvolutionTransformer, InformationTheoryTransformer, SocialStatusTransformer,
    NamespaceEcologyTransformer, AnticipatoryCommunicationTransformer, QuantitativeTransformer,
    CrossmodalTransformer, AudioTransformer, CrossLingualTransformer, DiscoverabilityTransformer,
    CognitiveFluencyTransformer, EmotionalResonanceTransformer, AuthenticityTransformer,
    ConflictTensionTransformer, ExpertiseAuthorityTransformer, CulturalContextTransformer,
    SuspenseMysteryTransformer, VisualMultimodalTransformer, UniversalNominativeTransformer,
    HierarchicalNominativeTransformer, NominativeInteractionTransformer, PureNominativePredictorTransformer,
    MultiScaleTransformer, MultiPerspectiveTransformer, ScaleInteractionTransformer
)

print("="*80)
print("ENHANCED GOLF ANALYSIS - NOMINATIVE ENRICHMENT TEST")
print("="*80)
print("\nTesting if nominative richness closes the gap")
print("Baseline: 40% R² with sparse nominatives")
print("Target: 60-80% R² with rich nominatives (field dynamics + course lore)")

# Load ENHANCED narratives
enhanced_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'golf_enhanced_narratives.json'

print(f"\n[1/6] Loading ENHANCED narratives...")
with open(enhanced_path) as f:
    results = json.load(f)

narratives = [r['narrative'] for r in results]
outcomes = np.array([int(r['won_tournament']) for r in results])

print(f"✓ {len(narratives)} enhanced narratives")
print(f"  Winners: {outcomes.sum()} ({100*outcomes.sum()/len(outcomes):.1f}%)")

# Sample narrative
sample = narratives[0]
sample_words = len(sample.split())
import re
sample_proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sample))

print(f"\n  Sample narrative: {sample_words} words, ~{sample_proper_nouns} proper nouns")
print(f"  Preview: {sample[:150]}...")

# Load π
pi_path = Path(__file__).parent / 'golf_narrativity.json'
with open(pi_path) as f:
    π = json.load(f)['π']

print(f"\n✓ Golf π = {π:.3f}")

# Apply ALL 33 transformers
print(f"\n[2/6] Applying ALL 33 transformers to ENHANCED narratives...")
print(f"  (Expected: More nominative features extracted due to richer text)")

transformers = [
    ('statistical', StatisticalTransformer(max_features=100)),
    ('nominative', NominativeAnalysisTransformer()),
    ('self_perception', SelfPerceptionTransformer()),
    ('narrative_potential', NarrativePotentialTransformer()),
    ('linguistic', LinguisticPatternsTransformer()),
    ('ensemble', EnsembleNarrativeTransformer()),
    ('relational', RelationalValueTransformer()),
    ('optics', OpticsTransformer()),
    ('framing', FramingTransformer()),
    ('phonetic', PhoneticTransformer()),
    ('temporal', TemporalEvolutionTransformer()),
    ('information_theory', InformationTheoryTransformer()),
    ('social_status', SocialStatusTransformer()),
    ('namespace', NamespaceEcologyTransformer()),
    ('anticipatory', AnticipatoryCommunicationTransformer()),
    ('quantitative', QuantitativeTransformer()),
    ('crossmodal', CrossmodalTransformer()),
    ('audio', AudioTransformer()),
    ('crosslingual', CrossLingualTransformer()),
    ('discoverability', DiscoverabilityTransformer()),
    ('cognitive_fluency', CognitiveFluencyTransformer()),
    ('emotional', EmotionalResonanceTransformer()),
    ('authenticity', AuthenticityTransformer()),
    ('conflict', ConflictTensionTransformer()),
    ('expertise', ExpertiseAuthorityTransformer()),
    ('cultural', CulturalContextTransformer()),
    ('suspense', SuspenseMysteryTransformer()),
    ('visual', VisualMultimodalTransformer()),
    ('universal_nominative', UniversalNominativeTransformer()),
    ('hierarchical_nominative', HierarchicalNominativeTransformer()),
    ('nominative_interaction', NominativeInteractionTransformer()),
    ('pure_nominative', PureNominativePredictorTransformer()),
    ('multi_scale', MultiScaleTransformer()),
    ('multi_perspective', MultiPerspectiveTransformer()),
    ('scale_interaction', ScaleInteractionTransformer()),
]

all_features = []

for idx, (name, transformer) in enumerate(transformers, 1):
    try:
        print(f"  [{idx}/35] {name}...", end=" ", flush=True)
        transformer.fit(narratives)
        features = transformer.transform(narratives)
        
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        if features.ndim == 0:
            print("skip (scalar)")
            continue
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if features.shape[0] != len(narratives) or features.ndim != 2:
            print(f"skip (shape {features.shape})")
            continue
        
        all_features.append(features)
        print(f"✓ {features.shape[1]} features")
        
    except Exception as e:
        print(f"error: {str(e)[:30]}")
        continue

# Combine
print(f"\n  Combining features...")
ж = np.hstack(all_features)

print(f"\n✓ GENOME EXTRACTED (ENHANCED)")
print(f"  Features: {ж.shape[1]}")
print(f"  Player-tournaments: {ж.shape[0]}")

# Load BASELINE results for comparison
baseline_path = Path(__file__).parent / 'golf_proper_results.json'
with open(baseline_path) as f:
    baseline = json.load(f)

baseline_features = baseline['features_extracted']
baseline_r2 = baseline['optimized']['test_r2']

print(f"\n[3/6] Comparison to baseline:")
print(f"  BASELINE features: {baseline_features}")
print(f"  ENHANCED features: {ж.shape[1]}")
print(f"  Feature increase: +{ж.shape[1] - baseline_features} ({100*(ж.shape[1] - baseline_features)/baseline_features:.1f}%)")

# Compute story quality
print(f"\n[4/6] Computing story quality...")
scaler = StandardScaler()
ж_norm = scaler.fit_transform(ж)

# Weight nominative/mental features
weights = np.ones(ж.shape[1])
ю = (ж_norm * weights / weights.sum()).sum(axis=1)
ю = (ю - ю.min()) / (ю.max() - ю.min())

print(f"✓ Story quality computed")

# Measure correlation
print(f"\n[5/6] Measuring correlation...")
r = np.corrcoef(ю, outcomes)[0, 1]
abs_r = abs(r)

print(f"✓ Basic |r| = {abs_r:.4f}")
print(f"  BASELINE |r|: {baseline['basic_r']:.4f}")
print(f"  Change: {abs_r - baseline['basic_r']:+.4f}")

# Optimize
print(f"\n[6/6] Optimizing (Ridge + feature selection)...")

X_train, X_test, y_train, y_test = train_test_split(
    ж, outcomes, test_size=0.3, random_state=42
)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

k = min(300, X_train_sc.shape[1])
selector = SelectKBest(mutual_info_regression, k=k)
selector.fit(X_train_sc, y_train)
X_train_sel = selector.transform(X_train_sc)
X_test_sel = selector.transform(X_test_sc)

model = Ridge(alpha=10.0)
model.fit(X_train_sel, y_train)

y_pred_train = model.predict(X_train_sel)
y_pred_test = model.predict(X_test_sel)

r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
r_test = np.corrcoef(y_pred_test, y_test)[0, 1]
r2_train = r_train ** 2
r2_test = r_test ** 2

print(f"✓ Optimized")
print(f"  Train R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
print(f"  Test R²: {r2_test:.4f} ({r2_test*100:.1f}%)")

# Save results
results_dict = {
    'domain': 'Golf (ENHANCED)',
    'π': π,
    'player_tournaments': len(results),
    'features_extracted': int(ж.shape[1]),
    'basic_r': float(abs_r),
    'optimized': {
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'features_selected': k
    },
    'baseline_comparison': {
        'baseline_features': baseline_features,
        'baseline_r2': baseline_r2,
        'enhanced_features': int(ж.shape[1]),
        'enhanced_r2': float(r2_test),
        'improvement': float(r2_test - baseline_r2)
    }
}

output_path = Path(__file__).parent / 'golf_enhanced_results.json'
with open(output_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*80)
print("ENHANCED GOLF ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 BASELINE vs ENHANCED COMPARISON")
print(f"="*80)

print(f"\nBaseline (sparse nominatives):")
print(f"  • Narrative length: ~150-250 words")
print(f"  • Proper nouns: ~5 per narrative")
print(f"  • Features extracted: {baseline_features}")
print(f"  • Test R²: {baseline_r2*100:.1f}%")

print(f"\nEnhanced (rich nominatives):")
print(f"  • Narrative length: ~192 words")
print(f"  • Proper nouns: ~15-20 per narrative")
print(f"  • Features extracted: {ж.shape[1]}")
print(f"  • Test R²: {r2_test*100:.1f}%")

print(f"\n📈 IMPROVEMENT")
print(f"="*80)
improvement = r2_test - baseline_r2
improvement_pct = 100 * improvement

print(f"R² Change: {baseline_r2*100:.1f}% → {r2_test*100:.1f}% ({improvement_pct:+.1f} points)")

if improvement > 0.15:
    print(f"\n✅ HYPOTHESIS CONFIRMED: Nominative enrichment significantly improves prediction")
    print(f"   Field dynamics and course lore close the gap toward tennis-level (93%)")
elif improvement > 0.05:
    print(f"\n✓ PARTIAL SUCCESS: Nominative enrichment helps moderately")
    print(f"   Some gap closed, but golf still differs from tennis")
elif improvement > -0.02:
    print(f"\n→ MINIMAL CHANGE: Nominative enrichment has limited impact")
    print(f"   Golf's 40% ceiling may be due to fundamental sport differences")
else:
    print(f"\n⚠ UNEXPECTED: Nominative enrichment decreased performance")
    print(f"   Possible overfitting or noise from synthetic enrichment")

print(f"\n📊 Context:")
print(f"  Tennis (π=0.75): 93% R²")
print(f"  Golf ENHANCED (π=0.70): {r2_test*100:.1f}% R²")
print(f"  Golf BASELINE (π=0.70): {baseline_r2*100:.1f}% R²")
print(f"  NFL (π=0.57): 14% R²")

gap_closed = improvement / (0.93 - baseline_r2) if 0.93 > baseline_r2 else 0
print(f"\n  Gap to tennis closed: {gap_closed*100:.1f}%")

print(f"\n" + "="*80)


