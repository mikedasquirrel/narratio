"""
Music/Spotify Complete Analysis - Full Framework Application

Apply complete framework to music (π=0.702, MID-HIGH):
1. Generate narratives from song/artist names
2. Apply transformers
3. Compute story quality
4. Measure |r|
5. Test genre-specific effects (mirroring movie genre breakthrough)
6. Optimize

Tests: Do song names + artist names predict popularity?
Genre hypothesis: Some genres narrative-driven, others production-driven
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
    PhoneticTransformer, EmotionalResonanceTransformer, CognitiveFluencyTransformer,
    CulturalContextTransformer, AuthenticityTransformer
)

print("="*80)
print("MUSIC/SPOTIFY COMPLETE ANALYSIS - FULL FRAMEWORK")
print("="*80)

# Load data
print("\n[1/8] Loading music data...")
music_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'spotify_songs.json'
with open(music_path) as f:
    data = json.load(f)
    songs = data['songs']

print(f"✓ {len(songs):,} songs")
print(f"✓ {len(data['genres'])} genres: {', '.join(data['genres'])}")
if data.get('data_type') == 'demo':
    print("  ℹ Using demo data. Replace with Spotify API for production.")

# Load π
pi_path = Path(__file__).parent / 'music_narrativity.json'
with open(pi_path) as f:
    pi_data = json.load(f)
    π = pi_data['π']

print(f"✓ Music π = {π:.3f} (MID-HIGH)")

# Generate narratives
print("\n[2/8] Generating music narratives...", end=" ", flush=True)
narratives = []
outcomes = []

for song in songs:
    # Create narrative from song + artist
    song_name = song['song_name']
    artist_name = song['artist_name']
    genre = song['genre']
    
    # Audio feature context
    af = song.get('audio_features', {})
    energy = af.get('energy', 0.5)
    valence = af.get('valence', 0.5)
    danceability = af.get('danceability', 0.5)
    
    # Build narrative
    narrative = f"{artist_name} - {song_name}. "
    narrative += f"A {genre} track "
    
    if energy > 0.7:
        narrative += "with high energy "
    elif energy < 0.3:
        narrative += "with low energy "
    
    if valence > 0.7:
        narrative += "and positive vibes. "
    elif valence < 0.3:
        narrative += "with melancholic tones. "
    else:
        narrative += "with balanced emotion. "
    
    if danceability > 0.7:
        narrative += "Very danceable."
    elif danceability < 0.3:
        narrative += "More contemplative."
    
    narratives.append(narrative)
    
    # Outcome: Popularity (binary: >50 = hit)
    popularity = song.get('popularity', 0)
    outcomes.append(int(popularity > 50))

outcomes = np.array(outcomes)

print(f"✓ {len(narratives):,} narratives")
print(f"  Hits (>50 popularity): {outcomes.sum():,}/{len(outcomes):,} ({100*outcomes.mean():.1f}%)")

# Apply transformers
print("\n[3/8] Applying transformers...", end=" ", flush=True)

transformers = [
    ('statistical', StatisticalTransformer(max_features=50)),
    ('nominative', NominativeAnalysisTransformer()),
    ('phonetic', PhoneticTransformer()),
    ('emotional', EmotionalResonanceTransformer()),
    ('cognitive', CognitiveFluencyTransformer()),
    ('linguistic', LinguisticPatternsTransformer()),
]

all_features = []
for name, transformer in transformers:
    try:
        transformer.fit(narratives)
        features = transformer.transform(narratives)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        all_features.append(features)
    except Exception as e:
        print(f"\n  Warning: {name} failed: {e}")
        continue

valid_features = []
for f in all_features:
    if not isinstance(f, np.ndarray):
        f = np.array(f)
    
    if f.ndim == 0:
        continue
    
    if f.shape[0] != len(narratives):
        continue
    
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    
    if f.ndim == 2:
        valid_features.append(f)

if not valid_features:
    raise ValueError("No valid features")

ж = np.hstack(valid_features)

print(f"✓ {ж.shape[1]} features")

# Compute story quality
print("\n[4/8] Computing story quality (ю)...", end=" ", flush=True)
scaler = StandardScaler()
ж_norm = scaler.fit_transform(ж)
ю = ж_norm.mean(axis=1)
ю = (ю - ю.min()) / (ю.max() - ю.min())
print("✓")

# Measure correlation
print("\n[5/8] Measuring |r|...", end=" ", flush=True)
r = np.corrcoef(ю, outcomes)[0, 1]
abs_r = abs(r)
print(f"✓ |r| = {abs_r:.4f}")

# Genre-specific analysis (mirroring movie genre breakthrough!)
print("\n[6/8] Analyzing genre-specific effects...")
print("  (Hypothesis: Some genres narrative-driven, others production-driven)")

genre_results = {}
genres = list(set(song['genre'] for song in songs))

for genre in sorted(genres):
    # Filter by genre
    genre_indices = [i for i, song in enumerate(songs) if song['genre'] == genre]
    
    if len(genre_indices) < 100:
        continue
    
    ж_genre = ж[genre_indices]
    outcomes_genre = outcomes[genre_indices]
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        ж_genre, outcomes_genre, test_size=0.3, random_state=42
    )
    
    scaler_g = StandardScaler()
    X_train_sc = scaler_g.fit_transform(X_train)
    X_test_sc = scaler_g.transform(X_test)
    
    selector = SelectKBest(mutual_info_regression, k=min(30, X_train_sc.shape[1]))
    selector.fit(X_train_sc, y_train)
    X_train_sel = selector.transform(X_train_sc)
    X_test_sel = selector.transform(X_test_sc)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train_sel, y_train)
    
    y_pred = model.predict(X_test_sel)
    r_genre = np.corrcoef(y_pred, y_test)[0, 1]
    r2_genre = r_genre ** 2
    
    genre_results[genre] = {
        'r': float(r_genre),
        'r2': float(r2_genre),
        'n_songs': len(genre_indices)
    }
    
    print(f"  {genre:12s}: r² = {r2_genre:.3f} (n={len(genre_indices):,})")

# Overall optimization
print("\n[7/8] Optimizing overall model...", end=" ", flush=True)
X_train, X_test, y_train, y_test = train_test_split(ж, outcomes, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

selector = SelectKBest(mutual_info_regression, k=min(50, X_train_sc.shape[1]))
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

print(f"✓ Train R²: {r2_train:.3f}, Test R²: {r2_test:.3f}")

# Calculate Д
κ = pi_data['forces']['κ']  # 0.50
Д = π * abs_r * κ
efficiency = Д / π

# Save
print("\n[8/8] Saving results...", end=" ", flush=True)

results = {
    'domain': 'Music/Spotify',
    'π': π,
    'songs': len(songs),
    'basic_r': float(abs_r),
    'Д': float(Д),
    'efficiency': float(efficiency),
    'passes_threshold': bool(efficiency > 0.5),
    'optimized': {
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'features': int(ж.shape[1]),
        'selected': int(X_train_sel.shape[1])
    },
    'genre_specific': genre_results
}

output_path = Path(__file__).parent / 'music_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("✓")

print("\n" + "="*80)
print("MUSIC/SPOTIFY ANALYSIS COMPLETE")
print("="*80)

print(f"\nMusic π = {π:.3f} (MID-HIGH)")
print(f"Basic |r| = {abs_r:.4f}")
print(f"Д = {Д:.3f}")
print(f"Efficiency (Д/π) = {efficiency:.3f} {'✓ PASS' if efficiency > 0.5 else '✗ FAIL'}")
print(f"Optimized R² = {r2_test*100:.1f}% (test)")

print(f"\nGenre-Specific Effects (mirroring movie genre breakthrough):")
sorted_genres = sorted(genre_results.items(), key=lambda x: x[1]['r2'], reverse=True)
for genre, stats in sorted_genres[:5]:
    print(f"  {genre:12s}: r² = {stats['r2']:.3f} (r = {stats['r']:.3f})")

print(f"\nSpectrum Position:")
print(f"  Movies (overall): π=0.65, r²≈0.04 (weak)")
print(f"  Music (overall):  π={π:.2f}, r²={r2_test:.3f}")
print(f"  Golf:             π=0.70, r²=0.977 (nominative enrichment)")
print(f"  Tennis:           π=0.75, r²=0.930 (rich nominative)")

print(f"\n✓ Results saved to: {output_path}")

