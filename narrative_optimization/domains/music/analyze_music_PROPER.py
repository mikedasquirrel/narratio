"""
Music PROPER Analysis - Nominative/Narrative Focus

THIS IS THE REAL QUESTION:
- Do song NAMES predict success? (nominative)
- Do artist NAMES matter? (nominative determinism)
- Do album NAMES predict? (nominative)
- Do LYRICS (actual narrative) predict success?

NOT: "Can we predict popularity from audio features?"
BUT: "Does narrative quality in names/lyrics predict outcomes?"

π = 0.702 (mid-high) → Should show SOME nominative/narrative effects
Test against Golf/Tennis nominative enrichment pattern
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.src.transformers import (
    NominativeAnalysisTransformer,
    PhoneticTransformer,
    EmotionalResonanceTransformer,
    CognitiveFluencyTransformer,
    LinguisticPatternsTransformer,
    StatisticalTransformer,
    AuthenticityTransformer,
    NarrativePotentialTransformer,
    UniversalNominativeTransformer,
    HierarchicalNominativeTransformer,
    PureNominativePredictorTransformer
)

print("="*80)
print("MUSIC PROPER ANALYSIS - NOMINATIVE/NARRATIVE FOCUS")
print("="*80)
print("\nTEST: Do song names, artist names, album names, and lyrics predict success?")
print("NOT: Do audio features predict success? (that's acoustic analysis, not narrative)\n")

# Load data
print("[1/9] Loading music data...")
music_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'spotify_songs.json'
with open(music_path) as f:
    data = json.load(f)
    songs = data['songs']

print(f"✓ {len(songs):,} songs loaded")

# Load π
pi_path = Path(__file__).parent / 'music_narrativity.json'
with open(pi_path) as f:
    pi_data = json.load(f)
    π = pi_data['π']

print(f"✓ Music π = {π:.3f} (MID-HIGH)")

# Extract PURE NOMINATIVE features
print("\n[2/9] Extracting NOMINATIVE features (names only)...")

song_names = [s['song_name'] for s in songs]
artist_names = [s['artist_name'] for s in songs]
album_names = [s['album_name'] for s in songs]

print(f"  • {len(set(song_names)):,} unique song names")
print(f"  • {len(set(artist_names)):,} unique artist names")
print(f"  • {len(set(album_names)):,} unique album names")

# Build NOMINATIVE NARRATIVES (names only, no audio features!)
print("\n[3/9] Building nominative narratives...")
nominative_narratives = []

for song in songs:
    # Pure nominative narrative - just the names
    narrative = f"{song['artist_name']} - {song['song_name']} (from {song['album_name']})"
    nominative_narratives.append(narrative)

print(f"✓ {len(nominative_narratives):,} nominative narratives created")

# Outcomes
outcomes = np.array([int(s['popularity'] > 50) for s in songs])
print(f"  Hits (>50): {outcomes.sum():,}/{len(outcomes):,} ({100*outcomes.mean():.1f}%)")

# Apply NOMINATIVE transformers
print("\n[4/9] Applying NOMINATIVE transformers...")
print("  (Focus: Name quality, phonetics, cognitive fluency, emotional resonance)")

transformers = [
    ('pure_nominative', PureNominativePredictorTransformer()),
    ('universal_nominative', UniversalNominativeTransformer()),
    ('hierarchical_nominative', HierarchicalNominativeTransformer()),
    ('nominative_analysis', NominativeAnalysisTransformer()),
    ('phonetic', PhoneticTransformer()),
    ('emotional', EmotionalResonanceTransformer()),
    ('cognitive_fluency', CognitiveFluencyTransformer()),
    ('authenticity', AuthenticityTransformer()),
    ('linguistic', LinguisticPatternsTransformer()),
    ('statistical', StatisticalTransformer(max_features=30)),
]

all_features = []
transformer_names = []

for name, transformer in transformers:
    try:
        print(f"  • {name}...", end=" ", flush=True)
        transformer.fit(nominative_narratives)
        features = transformer.transform(nominative_narratives)
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        if features.shape[0] == len(nominative_narratives):
            all_features.append(features)
            transformer_names.append(name)
            print(f"✓ ({features.shape[1]} features)")
        else:
            print("✗ (shape mismatch)")
    except Exception as e:
        print(f"✗ ({e})")
        continue

if not all_features:
    raise ValueError("No valid features extracted!")

ж_nominative = np.hstack(all_features)
print(f"\n✓ Total nominative features: {ж_nominative.shape[1]}")

# SEPARATE: Song name features
print("\n[5/9] Extracting SONG NAME features specifically...")
song_name_transformers = [
    ('song_nominative', NominativeAnalysisTransformer()),
    ('song_phonetic', PhoneticTransformer()),
    ('song_emotional', EmotionalResonanceTransformer()),
    ('song_cognitive', CognitiveFluencyTransformer()),
]

song_features = []
for name, transformer in song_name_transformers:
    try:
        transformer.fit(song_names)
        features = transformer.transform(song_names)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        song_features.append(features)
    except:
        continue

ж_songs = np.hstack(song_features) if song_features else np.zeros((len(songs), 1))
print(f"✓ Song name features: {ж_songs.shape[1]}")

# SEPARATE: Artist name features
print("\n[6/9] Extracting ARTIST NAME features specifically...")
artist_name_transformers = [
    ('artist_nominative', NominativeAnalysisTransformer()),
    ('artist_phonetic', PhoneticTransformer()),
    ('artist_cognitive', CognitiveFluencyTransformer()),
]

artist_features = []
for name, transformer in artist_name_transformers:
    try:
        transformer.fit(artist_names)
        features = transformer.transform(artist_names)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        artist_features.append(features)
    except:
        continue

ж_artists = np.hstack(artist_features) if artist_features else np.zeros((len(songs), 1))
print(f"✓ Artist name features: {ж_artists.shape[1]}")

# Compute story quality from NAMES ONLY
print("\n[7/9] Computing nominative story quality (ю)...")
scaler = StandardScaler()
ж_norm = scaler.fit_transform(ж_nominative)
ю_nominative = ж_norm.mean(axis=1)
ю_nominative = (ю_nominative - ю_nominative.min()) / (ю_nominative.max() - ю_nominative.min())

# Measure correlation
r_nominative = np.corrcoef(ю_nominative, outcomes)[0, 1]
abs_r_nominative = abs(r_nominative)
print(f"✓ Nominative |r| = {abs_r_nominative:.4f}")

# Song name only correlation
ж_songs_norm = StandardScaler().fit_transform(ж_songs)
ю_songs = ж_songs_norm.mean(axis=1)
ю_songs = (ю_songs - ю_songs.min()) / (ю_songs.max() - ю_songs.min())
r_songs = np.corrcoef(ю_songs, outcomes)[0, 1]
print(f"✓ Song name only |r| = {abs(r_songs):.4f}")

# Artist name only correlation
ж_artists_norm = StandardScaler().fit_transform(ж_artists)
ю_artists = ж_artists_norm.mean(axis=1)
ю_artists = (ю_artists - ю_artists.min()) / (ю_artists.max() - ю_artists.min())
r_artists = np.corrcoef(ю_artists, outcomes)[0, 1]
print(f"✓ Artist name only |r| = {abs(r_artists):.4f}")

# Optimize NOMINATIVE model
print("\n[8/9] Optimizing NOMINATIVE-only model...")
X_train, X_test, y_train, y_test = train_test_split(
    ж_nominative, outcomes, test_size=0.3, random_state=42
)

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

print(f"✓ Nominative-only R² (train): {r2_train:.3f}")
print(f"✓ Nominative-only R² (test): {r2_test:.3f}")

# Calculate Д (nominative only)
κ = pi_data['forces']['κ']
Д_nominative = π * abs_r_nominative * κ
efficiency_nominative = Д_nominative / π

# Genre-specific NOMINATIVE analysis
print("\n[Genre Analysis] Testing nominative effects by genre...")
genres = list(set(s['genre'] for s in songs))
genre_nominative_results = {}

for genre in sorted(genres):
    genre_indices = [i for i, s in enumerate(songs) if s['genre'] == genre]
    
    if len(genre_indices) < 100:
        continue
    
    ж_genre = ж_nominative[genre_indices]
    outcomes_genre = outcomes[genre_indices]
    
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        ж_genre, outcomes_genre, test_size=0.3, random_state=42
    )
    
    scaler_g = StandardScaler()
    X_train_sc_g = scaler_g.fit_transform(X_train_g)
    X_test_sc_g = scaler_g.transform(X_test_g)
    
    selector_g = SelectKBest(mutual_info_regression, k=min(30, X_train_sc_g.shape[1]))
    selector_g.fit(X_train_sc_g, y_train_g)
    X_train_sel_g = selector_g.transform(X_train_sc_g)
    X_test_sel_g = selector_g.transform(X_test_sc_g)
    
    model_g = Ridge(alpha=10.0)
    model_g.fit(X_train_sel_g, y_train_g)
    
    y_pred_g = model_g.predict(X_test_sel_g)
    r_genre = np.corrcoef(y_pred_g, y_test_g)[0, 1]
    r2_genre = r_genre ** 2
    
    genre_nominative_results[genre] = {
        'r': float(r_genre),
        'r2': float(r2_genre),
        'n_songs': len(genre_indices)
    }
    
    print(f"  {genre:12s}: r² = {r2_genre:.3f} (nominative-only)")

# Save results
print("\n[9/9] Saving NOMINATIVE analysis results...")

results = {
    'domain': 'Music/Spotify (NOMINATIVE ANALYSIS)',
    'approach': 'Pure nominative/narrative - names only, NO audio features',
    'π': π,
    'songs': len(songs),
    'nominative_analysis': {
        'basic_r': float(abs_r_nominative),
        'song_name_r': float(abs(r_songs)),
        'artist_name_r': float(abs(r_artists)),
        'optimized_r2': float(r2_test),
        'Д_nominative': float(Д_nominative),
        'efficiency': float(efficiency_nominative),
        'passes_threshold': bool(efficiency_nominative > 0.5)
    },
    'component_analysis': {
        'full_nominative': {
            'features': int(ж_nominative.shape[1]),
            'r': float(abs_r_nominative),
            'r2_test': float(r2_test)
        },
        'song_names_only': {
            'features': int(ж_songs.shape[1]),
            'r': float(abs(r_songs))
        },
        'artist_names_only': {
            'features': int(ж_artists.shape[1]),
            'r': float(abs(r_artists))
        }
    },
    'genre_specific_nominative': genre_nominative_results,
    'interpretation': 'Tests if NAMES ALONE (song/artist/album) predict success, independent of audio quality'
}

output_path = Path(__file__).parent / 'music_nominative_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("✓")

print("\n" + "="*80)
print("MUSIC NOMINATIVE ANALYSIS COMPLETE")
print("="*80)

print(f"\nMusic π = {π:.3f} (MID-HIGH)")
print(f"\nNOMINATIVE EFFECTS (names only):")
print(f"  • Full nominative |r| = {abs_r_nominative:.4f}")
print(f"  • Song names |r| = {abs(r_songs):.4f}")
print(f"  • Artist names |r| = {abs(r_artists):.4f}")
print(f"  • Optimized R² = {r2_test*100:.1f}%")
print(f"  • Д (nominative) = {Д_nominative:.3f}")
print(f"  • Efficiency = {efficiency_nominative:.3f} {'✓ PASS' if efficiency_nominative > 0.5 else '✗ FAIL'}")

print(f"\nComparison:")
print(f"  Golf (nominative enrichment): π=0.70, R²=97.7% (player names + context)")
print(f"  Music (pure nominative):      π={π:.2f}, R²={r2_test*100:.1f}% (song/artist names)")
print(f"  Tennis (rich nominative):     π=0.75, R²=93.0% (player names + rankings)")

print(f"\n✓ Results saved to: {output_path}")
print("\nNOTE: Still using demo data. Real Spotify + lyrics needed for production.")









