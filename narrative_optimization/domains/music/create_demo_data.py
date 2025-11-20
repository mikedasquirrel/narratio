"""
Create Demo Music Dataset

Generates realistic synthetic music data for framework testing.
Once Spotify API credentials are available, this can be replaced with real data.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Realistic song name patterns by genre
SONG_NAME_PATTERNS = {
    'pop': ['Love', 'Heart', 'Dream', 'Tonight', 'Forever', 'Dance', 'Feel', 'Shine', 'Beautiful', 'Perfect'],
    'rock': ['Thunder', 'Fire', 'Storm', 'Break', 'Wild', 'Free', 'Highway', 'Night', 'Shadow', 'Devil'],
    'hip-hop': ['Real', 'Money', 'Streets', 'Life', 'King', 'Boss', 'Ride', 'Game', 'City', 'Dreams'],
    'country': ['Home', 'Road', 'Truck', 'Beer', 'Girl', 'Rain', 'Farm', 'Whiskey', 'River', 'Sunset'],
    'electronic': ['Neon', 'Digital', 'Electric', 'Pulse', 'Wave', 'Sync', 'Flash', 'Echo', 'Spectrum', 'Frequency'],
    'jazz': ['Blue', 'Smooth', 'Cool', 'Velvet', 'Midnight', 'Satin', 'Silk', 'Mood', 'Groove', 'Soul'],
    'classical': ['Symphony', 'Sonata', 'Concerto', 'Prelude', 'Nocturne', 'Rhapsody', 'Waltz', 'Suite', 'March', 'Etude'],
    'indie': ['Broken', 'Lost', 'Fade', 'Ghost', 'Rust', 'Dust', 'Bones', 'Wolves', 'Cold', 'Empty']
}

ARTIST_NAME_PATTERNS = {
    'pop': ['The', 'Lil', 'Big', 'DJ', 'Lady', 'Young'],
    'rock': ['Black', 'Red', 'Iron', 'Stone', 'Wild', 'Dead'],
    'hip-hop': ['Lil', 'Young', 'Big', '$', '2', '21'],
    'country': ['Luke', 'Morgan', 'Blake', 'Keith', 'Jason', 'Kenny'],
    'electronic': ['DJ', 'The', 'Cosmic', 'Laser', 'Digital', 'Neon'],
    'jazz': ['Miles', 'John', 'Charlie', 'Ella', 'Billie', 'Duke'],
    'classical': ['Johann', 'Wolfgang', 'Ludwig', 'Franz', 'Frederic', 'Sergei'],
    'indie': ['The', 'Fleet', 'Beach', 'Arcade', 'Bon', 'Of']
}

def generate_song(genre, idx):
    """Generate a realistic synthetic song"""
    
    # Generate song name
    words = random.sample(SONG_NAME_PATTERNS[genre], k=random.randint(1, 3))
    song_name = ' '.join(words)
    if random.random() < 0.3:
        song_name += f" ({random.choice(['Remix', 'Acoustic', 'Live', 'Demo', 'Version'])})"
    
    # Generate artist name
    prefix = random.choice(ARTIST_NAME_PATTERNS.get(genre, ['The']))
    suffix = random.choice(['Wolf', 'Moon', 'Fire', 'Star', 'Wave', 'King', 'Queen', 'Soul'])
    artist_name = f"{prefix} {suffix}"
    
    # Audio features (realistic distributions per genre)
    if genre == 'pop':
        danceability = np.random.beta(8, 2)
        energy = np.random.beta(7, 3)
        valence = np.random.beta(6, 4)
    elif genre == 'rock':
        danceability = np.random.beta(5, 5)
        energy = np.random.beta(9, 2)
        valence = np.random.beta(5, 5)
    elif genre == 'hip-hop':
        danceability = np.random.beta(9, 2)
        energy = np.random.beta(8, 3)
        valence = np.random.beta(6, 4)
    elif genre == 'country':
        danceability = np.random.beta(6, 4)
        energy = np.random.beta(5, 5)
        valence = np.random.beta(7, 3)
    elif genre == 'electronic':
        danceability = np.random.beta(9, 1)
        energy = np.random.beta(9, 2)
        valence = np.random.beta(6, 4)
    elif genre == 'jazz':
        danceability = np.random.beta(5, 5)
        energy = np.random.beta(4, 6)
        valence = np.random.beta(5, 5)
    elif genre == 'classical':
        danceability = np.random.beta(2, 8)
        energy = np.random.beta(5, 5)
        valence = np.random.beta(5, 5)
    else:  # indie
        danceability = np.random.beta(5, 5)
        energy = np.random.beta(6, 4)
        valence = np.random.beta(4, 6)
    
    # Popularity (correlates somewhat with features, but with noise)
    # Better names, higher energy/danceability → slightly higher popularity
    name_quality = len(song_name) / 30  # Shorter names slightly better
    artist_quality = 0.5 if prefix in ['The', 'Lil', 'Young'] else 0.3
    
    popularity_base = (0.3 * danceability + 0.3 * energy + 0.2 * valence + 
                       0.1 * name_quality + 0.1 * artist_quality)
    popularity = int(np.clip(popularity_base * 100 + np.random.normal(0, 15), 0, 100))
    
    # Release date (last 10 years)
    days_ago = random.randint(0, 3650)
    release_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    return {
        'song_id': f'demo_{genre}_{idx}',
        'song_name': song_name,
        'artist_name': artist_name,
        'artist_id': f'artist_{genre}_{idx // 10}',
        'album_name': f'{artist_name} - Album',
        'release_date': release_date,
        'popularity': popularity,
        'genre': genre,
        'duration_ms': random.randint(120000, 300000),
        'explicit': random.random() < 0.15,
        'audio_features': {
            'danceability': round(danceability, 3),
            'energy': round(energy, 3),
            'key': random.randint(0, 11),
            'loudness': round(np.random.normal(-6, 3), 2),
            'mode': random.randint(0, 1),
            'speechiness': round(np.random.beta(2, 8), 3),
            'acousticness': round(np.random.beta(3, 7), 3),
            'instrumentalness': round(np.random.beta(2, 8), 3),
            'liveness': round(np.random.beta(2, 8), 3),
            'valence': round(valence, 3),
            'tempo': round(np.random.normal(120, 25), 1)
        }
    }

def create_demo_dataset(n_songs=50000):
    """Create demo dataset"""
    
    print("\n" + "="*80)
    print(f"CREATING DEMO MUSIC DATASET ({n_songs:,} songs)")
    print("="*80)
    
    genres = ['pop', 'rock', 'hip-hop', 'country', 'electronic', 'jazz', 'classical', 'indie']
    songs_per_genre = n_songs // len(genres)
    
    all_songs = []
    
    for genre in genres:
        print(f"\n[{genre}] Generating {songs_per_genre:,} songs...")
        for i in range(songs_per_genre):
            song = generate_song(genre, i)
            all_songs.append(song)
        print(f"  ✓ {songs_per_genre:,} {genre} songs created")
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'spotify_songs.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        'collected_at': datetime.now().isoformat(),
        'total_songs': len(all_songs),
        'genres': genres,
        'data_type': 'demo',
        'note': 'Synthetic demo data. Replace with real Spotify API data for production.',
        'songs': all_songs
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Created {len(all_songs):,} demo songs")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return all_songs

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    create_demo_dataset(50000)









