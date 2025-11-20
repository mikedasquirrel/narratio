"""
Spotify Data Collector

Collects 50K+ songs with:
- Song names, artist names
- Audio features (danceability, energy, valence, etc.)
- Popularity scores
- Genres, release dates
- Metadata
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import time
from pathlib import Path
from datetime import datetime

class SpotifyCollector:
    """Collect songs from Spotify API"""
    
    def __init__(self, client_id=None, client_secret=None):
        """
        Initialize Spotify client
        
        Get credentials from: https://developer.spotify.com/dashboard
        """
        if not client_id or not client_secret:
            print("=" * 80)
            print("SPOTIFY API CREDENTIALS NEEDED")
            print("=" * 80)
            print("\n1. Go to: https://developer.spotify.com/dashboard")
            print("2. Create an app")
            print("3. Get Client ID and Client Secret")
            print("\nThen run:")
            print("  collector = SpotifyCollector(client_id='YOUR_ID', client_secret='YOUR_SECRET')")
            print("=" * 80)
            return
        
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        
    def collect_songs(self, target_count=50000, genres=None):
        """
        Collect songs across multiple genres
        
        Args:
            target_count: Target number of songs
            genres: List of genres (default: diverse mix)
        """
        if not hasattr(self, 'sp'):
            print("❌ Spotify client not initialized. Provide credentials first.")
            return
        
        if genres is None:
            # Diverse genre mix for broad coverage
            genres = [
                'pop', 'rock', 'hip-hop', 'country', 'jazz', 'classical',
                'electronic', 'r-n-b', 'indie', 'metal', 'folk', 'blues',
                'reggae', 'latin', 'dance', 'alternative', 'punk', 'soul'
            ]
        
        all_songs = []
        songs_per_genre = target_count // len(genres)
        
        print(f"\n{'='*80}")
        print(f"COLLECTING {target_count:,} SONGS FROM SPOTIFY")
        print(f"{'='*80}\n")
        
        for genre in genres:
            print(f"[Genre: {genre}] Collecting {songs_per_genre:,} songs...")
            genre_songs = self._collect_genre(genre, songs_per_genre)
            all_songs.extend(genre_songs)
            print(f"  ✓ {len(genre_songs):,} songs collected")
            time.sleep(0.5)  # Rate limiting
        
        # Save
        output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'spotify_songs.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'collected_at': datetime.now().isoformat(),
                'total_songs': len(all_songs),
                'genres': genres,
                'songs': all_songs
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Collected {len(all_songs):,} songs")
        print(f"✓ Saved to: {output_path}")
        print(f"{'='*80}\n")
        
        return all_songs
    
    def _collect_genre(self, genre, count):
        """Collect songs for a specific genre"""
        songs = []
        offset = 0
        limit = 50  # Spotify max per request
        
        while len(songs) < count:
            try:
                # Search for genre
                results = self.sp.search(
                    q=f'genre:{genre}',
                    type='track',
                    limit=limit,
                    offset=offset
                )
                
                if not results['tracks']['items']:
                    break
                
                for track in results['tracks']['items']:
                    if not track or 'id' not in track:
                        continue
                    
                    # Get audio features
                    try:
                        audio_features = self.sp.audio_features([track['id']])[0]
                    except:
                        audio_features = {}
                    
                    # Extract data
                    song_data = {
                        'song_id': track['id'],
                        'song_name': track['name'],
                        'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                        'artist_id': track['artists'][0]['id'] if track['artists'] else None,
                        'album_name': track['album']['name'],
                        'release_date': track['album']['release_date'],
                        'popularity': track['popularity'],
                        'genre': genre,
                        'duration_ms': track['duration_ms'],
                        'explicit': track['explicit'],
                        'audio_features': {
                            'danceability': audio_features.get('danceability', None),
                            'energy': audio_features.get('energy', None),
                            'key': audio_features.get('key', None),
                            'loudness': audio_features.get('loudness', None),
                            'mode': audio_features.get('mode', None),
                            'speechiness': audio_features.get('speechiness', None),
                            'acousticness': audio_features.get('acousticness', None),
                            'instrumentalness': audio_features.get('instrumentalness', None),
                            'liveness': audio_features.get('liveness', None),
                            'valence': audio_features.get('valence', None),
                            'tempo': audio_features.get('tempo', None)
                        } if audio_features else {}
                    }
                    
                    songs.append(song_data)
                    
                    if len(songs) >= count:
                        break
                
                offset += limit
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  Warning: {e}")
                break
        
        return songs


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SPOTIFY DATA COLLECTOR")
    print("="*80)
    print("\nTo collect data:")
    print("\n1. Get Spotify API credentials:")
    print("   https://developer.spotify.com/dashboard")
    print("\n2. Run:")
    print("""
from music.data_collector import SpotifyCollector

collector = SpotifyCollector(
    client_id='your_client_id',
    client_secret='your_client_secret'
)
songs = collector.collect_songs(target_count=50000)
""")
    print("="*80 + "\n")









