"""
Music Dataset Collector (UNFILTERED APPROACH)

Collects broad, representative sample of music across all genres.
DOES NOT pre-select for "narrative" songs - lets transformers discover patterns.

Methodology:
1. Sample broadly across genres (proportional to popularity)
2. Include high AND low narrative songs
3. Let archetype transformers discover which songs have narrative
4. Empirically determine: Which genres are naturally narrative?

Target: 5,000 songs (unfiltered)

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import requests
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import random


class MusicCollectorUnfiltered:
    """
    Collect unfiltered music dataset for narrative pattern discovery.
    
    Key principle: Don't assume what "narrative music" is - discover it!
    """
    
    def __init__(self, output_dir='data/domains/music', genius_token=None, spotify_token=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.genius_token = genius_token  # From environment or config
        self.spotify_token = spotify_token
        
        self.songs = []
        
        # Genre distribution (proportional to real-world popularity)
        # NO BIAS toward "narrative" genres
        self.genre_targets = {
            'pop': 800,
            'rock': 700,
            'hip-hop': 600,
            'country': 500,
            'folk': 400,
            'r-and-b': 400,
            'indie': 300,
            'electronic': 200,  # Include even if low narrative!
            'metal': 200,
            'jazz': 150,
            'classical': 100,
            'world': 150,
            'blues': 100,
            'reggae': 100,
            'punk': 100,
            'soul': 100,
            'other': 100
        }
    
    def collect_spotify_top_tracks_by_genre(self, genre: str, limit: int) -> List[Dict]:
        """
        Collect top tracks for a genre from Spotify.
        
        Uses Spotify's genre seeds and popularity ranking.
        UNBIASED: Takes top tracks regardless of lyrical content.
        """
        if not self.spotify_token:
            print(f"Warning: No Spotify token, skipping Spotify collection for {genre}")
            return []
        
        # In production, would use actual Spotify API
        # For now, return structure
        print(f"Collecting {limit} {genre} tracks from Spotify...")
        
        # Simulated structure (replace with actual API calls)
        collected = []
        
        # Would implement:
        # 1. Search for genre
        # 2. Get top tracks by popularity
        # 3. Get track metadata
        # 4. NO filtering by lyrics!
        
        return collected
    
    def get_lyrics_from_genius(self, song_title: str, artist: str) -> Optional[str]:
        """
        Fetch lyrics from Genius API.
        
        Returns None if no lyrics found (instrumental, etc.) - that's data too!
        """
        if not self.genius_token:
            return None
        
        base_url = 'https://api.genius.com'
        headers = {'Authorization': f'Bearer {self.genius_token}'}
        
        try:
            # Search for song
            search_url = f'{base_url}/search'
            params = {'q': f'{song_title} {artist}'}
            response = requests.get(search_url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            hits = data.get('response', {}).get('hits', [])
            if not hits:
                return None
            
            # Get first result
            song_path = hits[0]['result']['path']
            
            # In production, would scrape lyrics page
            # For now, return None (lyrics scraping requires additional logic)
            return None
            
        except Exception as e:
            print(f"  Error fetching lyrics for {song_title}: {e}")
            return None
    
    def collect_billboard_charts(self, years: List[int], charts_per_year=52) -> List[Dict]:
        """
        Collect from Billboard charts (historical data).
        
        UNBIASED: Whatever was popular, regardless of narrative content.
        """
        print(f"Collecting Billboard chart data for {len(years)} years...")
        
        collected = []
        
        # Would implement Billboard scraping or API
        # Key: Take TOP songs, not "story songs"
        
        return collected
    
    def add_narrative_analysis(self, song: Dict) -> Dict:
        """
        Add narrative analysis flags.
        
        These are DISCOVERED, not assumed:
        - has_lyrics: Does song have lyrics?
        - lyrics_length: How many words?
        - has_story: Does transformer detect narrative?
        - journey_elements: Does transformer detect Campbell stages?
        """
        song['narrative_discovered'] = {
            'has_lyrics': bool(song.get('lyrics')),
            'lyrics_length': len(song.get('lyrics', '').split()) if song.get('lyrics') else 0,
            'has_narrative': None,  # Will be filled by transformer
            'narrative_score': None,  # Will be filled by transformer
            'discovered_archetype': None  # Will be filled by analysis
        }
        
        return song
    
    def run_unfiltered_collection(self) -> None:
        """
        Run complete unfiltered collection.
        
        Key principle: Sample broadly, discover patterns empirically.
        """
        print("="*70)
        print("UNFILTERED MUSIC COLLECTION")
        print("="*70)
        print("Methodology: Broad sampling → Empirical discovery")
        print("NOT pre-selecting for 'narrative' songs")
        print("="*70 + "\n")
        
        total_collected = 0
        
        for genre, target in self.genre_targets.items():
            print(f"\nCollecting {genre} ({target} songs)...")
            
            # Collect from multiple sources
            spotify_tracks = self.collect_spotify_top_tracks_by_genre(genre, target)
            
            # Add all to dataset (no filtering!)
            for track in spotify_tracks:
                # Get lyrics if available (but don't require them)
                lyrics = self.get_lyrics_from_genius(
                    track.get('title', ''),
                    track.get('artist', '')
                )
                
                track['lyrics'] = lyrics
                track['genre'] = genre
                
                # Add narrative analysis placeholders
                track = self.add_narrative_analysis(track)
                
                self.songs.append(track)
                total_collected += 1
            
            print(f"  Collected: {len(spotify_tracks)} {genre} songs")
            print(f"  Total so far: {total_collected}")
            
            time.sleep(1)  # Rate limiting
        
        # Save dataset
        self._save_dataset()
        
        print("\n" + "="*70)
        print("COLLECTION COMPLETE")
        print("="*70)
        print(f"Total songs: {len(self.songs)}")
        print(f"With lyrics: {sum([1 for s in self.songs if s.get('lyrics')])}")
        print(f"Genres: {len(self.genre_targets)}")
        print("\nNext step: Run archetype transformers on ALL songs")
        print("Discover: Which songs have narrative patterns?")
        print("="*70)
    
    def _save_dataset(self) -> None:
        """Save complete unfiltered dataset."""
        output_file = self.output_dir / 'music_unfiltered_dataset.json'
        
        dataset = {
            'metadata': {
                'total_songs': len(self.songs),
                'collection_method': 'UNFILTERED stratified random sampling',
                'principle': 'Broad collection, empirical discovery',
                'date_collected': time.strftime('%Y-%m-%d'),
                'genres': list(self.genre_targets.keys()),
                'with_lyrics': sum([1 for s in self.songs if s.get('lyrics')]),
                'narrative_to_be_discovered': True
            },
            'sampling_notes': [
                'NOT pre-filtered for narrative content',
                'Includes pop, EDM, instrumental (low narrative expected)',
                'Includes folk, hip-hop, country (high narrative expected)',
                'Transformers will empirically discover narrative patterns',
                'Goal: Learn which genres/types are naturally narrative'
            ],
            'songs': self.songs
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Dataset saved: {output_file}")


def main():
    """Run unfiltered music collection."""
    print("""
    UNFILTERED MUSIC COLLECTION
    
    This collector samples BROADLY across all genres.
    It does NOT pre-select for "narrative-focused" songs.
    
    Why? To empirically discover:
    - Which genres naturally have narrative structure?
    - Does narrative predict success in some genres but not others?
    - What is the distribution of narrative in music?
    - Can transformers detect story-songs from abstract songs?
    
    The archetype transformers will analyze ALL songs and reveal patterns.
    """)
    
    # Check for API tokens
    import os
    genius_token = os.environ.get('GENIUS_API_TOKEN')
    spotify_token = os.environ.get('SPOTIFY_API_TOKEN')
    
    if not genius_token or not spotify_token:
        print("\n⚠️  API tokens not found in environment")
        print("Set GENIUS_API_TOKEN and SPOTIFY_API_TOKEN")
        print("\nProceeding with demo mode (no actual collection)")
    
    collector = MusicCollectorUnfiltered(
        genius_token=genius_token,
        spotify_token=spotify_token
    )
    
    collector.run_unfiltered_collection()


if __name__ == '__main__':
    main()

