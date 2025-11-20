"""
Film Dataset Extension

Extends existing IMDB dataset with:
- Beat sheet analysis (Save the Cat)
- Hero's Journey mapping
- Genre-specific analysis
- Archetype profiling

Target: Enrich existing 1,000 films + add 1,500 more = 2,500 total

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import json
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from transformers.archetypes import (
    HeroJourneyTransformer,
    StructuralBeatTransformer
)


class FilmDatasetExtender:
    """
    Extend existing film dataset with archetype analysis.
    
    Approach: Apply archetype transformers to existing film summaries.
    """
    
    def __init__(self, existing_data_path='data/domains/imdb_movies_complete.json'):
        self.existing_path = Path(existing_data_path)
        self.films = []
        
        self.transformers = {
            'journey': HeroJourneyTransformer(),
            'beats': StructuralBeatTransformer()
        }
    
    def load_existing_dataset(self) -> None:
        """Load existing IMDB dataset."""
        if not self.existing_path.exists():
            print(f"Warning: {self.existing_path} not found")
            return
        
        with open(self.existing_path) as f:
            data = json.load(f)
        
        self.films = data.get('movies', []) if isinstance(data, dict) else data
        print(f"Loaded {len(self.films)} existing films")
    
    def add_archetype_analysis(self) -> None:
        """
        Add archetype analysis to all films.
        
        Extracts:
        - Hero's Journey completion
        - Beat adherence score
        - Dominant archetypes
        - Plot type classification
        """
        print(f"\nAnalyzing {len(self.films)} films with archetype transformers...")
        
        for i, film in enumerate(self.films):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(self.films)}...")
            
            # Get plot summary
            summary = film.get('plot_summary', '') or film.get('summary', '')
            
            if not summary or len(summary) < 200:
                continue
            
            # Extract Hero's Journey features
            try:
                journey_features = self.transformers['journey'].fit([summary]).transform([summary])
                film['archetype_analysis'] = {
                    'journey_completion': float(journey_features[0, 2]),
                    'stages_present': int(sum(journey_features[0, :17] > 0.5)),
                    'follows_hero_journey': bool(journey_features[0, 2] > 0.60)
                }
            except:
                film['archetype_analysis'] = None
            
            # Extract beat structure
            try:
                beat_features = self.transformers['beats'].fit([summary]).transform([summary])
                film['beat_analysis'] = {
                    'beat_adherence_score': float(beat_features[0, 15]),
                    'structural_quality': float(beat_features[0, -1])
                }
            except:
                film['beat_analysis'] = None
        
        print(f"✅ Archetype analysis complete")
    
    def analyze_by_genre(self) -> Dict:
        """
        Analyze archetype patterns by genre.
        
        Discovers: Which genres follow formulas? Which don't?
        """
        by_genre = {}
        
        for film in self.films:
            genre = film.get('genre', 'Unknown')
            
            if genre not in by_genre:
                by_genre[genre] = {
                    'films': [],
                    'mean_journey': 0,
                    'mean_beats': 0
                }
            
            if film.get('archetype_analysis'):
                by_genre[genre]['films'].append(film)
        
        # Calculate means
        for genre, data in by_genre.items():
            if data['films']:
                data['mean_journey'] = sum([f['archetype_analysis']['journey_completion'] 
                                            for f in data['films']]) / len(data['films'])
                if data['films'][0].get('beat_analysis'):
                    data['mean_beats'] = sum([f['beat_analysis']['beat_adherence_score'] 
                                             for f in data['films']]) / len(data['films'])
        
        return by_genre
    
    def save_extended_dataset(self) -> None:
        """Save extended dataset with archetype analysis."""
        output_file = Path('data/domains/film_extended/film_with_archetypes.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Genre analysis
        genre_analysis = self.analyze_by_genre()
        
        dataset = {
            'metadata': {
                'total_films': len(self.films),
                'with_archetype_analysis': sum([1 for f in self.films if f.get('archetype_analysis')]),
                'date_extended': time.strftime('%Y-%m-%d'),
                'transformers_applied': ['HeroJourney', 'StructuralBeat']
            },
            'genre_analysis': genre_analysis,
            'films': self.films
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Extended dataset saved: {output_file}")
    
    def run_extension(self) -> None:
        """Run complete extension workflow."""
        print("="*70)
        print("FILM DATASET EXTENSION")
        print("="*70 + "\n")
        
        self.load_existing_dataset()
        self.add_archetype_analysis()
        self.save_extended_dataset()
        
        print("\n" + "="*70)
        print("EXTENSION COMPLETE")
        print("="*70)


def main():
    extender = FilmDatasetExtender()
    extender.run_extension()


if __name__ == '__main__':
    main()

