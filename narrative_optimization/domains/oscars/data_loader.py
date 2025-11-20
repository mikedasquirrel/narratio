"""
Oscar Data Loader - Best Picture Nominees

Loads and validates oscar_nominees_complete.json
Preserves year-by-year competitive structure
Extracts rich nominative data (actors, characters, directors)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


class OscarDataLoader:
    """Load and process Oscar Best Picture nominees dataset"""
    
    def __init__(self, data_path=None):
        if data_path is None:
            # Default to project data directory
            self.data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'oscar_nominees_complete.json'
        else:
            self.data_path = Path(data_path)
    
    def load_raw_data(self):
        """Load raw Oscar data from JSON"""
        print(f"Loading Oscar data from {self.data_path}...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded data for {len(data)} years")
        return data
    
    def validate_structure(self, data):
        """Validate data structure"""
        print("Validating data structure...")
        
        issues = []
        total_films = 0
        total_winners = 0
        
        for year, films in data.items():
            if not isinstance(films, list):
                issues.append(f"Year {year}: films is not a list")
                continue
            
            year_winners = sum(1 for f in films if f.get('won_oscar', False))
            total_winners += year_winners
            total_films += len(films)
            
            if year_winners != 1:
                issues.append(f"Year {year}: has {year_winners} winners (should be 1)")
            
            for i, film in enumerate(films):
                required_fields = ['title', 'overview', 'cast', 'won_oscar']
                missing = [f for f in required_fields if f not in film]
                if missing:
                    issues.append(f"Year {year}, film {i}: missing fields {missing}")
        
        if issues:
            print(f"⚠ Found {len(issues)} validation issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ Data structure is valid")
        
        print(f"✓ Total: {total_films} films, {total_winners} winners across {len(data)} years")
        
        return total_films, total_winners
    
    def extract_nominatives(self, film):
        """Extract all nominative elements from film data"""
        nominatives = {
            'title': film.get('title', ''),
            'actors': [],
            'characters': [],
            'director': film.get('director', []),
            'keywords': film.get('keywords', []),
            'genres': film.get('genres', [])
        }
        
        # Extract from cast
        cast = film.get('cast', [])
        for cast_member in cast[:20]:  # Top 20
            if 'actor' in cast_member:
                nominatives['actors'].append(cast_member['actor'])
            if 'character' in cast_member and cast_member['character']:
                nominatives['characters'].append(cast_member['character'])
        
        return nominatives
    
    def create_full_narrative(self, film):
        """Create full narrative text for transformer input"""
        nominatives = self.extract_nominatives(film)
        
        # Combine all text elements
        narrative_parts = [
            nominatives['title'],
            film.get('overview', ''),
            film.get('tagline', ''),
            ' '.join(nominatives['actors'][:10]),
            ' '.join(nominatives['characters'][:10]),
            ' '.join(nominatives['keywords'][:10]),
            ' '.join(nominatives['director']),
            ' '.join(nominatives['genres'])
        ]
        
        full_narrative = ' '.join([p for p in narrative_parts if p])
        return full_narrative
    
    def process_dataset(self, data):
        """Process dataset into flat list with year metadata"""
        print("Processing dataset...")
        
        processed_films = []
        
        for year, films in data.items():
            # Count nominees for this year
            num_nominees = len(films)
            winner_title = next((f['title'] for f in films if f.get('won_oscar')), None)
            
            for film in films:
                nominatives = self.extract_nominatives(film)
                
                film_record = {
                    # Core identification
                    'year': int(year),
                    'title': film.get('title', ''),
                    'original_title': film.get('original_title', film.get('title', '')),
                    'won_oscar': int(film.get('won_oscar', False)),
                    
                    # Narrative text
                    'overview': film.get('overview', ''),
                    'tagline': film.get('tagline', ''),
                    'full_narrative': self.create_full_narrative(film),
                    
                    # Nominatives
                    'actors': nominatives['actors'],
                    'characters': nominatives['characters'],
                    'director': nominatives['director'],
                    'keywords': nominatives['keywords'],
                    'genres': nominatives['genres'],
                    
                    # Metadata
                    'num_actors': len(nominatives['actors']),
                    'num_characters': len(nominatives['characters']),
                    'num_keywords': len(nominatives['keywords']),
                    
                    # Competitive context
                    'num_nominees_this_year': num_nominees,
                    'winner_this_year': winner_title,
                    'is_winner': film.get('won_oscar', False),
                    
                    # Derived features
                    'word_count': len(film.get('overview', '').split()),
                    'cast_size': len(film.get('cast', [])),
                    'has_tagline': bool(film.get('tagline')),
                    'num_genres': len(nominatives['genres'])
                }
                
                processed_films.append(film_record)
        
        print(f"✓ Processed {len(processed_films)} film records")
        return processed_films
    
    def get_competitive_structure(self, processed_films):
        """Organize films by year for competitive analysis"""
        by_year = defaultdict(list)
        
        for film in processed_films:
            by_year[film['year']].append(film)
        
        return dict(by_year)
    
    def get_statistics(self, processed_films):
        """Get dataset statistics"""
        df = pd.DataFrame(processed_films)
        
        stats = {
            'total_films': len(processed_films),
            'total_years': df['year'].nunique(),
            'total_winners': df['won_oscar'].sum(),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'avg_nominees_per_year': df.groupby('year').size().mean(),
            'avg_cast_size': df['cast_size'].mean(),
            'avg_keywords': df['num_keywords'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'films_with_tagline': df['has_tagline'].sum(),
            
            # Genre distribution
            'genre_distribution': self._get_genre_distribution(processed_films),
            
            # Year distribution
            'year_distribution': df['year'].value_counts().sort_index().to_dict(),
            
            # Winners by year
            'winners_by_year': {
                row['year']: row['title'] 
                for _, row in df[df['won_oscar'] == 1].iterrows()
            }
        }
        
        return stats
    
    def _get_genre_distribution(self, processed_films):
        """Get distribution of genres across all films"""
        genre_counts = defaultdict(int)
        for film in processed_films:
            for genre in film['genres']:
                genre_counts[genre] += 1
        return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))
    
    def load_full_dataset(self):
        """
        Load and process full Oscar dataset
        
        Returns
        -------
        tuple
            (processed_films, competitive_structure, statistics)
        """
        print("="*80)
        print("OSCAR DATA LOADER - Best Picture Nominees")
        print("="*80)
        
        # Load and validate
        raw_data = self.load_raw_data()
        self.validate_structure(raw_data)
        
        # Process
        processed_films = self.process_dataset(raw_data)
        competitive_structure = self.get_competitive_structure(processed_films)
        stats = self.get_statistics(processed_films)
        
        return processed_films, competitive_structure, stats
    
    def save_processed_data(self, processed_films, output_path=None):
        """Save processed dataset"""
        if output_path is None:
            output_path = Path(__file__).parent / 'oscar_processed.json'
        
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_films, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved processed dataset: {output_path}")
        return output_path


def main():
    """Load and validate Oscar dataset"""
    loader = OscarDataLoader()
    
    # Load full dataset
    processed_films, competitive_structure, stats = loader.load_full_dataset()
    
    # Display statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total films: {stats['total_films']}")
    print(f"Total years: {stats['total_years']}")
    print(f"Total winners: {stats['total_winners']}")
    print(f"Year range: {stats['year_range'][0]}-{stats['year_range'][1]}")
    print(f"Avg nominees per year: {stats['avg_nominees_per_year']:.1f}")
    print(f"Avg cast size: {stats['avg_cast_size']:.1f}")
    print(f"Avg keywords: {stats['avg_keywords']:.1f}")
    print(f"Avg overview length: {stats['avg_word_count']:.0f} words")
    print(f"Films with tagline: {stats['films_with_tagline']} ({stats['films_with_tagline']/stats['total_films']*100:.1f}%)")
    
    print("\nTop 10 genres:")
    for genre, count in list(stats['genre_distribution'].items())[:10]:
        print(f"  {genre}: {count}")
    
    print("\nWinners by year:")
    for year in sorted(stats['winners_by_year'].keys()):
        title = stats['winners_by_year'][year]
        num_nominees = stats['year_distribution'][year]
        print(f"  {year}: {title} ({num_nominees} nominees)")
    
    # Show competitive structure
    print("\n" + "="*80)
    print("COMPETITIVE STRUCTURE SAMPLE")
    print("="*80)
    
    # Sample one year
    sample_year = max(competitive_structure.keys())
    print(f"\n{sample_year} Best Picture Competition:")
    for film in competitive_structure[sample_year]:
        status = "WINNER" if film['won_oscar'] else "nominee"
        print(f"  [{status}] {film['title']}")
        print(f"    Cast: {', '.join(film['actors'][:5])}")
        print(f"    Director: {', '.join(film['director'])}")
        print(f"    Genres: {', '.join(film['genres'][:3])}")
        print()
    
    # Save processed data
    loader.save_processed_data(processed_films)


if __name__ == '__main__':
    main()

