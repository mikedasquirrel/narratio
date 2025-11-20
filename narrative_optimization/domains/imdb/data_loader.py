"""
IMDB Data Loader - CMU Movie Summaries

Parses and merges:
- plot_summaries.txt: 42,306 plot summaries  
- movie.metadata.tsv: Box office, runtime, genres, release dates
- character.metadata.tsv: Cast and character names (nominatives)

Output: Unified dataset with rich narrative data
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime


class IMDBDataLoader:
    """Load and process CMU Movie Summaries dataset"""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default to project data directory
            self.data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'MovieSummaries'
        else:
            self.data_dir = Path(data_dir)
        
        self.plot_summaries_path = self.data_dir / 'plot_summaries.txt'
        self.movie_metadata_path = self.data_dir / 'movie.metadata.tsv'
        self.character_metadata_path = self.data_dir / 'character.metadata.tsv'
    
    def load_plot_summaries(self):
        """Load plot summaries from text file"""
        print("Loading plot summaries...")
        summaries = {}
        
        with open(self.plot_summaries_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    wikipedia_id, summary = parts
                    summaries[int(wikipedia_id)] = summary
        
        print(f"✓ Loaded {len(summaries)} plot summaries")
        return summaries
    
    def load_movie_metadata(self):
        """Load movie metadata from TSV"""
        print("Loading movie metadata...")
        
        # Column names based on README
        columns = [
            'wikipedia_id',
            'freebase_id',
            'title',
            'release_date',
            'box_office_revenue',
            'runtime',
            'languages',
            'countries',
            'genres'
        ]
        
        movies = pd.read_csv(
            self.movie_metadata_path,
            sep='\t',
            names=columns,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        print(f"✓ Loaded {len(movies)} movie metadata records")
        return movies
    
    def load_character_metadata(self):
        """Load character and cast metadata from TSV"""
        print("Loading character/cast metadata...")
        
        # Column names based on README
        columns = [
            'wikipedia_id',
            'freebase_id',
            'release_date',
            'character_name',
            'actor_dob',
            'actor_gender',
            'actor_height',
            'actor_ethnicity',
            'actor_name',
            'actor_age_at_release',
            'freebase_map_id',
            'freebase_character_id',
            'freebase_actor_id'
        ]
        
        characters = pd.read_csv(
            self.character_metadata_path,
            sep='\t',
            names=columns,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        
        print(f"✓ Loaded {len(characters)} character/actor records")
        return characters
    
    def aggregate_cast_data(self, characters_df):
        """Aggregate cast data per movie"""
        print("Aggregating cast data per movie...")
        
        cast_data = {}
        
        for wiki_id, group in characters_df.groupby('wikipedia_id'):
            # Extract actor names and character names
            actors = group['actor_name'].dropna().tolist()
            character_names = group['character_name'].dropna().tolist()
            
            # Count gender distribution
            gender_counts = group['actor_gender'].value_counts().to_dict()
            
            cast_data[int(wiki_id)] = {
                'actors': actors[:20],  # Top 20 actors
                'characters': character_names[:20],  # Top 20 characters
                'num_actors': len(actors),
                'num_female': gender_counts.get('F', 0),
                'num_male': gender_counts.get('M', 0),
                'cast_diversity': len(set(actors)) / max(len(actors), 1)
            }
        
        print(f"✓ Aggregated cast for {len(cast_data)} movies")
        return cast_data
    
    def parse_genres(self, genres_str):
        """Parse genres from Freebase format"""
        if pd.isna(genres_str):
            return []
        
        try:
            # Extract genre names from {"/m/id": "Genre Name"} format
            genres = []
            matches = re.findall(r':\s*"([^"]+)"', genres_str)
            genres = matches if matches else []
            return genres
        except:
            return []
    
    def parse_date(self, date_str):
        """Parse release date to year"""
        if pd.isna(date_str):
            return None
        
        try:
            # Try parsing as date
            date = pd.to_datetime(date_str)
            return date.year
        except:
            # Try extracting year directly
            match = re.search(r'(\d{4})', str(date_str))
            if match:
                return int(match.group(1))
            return None
    
    def calculate_success_score(self, row):
        """Calculate success score from box office and other metrics"""
        # Normalize box office by runtime and year
        box_office = row.get('box_office_revenue', 0) or 0
        runtime = row.get('runtime', 100) or 100
        year = row.get('release_year', 2000) or 2000
        
        if box_office <= 0:
            return 0.0
        
        # Adjust for inflation (simple approximation)
        inflation_factor = 1.0 + (2024 - year) * 0.03
        adjusted_box_office = box_office / inflation_factor
        
        # Normalize by runtime (per minute revenue)
        revenue_per_minute = adjusted_box_office / runtime
        
        # Log transform and normalize to [0, 1]
        score = np.log1p(revenue_per_minute) / 20.0  # Rough normalization
        score = min(1.0, max(0.0, score))
        
        return score
    
    def merge_all_data(self, summaries, movies, cast_data):
        """Merge all data sources into unified dataset"""
        print("Merging all data sources...")
        
        merged_data = []
        
        for idx, row in movies.iterrows():
            wiki_id = int(row['wikipedia_id'])
            
            # Skip if no plot summary
            if wiki_id not in summaries:
                continue
            
            # Parse genres
            genres = self.parse_genres(row['genres'])
            if not genres:
                continue  # Skip movies without genre info
            
            # Parse year
            year = self.parse_date(row['release_date'])
            if year is None or year < 1900 or year > 2025:
                continue  # Skip invalid years
            
            # Get cast data
            cast = cast_data.get(wiki_id, {})
            
            # Build movie record
            movie_record = {
                'wikipedia_id': wiki_id,
                'freebase_id': row['freebase_id'],
                'title': row['title'],
                'release_year': year,
                'plot_summary': summaries[wiki_id],
                'box_office_revenue': float(row['box_office_revenue']) if pd.notna(row['box_office_revenue']) else 0.0,
                'runtime': float(row['runtime']) if pd.notna(row['runtime']) else 100.0,
                'genres': genres,
                'primary_genre': genres[0] if genres else 'Unknown',
                'num_genres': len(genres),
                
                # Cast data (nominatives)
                'actors': cast.get('actors', []),
                'characters': cast.get('characters', []),
                'num_actors': cast.get('num_actors', 0),
                'num_female': cast.get('num_female', 0),
                'num_male': cast.get('num_male', 0),
                'cast_diversity': cast.get('cast_diversity', 0.0),
                
                # Derived features
                'has_box_office': row['box_office_revenue'] > 0 if pd.notna(row['box_office_revenue']) else False,
                'decade': (year // 10) * 10,
                'word_count': len(summaries[wiki_id].split()),
            }
            
            # Calculate success score
            movie_record['success_score'] = self.calculate_success_score(movie_record)
            
            # Create full narrative text (for transformer input)
            narrative_parts = [
                movie_record['title'],
                movie_record['plot_summary'],
                ' '.join(movie_record['actors'][:10]),  # Top 10 actors
                ' '.join(movie_record['characters'][:10]),  # Top 10 characters
                ' '.join(movie_record['genres'])
            ]
            movie_record['full_narrative'] = ' '.join(narrative_parts)
            
            merged_data.append(movie_record)
        
        print(f"✓ Merged {len(merged_data)} complete movie records")
        return merged_data
    
    def filter_dataset(self, data, min_box_office=10000, min_year=1980):
        """Filter to high-quality subset"""
        print(f"Filtering dataset (min_box_office=${min_box_office:,}, min_year={min_year})...")
        
        filtered = [
            movie for movie in data
            if movie['has_box_office'] and 
               movie['box_office_revenue'] >= min_box_office and
               movie['release_year'] >= min_year and
               movie['word_count'] >= 50  # Reasonable plot summary
        ]
        
        print(f"✓ Filtered to {len(filtered)} movies with reliable data")
        return filtered
    
    def save_dataset(self, data, output_path=None):
        """Save processed dataset to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'imdb_movies_complete.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved complete dataset: {output_path}")
        print(f"  Total movies: {len(data)}")
        
        return output_path
    
    def load_full_dataset(self, use_cache=True, filter_data=True):
        """
        Load and process full IMDB dataset
        
        Parameters
        ----------
        use_cache : bool
            If True, load from cached JSON if available
        filter_data : bool
            If True, filter to high-quality subset
        
        Returns
        -------
        list
            List of movie dictionaries
        """
        cache_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'imdb_movies_complete.json'
        
        # Try cache first
        if use_cache and cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data)} movies from cache")
            return data
        
        # Load from source files
        print("="* 80)
        print("IMDB DATA LOADER - CMU Movie Summaries")
        print("="* 80)
        
        summaries = self.load_plot_summaries()
        movies = self.load_movie_metadata()
        characters = self.load_character_metadata()
        
        cast_data = self.aggregate_cast_data(characters)
        merged_data = self.merge_all_data(summaries, movies, cast_data)
        
        if filter_data:
            merged_data = self.filter_dataset(merged_data)
        
        # Save to cache
        self.save_dataset(merged_data, cache_path)
        
        return merged_data
    
    def get_statistics(self, data):
        """Get dataset statistics"""
        stats = {
            'total_movies': len(data),
            'year_range': (min(m['release_year'] for m in data), max(m['release_year'] for m in data)),
            'genres': len(set(m['primary_genre'] for m in data)),
            'avg_box_office': np.mean([m['box_office_revenue'] for m in data]),
            'avg_runtime': np.mean([m['runtime'] for m in data]),
            'avg_word_count': np.mean([m['word_count'] for m in data]),
            'with_cast_data': sum(1 for m in data if m['num_actors'] > 0),
            'genre_distribution': pd.Series([m['primary_genre'] for m in data]).value_counts().to_dict(),
            'decade_distribution': pd.Series([m['decade'] for m in data]).value_counts().sort_index().to_dict()
        }
        
        return stats


def main():
    """Load and validate IMDB dataset"""
    loader = IMDBDataLoader()
    
    # Load full dataset
    data = loader.load_full_dataset(use_cache=False, filter_data=True)
    
    # Get statistics
    stats = loader.get_statistics(data)
    
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total movies: {stats['total_movies']:,}")
    print(f"Year range: {stats['year_range'][0]}-{stats['year_range'][1]}")
    print(f"Unique genres: {stats['genres']}")
    print(f"Avg box office: ${stats['avg_box_office']:,.0f}")
    print(f"Avg runtime: {stats['avg_runtime']:.1f} minutes")
    print(f"Avg plot length: {stats['avg_word_count']:.0f} words")
    print(f"Movies with cast data: {stats['with_cast_data']:,} ({stats['with_cast_data']/stats['total_movies']*100:.1f}%)")
    
    print("\nTop 10 genres:")
    for genre, count in list(stats['genre_distribution'].items())[:10]:
        print(f"  {genre}: {count:,}")
    
    print("\nDecade distribution:")
    for decade, count in stats['decade_distribution'].items():
        print(f"  {decade}s: {count:,}")
    
    # Sample movies
    print("\n" + "="*80)
    print("SAMPLE MOVIES")
    print("="*80)
    for movie in data[:3]:
        print(f"\nTitle: {movie['title']} ({movie['release_year']})")
        print(f"Genres: {', '.join(movie['genres'][:3])}")
        print(f"Box Office: ${movie['box_office_revenue']:,.0f}")
        print(f"Cast: {', '.join(movie['actors'][:5])}")
        print(f"Plot: {movie['plot_summary'][:200]}...")
        print(f"Success Score: {movie['success_score']:.3f}")


if __name__ == '__main__':
    main()

