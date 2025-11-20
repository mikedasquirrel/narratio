"""
Merge Movie Datasets - CMU + IMDB + MovieLens
Strategic merge for maximum coverage with rich features

Creates unified movie dataset:
- ~42K movies total from CMU base
- Enriched with IMDB detailed features
- Enhanced with MovieLens user ratings

Author: Narrative Optimization Framework
Date: November 16, 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MOVIE DATASET AMALGAMATION")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================

print("📂 Loading datasets...")

# Load CMU Movie Summaries (BASE)
cmu_metadata_path = Path('data/MovieSummaries/movie.metadata.tsv')
cmu_plots_path = Path('data/MovieSummaries/plot_summaries.txt')

print("  Loading CMU metadata...")
cmu_metadata = pd.read_csv(
    cmu_metadata_path,
    sep='\t',
    header=None,
    names=['wikipedia_id', 'freebase_id', 'title', 'release_date', 
           'box_office', 'runtime', 'languages', 'countries', 'genres']
)

print("  Loading CMU plot summaries...")
cmu_plots = pd.read_csv(
    cmu_plots_path,
    sep='\t',
    header=None,
    names=['wikipedia_id', 'plot_summary']
)

# Merge CMU data
cmu_complete = cmu_metadata.merge(cmu_plots, on='wikipedia_id', how='left')
print(f"  ✓ CMU: {len(cmu_complete):,} movies")

# Load IMDB Complete
print("  Loading IMDB data...")
with open('data/domains/imdb_movies_complete.json') as f:
    imdb_data = json.load(f)
imdb_df = pd.DataFrame(imdb_data)
print(f"  ✓ IMDB: {len(imdb_df):,} movies")

# Load MovieLens
print("  Loading MovieLens data...")
movielens_movies = pd.read_csv('data/ml-latest-small/movies.csv')
movielens_links = pd.read_csv('data/ml-latest-small/links.csv')
movielens_ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# Compute aggregated ratings
rating_stats = movielens_ratings.groupby('movieId').agg({
    'rating': ['mean', 'std', 'count']
}).reset_index()
rating_stats.columns = ['movieId', 'avg_rating', 'rating_std', 'num_ratings']

movielens_full = movielens_movies.merge(movielens_links, on='movieId', how='left')
movielens_full = movielens_full.merge(rating_stats, on='movieId', how='left')
print(f"  ✓ MovieLens: {len(movielens_full):,} movies")
print()

# ============================================================================
# STEP 2: NORMALIZE & PREPARE FOR JOINING
# ============================================================================

print("🔧 Normalizing data for joins...")

def normalize_title(title):
    """Normalize title for matching"""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    # Remove "the", "a", "an" at start
    title = re.sub(r'^(the|a|an)\s+', '', title)
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    # Remove extra whitespace
    title = ' '.join(title.split())
    return title

def extract_year_from_title(title):
    """Extract year from MovieLens format: 'Title (YYYY)'"""
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None

def extract_year(date_str):
    """Extract year from date string"""
    if pd.isna(date_str):
        return None
    # Try various formats
    try:
        if isinstance(date_str, int):
            return int(date_str)
        date_str = str(date_str)
        # YYYY-MM-DD
        if '-' in date_str:
            return int(date_str.split('-')[0])
        # YYYY
        if len(date_str) == 4 and date_str.isdigit():
            return int(date_str)
    except:
        pass
    return None

# Normalize CMU
cmu_complete['normalized_title'] = cmu_complete['title'].apply(normalize_title)
cmu_complete['year'] = cmu_complete['release_date'].apply(extract_year)

# Normalize IMDB
imdb_df['normalized_title'] = imdb_df['title'].apply(normalize_title)
imdb_df['year'] = imdb_df['release_year']

# Normalize MovieLens
movielens_full['year'] = movielens_full['title'].apply(extract_year_from_title)
movielens_full['title_clean'] = movielens_full['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
movielens_full['normalized_title'] = movielens_full['title_clean'].apply(normalize_title)

print(f"  ✓ Normalized all titles and years")
print()

# ============================================================================
# STEP 3: JOIN DATASETS
# ============================================================================

print("🔗 Joining datasets...")

# JOIN 1: CMU + IMDB (on wikipedia_id - direct match)
print("  Join 1: CMU ↔ IMDB (wikipedia_id)...")
merged = cmu_complete.merge(
    imdb_df,
    on='wikipedia_id',
    how='left',
    suffixes=('_cmu', '_imdb')
)
direct_matches = merged['title_imdb'].notna().sum()
print(f"    ✓ {direct_matches:,} direct matches")

# Preserve normalized_title and year for next join
# After merge, we may have normalized_title_cmu and normalized_title_imdb
# Use the CMU one as base (it's from the larger dataset)
if 'normalized_title_cmu' in merged.columns:
    merged['normalized_title'] = merged['normalized_title_cmu']
if 'year_cmu' in merged.columns:
    merged['year'] = merged['year_cmu']

# JOIN 2: Result + MovieLens (on normalized title + year)
print("  Join 2: Result ↔ MovieLens (title + year)...")
merged = merged.merge(
    movielens_full,
    left_on=['normalized_title', 'year'],
    right_on=['normalized_title', 'year'],
    how='left',
    suffixes=('', '_ml')
)
ml_matches = merged['movieId'].notna().sum()
print(f"    ✓ {ml_matches:,} MovieLens matches")
print()

# ============================================================================
# STEP 4: CREATE UNIFIED SCHEMA
# ============================================================================

print("📋 Creating unified schema...")

def safe_parse_json(val):
    """Safely parse JSON string"""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        # CMU uses Freebase format: {"/m/abc": "Genre Name"}
        if isinstance(val, str) and val.startswith('{'):
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return list(parsed.values())
            return parsed
        return []
    except:
        return []

def merge_genres(cmu_genres, imdb_genres, ml_genres):
    """Merge genres from all sources"""
    all_genres = set()
    
    # CMU genres
    for g in safe_parse_json(cmu_genres):
        all_genres.add(str(g))
    
    # IMDB genres
    if isinstance(imdb_genres, list):
        all_genres.update(imdb_genres)
    
    # MovieLens genres
    if pd.notna(ml_genres):
        all_genres.update(str(ml_genres).split('|'))
    
    return list(all_genres) if all_genres else []

# Build unified records
unified_movies = []

for idx, row in merged.iterrows():
    # Prefer IMDB > CMU for most fields
    title = row['title_imdb'] if pd.notna(row.get('title_imdb')) else row['title_cmu']
    
    # Parse plot
    plot_cmu = row.get('plot_summary')
    plot_imdb = row.get('plot_summary_imdb')
    plot = plot_imdb if pd.notna(plot_imdb) else plot_cmu
    
    # Merge genres
    genres = merge_genres(
        row.get('genres'),
        row.get('genres_imdb'),
        row.get('genres')
    )
    
    # Box office
    box_office = None
    if pd.notna(row.get('box_office_revenue')):
        box_office = float(row['box_office_revenue'])
    elif pd.notna(row.get('box_office')):
        try:
            box_office = float(row['box_office'])
        except:
            pass
    
    # Runtime
    runtime = row.get('runtime_imdb') if pd.notna(row.get('runtime_imdb')) else row.get('runtime')
    
    # Actors/characters from IMDB
    actors = row.get('actors') if isinstance(row.get('actors'), list) else []
    characters = row.get('characters') if isinstance(row.get('characters'), list) else []
    
    # MovieLens ratings
    avg_rating = row.get('avg_rating')
    num_ratings = row.get('num_ratings')
    rating_std = row.get('rating_std')
    
    # Feature flags
    has_plot = pd.notna(plot) and len(str(plot)) > 50
    has_cast = len(actors) > 0
    has_box_office = box_office is not None and box_office > 0
    has_ratings = pd.notna(avg_rating) and num_ratings > 0
    
    # Feature completeness score
    completeness = sum([
        has_plot,
        has_cast,
        has_box_office,
        has_ratings,
        len(genres) > 0,
        pd.notna(runtime)
    ]) / 6.0
    
    # Build full narrative
    narrative_parts = []
    if title:
        narrative_parts.append(f"{title}")
    if row['year']:
        narrative_parts.append(f"released {row['year']}")
    if genres:
        narrative_parts.append(f"{', '.join(genres[:3])} film")
    if actors and len(actors) > 0:
        narrative_parts.append(f"starring {', '.join(actors[:5])}")
    if has_plot:
        narrative_parts.append(f"Plot: {str(plot)[:500]}")
    
    full_narrative = ". ".join(narrative_parts) + "."
    
    # Convert numpy types to Python native types for JSON serialization
    def to_native(val):
        """Convert numpy/pandas types to Python native types"""
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        return val
    
    movie = {
        # IDs
        'movie_id': f"movie_{to_native(row['wikipedia_id'])}",
        'wikipedia_id': to_native(row['wikipedia_id']),
        'freebase_id': to_native(row.get('freebase_id')),
        'imdb_id': f"tt{int(row['imdbId']):07d}" if pd.notna(row.get('imdbId')) else None,
        'movielens_id': to_native(row.get('movieId')),
        
        # Core metadata
        'title': str(title) if title else None,
        'year': to_native(row['year']),
        'genres': genres,
        'runtime': to_native(runtime),
        
        # Narrative text
        'plot_summary': str(plot) if has_plot else None,
        'plot_length': len(str(plot)) if has_plot else 0,
        
        # Cast/Characters
        'actors': actors,
        'characters': characters,
        'num_actors': len(actors),
        'num_female': to_native(row.get('num_female', 0)),
        'num_male': to_native(row.get('num_male', 0)),
        'cast_diversity': to_native(row.get('cast_diversity', 0)),
        
        # Success metrics
        'box_office_revenue': float(box_office) if box_office else None,
        'avg_rating': to_native(avg_rating),
        'num_ratings': to_native(num_ratings),
        'rating_std': to_native(rating_std),
        
        # Derived features
        'has_plot': bool(has_plot),
        'has_cast': bool(has_cast),
        'has_box_office': bool(has_box_office),
        'has_ratings': bool(has_ratings),
        'feature_completeness': float(completeness),
        
        # Full narrative
        'full_narrative': full_narrative
    }
    
    unified_movies.append(movie)
    
    if (idx + 1) % 5000 == 0:
        print(f"    Processed {idx + 1:,} movies...")

print(f"  ✓ Created {len(unified_movies):,} unified movie records")
print()

# ============================================================================
# STEP 5: STATISTICS & SAVE
# ============================================================================

print("📊 Dataset statistics...")

df_unified = pd.DataFrame(unified_movies)

stats = {
    'total_movies': int(len(unified_movies)),
    'with_plot': int(df_unified['has_plot'].sum()),
    'with_cast': int(df_unified['has_cast'].sum()),
    'with_box_office': int(df_unified['has_box_office'].sum()),
    'with_ratings': int(df_unified['has_ratings'].sum()),
    'with_genres': int((df_unified['genres'].apply(len) > 0).sum()),
    'avg_completeness': float(df_unified['feature_completeness'].mean()),
    'rich_movies': int((df_unified['feature_completeness'] >= 0.6).sum()),
    'years_range': [
        int(df_unified['year'].min()) if df_unified['year'].notna().any() else None,
        int(df_unified['year'].max()) if df_unified['year'].notna().any() else None
    ]
}

print(f"  Total movies: {stats['total_movies']:,}")
print(f"  With plot summaries: {stats['with_plot']:,} ({stats['with_plot']/stats['total_movies']*100:.1f}%)")
print(f"  With cast info: {stats['with_cast']:,} ({stats['with_cast']/stats['total_movies']*100:.1f}%)")
print(f"  With box office: {stats['with_box_office']:,} ({stats['with_box_office']/stats['total_movies']*100:.1f}%)")
print(f"  With user ratings: {stats['with_ratings']:,} ({stats['with_ratings']/stats['total_movies']*100:.1f}%)")
print(f"  With genres: {stats['with_genres']:,} ({stats['with_genres']/stats['total_movies']*100:.1f}%)")
print(f"  Rich movies (60%+ complete): {stats['rich_movies']:,}")
print(f"  Average completeness: {stats['avg_completeness']:.1%}")
print(f"  Year range: {stats['years_range'][0]} - {stats['years_range'][1]}")
print()

# Save merged dataset
output_path = Path('data/domains/movies_merged_complete.json')
print(f"💾 Saving to {output_path}...")

output_data = {
    'metadata': {
        'created_at': datetime.now().isoformat(),
        'sources': {
            'cmu': 'CMU Movie Summary Corpus (42,306 movies)',
            'imdb': 'IMDB Complete (6,047 movies)',
            'movielens': 'MovieLens ml-latest-small (9,742 movies)'
        },
        'merge_strategy': 'CMU base with IMDB and MovieLens enrichment',
        'statistics': stats
    },
    'movies': unified_movies
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Saved {len(unified_movies):,} movies")
print()

print("="*80)
print("MERGE COMPLETE")
print("="*80)
print()
print(f"Output: {output_path}")
print(f"Total movies: {len(unified_movies):,}")
print(f"Ready for transformer analysis!")

