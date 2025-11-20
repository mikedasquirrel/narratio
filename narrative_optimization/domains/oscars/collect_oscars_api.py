"""
Oscar Best Picture API Collector

Uses TMDb API to collect ALL nominees with COMPLETE features:
- Full cast with character names
- Director, crew
- Plot summaries
- Settings/locations from keywords
- Production companies
- All nominative elements

Then analyzes: Winner vs nominees in gravitational narrative space
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict

# TMDb API (free, no key needed for basic searches)
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_KEY = "YOUR_KEY_HERE"  # Get free key from themoviedb.org

# Oscar Best Picture nominees by year (verified)
OSCAR_NOMINEES_BY_YEAR = {
    2024: ["Oppenheimer", "Killers of the Flower Moon", "Poor Things", "Barbie", "American Fiction", 
           "Anatomy of a Fall", "The Holdovers", "Maestro", "Past Lives", "The Zone of Interest"],
    2023: ["Everything Everywhere All at Once", "The Banshees of Inisherin", "All Quiet on the Western Front",
           "Avatar: The Way of Water", "Elvis", "The Fabelmans", "Tár", "Top Gun: Maverick", 
           "Triangle of Sadness", "Women Talking"],
    2022: ["CODA", "Belfast", "Don't Look Up", "Drive My Car", "Dune", "King Richard",
           "Licorice Pizza", "Nightmare Alley", "The Power of the Dog", "West Side Story"],
    2021: ["Nomadland", "The Father", "Judas and the Black Messiah", "Mank", "Minari",
           "Promising Young Woman", "Sound of Metal", "The Trial of the Chicago 7"],
    2020: ["Parasite", "1917", "Ford v Ferrari", "The Irishman", "Jojo Rabbit",
           "Joker", "Little Women", "Marriage Story", "Once Upon a Time in Hollywood"]
}

# Mark winners
WINNERS = {
    2024: "Oppenheimer",
    2023: "Everything Everywhere All at Once",
    2022: "CODA",
    2021: "Nomadland",
    2020: "Parasite"
}


class OscarAPICollector:
    """Collects complete Oscar data via TMDb API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "8265bd1679663a7ea12ac168da84d2e8"  # Public demo key
        self.base_url = "https://api.themoviedb.org/3"
        
    def search_movie(self, title: str, year: int) -> Dict:
        """Search for movie and get TMDb ID."""
        url = f"{self.base_url}/search/movie"
        params = {
            "api_key": self.api_key,
            "query": title,
            "year": year
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                return results[0]  # First match
        return None
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Get complete movie details."""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            "api_key": self.api_key,
            "append_to_response": "credits,keywords"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    
    def extract_complete_nominatives(self, movie_data: Dict) -> Dict:
        """Extract ALL nominative elements from movie."""
        
        nominatives = {
            # Film title
            'title': movie_data.get('title', ''),
            'original_title': movie_data.get('original_title', ''),
            
            # Cast (actors + character names)
            'cast': [
                {
                    'actor': person['name'],
                    'character': person.get('character', ''),
                    'order': person.get('order', 99)
                }
                for person in movie_data.get('credits', {}).get('cast', [])[:30]  # Top 30
            ],
            
            # Crew (director, writer, etc.)
            'director': [
                person['name'] 
                for person in movie_data.get('credits', {}).get('crew', [])
                if person.get('job') == 'Director'
            ],
            'writer': [
                person['name']
                for person in movie_data.get('credits', {}).get('crew', [])
                if person.get('job') in ['Writer', 'Screenplay']
            ],
            
            # Settings (from keywords and production countries)
            'keywords': [kw['name'] for kw in movie_data.get('keywords', {}).get('keywords', [])],
            'production_countries': [c['name'] for c in movie_data.get('production_countries', [])],
            
            # Companies
            'production_companies': [c['name'] for c in movie_data.get('production_companies', [])],
            
            # Metadata
            'genres': [g['name'] for g in movie_data.get('genres', [])],
            'runtime': movie_data.get('runtime'),
            'release_date': movie_data.get('release_date', ''),
            'overview': movie_data.get('overview', ''),
            'tagline': movie_data.get('tagline', '')
        }
        
        return nominatives
    
    def collect_year_nominees(self, year: int) -> List[Dict]:
        """Collect all nominees for a year."""
        nominees = OSCAR_NOMINEES_BY_YEAR.get(year, [])
        winner = WINNERS.get(year)
        
        print(f"\nCollecting {year} nominees ({len(nominees)} films)...")
        
        year_data = []
        
        for i, title in enumerate(nominees, 1):
            print(f"  {i}/{len(nominees)}: {title}...", end=' ')
            
            try:
                # Search
                search_result = self.search_movie(title, year)
                if not search_result:
                    print("✗ Not found")
                    continue
                
                movie_id = search_result['id']
                
                # Get details
                details = self.get_movie_details(movie_id)
                if not details:
                    print("✗ No details")
                    continue
                
                # Extract nominatives
                nominatives = self.extract_complete_nominatives(details)
                
                # Add metadata
                nominatives['year'] = year
                nominatives['won_oscar'] = (title == winner)
                nominatives['tmdb_id'] = movie_id
                
                year_data.append(nominatives)
                
                print(f"✓ ({len(nominatives['cast'])} cast, {len(nominatives['keywords'])} keywords)")
                
                time.sleep(0.25)  # Rate limiting
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"✓ Collected {len(year_data)}/{len(nominees)} for {year}")
        
        return year_data
    
    def collect_all_years(self, years: List[int]) -> Dict:
        """Collect all nominees for multiple years."""
        print("=" * 80)
        print("OSCAR BEST PICTURE - API COLLECTION")
        print("=" * 80)
        
        all_data = {}
        
        for year in years:
            year_data = self.collect_year_nominees(year)
            all_data[year] = year_data
        
        total_films = sum(len(films) for films in all_data.values())
        
        print(f"\n" + "=" * 80)
        print(f"COLLECTION COMPLETE")
        print(f"=" * 80)
        print(f"Total years: {len(years)}")
        print(f"Total films: {total_films}")
        
        return all_data


def main():
    """Collect complete Oscar data via API."""
    print("\nCollecting real Oscar data with complete nominative extraction...")
    print("Using TMDb API for full cast, characters, keywords, etc.\n")
    
    collector = OscarAPICollector()
    
    # Collect recent 5 years (can expand to 10+)
    years = [2024, 2023, 2022, 2021, 2020]
    
    data = collector.collect_all_years(years)
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / 'data/domains/oscar_nominees_complete.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved complete data: {output_path}")
    print(f"\nReady for analysis:")
    print(f"  - Winner vs nominees each year")
    print(f"  - Complete nominative breakdown")
    print(f"  - Gravitational analysis of competitive field")


if __name__ == "__main__":
    main()

