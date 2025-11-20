"""
Enrich Novels Dataset with Real Data

Replaces synthetic/template narratives with real plot summaries, character names,
and genres from Open Library, Wikipedia, and Google Books APIs.

Author: Narrative Framework System
Date: November 2025
"""

import json
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests

# Import API libraries
try:
    from olclient.openlibrary import OpenLibrary
except ImportError:
    print("Warning: openlibrary-client not found, trying alternate import")
    try:
        from openlibrary_client import OpenLibrary
    except:
        OpenLibrary = None

import wikipedia

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enrichment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Common words to filter from character names
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'about', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
    'just', 'should', 'now', 'new', 'york', 'times', 'first', 'one', 'two',
    'year', 'years', 'book', 'novel', 'story', 'chapter', 'page', 'author',
    'dr', 'mr', 'mrs', 'ms', 'miss', 'sir', 'lord', 'lady'
}


class NovelsEnricher:
    """Enriches novels dataset with real data from multiple APIs."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = self.dataset_path.parent / 'enrichment_checkpoint.json'
        self.novels = []
        self.processed_count = 0
        self.success_count = 0
        self.ol = None
        
        # Initialize Open Library client
        try:
            self.ol = OpenLibrary()
            logger.info("Open Library client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Open Library: {e}")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load novels dataset from JSON file."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.novels = json.load(f)
        logger.info(f"Loaded {len(self.novels)} novels")
        return self.novels
    
    def save_checkpoint(self, index: int):
        """Save checkpoint to resume processing."""
        checkpoint = {
            'last_processed_index': index,
            'processed_count': self.processed_count,
            'success_count': self.success_count
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        logger.info(f"Checkpoint saved at index {index}")
    
    def load_checkpoint(self) -> int:
        """Load checkpoint to resume processing."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Resuming from checkpoint at index {checkpoint['last_processed_index']}")
            return checkpoint['last_processed_index'] + 1
        return 0
    
    def fetch_from_openlibrary(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Fetch book data from Open Library."""
        if not self.ol:
            return None
        
        try:
            # Search for book
            search_query = f"{title} {author}"
            results = self.ol.search(search_query)
            
            if not results or not hasattr(results, 'docs') or not results.docs:
                return None
            
            # Get first result
            book = results.docs[0]
            
            data = {}
            
            # Extract description
            if hasattr(book, 'first_sentence') and book.first_sentence:
                data['description'] = book.first_sentence[0] if isinstance(book.first_sentence, list) else str(book.first_sentence)
            
            # Extract subjects as genres
            if hasattr(book, 'subject') and book.subject:
                data['genres'] = book.subject[:10]  # Limit to 10
            
            # Get work details if available
            if hasattr(book, 'key'):
                try:
                    work = self.ol.Work.get(book.key)
                    if hasattr(work, 'description'):
                        desc = work.description
                        if isinstance(desc, dict) and 'value' in desc:
                            data['description'] = desc['value']
                        elif isinstance(desc, str):
                            data['description'] = desc
                except:
                    pass
            
            logger.debug(f"Open Library found data for: {title}")
            return data if data else None
            
        except Exception as e:
            logger.debug(f"Open Library error for {title}: {e}")
            return None
    
    def fetch_from_wikipedia(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Fetch book data from Wikipedia."""
        try:
            # Try different search queries
            search_queries = [
                f"{title} (novel)",
                f"{title} ({author})",
                title
            ]
            
            for query in search_queries:
                try:
                    # Search for page
                    page = wikipedia.page(query, auto_suggest=False)
                    summary = page.summary
                    
                    # Only return if summary is substantial
                    if len(summary) > 200:
                        data = {
                            'description': summary,
                            'wikipedia_url': page.url
                        }
                        logger.debug(f"Wikipedia found data for: {title}")
                        return data
                        
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try first option
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        summary = page.summary
                        if len(summary) > 200 and (title.lower() in summary.lower() or author.lower() in summary.lower()):
                            return {
                                'description': summary,
                                'wikipedia_url': page.url
                            }
                    except:
                        continue
                        
                except wikipedia.exceptions.PageError:
                    continue
                    
            return None
            
        except Exception as e:
            logger.debug(f"Wikipedia error for {title}: {e}")
            return None
    
    def fetch_from_google_books(self, title: str, author: str) -> Optional[Dict[str, Any]]:
        """Fetch book data from Google Books API."""
        try:
            # Build query
            query = f"intitle:{title} inauthor:{author}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if 'items' not in data or not data['items']:
                return None
            
            # Get first result
            volume = data['items'][0]['volumeInfo']
            
            result = {}
            
            # Extract description
            if 'description' in volume:
                result['description'] = volume['description']
            
            # Extract categories as genres
            if 'categories' in volume:
                result['genres'] = volume['categories']
            
            # Extract publisher info
            if 'publisher' in volume:
                result['publisher'] = volume['publisher']
            
            if 'publishedDate' in volume:
                result['published_date'] = volume['publishedDate']
            
            # Extract page count
            if 'pageCount' in volume:
                result['page_count'] = volume['pageCount']
            
            logger.debug(f"Google Books found data for: {title}")
            return result if result else None
            
        except Exception as e:
            logger.debug(f"Google Books error for {title}: {e}")
            return None
    
    def extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text using pattern matching."""
        if not text:
            return []
        
        # Find capitalized words (potential names)
        # Pattern: Capital letter followed by lowercase letters, possibly with spaces
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)
        
        # Filter out common words and short names
        character_names = []
        seen = set()
        
        for match in matches:
            # Clean up
            name = match.strip()
            name_lower = name.lower()
            
            # Skip if common word, too short, or already seen
            if (len(name) < 3 or 
                name_lower in COMMON_WORDS or 
                name_lower in seen or
                name.lower() == 'the' or
                name.lower() == 'a'):
                continue
            
            # Add to results
            character_names.append(name)
            seen.add(name_lower)
            
            # Limit to reasonable number
            if len(character_names) >= 20:
                break
        
        return character_names
    
    def merge_data_sources(self, ol_data: Optional[Dict], wiki_data: Optional[Dict], 
                          gb_data: Optional[Dict]) -> Dict[str, Any]:
        """Merge data from multiple sources intelligently."""
        merged = {}
        
        # Collect all descriptions
        descriptions = []
        if wiki_data and 'description' in wiki_data:
            descriptions.append(('wikipedia', wiki_data['description']))
        if gb_data and 'description' in gb_data:
            descriptions.append(('google_books', gb_data['description']))
        if ol_data and 'description' in ol_data:
            descriptions.append(('openlibrary', ol_data['description']))
        
        # Choose best description (longest non-template one)
        best_description = None
        best_length = 0
        template_phrases = ['celebrated work of fiction', 'explores themes', 'compelling narrative']
        
        for source, desc in descriptions:
            # Check if it's not a template
            is_template = any(phrase in desc.lower() for phrase in template_phrases)
            if not is_template and len(desc) > best_length:
                best_description = desc
                best_length = len(desc)
        
        # If no non-template found, use longest
        if not best_description and descriptions:
            best_description = max(descriptions, key=lambda x: len(x[1]))[1]
        
        if best_description:
            merged['plot_summary'] = best_description
            merged['full_narrative'] = best_description
        
        # Merge genres from all sources
        genres = []
        for data in [ol_data, wiki_data, gb_data]:
            if data and 'genres' in data:
                if isinstance(data['genres'], list):
                    genres.extend(data['genres'])
                else:
                    genres.append(str(data['genres']))
        
        # Clean and deduplicate genres
        genres = list(set([g.strip() for g in genres if g and len(str(g).strip()) > 0]))
        merged['genres'] = genres[:15]  # Limit to 15
        
        # Add other metadata
        if gb_data:
            if 'publisher' in gb_data:
                merged['publisher'] = gb_data['publisher']
            if 'page_count' in gb_data:
                merged['page_count'] = gb_data['page_count']
        
        if wiki_data and 'wikipedia_url' in wiki_data:
            merged['wikipedia_url'] = wiki_data['wikipedia_url']
        
        return merged
    
    def enrich_novel(self, novel: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Enrich a single novel with real data."""
        title = novel.get('title', '')
        author = novel.get('author', '')
        
        logger.info(f"Processing: {title} by {author}")
        
        # Fetch from multiple sources
        ol_data = self.fetch_from_openlibrary(title, author)
        time.sleep(0.5)  # Rate limiting
        
        wiki_data = self.fetch_from_wikipedia(title, author)
        time.sleep(0.5)  # Rate limiting
        
        gb_data = self.fetch_from_google_books(title, author)
        time.sleep(0.5)  # Rate limiting
        
        # Check if we got any real data
        has_data = bool(ol_data or wiki_data or gb_data)
        
        if has_data:
            # Merge data from sources
            merged_data = self.merge_data_sources(ol_data, wiki_data, gb_data)
            
            # Update novel with real data
            if 'plot_summary' in merged_data:
                novel['plot_summary'] = merged_data['plot_summary']
                
                # Extract character names from plot summary
                character_names = self.extract_character_names(merged_data['plot_summary'])
                novel['character_names'] = character_names
            
            if 'full_narrative' in merged_data:
                novel['full_narrative'] = merged_data['full_narrative']
            
            if 'genres' in merged_data:
                novel['genres'] = merged_data['genres']
            
            if 'publisher' in merged_data:
                novel['publisher'] = merged_data['publisher']
            
            if 'page_count' in merged_data:
                novel['page_count'] = merged_data['page_count']
            
            if 'wikipedia_url' in merged_data:
                novel['wikipedia_url'] = merged_data['wikipedia_url']
            
            # Mark as enriched
            novel['data_enriched'] = True
            novel['enrichment_sources'] = []
            if ol_data:
                novel['enrichment_sources'].append('openlibrary')
            if wiki_data:
                novel['enrichment_sources'].append('wikipedia')
            if gb_data:
                novel['enrichment_sources'].append('google_books')
            
            logger.info(f"✓ Successfully enriched: {title}")
            return novel, True
        else:
            logger.warning(f"✗ No data found for: {title}")
            novel['data_enriched'] = False
            return novel, False
    
    def enrich_all(self, start_index: int = 0):
        """Enrich all novels in the dataset."""
        logger.info("="*80)
        logger.info("STARTING NOVELS DATASET ENRICHMENT")
        logger.info("="*80)
        logger.info(f"Total novels to process: {len(self.novels)}")
        logger.info(f"Starting from index: {start_index}")
        
        for i in range(start_index, len(self.novels)):
            self.novels[i], success = self.enrich_novel(self.novels[i])
            self.processed_count += 1
            if success:
                self.success_count += 1
            
            # Save checkpoint every 50 books
            if (i + 1) % 50 == 0:
                self.save_dataset()
                self.save_checkpoint(i)
                logger.info(f"Progress: {i + 1}/{len(self.novels)} ({self.success_count} successful)")
        
        # Final save
        self.save_dataset()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        
        logger.info("="*80)
        logger.info("ENRICHMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Total processed: {self.processed_count}")
        logger.info(f"Successfully enriched: {self.success_count}")
        logger.info(f"Success rate: {100 * self.success_count / self.processed_count:.1f}%")
    
    def save_dataset(self):
        """Save enriched dataset back to file."""
        logger.info(f"Saving dataset to {self.dataset_path}")
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.novels, f, indent=2, ensure_ascii=False)
        logger.info("Dataset saved successfully")


def main():
    """Main enrichment function."""
    # Path to dataset
    dataset_path = Path(__file__).parent / 'data' / 'novels_dataset.json'
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    # Create enricher
    enricher = NovelsEnricher(str(dataset_path))
    
    # Load dataset
    enricher.load_dataset()
    
    # Check for checkpoint
    start_index = enricher.load_checkpoint()
    
    # Enrich all novels
    enricher.enrich_all(start_index=start_index)
    
    logger.info("All done!")


if __name__ == '__main__':
    main()

