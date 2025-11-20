"""
Enrich Nonfiction Dataset with Real Data

Replaces synthetic/template descriptions with real summaries, subject categorization,
and metadata from Open Library, Wikipedia, and Google Books APIs.

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

# Nonfiction subject categories
NONFICTION_CATEGORIES = {
    'memoir': ['memoir', 'autobiography', 'biography', 'personal', 'life story'],
    'history': ['history', 'historical', 'war', 'military', 'ancient', 'medieval'],
    'science': ['science', 'biology', 'physics', 'chemistry', 'astronomy', 'nature'],
    'psychology': ['psychology', 'mental', 'brain', 'mind', 'cognitive'],
    'social_science': ['sociology', 'anthropology', 'culture', 'society', 'politics'],
    'business': ['business', 'management', 'leadership', 'entrepreneurship', 'startup'],
    'self_help': ['self-help', 'self help', 'improvement', 'motivation', 'success'],
    'true_crime': ['crime', 'murder', 'criminal', 'detective', 'investigation'],
    'journalism': ['journalism', 'investigative', 'reporter', 'news'],
    'economics': ['economics', 'economy', 'financial', 'money', 'wealth'],
    'philosophy': ['philosophy', 'philosophical', 'ethics', 'moral'],
    'religion': ['religion', 'spiritual', 'faith', 'theology'],
    'food': ['food', 'cooking', 'culinary', 'recipe', 'nutrition'],
    'travel': ['travel', 'journey', 'adventure', 'exploration']
}


class NonfictionEnricher:
    """Enriches nonfiction dataset with real data from multiple APIs."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = self.dataset_path.parent / 'enrichment_checkpoint.json'
        self.books = []
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
        """Load nonfiction dataset from JSON file."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.books = json.load(f)
        logger.info(f"Loaded {len(self.books)} nonfiction books")
        return self.books
    
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
            
            # Extract subjects
            if hasattr(book, 'subject') and book.subject:
                data['subjects'] = book.subject[:15]  # Limit to 15
            
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
                f"{title} (book)",
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
            
            # Extract categories
            if 'categories' in volume:
                result['categories'] = volume['categories']
            
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
    
    def categorize_subject(self, text: str, subjects: List[str] = None) -> List[str]:
        """Categorize nonfiction book by subject."""
        categories = []
        text_lower = text.lower() if text else ''
        
        # Check text against category keywords
        for category, keywords in NONFICTION_CATEGORIES.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)
        
        # Also check explicit subjects
        if subjects:
            for subject in subjects:
                subject_lower = subject.lower()
                for category, keywords in NONFICTION_CATEGORIES.items():
                    if any(keyword in subject_lower for keyword in keywords):
                        if category not in categories:
                            categories.append(category)
        
        return categories
    
    def assess_narrative_balance(self, text: str) -> Dict[str, float]:
        """Assess narrative vs expository balance in text."""
        if not text or len(text) < 100:
            return {'narrative_score': 0.5, 'expository_score': 0.5}
        
        text_lower = text.lower()
        
        # Narrative indicators
        narrative_words = ['story', 'narrative', 'tells', 'recounts', 'describes', 
                          'journey', 'experience', 'personal', 'life', 'lived']
        narrative_count = sum(1 for word in narrative_words if word in text_lower)
        
        # Expository indicators
        expository_words = ['analysis', 'examines', 'explores', 'investigates', 
                           'research', 'study', 'theory', 'argues', 'evidence', 'data']
        expository_count = sum(1 for word in expository_words if word in text_lower)
        
        # Calculate scores
        total = narrative_count + expository_count
        if total == 0:
            return {'narrative_score': 0.5, 'expository_score': 0.5}
        
        narrative_score = narrative_count / total
        expository_score = expository_count / total
        
        return {
            'narrative_score': round(narrative_score, 3),
            'expository_score': round(expository_score, 3)
        }
    
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
        template_phrases = ['significant work of nonfiction', 'explores important themes', 
                           'provides insightful analysis', 'compelling narrative and thorough research']
        
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
            merged['description'] = best_description
            merged['full_narrative'] = best_description
        
        # Merge subjects/categories
        subjects = []
        if ol_data and 'subjects' in ol_data:
            subjects.extend(ol_data['subjects'])
        if gb_data and 'categories' in gb_data:
            subjects.extend(gb_data['categories'])
        
        # Clean and deduplicate subjects
        subjects = list(set([s.strip() for s in subjects if s and len(str(s).strip()) > 0]))
        merged['subjects'] = subjects[:20]  # Limit to 20
        
        # Add other metadata
        if gb_data:
            if 'publisher' in gb_data:
                merged['publisher'] = gb_data['publisher']
            if 'page_count' in gb_data:
                merged['page_count'] = gb_data['page_count']
        
        if wiki_data and 'wikipedia_url' in wiki_data:
            merged['wikipedia_url'] = wiki_data['wikipedia_url']
        
        return merged
    
    def enrich_book(self, book: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Enrich a single nonfiction book with real data."""
        title = book.get('title', '')
        author = book.get('author', '')
        
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
            
            # Update book with real data
            if 'description' in merged_data:
                book['description'] = merged_data['description']
                
                # Categorize subject
                categories = self.categorize_subject(
                    merged_data['description'], 
                    merged_data.get('subjects', [])
                )
                book['subject_categories'] = categories
                
                # Assess narrative balance
                balance = self.assess_narrative_balance(merged_data['description'])
                book['narrative_balance'] = balance
            
            if 'full_narrative' in merged_data:
                book['full_narrative'] = merged_data['full_narrative']
            
            if 'subjects' in merged_data:
                book['subjects'] = merged_data['subjects']
            
            if 'publisher' in merged_data:
                book['publisher'] = merged_data['publisher']
            
            if 'page_count' in merged_data:
                book['page_count'] = merged_data['page_count']
            
            if 'wikipedia_url' in merged_data:
                book['wikipedia_url'] = merged_data['wikipedia_url']
            
            # Mark as enriched
            book['data_enriched'] = True
            book['enrichment_sources'] = []
            if ol_data:
                book['enrichment_sources'].append('openlibrary')
            if wiki_data:
                book['enrichment_sources'].append('wikipedia')
            if gb_data:
                book['enrichment_sources'].append('google_books')
            
            logger.info(f"✓ Successfully enriched: {title}")
            return book, True
        else:
            logger.warning(f"✗ No data found for: {title}")
            book['data_enriched'] = False
            return book, False
    
    def enrich_all(self, start_index: int = 0):
        """Enrich all books in the dataset."""
        logger.info("="*80)
        logger.info("STARTING NONFICTION DATASET ENRICHMENT")
        logger.info("="*80)
        logger.info(f"Total books to process: {len(self.books)}")
        logger.info(f"Starting from index: {start_index}")
        
        for i in range(start_index, len(self.books)):
            self.books[i], success = self.enrich_book(self.books[i])
            self.processed_count += 1
            if success:
                self.success_count += 1
            
            # Save checkpoint every 50 books
            if (i + 1) % 50 == 0:
                self.save_dataset()
                self.save_checkpoint(i)
                logger.info(f"Progress: {i + 1}/{len(self.books)} ({self.success_count} successful)")
        
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
            json.dump(self.books, f, indent=2, ensure_ascii=False)
        logger.info("Dataset saved successfully")


def main():
    """Main enrichment function."""
    # Path to dataset
    dataset_path = Path(__file__).parent / 'data' / 'nonfiction_dataset.json'
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    # Create enricher
    enricher = NonfictionEnricher(str(dataset_path))
    
    # Load dataset
    enricher.load_dataset()
    
    # Check for checkpoint
    start_index = enricher.load_checkpoint()
    
    # Enrich all books
    enricher.enrich_all(start_index=start_index)
    
    logger.info("All done!")


if __name__ == '__main__':
    main()

