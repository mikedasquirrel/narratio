"""
Nonfiction Data Loader

Loads nonfiction dataset and extracts:
- Narrative text (descriptions, excerpts)
- Nominatives (author names, book titles, subject names)
- Key figures mentioned in the book
- Topics and themes
- Multi-task outcomes
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class NonfictionDataLoader:
    """
    Loads and processes nonfiction dataset with comprehensive nominative extraction.
    
    Extracts:
    - Author names
    - Book titles
    - Key figures mentioned (historical figures, subjects)
    - Topics and themes
    - Full narrative text
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize nonfiction data loader."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        self.dataset_path = self.data_dir / 'nonfiction_dataset.json'
    
    def load_full_dataset(self, use_cache: bool = True, filter_data: bool = False) -> List[Dict[str, Any]]:
        """Load full nonfiction dataset with all nominative information."""
        print(f"Loading nonfiction dataset from {self.dataset_path}...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print(f"✓ Loaded {len(raw_data)} raw nonfiction books")
        
        # Process each book
        processed_books = []
        for book in raw_data:
            processed = self._process_book(book)
            if processed:
                processed_books.append(processed)
        
        print(f"✓ Processed {len(processed_books)} nonfiction books")
        
        return processed_books
    
    def _process_book(self, book: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single nonfiction book."""
        try:
            # Extract nominatives
            nominatives = self._extract_nominatives(book)
            
            # Extract key figures
            key_figures = self._extract_key_figures(book, nominatives)
            
            # Create full narrative
            full_narrative = self._create_full_narrative(book)
            
            # Create outcomes
            outcomes = self._create_outcomes(book)
            
            # Combine all information
            processed = {
                # Core identification
                'title': book.get('title', ''),
                'author': book.get('author', ''),
                'publication_year': book.get('publication_year'),
                
                # Narrative text
                'description': book.get('description', ''),
                'full_narrative': full_narrative,
                
                # Nominatives
                'author_name': book.get('author', ''),
                'book_title': book.get('title', ''),
                'key_figures': key_figures.get('names', []),
                'all_nominatives': nominatives.get('all_names', []),
                
                # Topics and themes
                'genres': book.get('genres', []),
                'topics': self._extract_topics(book),
                
                # Metadata
                'awards': book.get('awards', []),
                'won_major_award': book.get('won_major_award', False),
                'is_bestseller': book.get('is_bestseller', False),
                'on_best_list': book.get('on_best_list', False),
                
                # Outcomes
                'ratings': outcomes['ratings'],
                'awards_binary': outcomes['awards_binary'],
                'bestseller_binary': outcomes['bestseller_binary'],
                'critical_acclaim': outcomes['critical_acclaim'],
                'sales': outcomes['sales'],
                
                # For framework analysis
                'success_score': outcomes['composite_score'],
                'book_type': 'nonfiction'
            }
            
            return processed
            
        except Exception as e:
            print(f"  ⚠️  Error processing book {book.get('title', 'Unknown')}: {e}")
            return None
    
    def _extract_nominatives(self, book: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all nominative elements."""
        nominatives = {
            'author_name': book.get('author', ''),
            'book_title': book.get('title', ''),
            'all_names': []
        }
        
        # Add author name
        if nominatives['author_name']:
            nominatives['all_names'].append(nominatives['author_name'])
        
        # Extract names from title
        title_words = re.findall(r'\b[A-Z][a-z]+\b', nominatives['book_title'])
        nominatives['all_names'].extend(title_words)
        
        # Extract names from description
        description = book.get('description', '') + ' ' + book.get('full_narrative', '')
        extracted_names = self._extract_names_from_text(description)
        nominatives['all_names'].extend(extracted_names)
        
        # Remove duplicates
        seen = set()
        unique_names = []
        for name in nominatives['all_names']:
            name_lower = name.lower()
            if name_lower not in seen and len(name) > 1:
                seen.add(name_lower)
                unique_names.append(name)
        
        nominatives['all_names'] = unique_names
        
        return nominatives
    
    def _extract_names_from_text(self, text: str) -> List[str]:
        """Extract names from text (historical figures, subjects, etc.)."""
        if not text:
            return []
        
        # Capitalized words (potential names)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Title patterns
        title_patterns = re.findall(r'\b(?:Mr|Mrs|Ms|Dr|Professor|President|General|Captain)\.?\s+([A-Z][a-z]+)', text)
        
        all_candidates = capitalized_words + title_patterns
        
        # Filter common words
        skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those'}
        
        word_counts = {}
        for word in all_candidates:
            if word.lower() in skip_words or len(word) < 2:
                continue
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return names appearing at least 2 times
        names = [name for name, count in word_counts.items() if count >= 2]
        names = sorted(names, key=lambda x: word_counts[x], reverse=True)[:15]
        
        return names
    
    def _extract_key_figures(self, book: Dict[str, Any], nominatives: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key figures mentioned in the book."""
        return {
            'names': nominatives.get('all_names', [])[:10]  # Top 10 most mentioned
        }
    
    def _extract_topics(self, book: Dict[str, Any]) -> List[str]:
        """Extract topics and themes."""
        topics = book.get('genres', [])
        
        # Add topics from description
        description = book.get('description', '').lower()
        topic_keywords = {
            'history': ['history', 'historical', 'past', 'ancient', 'century'],
            'science': ['science', 'scientific', 'research', 'study', 'discovery'],
            'psychology': ['psychology', 'mental', 'mind', 'behavior', 'cognitive'],
            'business': ['business', 'company', 'corporate', 'management', 'strategy'],
            'politics': ['politics', 'political', 'government', 'policy', 'democracy'],
            'biography': ['biography', 'life', 'memoir', 'autobiography', 'story'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in description for keyword in keywords):
                if topic not in topics:
                    topics.append(topic)
        
        return topics
    
    def _create_full_narrative(self, book: Dict[str, Any]) -> str:
        """Create comprehensive narrative text."""
        parts = []
        
        if book.get('description'):
            parts.append(book['description'])
        
        if book.get('full_narrative'):
            parts.append(book['full_narrative'])
        
        return ' '.join(parts)
    
    def _create_outcomes(self, book: Dict[str, Any]) -> Dict[str, Any]:
        """Create multi-task outcome variables."""
        outcomes = {
            'ratings': book.get('average_rating'),
            'awards_binary': 1 if book.get('won_major_award') else 0,
            'bestseller_binary': 1 if book.get('is_bestseller') else 0,
            'critical_acclaim': book.get('critical_acclaim_score', 0.0),
            'sales': book.get('estimated_sales'),
            'composite_score': 0.0
        }
        
        # Create composite score
        scores = []
        if outcomes['ratings']:
            scores.append((outcomes['ratings'] - 2.0) / 3.0)
        if outcomes['awards_binary']:
            scores.append(1.0)
        if outcomes['bestseller_binary']:
            scores.append(0.8)
        if outcomes['critical_acclaim']:
            scores.append(outcomes['critical_acclaim'])
        
        outcomes['composite_score'] = np.mean(scores) if scores else 0.5
        
        return outcomes


def main():
    """Test data loader."""
    loader = NonfictionDataLoader()
    books = loader.load_full_dataset()
    
    print(f"\n✓ Loaded {len(books)} nonfiction books")
    if books:
        sample = books[0]
        print(f"\nSample book: {sample['title']}")
        print(f"  Author: {sample['author_name']}")
        print(f"  Key figures: {len(sample['key_figures'])}")
        print(f"  Topics: {', '.join(sample.get('topics', [])[:5])}")


if __name__ == '__main__':
    main()

