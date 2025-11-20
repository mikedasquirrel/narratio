"""
Google Ngrams Collector for Temporal Linguistics Analysis

Collects word frequency data from 1500-2024 to test if "history rhymes"
at predictable intervals (25, 50, 75 years).
"""

import requests
import time
import json
from pathlib import Path
from typing import List, Dict
import numpy as np


class NgramsCollector:
    """Collect word frequency data from Google Ngrams."""
    
    def __init__(self, use_comprehensive: bool = False):
        """Initialize collector with word categories.
        
        Args:
            use_comprehensive: If True, use 553-word comprehensive vocabulary
        """
        
        if use_comprehensive:
            # Load comprehensive vocabulary from file
            vocab_path = Path(__file__).parent / 'comprehensive_vocabulary.py'
            if vocab_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("comprehensive_vocabulary", vocab_path)
                vocab_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vocab_module)
                self.word_categories = vocab_module.COMPREHENSIVE_VOCABULARY
            else:
                print(f"Error: {vocab_path} not found, using demo set")
                self.word_categories = self._get_demo_categories()
        else:
            # Original smaller set for quick testing
            self.word_categories = self._get_demo_categories()
        
        self.all_words = []
        for category in self.word_categories.values():
            self.all_words.extend(category['words'])
        
        # Remove duplicates
        self.all_words = list(set(self.all_words))
    
    def _get_demo_categories(self):
        """Get demo word categories for quick testing."""
        return {
            'war_vocabulary': {
                'words': ['battle', 'conflict', 'trench', 'tank', 'drone', 'insurgent',
                         'warfare', 'combat', 'siege', 'artillery', 'bayonet', 'missile'],
                'hypothesis': 'H2_crisis_rhyming',
                'expected_cycle': 75,  # years
                'description': 'War-related terms peak during major conflicts'
            },
            
            'economic_terms': {
                'words': ['speculation', 'bubble', 'crash', 'prosperity', 'depression',
                         'inflation', 'recession', 'boom', 'panic', 'crisis', 'recovery'],
                'hypothesis': 'H2_crisis_rhyming',
                'expected_cycle': 25,  # Economic cycles shorter
                'description': 'Economic terms peak during financial events'
            },
            
            'technology_words': {
                'words': ['wire', 'tube', 'chip', 'web', 'cloud', 'network',
                         'digital', 'virtual', 'cyber', 'algorithm', 'data'],
                'hypothesis': 'H3_tech_innovation',
                'expected_cycle': 30,
                'description': 'Tech metaphors cycle with innovation waves'
            },
            
            'approval_slang': {
                'words': ['groovy', 'rad', 'cool', 'dope', 'lit', 'fire',
                         'neat', 'swell', 'keen', 'hip', 'sick'],
                'hypothesis': 'H1_generation_cycle',
                'expected_cycle': 25,
                'description': 'Slang approval words cycle generationally'
            },
            
            'victorian_terms': {
                'words': ['splendid', 'capital', 'dreadful', 'frightful', 'proper',
                         'ghastly', 'beastly', 'jolly', 'queer', 'fancy'],
                'hypothesis': 'H4_victorian_revival',
                'expected_cycle': 120,
                'description': 'Victorian words may be reviving after 100+ years'
            },
            
            'emotion_words': {
                'words': ['anxiety', 'melancholy', 'despair', 'jubilation', 'contentment',
                         'anguish', 'ecstasy', 'dread', 'bliss', 'terror'],
                'hypothesis': 'general_cyclicity',
                'expected_cycle': None,
                'description': 'Emotion words may reflect cultural mood cycles'
            }
        }
    
    def fetch_ngram(self, word: str, start_year: int = 1500, end_year: int = 2019) -> Dict:
        """
        Fetch frequency data for a word from Google Ngrams.
        
        Args:
            word: Word to analyze
            start_year: Start year
            end_year: End year (Ngrams only goes to 2019)
            
        Returns:
            Dictionary with frequency time series
        """
        # Google Ngrams Viewer API (unofficial but works)
        url = "https://books.google.com/ngrams/json"
        
        params = {
            'content': word,
            'year_start': start_year,
            'year_end': end_year,
            'corpus': '26',  # English 2019 corpus
            'smoothing': '3'  # Light smoothing
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                timeseries = data[0]['timeseries']
                
                # Create year-value pairs
                years = list(range(start_year, end_year + 1))
                
                return {
                    'word': word,
                    'years': years,
                    'frequencies': timeseries,
                    'start_year': start_year,
                    'end_year': end_year,
                    'success': True
                }
            else:
                return {'word': word, 'success': False, 'error': 'No data returned'}
                
        except Exception as e:
            return {'word': word, 'success': False, 'error': str(e)}
    
    def collect_all_words(self, delay: float = 1.0) -> List[Dict]:
        """
        Collect ngram data for all words.
        
        Args:
            delay: Seconds between requests (rate limiting)
            
        Returns:
            List of word frequency dictionaries
        """
        print(f"\n{'='*80}")
        print("GOOGLE NGRAMS COLLECTION - Temporal Linguistics")
        print(f"{'='*80}\n")
        
        print(f"Collecting frequency data for {len(self.all_words)} words")
        print(f"Time span: 1500-2019 (520 years)\n")
        
        word_data = []
        
        total = len(self.all_words)
        for idx, word in enumerate(self.all_words, 1):
            print(f"[{idx:3d}/{total}] Fetching '{word:20s}'...", end=" ")
            
            result = self.fetch_ngram(word)
            
            if result['success']:
                word_data.append(result)
                print(f"✓ ({len(result['frequencies'])} data points)")
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")
            
            # Rate limiting
            time.sleep(delay)
            
            # Progress indicator every 10 words
            if idx % 10 == 0:
                print(f"      Progress: {idx/total*100:.1f}% complete")
        
        print(f"\n✓ Successfully collected {len(word_data)}/{total} words")
        
        return word_data
    
    def save_data(self, word_data: List[Dict], output_path: Path = None):
        """Save collected word data."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'temporal_linguistics' / 'word_frequencies.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output = {
            'collection_date': time.strftime('%Y-%m-%d'),
            'total_words': len(word_data),
            'time_span': '1500-2019',
            'categories': self.word_categories,
            'words': word_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved {len(word_data)} words to: {output_path}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("COLLECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total words: {len(word_data)}")
        print(f"Time span: 1500-2019 (520 years)")
        print(f"Data points per word: {len(word_data[0]['frequencies']) if word_data else 0}")
        
        # Category breakdown
        print(f"\nBy category:")
        for cat_name, cat_data in self.word_categories.items():
            cat_words = [w for w in word_data if w['word'] in cat_data['words']]
            print(f"  {cat_name:20s} {len(cat_words)} words")


def main():
    """Run Google Ngrams collection."""
    import sys
    
    # Use comprehensive vocabulary if requested
    use_comprehensive = '--comprehensive' in sys.argv or '-c' in sys.argv
    
    collector = NgramsCollector(use_comprehensive=use_comprehensive)
    
    print(f"\n⚠️  This will make {len(collector.all_words)} API requests to Google Ngrams")
    print(f"⚠️  With 1s delay, this takes ~{len(collector.all_words)/60:.0f} minutes\n")
    
    word_data = collector.collect_all_words(delay=1.0)
    collector.save_data(word_data)
    
    print(f"\n{'='*80}")
    print("✓ Google Ngrams collection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

