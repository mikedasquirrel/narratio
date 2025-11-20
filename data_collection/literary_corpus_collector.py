"""
Literary & Documentary Narrative Corpus Collector

Collects large-scale corpus of traditional narratives:
- Novels (Gutenberg, Google Books)
- Short stories
- Biographies (Wikipedia)
- Documentaries (transcripts/descriptions)
- Historical narratives
- Literary works across eras

CRITICAL:
- Preserve COMPLETE texts (not summaries)
- Maintain sequential structure
- Keep paragraph spacing
- No categorization (let AI discover)

Target: 10K-50K complete narratives

Author: Narrative Optimization Framework
Date: November 2025
"""

import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Wikipedia API for biographies and summaries
# Gutenberg API for public domain literature
# All free, no API keys needed


class GutenbergCollector:
    """
    Collect public domain literature from Project Gutenberg.
    
    Available: 60,000+ books
    Target: 5,000-10,000 novels and short story collections
    """
    
    def __init__(self, output_dir: str = 'data/literary_corpus/gutenberg'):
        """Initialize collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = 'https://gutendex.com/books/'
        
    def collect_literature(
        self,
        target_count: int = 5000,
        languages: List[str] = ['en'],
        min_downloads: int = 100  # Popular books = culturally successful
    ):
        """
        Collect literature from Gutenberg.
        
        Parameters
        ----------
        target_count : int
            Target number of books
        languages : list of str
            Languages to collect
        min_downloads : int
            Minimum downloads (proxy for cultural success)
        """
        print(f"\n{'='*80}")
        print("COLLECTING PROJECT GUTENBERG LITERATURE")
        print(f"{'='*80}\n")
        print(f"Target: {target_count:,} books")
        print(f"Languages: {languages}")
        print(f"Minimum downloads: {min_downloads:,} (success threshold)\n")
        
        collected = []
        page = 1
        
        while len(collected) < target_count:
            print(f"Fetching page {page}... (collected: {len(collected):,}/{target_count:,})")
            
            try:
                # Gutendex API (free, no auth)
                response = requests.get(
                    self.base_url,
                    params={
                        'languages': ','.join(languages),
                        'page': page
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"  ✗ Error: Status {response.status_code}")
                    break
                
                data = response.json()
                books = data.get('results', [])
                
                for book in books:
                    # Filter: must have text format and sufficient downloads
                    downloads = book.get('download_count', 0)
                    
                    if downloads < min_downloads:
                        continue
                    
                    # Get plain text URL
                    text_url = None
                    for format_type, urls in book.get('formats', {}).items():
                        if 'text/plain' in format_type:
                            text_url = urls
                            break
                    
                    if not text_url:
                        continue
                    
                    # Collect book metadata
                    book_data = {
                        'gutenberg_id': book['id'],
                        'title': book['title'],
                        'authors': [a['name'] for a in book.get('authors', [])],
                        'subjects': book.get('subjects', []),
                        'bookshelves': book.get('bookshelves', []),
                        'languages': book.get('languages', []),
                        'download_count': downloads,
                        'text_url': text_url,
                        'outcome': downloads,  # Cultural success = download count
                        'source': 'gutenberg',
                        'collected_at': time.time()
                    }
                    
                    collected.append(book_data)
                    
                    if len(collected) >= target_count:
                        break
                
                # Check if more pages
                if not data.get('next'):
                    print("  ℹ No more pages available")
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ✗ Error on page {page}: {e}")
                break
        
        # Save metadata
        metadata_file = self.output_dir / 'gutenberg_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(collected, f, indent=2)
        
        print(f"\n✓ Collected {len(collected):,} books")
        print(f"✓ Metadata saved: {metadata_file}")
        print(f"\nNext: Download full texts using download_gutenberg_texts.py\n")
        
        return collected


class WikipediaBiographyCollector:
    """
    Collect biography narratives from Wikipedia.
    
    Biographies are pure narratives:
    - Life arcs (birth → death)
    - Character development
    - Events and conflicts
    - Clear outcomes (historical impact)
    """
    
    def __init__(self, output_dir: str = 'data/literary_corpus/biographies'):
        """Initialize collector."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_url = 'https://en.wikipedia.org/w/api.php'
    
    def collect_biographies(
        self,
        target_count: int = 2000,
        categories: List[str] = None
    ):
        """
        Collect biographies from Wikipedia.
        
        Parameters
        ----------
        target_count : int
            Target number of biographies
        categories : list of str, optional
            Wikipedia categories to collect from
        """
        if categories is None:
            categories = [
                'Category:Scientists',
                'Category:Writers',
                'Category:Artists',
                'Category:Politicians',
                'Category:Athletes',
                'Category:Business_people',
                'Category:Musicians',
                'Category:Philosophers'
            ]
        
        print(f"\n{'='*80}")
        print("COLLECTING WIKIPEDIA BIOGRAPHIES")
        print(f"{'='*80}\n")
        print(f"Target: {target_count:,} biographies")
        print(f"Categories: {len(categories)}\n")
        
        collected = []
        
        for category in categories:
            if len(collected) >= target_count:
                break
            
            print(f"Collecting from: {category}")
            
            # Get category members
            members = self._get_category_members(category, limit=target_count // len(categories))
            
            print(f"  Found: {len(members)} articles")
            
            for member in members:
                if len(collected) >= target_count:
                    break
                
                # Get article content
                content = self._get_article_content(member)
                
                if content and len(content) > 500:
                    biography = {
                        'title': member,
                        'text': content,
                        'category': category,
                        'length': len(content),
                        'source': 'wikipedia_biography',
                        'collected_at': time.time()
                    }
                    
                    collected.append(biography)
                
                if len(collected) % 50 == 0:
                    print(f"  Collected: {len(collected)}")
                
                time.sleep(0.1)  # Rate limiting
        
        # Save
        output_file = self.output_dir / 'wikipedia_biographies.json'
        with open(output_file, 'w') as f:
            json.dump(collected, f, indent=2)
        
        print(f"\n✓ Collected {len(collected):,} biographies")
        print(f"✓ Saved: {output_file}\n")
        
        return collected
    
    def _get_category_members(self, category: str, limit: int = 500) -> List[str]:
        """Get articles in category."""
        members = []
        
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmlimit': min(limit, 500),
            'format': 'json'
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=30)
            data = response.json()
            
            for member in data.get('query', {}).get('categorymembers', []):
                members.append(member['title'])
        
        except Exception as e:
            print(f"  ✗ Error getting category members: {e}")
        
        return members
    
    def _get_article_content(self, title: str) -> Optional[str]:
        """Get article text content."""
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=30)
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                return page_data.get('extract', '')
        
        except:
            return None
        
        return None


class LiteraryNarrativeBatchAnalyzer:
    """
    Batch analyze literary narratives with new renovation framework.
    
    Applies:
    - Multi-stream detection (character arcs, plot, themes, symbols)
    - Sequential preservation (order, spacing, rhythm)
    - Temporal dynamics (τ/ς/ρ)
    - Unsupervised discovery (natural patterns)
    - Cross-cultural framework detection
    """
    
    def __init__(self, output_dir: str = 'results/literary_analysis'):
        """Initialize batch analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Will lazy-load analysis modules
        self.processors_loaded = False
    
    def analyze_literary_corpus(
        self,
        narratives: List[Dict],
        batch_size: int = 100,
        checkpoint_every: int = 500
    ):
        """
        Analyze complete literary corpus.
        
        Process:
        1. Multi-stream detection (concurrent narratives within each text)
        2. Temporal analysis (compression, rhythm, pacing)
        3. Unsupervised discovery (find natural patterns)
        4. Cross-cultural framework detection
        
        Results:
        - Discovered patterns (NOT imposed archetypes)
        - Stream statistics (how many stories per narrative)
        - Temporal constants (test ς × τ ≈ 0.3)
        - Cultural framework fits
        """
        print(f"\n{'='*80}")
        print("LITERARY CORPUS - MULTI-STREAM TEMPORAL ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Corpus: {len(narratives):,} narratives")
        print(f"Analysis: Multi-stream + Temporal + Unsupervised + Cross-cultural")
        print(f"\nThis will reveal:")
        print(f"  - How many concurrent stories in each narrative")
        print(f"  - Temporal patterns (τ/ς/ρ) across literature")
        print(f"  - Natural clusters (AI discovers, not imposed)")
        print(f"  - Cultural framework fits")
        print(f"\n{'='*80}\n")
        
        # Load analysis infrastructure
        if not self.processors_loaded:
            self._load_processors()
        
        results = {
            'corpus_size': len(narratives),
            'analyses': [],
            'stream_statistics': {'counts': [], 'avg': 0},
            'temporal_patterns': [],
            'discovered_patterns': None
        }
        
        # Process in batches
        for batch_start in range(0, len(narratives), batch_size):
            batch_end = min(batch_start + batch_size, len(narratives))
            batch = narratives[batch_start:batch_end]
            
            print(f"[BATCH {batch_start}-{batch_end}] Processing {len(batch)} narratives...")
            
            batch_results = self._analyze_batch(batch)
            results['analyses'].extend(batch_results)
            
            # Update statistics
            stream_counts = [r.get('n_streams', 0) for r in batch_results if 'n_streams' in r]
            results['stream_statistics']['counts'].extend(stream_counts)
            
            if stream_counts:
                avg_streams = sum(stream_counts) / len(stream_counts)
                print(f"  ✓ Avg streams/narrative: {avg_streams:.1f}")
            
            # Checkpoint
            if batch_end % checkpoint_every == 0:
                self._save_checkpoint(results, batch_end)
        
        # Final statistics
        if results['stream_statistics']['counts']:
            results['stream_statistics']['avg'] = sum(results['stream_statistics']['counts']) / len(results['stream_statistics']['counts'])
            results['stream_statistics']['max'] = max(results['stream_statistics']['counts'])
            results['stream_statistics']['min'] = min(results['stream_statistics']['counts'])
        
        # Save final results
        final_file = self.output_dir / 'literary_corpus_complete_analysis.json'
        with open(final_file, 'w') as f:
            # Don't serialize embeddings (too large)
            serializable = {k: v for k, v in results.items() if k != 'analyses'}
            serializable['n_analyzed'] = len(results['analyses'])
            json.dump(serializable, f, indent=2)
        
        print(f"\n{'='*80}")
        print("LITERARY CORPUS ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Analyzed: {len(results['analyses']):,} narratives")
        print(f"Avg streams per narrative: {results['stream_statistics'].get('avg', 0):.1f}")
        print(f"Results: {final_file}")
        print(f"{'='*80}\n")
        
        return results
    
    def _load_processors(self):
        """Lazy load analysis processors."""
        print("Loading AI analysis infrastructure...")
        
        sys.path.insert(0, str(Path(__file__).parent.parent / 'narrative_optimization' / 'src'))
        
        from analysis.multi_stream_narrative_processor import MultiStreamNarrativeProcessor
        from analysis.sequential_narrative_processor import SequentialNarrativeProcessor
        from transformers.temporal.temporal_compression import TemporalCompressionTransformer
        
        self.multi_stream = MultiStreamNarrativeProcessor()
        self.sequential = SequentialNarrativeProcessor()
        self.temporal = TemporalCompressionTransformer(domain='novel')
        
        print("✓ AI infrastructure loaded\n")
        
        self.processors_loaded = True
    
    def _analyze_batch(self, batch: List[Dict]) -> List[Dict]:
        """Analyze batch of narratives."""
        batch_results = []
        
        for narrative_data in batch:
            text = narrative_data.get('text', narrative_data.get('content', ''))
            narrative_id = narrative_data.get('title', narrative_data.get('id', 'unknown'))
            
            if not text or len(text) < 500:
                continue
            
            try:
                # Multi-stream detection
                stream_result = self.multi_stream.discover_streams(text, narrative_id)
                
                # Sequential analysis
                # seq_result = self.sequential.process_narrative_sequential(text, narrative_id)
                
                batch_results.append({
                    'narrative_id': narrative_id,
                    'n_streams': stream_result.get('n_streams', 0),
                    'stream_analysis': {k: v for k, v in stream_result.items() if k != 'streams'},  # Don't save full data
                    'length': len(text),
                    'source': narrative_data.get('source', 'unknown')
                })
            
            except Exception as e:
                print(f"  ✗ Error analyzing {narrative_id}: {e}")
                continue
        
        return batch_results
    
    def _save_checkpoint(self, results: Dict, count: int):
        """Save checkpoint."""
        checkpoint_file = self.output_dir / f'checkpoint_{count}.json'
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'count': count,
                'stream_stats': results['stream_statistics']
            }, f, indent=2)


def quick_collect_and_analyze_literature():
    """
    Quick start: Collect and analyze literature in one go.
    
    This will:
    1. Collect metadata for 1000 popular books from Gutenberg
    2. Download 100 complete texts (pilot)
    3. Run multi-stream analysis
    4. Discover patterns
    5. Report findings
    """
    print(f"\n{'='*80}")
    print("QUICK START: LITERATURE ANALYSIS")
    print(f"{'='*80}\n")
    print("Phase 1: Collect metadata (Gutenberg API)")
    print("Phase 2: Analyze existing + download samples")
    print("Phase 3: Multi-stream + temporal discovery\n")
    
    # Phase 1: Collect metadata
    collector = GutenbergCollector()
    
    print("[Phase 1] Collecting book metadata from Gutenberg...")
    books = collector.collect_literature(
        target_count=1000,
        min_downloads=500  # Popular books only for pilot
    )
    
    print(f"\n[Phase 2] Books available for analysis:")
    print(f"  Total: {len(books):,}")
    print(f"  With text URLs: {sum(1 for b in books if b.get('text_url')):,}\n")
    
    # Phase 3: For pilot, analyze Wikipedia plot summaries we already have
    print("[Phase 3] Starting with existing narrative data...")
    
    # Check what literary data we have
    literary_paths = [
        Path('data/domains/mythology/mythology_complete_dataset.json'),
        Path('data/domains/imdb_movies_complete.json'),  # Has plot summaries
    ]
    
    existing_narratives = []
    
    for path in literary_paths:
        if path.exists():
            print(f"  Loading: {path.name}")
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing_narratives.extend(data)
                    print(f"    ✓ Added {len(data):,} narratives")
    
    if existing_narratives:
        print(f"\n  Total existing narratives: {len(existing_narratives):,}")
        print(f"  Beginning multi-stream analysis...\n")
        
        analyzer = LiteraryNarrativeBatchAnalyzer()
        results = analyzer.analyze_literary_corpus(
            existing_narratives[:200],  # Pilot with 200
            batch_size=50
        )
        
        return results
    else:
        print("\n  ⊘ No existing narrative data found")
        print("  Recommendation: Run Gutenberg text download first\n")
        
        return None


if __name__ == '__main__':
    quick_collect_and_analyze_literature()

