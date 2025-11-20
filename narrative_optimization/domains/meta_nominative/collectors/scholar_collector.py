"""
Google Scholar Collector for Meta-Nominative Determinism Research

Collects papers and citation data from Google Scholar.
Note: Uses scholarly library or custom scraping with rate limiting.
"""

import time
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import random


class GoogleScholarCollector:
    """Collect nominative determinism papers from Google Scholar."""
    
    def __init__(self):
        """Initialize Google Scholar collector."""
        self.papers = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Search queries
        self.search_queries = [
            "nominative determinism",
            "implicit egotism names",
            "name-letter effect",
            "name career correspondence",
            "surname occupation correlation"
        ]
        
        # Key papers (seed list for validation)
        self.key_papers = [
            {"title": "Why Susie sells seashells", "author": "Pelham"},
            {"title": "Moniker maladies", "author": "Pelham"},
            {"title": "spurious name similarity effects", "author": "Simonsohn"},
            {"title": "What's in a surname", "author": "Jung"},
        ]
    
    def search_scholar(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search Google Scholar for papers.
        
        Args:
            query: Search query
            max_results: Maximum results to retrieve
            
        Returns:
            List of paper dictionaries
        """
        print(f"\nSearching Google Scholar: '{query}'...")
        papers = []
        
        base_url = "https://scholar.google.com/scholar"
        
        # Paginate through results
        for start in range(0, max_results, 10):
            params = {
                'q': query,
                'start': start,
                'hl': 'en'
            }
            
            try:
                # Add random delay to avoid blocking
                time.sleep(random.uniform(3, 7))
                
                response = self.session.get(base_url, params=params, timeout=30)
                
                if response.status_code == 429:
                    print("  Rate limited, waiting longer...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                
                # Parse results
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', class_='gs_r gs_or gs_scl')
                
                if not results:
                    # Try alternative class names
                    results = soup.find_all('div', {'data-rp': True})
                
                for result in results:
                    paper = self._parse_scholar_result(result)
                    if paper:
                        papers.append(paper)
                
                print(f"  Found {len(papers)} papers so far...")
                
                # Check if there are more results
                if not results or len(results) < 10:
                    break
                    
            except Exception as e:
                print(f"  Error on page {start//10 + 1}: {e}")
                break
        
        print(f"✓ Retrieved {len(papers)} papers from '{query}'")
        return papers
    
    def _parse_scholar_result(self, result) -> Optional[Dict]:
        """Parse a single Google Scholar search result."""
        try:
            # Title
            title_elem = result.find('h3', class_='gs_rt')
            if not title_elem:
                return None
            
            # Remove citation formatting
            for citation in title_elem.find_all('span', class_='gs_ctu'):
                citation.decompose()
            
            title = title_elem.get_text(strip=True)
            
            # Link
            link_elem = title_elem.find('a')
            link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else None
            
            # Authors and publication info
            info_elem = result.find('div', class_='gs_a')
            authors_text = ""
            year = None
            journal = ""
            
            if info_elem:
                info_text = info_elem.get_text()
                parts = info_text.split(' - ')
                
                if len(parts) >= 1:
                    authors_text = parts[0]
                if len(parts) >= 2:
                    journal = parts[1]
                
                # Extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', info_text)
                if year_match:
                    year = int(year_match.group())
            
            # Parse authors
            authors = self._parse_authors(authors_text)
            
            # Snippet/abstract
            snippet_elem = result.find('div', class_='gs_rs')
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
            
            # Citations
            citation_elem = result.find('div', class_='gs_fl')
            citations = 0
            if citation_elem:
                cited_by = citation_elem.find('a', string=re.compile(r'Cited by'))
                if cited_by:
                    cite_match = re.search(r'Cited by (\d+)', cited_by.text)
                    if cite_match:
                        citations = int(cite_match.group(1))
            
            # Extract research topic
            research_topic = self._infer_topic(title, snippet)
            
            # Try to extract effect size from snippet
            effect_size = self._extract_effect_size(snippet)
            
            return {
                'title': title,
                'authors': authors,
                'lead_author': authors[0] if authors else None,
                'year': year,
                'journal': journal,
                'abstract': snippet,  # Only snippet available
                'link': link,
                'citations': citations,
                'research_topic': research_topic,
                'effect_size': effect_size,
                'finding_type': self._classify_finding(snippet, effect_size),
                'source': 'google_scholar'
            }
            
        except Exception as e:
            print(f"    Error parsing result: {e}")
            return None
    
    def _parse_authors(self, authors_text: str) -> List[Dict]:
        """Parse author string into structured list."""
        if not authors_text:
            return []
        
        authors = []
        
        # Split by commas or 'and'
        author_names = re.split(r',\s*|\s+and\s+', authors_text)
        
        for name in author_names[:10]:  # Limit to first 10
            name = name.strip()
            if not name or len(name) < 2:
                continue
            
            # Handle "Last, First" or "First Last" format
            parts = name.split()
            if len(parts) >= 2:
                # Heuristic: if comma, it's "Last, First"
                if ',' in name:
                    last_name = parts[0].replace(',', '')
                    first_name = ' '.join(parts[1:])
                else:
                    # Assume "First Last" or "First Middle Last"
                    first_name = ' '.join(parts[:-1])
                    last_name = parts[-1]
                
                authors.append({
                    'last_name': last_name,
                    'first_name': first_name,
                    'full_name': f"{first_name} {last_name}".strip()
                })
        
        return authors
    
    def _infer_topic(self, title: str, abstract: str) -> str:
        """Infer research topic from title and abstract."""
        text = f"{title} {abstract}".lower()
        
        topics = {
            'dentists': ['dentist', 'dental'],
            'hurricanes': ['hurricane', 'storm'],
            'lawyers': ['lawyer', 'attorney'],
            'doctors': ['physician', 'doctor', 'medical'],
            'marriage': ['marriage', 'mate', 'partner'],
            'geography': ['geographic', 'location', 'city'],
            'occupations': ['occupation', 'career', 'profession'],
            'brands': ['brand', 'company'],
            'names_general': ['name', 'naming']
        }
        
        detected = []
        for topic, keywords in topics.items():
            if any(kw in text for kw in keywords):
                detected.append(topic)
        
        return ', '.join(detected) if detected else 'general'
    
    def _extract_effect_size(self, text: str) -> Optional[Dict]:
        """Extract effect size from text."""
        if not text:
            return None
        
        patterns = {
            'correlation': r'r\s*=\s*([0-9.]+)',
            'cohens_d': r'd\s*=\s*([0-9.]+)',
            'odds_ratio': r'OR\s*=\s*([0-9.]+)',
        }
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return {
                        'metric': metric,
                        'value': float(matches[0])
                    }
                except ValueError:
                    continue
        
        return None
    
    def _classify_finding(self, abstract: str, effect_size: Optional[Dict]) -> str:
        """Classify finding type."""
        if not abstract:
            return 'unknown'
        
        text = abstract.lower()
        
        if any(term in text for term in ['no significant', 'not significant', 'null']):
            return 'null'
        elif any(term in text for term in ['significant', 'effect', 'relationship']):
            return 'positive'
        elif effect_size and effect_size['value'] > 0.15:
            return 'positive'
        else:
            return 'unknown'
    
    def collect_all(self, max_per_query: int = 40) -> List[Dict]:
        """
        Collect papers from all search queries.
        
        Args:
            max_per_query: Maximum results per query
            
        Returns:
            List of unique papers
        """
        print(f"\n{'='*80}")
        print("GOOGLE SCHOLAR SEARCH - Nominative Determinism Papers")
        print(f"{'='*80}\n")
        
        all_papers = []
        seen_titles = set()
        
        for query in self.search_queries:
            papers = self.search_scholar(query, max_per_query)
            
            # Deduplicate by title
            for paper in papers:
                title_normalized = paper['title'].lower().strip()
                if title_normalized not in seen_titles:
                    seen_titles.add(title_normalized)
                    all_papers.append(paper)
        
        self.papers = all_papers
        
        print(f"\n{'='*80}")
        print(f"✓ Total unique papers collected: {len(all_papers)}")
        print(f"{'='*80}\n")
        
        return all_papers
    
    def save_papers(self, output_path: Optional[Path] = None) -> Path:
        """Save collected papers to JSON."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative' / 'papers_scholar.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'source': 'google_scholar',
                'collection_date': time.strftime('%Y-%m-%d'),
                'total_papers': len(self.papers),
                'papers': self.papers
            }, f, indent=2)
        
        print(f"✓ Saved {len(self.papers)} papers to: {output_path}")
        return output_path
    
    def collect(self, max_per_query: int = 40) -> List[Dict]:
        """Complete collection pipeline."""
        papers = self.collect_all(max_per_query)
        self.save_papers()
        
        # Summary
        print(f"\nCOLLECTION SUMMARY:")
        print(f"  Total papers: {len(papers)}")
        print(f"  With citations: {sum(1 for p in papers if p.get('citations', 0) > 0)}")
        print(f"  Average citations: {sum(p.get('citations', 0) for p in papers) / len(papers):.1f}")
        print(f"  With effect sizes: {sum(1 for p in papers if p['effect_size'])}")
        
        return papers


def main():
    """Run Google Scholar collection."""
    print("⚠️  Google Scholar scraping requires careful rate limiting")
    print("⚠️  Collection will be slow to avoid blocking\n")
    
    collector = GoogleScholarCollector()
    papers = collector.collect(max_per_query=30)
    
    print(f"\n{'='*80}")
    print("✓ Google Scholar collection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

