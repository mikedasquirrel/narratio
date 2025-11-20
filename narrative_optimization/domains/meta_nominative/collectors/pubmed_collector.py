"""
PubMed Collector for Meta-Nominative Determinism Research

Collects papers on nominative determinism from PubMed using the Entrez API.
Extracts: authors, titles, abstracts, publication info, and metadata.
"""

import requests
import time
import json
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET
import re
from pathlib import Path


class PubMedCollector:
    """Collect nominative determinism papers from PubMed."""
    
    def __init__(self, email: str = "research@nominative.study"):
        """
        Initialize PubMed collector.
        
        Args:
            email: Email for PubMed API (required by NCBI)
        """
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.papers = []
        
        # Search terms for nominative determinism research
        self.search_terms = [
            "nominative determinism",
            "name-career correspondence",
            "implicit egotism",
            "name-letter effect",
            "nominal determinism",
            "name similarity effect",
            "nominative fit",
            "surname congruence"
        ]
    
    def search_papers(self, max_results: int = 200) -> List[str]:
        """
        Search PubMed for relevant papers.
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        print(f"\n{'='*80}")
        print("PUBMED SEARCH - Nominative Determinism Papers")
        print(f"{'='*80}\n")
        
        all_pmids = set()
        
        for term in self.search_terms:
            print(f"Searching: '{term}'...")
            
            # Build query
            query = f'"{term}"[Title/Abstract] OR "{term}"[MeSH Terms]'
            
            # Search endpoint
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'email': self.email,
                'usehistory': 'y'
            }
            
            try:
                response = requests.get(search_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                pmids = data.get('esearchresult', {}).get('idlist', [])
                all_pmids.update(pmids)
                print(f"  Found {len(pmids)} papers (total unique: {len(all_pmids)})")
                
                time.sleep(0.34)  # NCBI rate limit: 3 requests/second
                
            except Exception as e:
                print(f"  Error searching '{term}': {e}")
                continue
        
        pmid_list = list(all_pmids)[:max_results]
        print(f"\n✓ Total unique papers found: {len(pmid_list)}")
        return pmid_list
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch full details for papers by PMID.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of paper dictionaries with all metadata
        """
        print(f"\nFetching details for {len(pmids)} papers...")
        papers = []
        
        # Fetch in batches of 50
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}...", end=" ")
            
            # Fetch endpoint
            fetch_url = f"{self.base_url}efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                response = requests.get(fetch_url, params=params, timeout=60)
                response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    paper = self._parse_article(article)
                    if paper:
                        papers.append(paper)
                
                print(f"{len(papers)} total")
                time.sleep(0.34)  # Rate limit
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        print(f"\n✓ Successfully retrieved {len(papers)} papers")
        self.papers = papers
        return papers
    
    def _parse_article(self, article: ET.Element) -> Optional[Dict]:
        """Parse PubMed article XML into structured dictionary."""
        try:
            medline = article.find('.//MedlineCitation')
            if medline is None:
                return None
            
            # PMID
            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Article details
            article_elem = medline.find('.//Article')
            if article_elem is None:
                return None
            
            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                label = abstract_elem.get('Label', '')
                text = abstract_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            
            # Authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                lastname = author_elem.find('.//LastName')
                forename = author_elem.find('.//ForeName')
                initials = author_elem.find('.//Initials')
                
                if lastname is not None:
                    author = {
                        'last_name': lastname.text,
                        'first_name': forename.text if forename is not None else "",
                        'initials': initials.text if initials is not None else "",
                        'full_name': f"{forename.text if forename is not None else ''} {lastname.text}".strip()
                    }
                    authors.append(author)
            
            # Journal
            journal_elem = article_elem.find('.//Journal')
            journal_title = ""
            if journal_elem is not None:
                journal_title_elem = journal_elem.find('.//Title')
                journal_title = journal_title_elem.text if journal_title_elem is not None else ""
            
            # Publication date
            pub_date_elem = journal_elem.find('.//PubDate') if journal_elem is not None else None
            year = None
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find('.//Year')
                year = int(year_elem.text) if year_elem is not None else None
            
            # Keywords/MeSH terms
            keywords = []
            for keyword_elem in medline.findall('.//Keyword'):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)
            
            for mesh_elem in medline.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    keywords.append(mesh_elem.text)
            
            # Extract research topic from title/abstract/keywords
            research_topic = self._extract_research_topic(title, abstract, keywords)
            
            # Try to extract effect size from abstract
            effect_size = self._extract_effect_size(abstract)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'lead_author': authors[0] if authors else None,
                'journal': journal_title,
                'year': year,
                'keywords': keywords,
                'research_topic': research_topic,
                'effect_size': effect_size,
                'sample_size': self._extract_sample_size(abstract),
                'finding_type': self._classify_finding(abstract, effect_size),
                'source': 'pubmed',
                'full_text_available': False  # Would need separate API call
            }
            
        except Exception as e:
            print(f"    Error parsing article: {e}")
            return None
    
    def _extract_research_topic(self, title: str, abstract: str, keywords: List[str]) -> str:
        """Extract what domain/topic the research studied."""
        text = f"{title} {abstract}".lower()
        
        # Common topics in nominative determinism research
        topics = {
            'dentists': ['dentist', 'dental', 'orthodont'],
            'hurricanes': ['hurricane', 'storm', 'weather', 'cyclone'],
            'lawyers': ['lawyer', 'attorney', 'legal profession'],
            'doctors': ['physician', 'doctor', 'medical profession'],
            'names_general': ['name', 'naming', 'nomenclature'],
            'occupations': ['occupation', 'career', 'profession', 'job'],
            'geography': ['geographic', 'location', 'city', 'place name'],
            'marriage': ['marriage', 'mate selection', 'partner choice'],
            'brands': ['brand', 'product name', 'company name'],
            'academic': ['academic', 'university', 'researcher']
        }
        
        detected_topics = []
        for topic, terms in topics.items():
            if any(term in text for term in terms):
                detected_topics.append(topic)
        
        return ', '.join(detected_topics) if detected_topics else 'general nominative effects'
    
    def _extract_effect_size(self, text: str) -> Optional[Dict]:
        """Extract effect size from abstract text."""
        if not text:
            return None
        
        # Common effect size patterns
        patterns = {
            'correlation': r'r\s*=\s*([0-9.]+)',
            'correlation_alt': r'\(r\s*=\s*([0-9.]+)',
            'cohens_d': r'd\s*=\s*([0-9.]+)',
            'odds_ratio': r'OR\s*=\s*([0-9.]+)',
            'beta': r'β\s*=\s*([0-9.]+)',
            'r_squared': r'R2?\s*=\s*([0-9.]+)'
        }
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    return {
                        'metric': metric,
                        'value': value,
                        'text_context': text[max(0, text.find(matches[0])-50):text.find(matches[0])+100]
                    }
                except ValueError:
                    continue
        
        return None
    
    def _extract_sample_size(self, text: str) -> Optional[int]:
        """Extract sample size from abstract."""
        if not text:
            return None
        
        # Look for "n = XXX" or "N = XXX" patterns
        patterns = [
            r'[Nn]\s*=\s*([0-9,]+)',
            r'sample of ([0-9,]+)',
            r'([0-9,]+) participants',
            r'([0-9,]+) subjects'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Remove commas and convert
                    n = int(matches[0].replace(',', ''))
                    if 10 < n < 1000000:  # Sanity check
                        return n
                except ValueError:
                    continue
        
        return None
    
    def _classify_finding(self, abstract: str, effect_size: Optional[Dict]) -> str:
        """Classify finding as positive, null, or negative."""
        if not abstract:
            return 'unknown'
        
        text = abstract.lower()
        
        # Null result indicators
        null_indicators = [
            'no significant',
            'not significant',
            'no effect',
            'no association',
            'no relationship',
            'failed to find',
            'null result'
        ]
        
        # Positive result indicators
        positive_indicators = [
            'significant effect',
            'significant association',
            'significant relationship',
            'predicted',
            'supported',
            'confirmed'
        ]
        
        if any(indicator in text for indicator in null_indicators):
            return 'null'
        elif any(indicator in text for indicator in positive_indicators):
            return 'positive'
        elif effect_size and effect_size['value'] > 0.1:
            return 'positive'
        elif effect_size and effect_size['value'] <= 0.1:
            return 'weak_or_null'
        else:
            return 'unknown'
    
    def save_papers(self, output_path: Optional[Path] = None) -> Path:
        """Save collected papers to JSON file."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative' / 'papers_pubmed.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'source': 'pubmed',
                'collection_date': time.strftime('%Y-%m-%d'),
                'total_papers': len(self.papers),
                'papers': self.papers
            }, f, indent=2)
        
        print(f"\n✓ Saved {len(self.papers)} papers to: {output_path}")
        return output_path
    
    def collect(self, max_results: int = 200) -> List[Dict]:
        """
        Complete collection pipeline.
        
        Args:
            max_results: Maximum papers to collect
            
        Returns:
            List of paper dictionaries
        """
        pmids = self.search_papers(max_results)
        papers = self.fetch_paper_details(pmids)
        self.save_papers()
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("COLLECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total papers: {len(papers)}")
        print(f"Papers with abstracts: {sum(1 for p in papers if p['abstract'])}")
        print(f"Papers with effect sizes: {sum(1 for p in papers if p['effect_size'])}")
        print(f"Average authors per paper: {sum(len(p['authors']) for p in papers) / len(papers):.1f}")
        
        year_dist = {}
        for p in papers:
            if p['year']:
                year_dist[p['year']] = year_dist.get(p['year'], 0) + 1
        print(f"Year range: {min(year_dist.keys())} - {max(year_dist.keys())}")
        
        finding_dist = {}
        for p in papers:
            finding_dist[p['finding_type']] = finding_dist.get(p['finding_type'], 0) + 1
        print(f"Findings: {finding_dist}")
        
        return papers


def main():
    """Run PubMed collection."""
    collector = PubMedCollector()
    papers = collector.collect(max_results=100)
    
    print(f"\n{'='*80}")
    print("✓ PubMed collection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

