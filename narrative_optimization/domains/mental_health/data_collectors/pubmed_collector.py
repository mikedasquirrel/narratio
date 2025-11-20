"""
PubMed Article Counter

Collects publication counts for mental health disorders using PubMed E-utilities API.

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/

Author: Narrative Optimization Research
Date: November 2025
"""

import requests
import time
from typing import Dict, List
from xml.etree import ElementTree as ET


class PubMedCollector:
    """
    Collect PubMed article counts for mental health disorders.
    
    Uses NCBI E-utilities API (free, no key required for low volume).
    """
    
    ESEARCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    def __init__(self, rate_limit_delay: float = 0.4):
        """
        Initialize PubMed collector.
        
        Parameters
        ----------
        rate_limit_delay : float
            Seconds between requests (NCBI limit: 3/second without key)
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
    
    def count_articles_for_disorder(self, disorder_name: str,
                                   include_synonyms: bool = True) -> Dict:
        """
        Count PubMed articles for a disorder.
        
        Parameters
        ----------
        disorder_name : str
            Name of disorder
        include_synonyms : bool
            Search with common synonyms/acronyms
        
        Returns
        -------
        dict
            Article count statistics
        """
        try:
            # Build search query
            query_terms = [disorder_name]
            
            # Add common synonyms/acronyms
            if include_synonyms:
                synonyms = self._get_synonyms(disorder_name)
                query_terms.extend(synonyms)
            
            # Combine with OR
            query = ' OR '.join(f'"{term}"' for term in query_terms)
            
            params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': 0  # We only want the count
            }
            
            response = self.session.get(self.ESEARCH_BASE, params=params, timeout=10)
            
            # Respect rate limit
            time.sleep(self.rate_limit_delay)
            
            if response.status_code != 200:
                return {
                    'disorder_name': disorder_name,
                    'error': f'API returned status {response.status_code}',
                    'article_count': 0
                }
            
            data = response.json()
            count = int(data.get('esearchresult', {}).get('count', 0))
            
            return {
                'disorder_name': disorder_name,
                'article_count': count,
                'search_query': query
            }
            
        except Exception as e:
            return {
                'disorder_name': disorder_name,
                'error': str(e),
                'article_count': 0
            }
    
    def _get_synonyms(self, disorder_name: str) -> List[str]:
        """Get common synonyms and acronyms for disorders."""
        synonyms_map = {
            'Major Depressive Disorder': ['MDD', 'Depression', 'Clinical Depression'],
            'Generalized Anxiety Disorder': ['GAD', 'Anxiety'],
            'Post-Traumatic Stress Disorder': ['PTSD'],
            'Obsessive-Compulsive Disorder': ['OCD'],
            'Attention-Deficit/Hyperactivity Disorder': ['ADHD', 'ADD'],
            'Bipolar Disorder': ['Bipolar', 'Manic Depression'],
            'Schizophrenia': ['Schizophrenic Disorder'],
            'Borderline Personality Disorder': ['BPD'],
            'Antisocial Personality Disorder': ['ASPD'],
            'Narcissistic Personality Disorder': ['NPD']
        }
        
        return synonyms_map.get(disorder_name, [])
    
    def collect_for_all_disorders(self, disorders: List[Dict]) -> List[Dict]:
        """Collect PubMed counts for all disorders."""
        results = []
        total = len(disorders)
        
        print(f"Collecting PubMed article counts for {total} disorders...")
        
        for i, disorder in enumerate(disorders, 1):
            disorder_name = disorder.get('disorder_name', disorder.get('name', ''))
            
            if not disorder_name:
                continue
            
            if i % 50 == 0:
                print(f"Progress: {i}/{total}...")
            
            article_data = self.count_articles_for_disorder(disorder_name)
            
            disorder_with_articles = disorder.copy()
            disorder_with_articles['pubmed_data'] = article_data
            results.append(disorder_with_articles)
        
        print(f"✅ PubMed collection complete: {total} disorders\n")
        
        return results


if __name__ == '__main__':
    # Demo
    collector = PubMedCollector()
    
    result = collector.count_articles_for_disorder('Schizophrenia')
    print(f"Schizophrenia: {result['article_count']:,} articles")

