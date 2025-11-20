"""
Supreme Court Data Collector

Collects comprehensive data from multiple sources:
- CourtListener API: Opinions, briefs, oral arguments
- Supreme Court Database: Vote data, case metadata
- Citation data: Future influence metrics

Collects ALL Supreme Court cases from 1789-present (~30,000+ cases)
with as much detail as possible for each.

Author: Narrative Optimization Framework
Date: November 17, 2025
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupremeCourtDataCollector:
    """
    Comprehensive Supreme Court data collector.
    
    Sources:
    1. CourtListener API (primary - free, comprehensive)
    2. Oyez for oral arguments
    3. Citation tracking for influence metrics
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize collector.
        
        Parameters
        ----------
        api_key : str, optional
            CourtListener API key (improves rate limits)
            Get from: https://www.courtlistener.com/api/
        """
        self.api_key = api_key
        self.base_url = "https://www.courtlistener.com/api/rest/v3"
        
        # Rate limiting
        self.requests_per_hour = 5000 if api_key else 100
        self.request_count = 0
        self.start_time = time.time()
        
        # Checkpoint management
        self.checkpoint_file = Path(__file__).parent / 'collection_checkpoint.json'
        self.data_file = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'supreme_court_complete.json'
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.cases = []
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load existing data from checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)
                self.cases = checkpoint.get('cases', [])
                self.request_count = checkpoint.get('request_count', 0)
                logger.info(f"Loaded checkpoint: {len(self.cases)} cases collected")
        
        if self.data_file.exists():
            with open(self.data_file) as f:
                existing = json.load(f)
                if len(existing) > len(self.cases):
                    self.cases = existing
                    logger.info(f"Loaded existing data: {len(self.cases)} cases")
    
    def save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'cases': self.cases,
                'request_count': self.request_count,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save full data
        with open(self.data_file, 'w') as f:
            json.dump(self.cases, f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(self.cases)} cases")
    
    def check_rate_limit(self):
        """Check and enforce rate limiting."""
        elapsed = time.time() - self.start_time
        if elapsed < 3600:  # Within first hour
            if self.request_count >= self.requests_per_hour:
                sleep_time = 3600 - elapsed
                logger.warning(f"Rate limit reached. Sleeping {sleep_time:.0f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.start_time = time.time()
        else:
            # Reset counter after hour
            self.request_count = 0
            self.start_time = time.time()
    
    def api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with rate limiting.
        
        Parameters
        ----------
        endpoint : str
            API endpoint (e.g., '/opinions/')
        params : dict
            Query parameters
        
        Returns
        -------
        response : dict or None
        """
        self.check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Token {self.api_key}'
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60s")
                time.sleep(60)
                return self.api_request(endpoint, params)
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def collect_supreme_court_cases(
        self,
        start_year: int = 1789,
        end_year: Optional[int] = None,
        max_cases: Optional[int] = None
    ):
        """
        Collect all Supreme Court cases.
        
        Parameters
        ----------
        start_year : int
            Starting year (default: 1789 - founding)
        end_year : int, optional
            Ending year (default: current year)
        max_cases : int, optional
            Maximum cases to collect (for testing)
        """
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Collecting Supreme Court cases: {start_year}-{end_year}")
        logger.info(f"Target: {'unlimited' if max_cases is None else max_cases} cases")
        
        # Get existing case IDs to avoid duplicates
        existing_ids = {case['case_id'] for case in self.cases if 'case_id' in case}
        logger.info(f"Already have {len(existing_ids)} cases, collecting more...")
        
        # Query Supreme Court cluster (court_id for SCOTUS)
        page = 1
        new_cases_count = 0
        
        while True:
            if max_cases and len(self.cases) >= max_cases:
                logger.info(f"Reached max_cases limit: {max_cases}")
                break
            
            # Search opinions from Supreme Court
            params = {
                'court': 'scotus',  # Supreme Court
                'filed_after': f'{start_year}-01-01',
                'filed_before': f'{end_year}-12-31',
                'page': page,
                'page_size': 100,  # Max per page
                'order_by': '-date_filed'  # Newest first
            }
            
            logger.info(f"Fetching page {page}...")
            data = self.api_request('/opinions/', params)
            
            if not data or 'results' not in data:
                logger.warning(f"No results on page {page}, stopping")
                break
            
            results = data['results']
            if not results:
                logger.info("No more results")
                break
            
            # Process each opinion
            for opinion_data in results:
                case_id = self._extract_case_id(opinion_data)
                
                # Skip if already have this case
                if case_id in existing_ids:
                    continue
                
                # Extract comprehensive case data
                case = self._process_opinion(opinion_data)
                if case:
                    self.cases.append(case)
                    existing_ids.add(case_id)
                    new_cases_count += 1
                    
                    # Log progress
                    if new_cases_count % 100 == 0:
                        logger.info(f"Collected {new_cases_count} new cases ({len(self.cases)} total)")
                        self.save_checkpoint()
            
            # Check if there are more pages
            if not data.get('next'):
                logger.info("Reached last page")
                break
            
            page += 1
            
            # Save checkpoint every 10 pages
            if page % 10 == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint()
        logger.info(f"Collection complete: {len(self.cases)} total cases ({new_cases_count} new)")
    
    def _extract_case_id(self, opinion_data: Dict) -> str:
        """Extract unique case identifier."""
        # Use CourtListener opinion ID
        return str(opinion_data.get('id', 'unknown'))
    
    def _process_opinion(self, opinion_data: Dict) -> Optional[Dict]:
        """
        Process single opinion into comprehensive case data.
        
        Parameters
        ----------
        opinion_data : dict
            Raw opinion data from CourtListener
        
        Returns
        -------
        case : dict
            Processed case with all narratives and outcomes
        """
        try:
            case_id = self._extract_case_id(opinion_data)
            
            # Extract basic info
            case = {
                'case_id': case_id,
                'year': self._extract_year(opinion_data),
                'case_name': opinion_data.get('case_name', 'Unknown'),
                'docket_number': opinion_data.get('docket_number', ''),
                'date_filed': opinion_data.get('date_filed', ''),
                
                # Narratives (multiple types)
                'majority_opinion': '',
                'concurring_opinion': '',
                'dissenting_opinion': '',
                'plurality_opinion': '',
                'opinion_full_text': '',
                
                # We'll collect these separately if available
                'petitioner_brief': '',
                'respondent_brief': '',
                'oral_arguments': '',
                
                # Outcomes (multiple metrics)
                'outcome': {},
                'metadata': {}
            }
            
            # Extract opinion text
            opinion_text = opinion_data.get('plain_text', '') or opinion_data.get('html', '')
            if opinion_text:
                case['opinion_full_text'] = self._clean_text(opinion_text)
            
            # Determine opinion type
            opinion_type = opinion_data.get('type', 'unknown')
            if 'majority' in opinion_type.lower() or 'lead' in opinion_type.lower():
                case['majority_opinion'] = case['opinion_full_text']
            elif 'dissent' in opinion_type.lower():
                case['dissenting_opinion'] = case['opinion_full_text']
            elif 'concur' in opinion_type.lower():
                case['concurring_opinion'] = case['opinion_full_text']
            elif 'plurality' in opinion_type.lower():
                case['plurality_opinion'] = case['opinion_full_text']
            else:
                # Default to majority if unclear
                case['majority_opinion'] = case['opinion_full_text']
            
            # Extract cluster info for more details
            cluster_id = opinion_data.get('cluster')
            if cluster_id:
                cluster_data = self._get_cluster_details(cluster_id)
                if cluster_data:
                    case = self._enrich_with_cluster_data(case, cluster_data)
            
            # Extract outcome metrics
            case['outcome'] = self._extract_outcomes(opinion_data, case)
            
            # Extract metadata
            case['metadata'] = self._extract_metadata(opinion_data, case)
            
            # Validate we have enough data
            if not self._validate_case(case):
                return None
            
            return case
            
        except Exception as e:
            logger.error(f"Error processing opinion: {e}")
            return None
    
    def _extract_year(self, opinion_data: Dict) -> int:
        """Extract year from date_filed."""
        date_str = opinion_data.get('date_filed', '')
        if date_str:
            try:
                return int(date_str.split('-')[0])
            except:
                pass
        return 0
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML and formatting from text."""
        # Remove HTML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers, footnotes, etc
        text = re.sub(r'\[\d+\]', '', text)
        return text.strip()
    
    def _get_cluster_details(self, cluster_url: str) -> Optional[Dict]:
        """
        Get cluster details (contains case-level info).
        
        Parameters
        ----------
        cluster_url : str
            URL to opinion cluster
        
        Returns
        -------
        cluster_data : dict or None
        """
        # Extract cluster ID from URL
        if isinstance(cluster_url, str):
            cluster_id = cluster_url.rstrip('/').split('/')[-1]
        else:
            cluster_id = cluster_url
        
        # Get cluster data
        data = self.api_request(f'/clusters/{cluster_id}/')
        return data
    
    def _enrich_with_cluster_data(self, case: Dict, cluster_data: Dict) -> Dict:
        """
        Enrich case with cluster-level data.
        
        Cluster contains:
        - Case citations (Shepard's citations)
        - Related opinions
        - Precedential status
        """
        if not cluster_data:
            return case
        
        # Add citation info
        case['metadata']['citation_count'] = cluster_data.get('citation_count', 0)
        case['metadata']['citations'] = cluster_data.get('citations', [])
        
        # Precedential status
        case['metadata']['precedential_status'] = cluster_data.get('precedential_status', 'unknown')
        
        # Nature of suit
        case['metadata']['nature_of_suit'] = cluster_data.get('nature_of_suit', '')
        
        # Panel info (justices)
        case['metadata']['panel'] = cluster_data.get('panel', [])
        
        # Sub-opinions in cluster
        sub_opinions = cluster_data.get('sub_opinions', [])
        case['metadata']['opinion_count'] = len(sub_opinions)
        
        return case
    
    def _extract_outcomes(self, opinion_data: Dict, case: Dict) -> Dict:
        """
        Extract all outcome metrics.
        
        Multiple outcomes to test different hypotheses:
        1. Vote margin (primary)
        2. Winner (petitioner/respondent)
        3. Unanimous vs split
        4. Citation count (influence)
        5. Precedent-setting (landmark status)
        """
        outcomes = {
            'vote_margin': 0,  # Range: 0 (5-4) to 8 (9-0)
            'unanimous': False,
            'winner': 'unknown',  # petitioner, respondent, or mixed
            'citation_count': 0,
            'precedent_setting': False,
            'overturned': False
        }
        
        # Try to extract vote from opinion text or metadata
        # Note: Vote data often requires separate database
        # For now, mark as needing supplementation
        
        outcomes['citation_count'] = case['metadata'].get('citation_count', 0)
        
        # Heuristic: High citation + precedential = precedent-setting
        if outcomes['citation_count'] > 1000:
            outcomes['precedent_setting'] = True
        
        return outcomes
    
    def _extract_metadata(self, opinion_data: Dict, case: Dict) -> Dict:
        """Extract metadata for analysis."""
        metadata = {
            'court': 'scotus',
            'author': opinion_data.get('author_str', ''),
            'author_id': opinion_data.get('author', ''),
            'per_curiam': opinion_data.get('per_curiam', False),
            'opinion_type': opinion_data.get('type', 'unknown'),
            'download_url': opinion_data.get('download_url', ''),
            'absolute_url': opinion_data.get('absolute_url', ''),
            'word_count': len(case.get('opinion_full_text', '').split()),
            'citation_count': 0,  # Will be enriched
            'area_of_law': 'unknown'  # Will be enriched
        }
        
        return metadata
    
    def _validate_case(self, case: Dict) -> bool:
        """
        Validate case has minimum data for analysis.
        
        Requirements:
        - Has at least one opinion text (>500 chars)
        - Has year
        - Has case name
        """
        # Check opinion text
        opinion_texts = [
            case.get('majority_opinion', ''),
            case.get('dissenting_opinion', ''),
            case.get('concurring_opinion', ''),
            case.get('opinion_full_text', '')
        ]
        
        has_text = any(len(text) > 500 for text in opinion_texts)
        has_year = case.get('year', 0) > 1788
        has_name = len(case.get('case_name', '')) > 5
        
        return has_text and has_year and has_name
    
    def supplement_with_vote_data(self):
        """
        Supplement with vote data from Supreme Court Database.
        
        This adds:
        - Actual vote counts (7-2, 5-4, etc.)
        - Justice votes
        - Case disposition
        """
        logger.info("Vote data supplementation would require Supreme Court Database")
        logger.info("For now, we focus on opinion text and citation data")
        # TODO: Integrate with SCDB in future enhancement
    
    def enrich_with_citations(self):
        """
        Enrich cases with forward citation counts.
        
        Uses CourtListener citation network.
        """
        logger.info(f"Enriching {len(self.cases)} cases with citations...")
        
        for i, case in enumerate(self.cases):
            if i % 100 == 0:
                logger.info(f"Enriching citations: {i}/{len(self.cases)}")
                self.save_checkpoint()
            
            case_id = case['case_id']
            
            # Query citations to this case
            params = {
                'cites': case_id,
                'page_size': 1  # Just need count
            }
            
            data = self.api_request('/opinions/', params)
            if data:
                citation_count = data.get('count', 0)
                case['outcome']['citation_count'] = citation_count
                case['metadata']['citation_count'] = citation_count
        
        self.save_checkpoint()
        logger.info("Citation enrichment complete")
    
    def collect_sample(self, n_cases: int = 1000):
        """
        Collect sample of recent cases for testing.
        
        Parameters
        ----------
        n_cases : int
            Number of cases to collect
        """
        logger.info(f"Collecting sample of {n_cases} recent cases...")
        
        # Get most recent cases
        self.collect_supreme_court_cases(
            start_year=2000,  # Modern era
            end_year=datetime.now().year,
            max_cases=n_cases
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.cases:
            return {'total_cases': 0}
        
        stats = {
            'total_cases': len(self.cases),
            'year_range': (
                min(c['year'] for c in self.cases if c.get('year', 0) > 0),
                max(c['year'] for c in self.cases if c.get('year', 0) > 0)
            ),
            'with_majority_opinion': sum(1 for c in self.cases if len(c.get('majority_opinion', '')) > 500),
            'with_dissent': sum(1 for c in self.cases if len(c.get('dissenting_opinion', '')) > 500),
            'with_citations': sum(1 for c in self.cases if c['outcome'].get('citation_count', 0) > 0),
            'avg_opinion_length': int(np.mean([
                len(c.get('majority_opinion', '').split()) 
                for c in self.cases 
                if c.get('majority_opinion')
            ])) if self.cases else 0,
            'precedent_setting': sum(1 for c in self.cases if c['outcome'].get('precedent_setting', False))
        }
        
        return stats


def main():
    """Main collection script."""
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Collect Supreme Court data')
    parser.add_argument('--api-key', help='CourtListener API key')
    parser.add_argument('--sample', type=int, help='Collect sample of N cases')
    parser.add_argument('--start-year', type=int, default=1789, help='Start year')
    parser.add_argument('--end-year', type=int, help='End year')
    parser.add_argument('--enrich-citations', action='store_true', help='Enrich with citation counts')
    parser.add_argument('--stats-only', action='store_true', help='Show stats and exit')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = SupremeCourtDataCollector(api_key=args.api_key)
    
    if args.stats_only:
        stats = collector.get_statistics()
        print("\n" + "="*80)
        print("SUPREME COURT DATA COLLECTION STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key:25s}: {value}")
        print("="*80)
        return
    
    # Collect data
    if args.sample:
        collector.collect_sample(n_cases=args.sample)
    else:
        collector.collect_supreme_court_cases(
            start_year=args.start_year,
            end_year=args.end_year
        )
    
    # Enrich with citations if requested
    if args.enrich_citations:
        collector.enrich_with_citations()
    
    # Show final stats
    stats = collector.get_statistics()
    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    for key, value in stats.items():
        print(f"{key:25s}: {value}")
    print("="*80)
    print(f"\nData saved to: {collector.data_file}")


if __name__ == '__main__':
    main()

