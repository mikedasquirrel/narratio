"""
Researcher Metadata Collector for Meta-Nominative Analysis

Collects metadata about researchers: h-index, institution prestige, career stage, etc.
"""

import time
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import requests
from collections import defaultdict


class ResearcherMetadataCollector:
    """Collect metadata about nominative determinism researchers."""
    
    def __init__(self):
        """Initialize metadata collector."""
        self.researchers = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Institution prestige tiers (QS World University Rankings approximate)
        self.institution_tiers = {
            'harvard': 1, 'stanford': 1, 'mit': 1, 'cambridge': 1, 'oxford': 1,
            'yale': 1, 'princeton': 1, 'columbia': 1, 'caltech': 1, 'chicago': 1,
            'ucl': 2, 'penn': 2, 'cornell': 2, 'michigan': 2, 'toronto': 2,
            'northwestern': 2, 'duke': 2, 'johns hopkins': 2, 'ucsd': 2,
            'state university': 3, 'public': 3, 'regional': 4, 'college': 4
        }
    
    def extract_unique_researchers(self, papers: List[Dict]) -> Dict[str, Dict]:
        """
        Extract unique researchers from paper list.
        
        Args:
            papers: List of paper dictionaries with author info
            
        Returns:
            Dictionary of researcher data keyed by name
        """
        print(f"\n{'='*80}")
        print("EXTRACTING UNIQUE RESEARCHERS")
        print(f"{'='*80}\n")
        
        researcher_papers = defaultdict(list)
        
        for paper in papers:
            authors = paper.get('authors', [])
            for i, author in enumerate(authors):
                if not author or 'full_name' not in author:
                    continue
                
                name = author['full_name'].strip()
                if not name or len(name) < 3:
                    continue
                
                # Store paper association
                researcher_papers[name].append({
                    'paper_id': paper.get('pmid') or paper.get('title', '')[:50],
                    'title': paper.get('title', ''),
                    'year': paper.get('year'),
                    'is_lead': (i == 0),
                    'effect_size': paper.get('effect_size'),
                    'finding_type': paper.get('finding_type'),
                    'research_topic': paper.get('research_topic', '')
                })
        
        print(f"Found {len(researcher_papers)} unique researchers")
        print(f"Total papers: {sum(len(papers) for papers in researcher_papers.values())}")
        
        # Build researcher profiles
        for name, papers in researcher_papers.items():
            self.researchers[name] = {
                'name': name,
                'papers': papers,
                'paper_count': len(papers),
                'lead_author_count': sum(1 for p in papers if p['is_lead']),
                'years_active': self._get_active_years(papers),
                'topics_studied': self._get_topics(papers),
                'h_index': None,  # To be filled
                'institution': None,  # To be filled
                'institution_tier': None,  # To be filled
                'years_since_phd': None,  # To be filled
                'career_stage': None,  # To be filled
                'average_effect_size': self._calculate_avg_effect_size(papers)
            }
        
        return self.researchers
    
    def _get_active_years(self, papers: List[Dict]) -> Dict:
        """Calculate years active for researcher."""
        years = [p['year'] for p in papers if p['year']]
        if not years:
            return {'first': None, 'last': None, 'span': None}
        
        return {
            'first': min(years),
            'last': max(years),
            'span': max(years) - min(years) + 1
        }
    
    def _get_topics(self, papers: List[Dict]) -> List[str]:
        """Get unique topics studied by researcher."""
        topics = set()
        for paper in papers:
            topic = paper.get('research_topic', '')
            if topic and topic != 'unknown':
                # Split comma-separated topics
                for t in topic.split(','):
                    topics.add(t.strip())
        return list(topics)
    
    def _calculate_avg_effect_size(self, papers: List[Dict]) -> Optional[float]:
        """Calculate average effect size reported by researcher."""
        effect_sizes = []
        
        for paper in papers:
            effect = paper.get('effect_size')
            if effect and isinstance(effect, dict):
                value = effect.get('value')
                if value is not None:
                    # Normalize different metrics to correlation r
                    metric = effect.get('metric', '').lower()
                    if 'cohens_d' in metric:
                        # Convert Cohen's d to r (approximation)
                        r = value / ((value**2 + 4)**0.5)
                        effect_sizes.append(r)
                    elif 'odds' in metric:
                        # Convert OR to r (log approximation)
                        if value > 0:
                            r = (value - 1) / (value + 1)
                            effect_sizes.append(abs(r))
                    else:
                        effect_sizes.append(abs(value))
        
        return sum(effect_sizes) / len(effect_sizes) if effect_sizes else None
    
    def enrich_with_google_scholar(self, researcher_name: str, delay: float = 5.0) -> Dict:
        """
        Fetch h-index and institution from Google Scholar profile.
        
        Args:
            researcher_name: Full name of researcher
            delay: Seconds to wait between requests
            
        Returns:
            Dictionary with h-index and institution
        """
        time.sleep(delay)  # Rate limiting
        
        try:
            # Search for author profile
            search_url = "https://scholar.google.com/citations"
            params = {
                'view_op': 'search_authors',
                'mauthors': researcher_name,
                'hl': 'en'
            }
            
            response = self.session.get(search_url, params=params, timeout=30)
            
            if response.status_code == 429:
                print(f"  Rate limited, skipping {researcher_name}")
                return {'h_index': None, 'institution': None}
            
            # Parse response (basic extraction)
            text = response.text
            
            # Try to extract h-index
            h_match = re.search(r'h-index.*?(\d+)', text, re.IGNORECASE)
            h_index = int(h_match.group(1)) if h_match else None
            
            # Try to extract institution
            inst_match = re.search(r'<div class="gs_ai_aff">(.*?)</div>', text)
            institution = inst_match.group(1) if inst_match else None
            
            return {
                'h_index': h_index,
                'institution': institution
            }
            
        except Exception as e:
            print(f"  Error fetching data for {researcher_name}: {e}")
            return {'h_index': None, 'institution': None}
    
    def infer_institution_tier(self, institution: Optional[str]) -> Optional[int]:
        """Infer institution prestige tier from name."""
        if not institution:
            return None
        
        inst_lower = institution.lower()
        
        for key, tier in self.institution_tiers.items():
            if key in inst_lower:
                return tier
        
        # Default to tier 3 if unknown
        return 3
    
    def infer_career_stage(self, years_active: Dict) -> Optional[str]:
        """Infer career stage from publication history."""
        span = years_active.get('span')
        last_year = years_active.get('last')
        
        if not span or not last_year:
            return None
        
        # Calculate current year
        current_year = 2025
        years_since_last = current_year - last_year
        
        # Heuristics
        if span <= 5:
            return 'early'
        elif span <= 15:
            return 'mid'
        else:
            return 'late'
    
    def estimate_years_since_phd(self, years_active: Dict, paper_count: int) -> Optional[int]:
        """Estimate years since PhD based on publication history."""
        first_year = years_active.get('first')
        
        if not first_year:
            return None
        
        # Assume first publication ~2 years after PhD
        estimated_phd_year = first_year - 2
        current_year = 2025
        
        years_since = current_year - estimated_phd_year
        
        # Sanity checks
        if years_since < 0 or years_since > 50:
            return None
        
        return years_since
    
    def enrich_all_researchers(self, use_google_scholar: bool = False) -> Dict[str, Dict]:
        """
        Enrich all researchers with metadata.
        
        Args:
            use_google_scholar: Whether to fetch from Google Scholar (slow!)
            
        Returns:
            Dictionary of enriched researcher data
        """
        print(f"\n{'='*80}")
        print(f"ENRICHING RESEARCHER METADATA")
        print(f"{'='*80}\n")
        
        total = len(self.researchers)
        
        for i, (name, data) in enumerate(self.researchers.items(), 1):
            print(f"[{i}/{total}] {name}...", end=" ")
            
            # Infer career stage
            data['career_stage'] = self.infer_career_stage(data['years_active'])
            
            # Estimate years since PhD
            data['years_since_phd'] = self.estimate_years_since_phd(
                data['years_active'], 
                data['paper_count']
            )
            
            # Optionally fetch from Google Scholar
            if use_google_scholar and i <= 20:  # Limit to first 20 to avoid blocking
                scholar_data = self.enrich_with_google_scholar(name)
                data['h_index'] = scholar_data['h_index']
                data['institution'] = scholar_data['institution']
                data['institution_tier'] = self.infer_institution_tier(scholar_data['institution'])
                print(f"✓ (h={data['h_index']}, tier={data['institution_tier']})")
            else:
                # Use heuristics
                data['h_index'] = self._estimate_h_index(data['paper_count'], data['years_active'])
                data['institution_tier'] = 2  # Default to tier 2
                print(f"✓ (estimated)")
        
        print(f"\n✓ Enriched {len(self.researchers)} researchers")
        return self.researchers
    
    def _estimate_h_index(self, paper_count: int, years_active: Dict) -> Optional[int]:
        """Estimate h-index from paper count and career length."""
        span = years_active.get('span', 1)
        if not span:
            return None
        
        # Rough heuristic: h-index ≈ sqrt(papers) * (span/10)
        estimated = int((paper_count ** 0.5) * (span / 10))
        return max(1, min(estimated, paper_count))
    
    def save_metadata(self, output_path: Optional[Path] = None) -> Path:
        """Save researcher metadata to JSON."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative' / 'researchers_metadata.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'collection_date': time.strftime('%Y-%m-%d'),
                'total_researchers': len(self.researchers),
                'researchers': self.researchers
            }, f, indent=2)
        
        print(f"\n✓ Saved metadata for {len(self.researchers)} researchers to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("RESEARCHER METADATA SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nTotal researchers: {len(self.researchers)}")
        
        # Career stages
        stages = defaultdict(int)
        for r in self.researchers.values():
            stage = r.get('career_stage', 'unknown')
            stages[stage] += 1
        print(f"\nCareer stages: {dict(stages)}")
        
        # Paper counts
        paper_counts = [r['paper_count'] for r in self.researchers.values()]
        if paper_counts:
            print(f"\nPapers per researcher: {sum(paper_counts)/len(paper_counts):.1f} avg")
            print(f"  Range: {min(paper_counts)} - {max(paper_counts)}")
        
        # Effect sizes
        avg_effects = [r['average_effect_size'] for r in self.researchers.values() if r['average_effect_size']]
        if avg_effects:
            print(f"\nAverage effect size reported: {sum(avg_effects)/len(avg_effects):.3f}")
            print(f"  Range: {min(avg_effects):.3f} - {max(avg_effects):.3f}")
        
        # Topics
        all_topics = set()
        for r in self.researchers.values():
            all_topics.update(r['topics_studied'])
        print(f"\nUnique research topics: {len(all_topics)}")
        print(f"  Topics: {', '.join(list(all_topics)[:10])}")


def main():
    """Run metadata collection on existing papers."""
    from pathlib import Path
    import json
    
    # Load papers from PubMed/Scholar collections
    data_dir = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    papers = []
    for source_file in ['papers_pubmed.json', 'papers_scholar.json']:
        path = data_dir / source_file
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                papers.extend(data.get('papers', []))
    
    if not papers:
        print("No papers found. Run paper collectors first.")
        return
    
    print(f"Loaded {len(papers)} papers")
    
    collector = ResearcherMetadataCollector()
    collector.extract_unique_researchers(papers)
    collector.enrich_all_researchers(use_google_scholar=False)  # Set True for real Scholar data
    collector.save_metadata()
    collector.print_summary()


if __name__ == "__main__":
    main()

