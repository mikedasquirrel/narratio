"""
Paper Parser for Meta-Nominative Analysis

Consolidates papers from multiple sources, deduplicates, and creates unified dataset.
"""

import json
import re
from typing import List, Dict, Optional, Set
from pathlib import Path
from collections import defaultdict
import difflib


class PaperParser:
    """Parse and consolidate papers from multiple sources."""
    
    def __init__(self):
        """Initialize paper parser."""
        self.papers = []
        self.deduplicated_papers = []
        self.author_index = defaultdict(list)
        
    def load_from_sources(self, data_dir: Path) -> List[Dict]:
        """
        Load papers from all source files.
        
        Args:
            data_dir: Directory containing paper JSON files
            
        Returns:
            List of all papers
        """
        print(f"\n{'='*80}")
        print("LOADING PAPERS FROM ALL SOURCES")
        print(f"{'='*80}\n")
        
        sources = ['papers_pubmed.json', 'papers_scholar.json', 'papers_manual.json']
        all_papers = []
        
        for source_file in sources:
            path = data_dir / source_file
            print(f"Checking: {path}")
            if path.exists():
                print(f"Loading {source_file}...", end=" ")
                with open(path) as f:
                    data = json.load(f)
                    papers = data.get('papers', [])
                    all_papers.extend(papers)
                    print(f"✓ {len(papers)} papers")
            else:
                print(f"  {source_file} not found at {path}")
        
        self.papers = all_papers
        print(f"\n✓ Total papers loaded: {len(all_papers)}")
        return all_papers
    
    def deduplicate(self, papers: List[Dict]) -> List[Dict]:
        """
        Deduplicate papers based on title similarity.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Deduplicated list
        """
        print(f"\n{'='*80}")
        print("DEDUPLICATING PAPERS")
        print(f"{'='*80}\n")
        
        seen_titles = {}
        unique_papers = []
        
        for paper in papers:
            title = paper.get('title', '').strip().lower()
            if not title:
                continue
            
            # Check for similar titles
            is_duplicate = False
            for seen_title in seen_titles.keys():
                similarity = difflib.SequenceMatcher(None, title, seen_title).ratio()
                if similarity > 0.85:  # 85% similarity threshold
                    # Merge data from duplicate
                    existing_idx = seen_titles[seen_title]
                    unique_papers[existing_idx] = self._merge_papers(
                        unique_papers[existing_idx],
                        paper
                    )
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles[title] = len(unique_papers)
                unique_papers.append(paper)
        
        print(f"Original papers: {len(papers)}")
        print(f"Unique papers: {len(unique_papers)}")
        print(f"Duplicates removed: {len(papers) - len(unique_papers)}")
        
        self.deduplicated_papers = unique_papers
        return unique_papers
    
    def _merge_papers(self, paper1: Dict, paper2: Dict) -> Dict:
        """Merge data from two duplicate papers."""
        merged = paper1.copy()
        
        # Keep longer abstract
        if len(paper2.get('abstract', '')) > len(merged.get('abstract', '')):
            merged['abstract'] = paper2['abstract']
        
        # Keep more complete author list
        if len(paper2.get('authors', [])) > len(merged.get('authors', [])):
            merged['authors'] = paper2['authors']
            merged['lead_author'] = paper2.get('lead_author')
        
        # Prefer effect size if missing
        if not merged.get('effect_size') and paper2.get('effect_size'):
            merged['effect_size'] = paper2['effect_size']
        
        # Prefer sample size if missing
        if not merged.get('sample_size') and paper2.get('sample_size'):
            merged['sample_size'] = paper2['sample_size']
        
        # Merge sources
        sources = set([merged.get('source', '')])
        sources.add(paper2.get('source', ''))
        merged['sources'] = list(sources - {''})
        
        return merged
    
    def normalize_effect_sizes(self, papers: List[Dict]) -> List[Dict]:
        """
        Normalize all effect sizes to correlation r.
        
        Args:
            papers: List of papers
            
        Returns:
            Papers with normalized effect sizes
        """
        print(f"\n{'='*80}")
        print("NORMALIZING EFFECT SIZES")
        print(f"{'='*80}\n")
        
        normalized_count = 0
        
        for paper in papers:
            effect = paper.get('effect_size')
            if not effect or not isinstance(effect, dict):
                continue
            
            metric = effect.get('metric', '').lower()
            value = effect.get('value')
            
            if value is None:
                continue
            
            # Convert to correlation r
            if 'correlation' in metric or metric == 'r':
                r = abs(value)
            elif 'cohens_d' in metric or metric == 'd':
                # Convert Cohen's d to r
                r = value / ((value**2 + 4)**0.5)
            elif 'odds' in metric or 'or' in metric:
                # Convert odds ratio to r (approximation)
                if value > 0:
                    r = abs((value - 1) / (value + 1))
                else:
                    r = 0
            elif 'beta' in metric or 'β' in metric:
                # Beta coefficients are similar to correlations
                r = abs(value)
            elif 'r_squared' in metric or 'r2' in metric:
                # R² to r
                r = value ** 0.5
            else:
                # Unknown metric, keep as is
                r = abs(value)
            
            # Clamp to [0, 1]
            r = max(0, min(1, r))
            
            paper['effect_size_normalized'] = {
                'r': r,
                'original_metric': metric,
                'original_value': value
            }
            
            normalized_count += 1
        
        print(f"✓ Normalized {normalized_count} effect sizes to correlation r")
        return papers
    
    def build_author_index(self, papers: List[Dict]) -> Dict[str, List[int]]:
        """
        Build index mapping authors to their paper indices.
        
        Args:
            papers: List of papers
            
        Returns:
            Dictionary mapping author name to paper indices
        """
        print(f"\n{'='*80}")
        print("BUILDING AUTHOR INDEX")
        print(f"{'='*80}\n")
        
        self.author_index = defaultdict(list)
        
        for i, paper in enumerate(papers):
            authors = paper.get('authors', [])
            for author in authors:
                if not author or 'full_name' not in author:
                    continue
                
                name = author['full_name'].strip()
                if name:
                    self.author_index[name].append(i)
        
        print(f"✓ Indexed {len(self.author_index)} unique authors")
        if len(self.author_index) > 0:
            print(f"  Average papers per author: {sum(len(papers) for papers in self.author_index.values()) / len(self.author_index):.1f}")
        
        return dict(self.author_index)
    
    def validate_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Validate and filter papers for quality.
        
        Args:
            papers: List of papers
            
        Returns:
            Filtered list of valid papers
        """
        print(f"\n{'='*80}")
        print("VALIDATING PAPERS")
        print(f"{'='*80}\n")
        
        valid_papers = []
        
        required_fields = ['title', 'authors']
        
        for paper in papers:
            # Check required fields
            if not all(paper.get(field) for field in required_fields):
                continue
            
            # Must have at least one author
            if not paper.get('authors') or len(paper['authors']) == 0:
                continue
            
            # Title must be reasonable length
            title = paper.get('title', '')
            if len(title) < 10 or len(title) > 500:
                continue
            
            valid_papers.append(paper)
        
        print(f"Original papers: {len(papers)}")
        print(f"Valid papers: {len(valid_papers)}")
        print(f"Filtered out: {len(papers) - len(valid_papers)}")
        
        return valid_papers
    
    def enrich_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Add derived fields to papers.
        
        Args:
            papers: List of papers
            
        Returns:
            Enriched papers
        """
        print(f"\n{'='*80}")
        print("ENRICHING PAPERS WITH DERIVED FIELDS")
        print(f"{'='*80}\n")
        
        for paper in papers:
            # Add unique ID
            paper['paper_id'] = self._generate_paper_id(paper)
            
            # Count authors
            paper['author_count'] = len(paper.get('authors', []))
            
            # Extract first/last author names
            authors = paper.get('authors', [])
            if authors:
                paper['first_author_name'] = authors[0].get('full_name', '')
                paper['last_author_name'] = authors[-1].get('full_name', '')
            
            # Classify paper quality (heuristic)
            paper['quality_score'] = self._estimate_quality(paper)
            
            # Flag papers with strong findings
            effect = paper.get('effect_size_normalized', {})
            paper['strong_finding'] = effect.get('r', 0) > 0.25
            
        print(f"✓ Enriched {len(papers)} papers")
        return papers
    
    def _generate_paper_id(self, paper: Dict) -> str:
        """Generate unique ID for paper."""
        # Use PMID if available
        if paper.get('pmid'):
            return f"pmid_{paper['pmid']}"
        
        # Otherwise use title hash
        title = paper.get('title', '').strip().lower()
        title_hash = str(hash(title))[-8:]
        return f"paper_{title_hash}"
    
    def _estimate_quality(self, paper: Dict) -> float:
        """Estimate paper quality score (0-1)."""
        score = 0.0
        
        # Has abstract
        if paper.get('abstract') and len(paper.get('abstract', '')) > 100:
            score += 0.3
        
        # Has effect size
        if paper.get('effect_size'):
            score += 0.2
        
        # Has sample size
        if paper.get('sample_size'):
            score += 0.2
        
        # Recent publication
        year = paper.get('year')
        if year and year >= 2000:
            score += 0.15
        
        # Multiple authors
        if paper.get('author_count', 0) >= 2:
            score += 0.15
        
        return min(1.0, score)
    
    def save_consolidated(self, papers: List[Dict], output_path: Optional[Path] = None) -> Path:
        """Save consolidated dataset."""
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative' / 'papers_consolidated.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive metadata
        metadata = {
            'total_papers': len(papers),
            'papers_with_abstracts': sum(1 for p in papers if p.get('abstract')),
            'papers_with_effect_sizes': sum(1 for p in papers if p.get('effect_size_normalized')),
            'papers_with_sample_sizes': sum(1 for p in papers if p.get('sample_size')),
            'total_authors': len(self.author_index),
            'year_range': self._get_year_range(papers),
            'topics': self._get_topic_distribution(papers),
            'finding_types': self._get_finding_distribution(papers),
            'quality_distribution': self._get_quality_distribution(papers)
        }
        
        output_data = {
            'metadata': metadata,
            'papers': papers,
            'author_index': dict(self.author_index)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved consolidated dataset to: {output_path}")
        return output_path
    
    def _get_year_range(self, papers: List[Dict]) -> Dict:
        """Get year range of papers."""
        years = [p['year'] for p in papers if p.get('year')]
        if not years:
            return {'min': None, 'max': None}
        return {'min': min(years), 'max': max(years)}
    
    def _get_topic_distribution(self, papers: List[Dict]) -> Dict:
        """Get distribution of research topics."""
        topics = defaultdict(int)
        for paper in papers:
            topic = paper.get('research_topic', 'unknown')
            topics[topic] += 1
        return dict(topics)
    
    def _get_finding_distribution(self, papers: List[Dict]) -> Dict:
        """Get distribution of finding types."""
        findings = defaultdict(int)
        for paper in papers:
            finding = paper.get('finding_type', 'unknown')
            findings[finding] += 1
        return dict(findings)
    
    def _get_quality_distribution(self, papers: List[Dict]) -> Dict:
        """Get distribution of quality scores."""
        high = sum(1 for p in papers if p.get('quality_score', 0) >= 0.7)
        medium = sum(1 for p in papers if 0.4 <= p.get('quality_score', 0) < 0.7)
        low = sum(1 for p in papers if p.get('quality_score', 0) < 0.4)
        return {'high': high, 'medium': medium, 'low': low}
    
    def process_all(self, data_dir: Path) -> List[Dict]:
        """
        Run complete processing pipeline.
        
        Args:
            data_dir: Directory with source files
            
        Returns:
            Processed papers
        """
        papers = self.load_from_sources(data_dir)
        papers = self.validate_papers(papers)
        papers = self.deduplicate(papers)
        papers = self.normalize_effect_sizes(papers)
        papers = self.enrich_papers(papers)
        self.build_author_index(papers)
        self.save_consolidated(papers)
        
        self.print_summary()
        
        return papers
    
    def print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*80}")
        print("PAPER PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        papers = self.deduplicated_papers
        
        print(f"\nFinal dataset:")
        print(f"  Total papers: {len(papers)}")
        print(f"  Unique authors: {len(self.author_index)}")
        print(f"  Papers with effect sizes: {sum(1 for p in papers if p.get('effect_size_normalized'))}")
        
        # Effect size distribution
        effects = [p['effect_size_normalized']['r'] for p in papers if p.get('effect_size_normalized')]
        if effects:
            print(f"\nEffect size distribution (r):")
            print(f"  Mean: {sum(effects)/len(effects):.3f}")
            print(f"  Median: {sorted(effects)[len(effects)//2]:.3f}")
            print(f"  Range: {min(effects):.3f} - {max(effects):.3f}")
        
        print()


def main():
    """Run paper parser."""
    data_dir = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    parser = PaperParser()
    papers = parser.process_all(data_dir)
    
    print(f"{'='*80}")
    print("✓ Paper parsing complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

