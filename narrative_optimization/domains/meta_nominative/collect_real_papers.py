"""
Collect Real Papers from PubMed for Meta-Nominative Analysis

Retrieves actual published research on nominative determinism with real authors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.domains.meta_nominative.collectors.pubmed_collector import PubMedCollector
from narrative_optimization.domains.meta_nominative.extractors.paper_parser import PaperParser
from narrative_optimization.domains.meta_nominative.collectors.researcher_metadata_collector import ResearcherMetadataCollector


def main():
    """Collect real papers from PubMed."""
    print(f"\n{'='*80}")
    print("COLLECTING REAL NOMINATIVE DETERMINISM PAPERS")
    print(f"{'='*80}\n")
    
    print("Step 1: Collecting from PubMed...")
    print("This will retrieve real published papers with actual authors.\n")
    
    # Collect from PubMed
    pubmed = PubMedCollector(email="research@analysis.study")
    papers = pubmed.collect(max_results=100)
    
    print(f"\nStep 2: Processing and consolidating papers...")
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    parser = PaperParser()
    papers = parser.process_all(data_dir)
    
    if len(papers) < 20:
        print(f"\n⚠️  WARNING: Only {len(papers)} papers collected.")
        print("This may not be sufficient for robust statistical analysis.")
        print("Consider:")
        print("  1. Running Google Scholar collector (slower but more results)")
        print("  2. Adding manual papers from known publications")
        return
    
    print(f"\nStep 3: Extracting researcher metadata...")
    metadata_collector = ResearcherMetadataCollector()
    metadata_collector.extract_unique_researchers(papers)
    metadata_collector.enrich_all_researchers(use_google_scholar=False)
    metadata_collector.save_metadata()
    metadata_collector.print_summary()
    
    print(f"\n{'='*80}")
    print("✓ REAL DATA COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nCollected:")
    print(f"  {len(papers)} real papers")
    print(f"  {len(metadata_collector.researchers)} real researchers")
    print(f"\nNext steps:")
    print("  python3 narrative_optimization/domains/meta_nominative/feature_extraction/name_field_fit.py")
    print("  python3 narrative_optimization/domains/meta_nominative/feature_extraction/name_characteristics.py")
    print("  python3 narrative_optimization/domains/meta_nominative/analyze_meta_nominative_complete.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

