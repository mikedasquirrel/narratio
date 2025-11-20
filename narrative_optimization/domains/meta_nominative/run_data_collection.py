"""
Unified Data Collection Pipeline for Meta-Nominative Analysis

Runs all collectors and consolidates papers into final dataset.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from narrative_optimization.domains.meta_nominative.extractors.paper_parser import PaperParser
from narrative_optimization.domains.meta_nominative.collectors.researcher_metadata_collector import ResearcherMetadataCollector


def main():
    """
    Run complete data collection pipeline.
    
    Steps:
    1. Parse and consolidate papers from all sources (manual, PubMed, Scholar)
    2. Extract unique researchers
    3. Enrich with metadata
    4. Save consolidated dataset
    """
    print(f"\n{'='*80}")
    print("META-NOMINATIVE DETERMINISM DATA COLLECTION")
    print(f"{'='*80}\n")
    
    # Get to novelization root, then to data
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'meta_nominative'
    
    # PHASE 1: Parse and consolidate papers
    print("\nPHASE 1: Paper Consolidation")
    print("-" * 80)
    
    parser = PaperParser()
    papers = parser.process_all(data_dir)
    
    # PHASE 2: Extract researcher metadata
    print(f"\n{'='*80}")
    print("PHASE 2: Researcher Metadata Collection")
    print("-" * 80)
    
    metadata_collector = ResearcherMetadataCollector()
    metadata_collector.extract_unique_researchers(papers)
    metadata_collector.enrich_all_researchers(use_google_scholar=False)
    metadata_collector.save_metadata()
    metadata_collector.print_summary()
    
    # Final summary
    print(f"\n{'='*80}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Final dataset:")
    print(f"  Papers: {len(papers)}")
    print(f"  Researchers: {len(metadata_collector.researchers)}")
    print(f"  Papers with effect sizes: {sum(1 for p in papers if p.get('effect_size_normalized'))}")
    print(f"\nFiles created:")
    print(f"  {data_dir / 'papers_consolidated.json'}")
    print(f"  {data_dir / 'researchers_metadata.json'}")
    print(f"\nNext steps:")
    print(f"  1. Calculate name-field fit scores")
    print(f"  2. Apply transformers to extract features")
    print(f"  3. Run statistical analysis")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

