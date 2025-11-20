"""
UFC/MMA Domain - Narrative Analysis

Pure 1v1 matchups testing individual narrative power.
Expected п ≈ 0.73 (highest individual sport).
Potential 3rd PASSING domain if |r| > 0.85.

Modules:
- collect_ufc_data: Comprehensive fight data collection
- generate_fighter_narratives: Nominative-rich fight narratives
- analyze_ufc_complete: Apply ALL 25 transformers
- discover_ufc_contexts: Data-first context discovery
- test_ufc_betting_edge: Narrative vs Vegas odds
- ufc_fighter_analyzer: Individual fighter analysis
"""

__version__ = "1.0.0"
__author__ = "Narrative Optimization Framework"
__domain__ = "UFC"
__expected_narrativity__ = 0.73
__pass_threshold__ = 0.85  # Required |r| to pass with п=0.73, κ=0.6

