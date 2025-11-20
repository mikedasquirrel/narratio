# Comprehensive Reanalysis Master Log

**Started**: 2025-11-16T08:29:50.952155
**Framework**: November 2025 Renovation
**Transformers**: Full suite with renovation features
**Mode**: Full

---


## Processing Domain: nba
Sample size: 11,979
Timestamp: 2025-11-16T08:29:50.952358

✓ Complete: 20 patterns discovered
  Significant: 19

[2/4] oscars
--------------------------------------------------------------------------------

## Processing Domain: oscars
Sample size: 1,500
Timestamp: 2025-11-16T08:53:16.801803

✗ Error: Found array with 0 sample(s) (shape=(0, 5000)) while a minimum of 1 is required by TfidfTransformer.

[3/4] music
--------------------------------------------------------------------------------

## Processing Domain: music
Sample size: 1,500
Timestamp: 2025-11-16T08:53:19.134178

✗ Failed: Failed to load domain

[4/4] wwe
--------------------------------------------------------------------------------

## Processing Domain: wwe
Sample size: 1,500
Timestamp: 2025-11-16T08:53:19.352392

✗ Failed: Failed to load domain

================================================================================
BATCH SUMMARY: Entertainment
================================================================================
Total: 4
✓ Successful: 1
✗ Failed: 3

  ✓ movies: 20 patterns, 19 significant (π=0.65)
  ✗ oscars: Found array with 0 sample(s) (shape=(0, 5000)) while a minimum of 1 is required by TfidfTransformer.
  ✗ music: Failed to load domain
  ✗ wwe: Failed to load domain
================================================================================

✓ Saved batch results: narrative_optimization/results/comprehensive_reanalysis/entertainment_results.json
