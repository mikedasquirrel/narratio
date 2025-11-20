# Movie Dataset Analysis - Complete Pipeline

## Overview

This pipeline merges three major movie datasets and runs all applicable narrative transformers to analyze movie narratives at scale.

## Datasets Merged

1. **CMU Movie Summary Corpus** (42,306 movies)
   - Source: Carnegie Mellon University
   - Plot summaries from Wikipedia
   - Comprehensive metadata

2. **IMDB Movies Complete** (6,047 movies)
   - Detailed cast and character information
   - Box office revenue
   - Rich narrative features

3. **MovieLens ml-latest-small** (9,742 movies)
   - Real user ratings (100,836 ratings from 610 users)
   - Links to IMDB for joining
   - Engagement metrics

## Merged Dataset Output

**File:** `data/domains/movies_merged_complete.json`

**Expected Coverage:**
- ~42,000 total movies
- ~6,000 with plot summaries (~15%)
- ~6,000 with cast information (~15%)
- ~8,000 with box office data (~20%)
- ~9,000 with user ratings (~22%)
- ~6,000 "rich" movies with 60%+ features complete

**Schema:**
```json
{
  "movie_id": "unique_id",
  "wikipedia_id": 123,
  "imdb_id": "tt0114709",
  "movielens_id": 1,
  "title": "Movie Title",
  "year": 2023,
  "genres": ["Action", "Drama"],
  "runtime": 120.0,
  "plot_summary": "Full plot text...",
  "actors": ["Actor 1", "Actor 2"],
  "characters": ["Character 1", "Character 2"],
  "box_office_revenue": 1000000.0,
  "avg_rating": 4.2,
  "num_ratings": 500,
  "has_plot": true,
  "has_cast": true,
  "has_box_office": true,
  "has_ratings": true,
  "feature_completeness": 0.83,
  "full_narrative": "Constructed narrative combining all fields..."
}
```

## Transformers Tested (25 Total)

### Text-Based (NLP) - Require plot_summary
1. Linguistic Patterns - linguistic features
2. Nominative Analysis - name/entity analysis
3. Emotional Resonance - emotional content
4. Conflict & Tension - conflict detection
5. Suspense & Mystery - suspense elements
6. Narrative Potential - story potential
7. Authenticity - authenticity signals
8. Cultural Context - cultural references
9. Phonetic Analysis - sound patterns
10. Information Theory - info density

### Statistical - Use structured metadata
11. Statistical - statistical features
12. Quantitative - quantitative patterns
13. Social Status - status markers
14. Optics - observability
15. Framing - framing effects
16. Discoverability - discovery patterns

### Hybrid - Multiple data types
17. Ensemble Narrative - ensemble approach
18. Universal Nominative - universal patterns
19. Nominative Richness - name richness
20. Hierarchical Nominative - hierarchical names
21. Cognitive Fluency - cognitive ease
22. Self-Perception - self-reference
23. Expertise & Authority - authority signals
24. Namespace Ecology - naming ecology
25. Relational Value - relational features

## Usage

### Quick Start (Run Everything)

```bash
chmod +x run_movie_analysis_complete.sh
./run_movie_analysis_complete.sh
```

This will:
1. Merge all three datasets (~5 minutes)
2. Run all 25 transformers (~45-60 minutes)
3. Generate results and logs

### Step by Step

```bash
# Step 1: Merge datasets
python3 scripts/merge_movie_datasets.py

# Step 2: Run transformers
python3 run_all_transformers_movies.py
```

## Progress Tracking

The system generates multiple output files to track progress:

### Real-Time Monitoring

```bash
# Watch console output
tail -f movie_transformer_progress.log

# Check JSON progress (which transformer is running)
cat movie_transformer_progress.json
```

### Progress Files

1. **movie_transformer_progress.log** - Detailed timestamped log
2. **movie_transformer_progress.json** - Real-time progress (JSON)
3. **movie_transformer_results.json** - Final results with rankings

### Example Log Output

```
[14:23:15] INFO     | Loading merged movie dataset...
[14:23:18] INFO     | ✓ Loaded 42,306 movies
[14:23:18] INFO     | Configured 25 transformers
[14:23:18] INFO     | 
[14:23:18] INFO     | ================================================================================
[14:23:18] INFO     | BEGINNING TRANSFORMER ANALYSIS
[14:23:18] INFO     | ================================================================================
[14:23:18] INFO     | 
[14:23:18] INFO     | ================================================================================
[14:23:18] INFO     | Transformer 1/25: Linguistic Patterns
[14:23:18] INFO     | Category: NLP | Data Type: text | Requirements: ['has_plot']
[14:23:18] INFO     |   Valid movies: 6,047 / 42,306 (14.3%)
[14:23:18] INFO     |   Train: 4,837 | Test: 1,210
[14:23:20] INFO     |   Extracting text features...
[14:23:21] INFO     |   ✓ Features extracted (train samples: 4837)
[14:23:21] INFO     |   Fitting Linguistic Patterns...
[14:23:45] INFO     |   ✓ Generated 156 features
[14:23:45] INFO     |   Transforming test set...
[14:23:47] INFO     |   ✓ Test set transformed
[14:23:47] INFO     |   Evaluating predictive power...
[14:23:48] INFO     |   ✓ COMPLETE
[14:23:48] INFO     |     Train R²: 0.4523 | Test R²: 0.3891
[14:23:48] INFO     |     Train RMSE: 0.8234 | Test RMSE: 0.8567
[14:23:48] INFO     |     Time: 30.2s
```

## Results Analysis

### View Top Transformers

```bash
# Pretty print results
cat movie_transformer_results.json | python3 -m json.tool | less

# Quick summary
python3 -c "import json; r=json.load(open('movie_transformer_results.json')); print('Top 5:'); [print(f'{i+1}. {t[\"name\"]}: R²={t[\"test_r2\"]:.4f}') for i,t in enumerate(r['top_10'][:5])]"
```

### Results Structure

```json
{
  "metadata": {
    "created_at": "2025-11-16T...",
    "total_movies": 42306,
    "successful_transformers": 23,
    "failed_transformers": 2,
    "execution_time": 3245.6
  },
  "results": [...],
  "top_10": [
    {
      "name": "Transformer Name",
      "category": "NLP",
      "test_r2": 0.4523,
      "features_generated": 156,
      "valid_movies": 6047,
      "time_seconds": 30.2
    }
  ],
  "category_summary": {
    "NLP": {
      "count": 10,
      "avg_r2": 0.3456,
      "max_r2": 0.4523
    }
  }
}
```

## Troubleshooting

### Issue: Merged dataset not found

```bash
# Run merge first
python3 scripts/merge_movie_datasets.py
```

### Issue: Transformer fails

Check `movie_transformer_progress.log` for detailed error messages. The pipeline continues even if individual transformers fail.

### Issue: Out of memory

Reduce dataset size by filtering:

```python
# In run_all_transformers_movies.py, add after loading:
all_movies = all_movies[:10000]  # Test with 10K movies
```

## Expected Execution Time

- **Merge:** ~5 minutes
- **Transformers:** ~45-60 minutes (varies by system)
  - Text transformers: ~2-3 min each
  - Statistical transformers: ~30 sec each
  - Hybrid transformers: ~1-2 min each
- **Total:** ~50-65 minutes

## Output Files Summary

```
data/domains/movies_merged_complete.json    # 42K merged movies (~200MB)
movie_transformer_results.json              # Final results (~50KB)
movie_transformer_progress.log              # Detailed log (~500KB)
movie_transformer_progress.json             # Real-time progress (~5KB)
```

## Next Steps

After analysis completes:

1. Review top transformers in results
2. Analyze which features predict success
3. Compare to sports domain results
4. Add movie route to Flask app for visualization
5. Run deeper analysis on high-performing transformers

## Integration with Web App

To add to the Flask app:

1. Create `routes/movies.py` (similar to `routes/music.py`)
2. Add movie blueprint to `app.py`
3. Create `templates/movies_dashboard.html`
4. View results at `http://127.0.0.1:5000/movies`

---

**Questions?** Check logs or contact the development team.

