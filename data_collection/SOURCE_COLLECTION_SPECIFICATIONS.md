# Source-by-Source Narrative Collection Specifications

**Purpose**: Systematically collect large narrative corpora for unsupervised AI discovery  
**Philosophy**: Preserve EVERYTHING - sequence, spacing, ordering, timing  
**Method**: AI analysis, NO hardcoded categories, let patterns emerge

---

## Collection Principles

1. **Preserve Sequence**: Maintain exact ordering of narrative elements
2. **Preserve Spacing**: Keep paragraph breaks, pauses, gaps, rhythm
3. **No Categorization**: Don't label genres, types, themes - let AI discover
4. **Source Integrity**: Process each source separately, then meta-analyze
5. **Completeness**: Collect full narratives, not summaries
6. **Variety**: Maximum diversity in forms, cultures, eras, domains

---

## Source 1: Project Gutenberg (Literature)

**Target**: 10,000 complete novels and short stories  
**Why**: Rich narrative structure, complete texts, public domain

### Collection Strategy

```python
# Use Gutenberg API or direct download
sources = {
    'gutenberg_novels': {
        'target': 5000,
        'format': 'full_text',
        'time_period': 'all',  # 1600s-1920s
        'languages': ['english'],  # Start, expand to multilingual
        'outcomes': 'citation_count',  # Works still read/cited
    }
}
```

### What to Capture

- **Full text** (preserve all spacing, paragraphs, chapters)
- **Chapter boundaries** (mark but don't break sequence)
- **Paragraph breaks** (spacing rhythm)
- **Citation count** (still read = successful narrative)
- **Publication date** (for temporal decay analysis)

### Sequential Processing

```python
for novel in gutenberg_iterator():
    narrative_data = {
        'text': novel.full_text,  # COMPLETE, unmodified
        'chapters': novel.chapter_positions,  # Indices, not breaks
        'paragraph_breaks': novel.paragraph_positions,  # Spacing data
        'length': len(novel.full_text),
        'outcome': novel.citation_count
    }
    
    # Process sequentially
    sequence = process_sequential(narrative_data['text'])
    # sequence contains: embeddings, spacing, progression, rhythm
```

---

## Source 2: Movie Plot Summaries (CMU/Wikipedia)

**Target**: 50,000 movie plots  
**Why**: Compressed narratives, clear outcomes, diverse genres

### Collection Strategy

- CMU Movie Summary Corpus (42K movies)
- Wikipedia plot summaries
- IMDB plot keywords (for outcome correlation)

### What to Capture

- **Plot summary** (complete, as written)
- **Sentence ordering** (sequence matters)
- **Box office** (outcome)
- **Ratings** (IMDB, Rotten Tomatoes)
- **Awards** (Oscars, etc.)

### Sequential Processing

Plots are already compressed - preserve their compression:
- Sentence order is critical
- Paragraph breaks indicate act structure
- Length variation is data (some plots detailed, some sparse)

---

## Source 3: Sports Game Narratives

**Target**: 100,000+ game narratives across sports  
**Why**: Real outcomes, repeated structure, temporal variation

### Sources

- Existing: NBA (11K), NFL (3K), Tennis (75K), UFC (7K), Golf (7K), MLB (23K)
- Process game descriptions sequentially

### What to Capture

- **Pre-game narrative** (setup)
- **Play-by-play structure** (if available, preserve sequence)
- **Post-game narrative** (resolution)
- **Temporal markers** (quarter, inning, round - timing data)
- **Outcome** (win/loss, margin)

### Sequential Processing

Game narratives have natural temporal markers:
- Quarter/inning boundaries = act breaks
- Play sequences = beats
- Momentum shifts = progression changes
- Score changes = intensity markers

---

## Source 4: Historical Events (Wikipedia)

**Target**: 10,000 historical event narratives  
**Why**: Varied outcomes, real-world validation, long temporal scales

### Collection Strategy

```python
events = {
    'battles': 2000,          # Clear outcomes
    'discoveries': 1000,      # Scientific breakthroughs
    'political_events': 2000, # Revolutions, elections
    'cultural_movements': 1000, # Art movements, social changes
    'biographies': 4000       # Individual life narratives
}
```

### What to Capture

- **Event narrative** (Wikipedia article text, maintain structure)
- **Temporal sequence** (dates, durations)
- **Outcome** (success/failure, impact, persistence)
- **Contemporary descriptions** (how it was narrated at the time)

---

## Source 5: News Stories (Archive.org)

**Target**: 50,000 news articles with outcomes  
**Why**: Short-form narratives, measurable decay, real-world outcomes

### Collection Strategy

- Historical news archives
- Stories with measurable outcomes (predictions, investigations)
- Follow-up tracking (did narrative prove accurate?)

### What to Capture

- **Article text** (complete, maintain paragraph structure)
- **Publication date** (for decay analysis)
- **Outcome** (prediction accuracy, investigation results)
- **Citation/sharing** (narrative impact)
- **Temporal decay** (attention over time)

### Sequential Processing

News has constrained structure but varies:
- Lead paragraph (setup)
- Development (evidence)
- Quotes (voices)
- Conclusion
- Preserve exact sequence - compression patterns matter

---

## Source 6: Scientific Abstracts (arXiv)

**Target**: 20,000 abstracts  
**Why**: Constrained narrative form, clear outcomes (citations)

### Collection Strategy

- arXiv papers across fields
- Abstract = compressed narrative of research
- Citations = outcome measure

### What to Capture

- **Abstract text** (complete)
- **Sequential structure** (background → methods → results → implications)
- **Citations** (5-year citation count)
- **Field** (for cross-field comparison)

### Sequential Processing

Abstracts follow strict sequence:
- Position of elements matters (results before methods = bad)
- Spacing between elements (implied by structure)
- Compression ratio (how much research into 250 words)

---

## Source 7: Religious Texts (Multiple Traditions)

**Target**: 5,000 religious narratives  
**Why**: Cross-cultural, ancient, high persistence (ultimate outcome)

### Collection Strategy

```python
traditions = {
    'bible': ['parables', 'narratives', 'psalms'],
    'quran': ['surah_narratives', 'prophet_stories'],
    'buddhist': ['jataka_tales', 'sutras'],
    'hindu': ['mahabharata', 'ramayana', 'upanishads'],
    'torah': ['narrative_portions'],
}
```

### What to Capture

- **Original text** (maintain structure, translations preserved)
- **Sequential structure** (verse order critical)
- **Interpretive layers** (if available from commentaries)
- **Persistence** (still studied/taught after millennia)

### Sequential Processing

Religious texts have unique temporal properties:
- Designed for memorization (rhythm matters)
- Interpretive flexibility (same sequence, multiple meanings)
- Cultural transmission (ultimate long-term outcome)

---

## Source 8: Music Lyrics (Genius)

**Target**: 20,000 songs  
**Why**: Constrained temporal form, varied genres, clear outcomes

### Collection Strategy

- Genius API
- Multiple genres (pop, rock, hip-hop, country, classical)
- Lyrics = compressed narrative

### What to Capture

- **Lyrics** (exact order, line breaks, spacing)
- **Verse/chorus structure** (sequential pattern)
- **Temporal duration** (song length)
- **Popularity** (streams, chart position)

### Sequential Processing

Songs have strict temporal constraints:
- 3-4 minutes average
- Verse-chorus structure = acts
- Line breaks = pauses (rhythm)
- Repetition is intentional (not redundancy)

---

## Source 9: Video Game Narratives

**Target**: 5,000 game narratives  
**Why**: Interactive narratives, player agency, emergent stories

### Collection Strategy

- Game plot summaries (Wikipedia, gaming sites)
- Player-created narratives (forums, reviews)
- Critical reviews

### What to Capture

- **Main plot** (designed narrative)
- **Player experience** (emergent narrative)
- **Sequential structure** (act progression)
- **Success metrics** (sales, ratings, awards)

---

## Source 10: Social Media Story Threads

**Target**: 10,000 story threads  
**Why**: Modern narrative form, real-time outcomes, natural variety

### Collection Strategy

- Reddit story subreddits (r/stories, r/nosleep)
- Twitter threads (long-form stories)
- Upvotes/engagement = outcome

### What to Capture

- **Thread** (complete, maintain post order)
- **Temporal spacing** (gaps between posts)
- **Engagement** (upvotes, comments)
- **Sequential reveals** (information ordering)

---

## Meta-Collection Infrastructure

### Universal Narrative Record

```python
@dataclass
class NarrativeRecord:
    """
    Universal format for ANY narrative, ANY source.
    
    Preserves EVERYTHING without categorization.
    """
    # Identity
    narrative_id: str
    source: str
    
    # Content (COMPLETE, unmodified)
    text: str
    
    # Structure (positions, not breaks)
    element_boundaries: List[int]  # Character positions
    spacing_data: List[float]  # Gaps between elements
    
    # Temporal
    duration: Optional[float]  # If applicable (film length, reading time)
    creation_date: Optional[str]
    
    # Outcomes (if any)
    outcome_measures: Dict[str, Any]
    
    # Metadata (NO categorization labels)
    metadata: Dict[str, Any]
    
    # Processing (filled by AI)
    sequential_embeddings: Optional[np.ndarray] = None
    progression_pattern: Optional[Dict] = None
    rhythm_analysis: Optional[Dict] = None
    mysterious_dimensions: Optional[Dict] = None
```

### Background Processing Queue

```python
class NarrativeProcessingQueue:
    """
    Manages background processing of large corpora.
    
    Features:
    - Parallel processing (multiple sources simultaneously)
    - Checkpoint/resume (handle interruptions)
    - Progress tracking
    - Error handling
    """
    
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.queue = []
        self.completed = []
        self.failed = []
    
    def add_source(self, source_spec: Dict):
        """Add source to processing queue."""
        self.queue.append(source_spec)
    
    def process_all(self):
        """Process all sources in background."""
        # TODO: Implement parallel processing
        # For now, sequential
        
        for source_spec in self.queue:
            try:
                process_source_background(source_spec)
                self.completed.append(source_spec['name'])
            except Exception as e:
                self.failed.append({
                    'source': source_spec['name'],
                    'error': str(e)
                })
```

---

## Expected Corpus Statistics

After complete collection:

| Source | Count | Avg Length | Total Tokens | Temporal Range |
|--------|-------|------------|--------------|----------------|
| **Gutenberg** | 10,000 | 80K words | 800M | 1600-1920 |
| **Movie Plots** | 50,000 | 500 words | 25M | 1920-2025 |
| **Sports** | 100,000 | 300 words | 30M | 1990-2025 |
| **Historical** | 10,000 | 2K words | 20M | All time |
| **News** | 50,000 | 800 words | 40M | 1990-2025 |
| **Scientific** | 20,000 | 250 words | 5M | 2000-2025 |
| **Religious** | 5,000 | 1K words | 5M | Ancient |
| **Music** | 20,000 | 200 words | 4M | 1950-2025 |
| **Games** | 5,000 | 1K words | 5M | 1980-2025 |
| **Social** | 10,000 | 500 words | 5M | 2020-2025 |
| **TOTAL** | **280,000** | - | **939M tokens** | All eras |

---

## Processing Timeline

**Month 1-2**: Infrastructure + Pilots
- Build sequential processor (done)
- Test on 1,000 narratives per source
- Validate preservation of sequence/spacing

**Month 3-4**: Large-Scale Collection
- Process Gutenberg (10K novels)
- Process movie plots (50K)
- Process sports (100K - already partially collected)

**Month 5-6**: Unsupervised Discovery
- Run pattern discovery on each source
- Identify source-specific patterns
- Find cross-source universals

**Month 7-8**: Meta-Analysis
- Compare patterns across sources
- Test universality hypotheses
- Formalize mysterious dimensions

**Month 9-10**: Validation
- Test if discovered patterns predict outcomes
- Cross-source transfer learning
- Temporal isomorphism tests

---

## Success Criteria

1. **Sequential Integrity**: Can reconstruct narrative order from embeddings (accuracy > 90%)

2. **Universal Patterns**: Find patterns appearing in ALL sources (if they exist)

3. **Prediction Without Understanding**: Mysterious dimensions predict outcomes (R² > 0.30) WITHOUT us knowing what they mean

4. **Temporal Isomorphism**: Patterns at 50% completion match across sources (r > 0.60)

5. **Better Than Theory**: AI-discovered patterns outperform Campbell/Jung (ΔR² > 0.10)

---

## Philosophy: Why Keep Mechanisms Elusive

**Traditional Approach**: 
- Define what to look for (hero's journey, archetypes)
- Build features to detect those patterns
- Validate if patterns exist
- PROBLEM: Only find what you look for

**Our Approach**:
- Let AI embed narratives (captures ALL structure)
- Discover natural patterns (unsupervised clustering)
- Correlate with outcomes (which patterns win?)
- DO NOT interpret dimensions (let them remain mysterious)
- BENEFIT: Find patterns theory didn't anticipate

**Why Elusive is Better**:
- Over-interpretation constrains analysis
- Naming patterns prematurely biases future analysis
- Mechanisms may be emergent, not reductive
- Sometimes "works but we don't know why" is honest answer
- Preserves discovery potential (not locked into interpretation)

**Quote**: "The Tao that can be told is not the eternal Tao. The name that can be named is not the eternal name."

Applied: Patterns that can be explained may not be the patterns that actually govern.

---

## Output Structure

```python
processed_corpus = {
    'source': str,
    'narratives': [
        {
            'id': str,
            'sequential_elements': [
                {
                    'index': int,
                    'text': str,
                    'position': float,  # % through narrative
                    'spacing_before': float,
                    'embedding_cache_key': str,  # Reference to cached embedding
                }
            ],
            'rhythm': {
                'rho': float,
                'mean_spacing': float,
                'acceleration': float,
                # ... rhythm analysis
            },
            'progression': {
                'path_length': float,
                'directionality': float,
                'trajectory_3d': List[List[float]],
                # ... progression analysis
            },
            'mysterious_dimensions': {
                'pca_components': List[float],
                'ica_components': List[float],
                # Measured, not interpreted
            },
            'outcome': Any
        }
    ],
    'discovered_patterns': [
        {
            'pattern_id': str,
            'size': int,
            'centroid': List[float],  # In latent space
            'signature': Dict,  # Dimensional profile
            'correlation_with_outcome': float,
            'note': 'Pattern exists. Meaning elusive.'
        }
    ]
}
```

---

## Background Processing Commands

```bash
# Process each source independently
python -m narrative_optimization.data_collection.process_source \
    --source gutenberg \
    --target 10000 \
    --output data/processed_narratives/gutenberg/

python -m narrative_optimization.data_collection.process_source \
    --source movie_plots \
    --target 50000 \
    --output data/processed_narratives/movies/

# Discover patterns in each source
python -m narrative_optimization.src.analysis.discover_patterns \
    --source_dir data/processed_narratives/gutenberg/ \
    --output results/patterns/gutenberg_patterns.json

# Meta-analysis across sources
python -m narrative_optimization.src.analysis.meta_discover \
    --sources data/processed_narratives/*/ \
    --output results/universal_patterns.json
```

---

## Conclusion

This infrastructure enables:

1. **Large-scale systematic collection** (280K narratives)
2. **Sequential integrity** (preserve order, spacing, timing)
3. **AI-driven discovery** (unsupervised, no preconceptions)
4. **Source-by-source** (process independently, then meta-analyze)
5. **Pattern emergence** (let data reveal structure)
6. **Mysterious efficacy** (patterns predict without explaining)

**Core Innovation**: Accepting that some patterns work WITHOUT us understanding why. Better analysis through embracing mystery rather than forcing interpretation.

**Status**: Infrastructure designed, ready for implementation.  
**Next**: Begin collection with Gutenberg (easiest), process sequentially, discover patterns without presupposition.

