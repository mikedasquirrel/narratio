# Data Collection Guide: Classical Archetype Domains

**Date**: November 13, 2025  
**Purpose**: Comprehensive guide for collecting archetype datasets across all 6 new domains

---

## Overview

To complete the archetype integration, we need to collect ~10,000 narrative samples across 6 domains:

| Domain | Target | Sources | Estimated Time |
|--------|--------|---------|----------------|
| Classical Literature | 800 | Gutenberg, Google Books | 3-4 days |
| Mythology | 1,000 | Wikipedia, Sacred Texts | 4-5 days |
| Scripture | 500 | Bible, Jataka, Sufi tales | 2-3 days |
| Film Extended | 2,500 | IMDb, TMDb APIs | 3-4 days |
| Music Narrative | 4,000 | Genius, Spotify APIs | 2-3 days |
| Stage Drama | 500 | Drama databases | 2-3 days |
| **TOTAL** | **9,300** | **Multiple APIs** | **2-3 weeks** |

---

## 1. Mythology & Folklore (Priority 1)

### Target: 1,000 myths

**Why First**: Validate Campbell on his source material (critical validation)

**Sources**:
- Wikipedia Mythology API (free, comprehensive)
- Theoi Project (Greek mythology)
- Sacred Texts Archive
- Norse mythology databases

**Script**: `collect_mythology_dataset.py` ✅ Created

**Run**:
```bash
cd narrative_optimization/scripts
python collect_mythology_dataset.py
```

**Expected Output**:
- `data/domains/mythology/mythology_complete_dataset.json`
- 1,000 myths with summaries, full text, deity names
- Cultural persistence scores

**Validation Test After Collection**:
```python
# Should show Campbell validated on mythology
results = discover_journey_patterns(myth_texts, myth_outcomes)
assert results['theoretical_validation']['summary']['campbell_validated'] == True
assert results['theoretical_validation']['summary']['correlation'] > 0.85
```

---

## 2. Classical Literature (Priority 2)

### Target: 800 works

**Breakdown**:
- Epic poetry: 50 (Homer, Virgil, Beowulf)
- Classical novels: 300 (Dickens, Austen, Tolstoy, etc.)
- Modernist: 250 (Joyce, Woolf, Faulkner)
- Postmodern: 200 (Pynchon, DeLillo, Wallace)

**Sources**:
- Project Gutenberg API (free, public domain)
- Google Books API (summaries, metadata)
- Open Library API
- Goodreads API (ratings, reviews)

**Script Needed**: `collect_literature_dataset.py`

**Key Fields**:
```json
{
  "title": "The Odyssey",
  "author": "Homer",
  "year": -800,
  "period": "epic",
  "genre": "epic poetry",
  "summary": "...",
  "full_text": "...",
  "outcomes": {
    "still_taught": true,
    "citations": 50000,
    "translations": 100,
    "cultural_impact": 0.95
  }
}
```

**Validation Test**:
```python
# Test π → journey completion correlation
assert correlation(pi_values, journey_completions) > 0.70
# Test temporal evolution (decreasing journey over time)
assert slope_journey_vs_year < 0
```

---

## 3. Film Extended (Priority 3)

### Target: 2,500 films

**Why**: Test if Snyder's Hollywood formula empirically works

**Extend Existing**:
- Already have IMDB dataset
- Add: Beat sheet analysis
- Add: Hero's Journey mapping
- Add: Genre-specific analysis

**Sources**:
- IMDb datasets (free download)
- OMDb API
- The Movie Database (TMDb) API
- Box Office Mojo

**Script Needed**: `extend_film_dataset.py`

**New Fields to Add**:
```json
{
  "beat_sheet": {
    "catalyst_position": 0.12,
    "midpoint_position": 0.50,
    "all_is_lost_position": 0.75
  },
  "journey_analysis": {
    "stages_present": 12,
    "completion_score": 0.78
  }
}
```

**Validation Tests**:
```python
# Snyder beats predict box office?
assert R²(box_office ~ beat_adherence) > 0.45  # For blockbusters
# Refusal more important than in myths?
assert w_film['refusal'] > w_mythology['refusal'] + 0.20
```

---

## 4. Scripture & Parables (Priority 4)

### Target: 500 texts

**Breakdown**:
- Biblical parables: 120
- Buddhist Jataka: 150
- Sufi stories: 80
- Hasidic tales: 60
- Zen koans: 50
- Aesop's fables: 40

**Sources**:
- Bible APIs (multiple translations)
- Sacred Texts Archive
- Jataka Tales Project
- Sufi story compilations

**Script Needed**: `collect_scripture_dataset.py`

**Key Fields**:
```json
{
  "title": "The Prodigal Son",
  "tradition": "Christian",
  "full_text": "...",
  "explicit_moral": "...",
  "implicit_lesson": "...",
  "outcomes": {
    "memorability": 0.95,
    "still_taught": true,
    "cross_cultural": true
  }
}
```

---

## 5. Music Narrative (Priority 5)

### Target: 4,000 songs

**Focus**: Narrative-heavy songs (story songs, concept albums, hip-hop narratives)

**Breakdown**:
- Concept albums: 200
- Story songs: 800
- Hip-hop narrative: 600
- Opera/musical: 400
- Ballads: 500
- Other genres: 1,500

**Sources**:
- Genius Lyrics API (free tier: 1,000 requests/day)
- Spotify API (metadata, popularity)
- MusicBrainz (comprehensive metadata)
- Last.fm API

**Script Needed**: `collect_music_narrative_dataset.py`

**Key Fields**:
```json
{
  "title": "Stan",
  "artist": "Eminem",
  "album": "The Marshall Mathers LP",
  "lyrics": "...",
  "narrative_clarity": 0.85,
  "emotional_arc": [0.3, 0.5, -0.2, -0.8],
  "outcomes": {
    "chart_peak": 1,
    "streams": 500000000,
    "cultural_impact": 0.90
  }
}
```

---

## 6. Stage Drama (Priority 6)

### Target: 500 plays

**Breakdown**:
- Greek tragedy: 50
- Greek comedy: 30
- Shakespeare: 100
- Restoration: 40
- Modern drama: 150
- Contemporary: 100
- Musical theatre: 30

**Sources**:
- Project Gutenberg (drama section)
- Internet Shakespeare
- Drama Online
- Open-source play databases

**Script Needed**: `collect_stage_drama_dataset.py`

---

## Automated Collection Workflow

### Step 1: Run All Collectors (Parallel)

```bash
# Run in parallel (different terminals)
python scripts/collect_mythology_dataset.py &
python scripts/collect_literature_dataset.py &
python scripts/collect_scripture_dataset.py &
python scripts/extend_film_dataset.py &
python scripts/collect_music_narrative_dataset.py &
python scripts/collect_stage_drama_dataset.py &
```

### Step 2: Monitor Progress

```bash
# Check collection status
ls -lh data/domains/*/

# Count collected samples
wc -l data/domains/*/*.json
```

### Step 3: Process with Transformers

```bash
# Once data collected, extract features
python scripts/process_archetype_domains.py --all
```

---

## Rate Limiting & Ethics

### API Rate Limits

- **Wikipedia**: 200 requests/sec (respectful: 2/sec)
- **Genius Lyrics**: 1,000 requests/day (free tier)
- **Spotify**: 180 requests/minute
- **IMDb**: No official API (use datasets)

### Best Practices

1. **Respect robots.txt**: Check before scraping
2. **Rate limit**: Add delays between requests
3. **Cache responses**: Don't re-request same data
4. **Attribution**: Credit sources in dataset
5. **License compliance**: Check usage rights

---

## Data Quality Assurance

### Minimum Requirements

Each sample must have:
- ✅ Unique identifier
- ✅ Full narrative text (>500 words for myths, >100 for parables)
- ✅ Metadata (author, culture, date)
- ✅ Outcome measure (success metric)
- ✅ Source attribution

### Quality Checks

```python
def validate_sample(sample):
    assert len(sample['full_text']) >= 500
    assert 'outcome_measures' in sample
    assert sample['culture'] in known_cultures
    assert 'source' in sample
```

---

## Estimated Collection Timeline

### Week 1: High-Priority Domains
- Day 1-2: Mythology (1,000 myths)
- Day 3-4: Literature (800 works)
- Day 5: Scripture (500 texts)

### Week 2: Extended Domains
- Day 6-7: Film extension (2,500 films)
- Day 8-9: Music narrative (4,000 songs)
- Day 10: Stage drama (500 plays)

### Week 3: Processing & Validation
- Day 11-12: Feature extraction with all transformers
- Day 13-14: Run validation experiments
- Day 15: Document discoveries

**Total: 2-3 weeks focused work**

---

## Next Actions

### Immediate
1. Review and run `collect_mythology_dataset.py`
2. Create remaining 5 collection scripts
3. Set up API keys (Genius, Spotify, etc.)

### Short-Term
4. Run all collectors in parallel
5. Monitor progress and handle errors
6. Process collected data with transformers

### Validation
7. Run formula validation tests
8. Document discoveries
9. Generate publication-ready results

---

**Collection infrastructure ready. Awaiting execution.**

