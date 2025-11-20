# Domain Testing Analysis - Which Dataset is Most Efficient?

**Question:** Which domain can most efficiently test transformers?

**Date:** November 16, 2025  
**Note:** Transformer counts in this doc are historical. Use `python -m narrative_optimization.tools.list_transformers` for current list (100+).

---

## Available Domains Comparison

| Domain | Records | File Size | π (Narrativity) | Temporal Context | Richness | Load Speed |
|--------|---------|-----------|-----------------|------------------|----------|------------|
| **NBA** | 11,976 | 179M | ~0.4-0.5 | ✅ Yes | Very Rich | Fast |
| **Golf** | 7,700 | 25M | ~0.85 | ✅ Yes | VERY Rich | Very Fast |
| **Tennis** | Large | 326M | ~0.6-0.7 | ✅ Yes | Rich | Slow |
| **Movies** | 6,047 | 87M | ~0.8-0.9 | ⚠️ Limited | Very Rich | Medium |
| **NFL** | ~8,000 | 31M | ~0.4-0.5 | ✅ Yes | Rich | Fast |
| **MLB** | 23,264 | 153M | ~0.3-0.4 | ✅ Yes | Rich | Slow |
| **Startups** | ~400 | 402K | ~0.9 | ⚠️ Limited | Medium | Very Fast |
| **UFC** | Medium | 10M | ~0.5-0.6 | ⚠️ Limited | Medium | Fast |
| **Oscars** | ~1,000 | 202K | ~0.8-0.9 | ✅ Limited | High | Very Fast |

---

## 🏆 RECOMMENDED: **Golf**

### Why Golf is PERFECT for Comprehensive Testing:

#### ✅ **Optimal Size**
- 7,700 tournaments (large enough for statistical power)
- 25M file (loads quickly)
- Not too big, not too small - Goldilocks zone

#### ✅ **Perfect for ALL Transformer Types**

**Nominative Transformers:**
- ✅ Rich player names (Tiger Woods, Rory McIlroy, etc.)
- ✅ Tournament names (Masters, Open Championship)
- ✅ Course names (Augusta National, Pebble Beach)
- ✅ **This is where nominative richness was discovered! (97.7% R²)**

**Temporal Transformers:**
- ✅ Season-long context (PGA Tour seasons)
- ✅ Career trajectories (player development)
- ✅ Historical rivalries
- ✅ Tournament progression (rounds 1-4)
- ✅ Momentum patterns (winning streaks)
- ✅ Perfect for `temporal/` subdirectory transformers

**Universal/Meta Transformers:**
- ✅ Hero's journey arcs (underdog victories)
- ✅ Universal themes (redemption, struggle, triumph)
- ✅ Cross-domain patterns (golf tournaments ≈ tennis tournaments)
- ✅ Meta-narrative (golf commentary is self-aware)

**Structural Transformers:**
- ✅ Tension buildup (final round drama)
- ✅ Conflict (player vs course, player vs player)
- ✅ Pacing (4-round arc structure)
- ✅ Suspense (leaderboard changes)

**Contextual Transformers:**
- ✅ Cultural context (major championships vs regular)
- ✅ Competitive context (world rankings, favorites)
- ✅ Temporal context (career stage, form)

#### ✅ **Known High Performance**
- Golf breakthrough: 39.6% → 97.7% R² with nominative richness
- Highest π of all sports (~0.85)
- Proven that narrative features work extremely well

#### ✅ **Data Quality**
- Clean, structured data
- Rich narratives available
- Enhanced with player/tournament details
- Temporal context included

#### ✅ **Efficiency**
- Loads in ~3-5 seconds
- Processes quickly (medium dataset size)
- Won't overwhelm memory
- Fast iteration for testing

---

## 🥈 ALTERNATIVE: **NBA** (Already Proven)

### Why NBA is Also Great:

#### ✅ **We've Already Tested It!**
- Know it works (97% transformer success rate today)
- 11,976 games
- Rich temporal + player data
- Fast and reliable

#### ⚠️ **Limitations vs Golf:**
- Lower π (~0.4-0.5 vs Golf's ~0.85)
- Won't test high-π transformers as thoroughly
- Larger file size (179M vs 25M)

#### ✅ **Best For:**
- Quick validation that nothing broke
- Regression testing
- Performance benchmarking
- Known baseline

---

## 🥉 THIRD OPTION: **Movies** (Different Domain Type)

### Why Movies Would Be Interesting:

#### ✅ **Very Different from Sports**
- High π (~0.8-0.9)
- Pure narrative domain
- Tests universality of transformers

#### ✅ **Rich Narratives**
- Plot summaries
- Character descriptions
- Reviews and commentary

#### ⚠️ **Limitations:**
- Limited temporal progression (movies are single events)
- Won't test temporal transformers as well
- Mixed data quality

---

## 📊 Efficiency Matrix

| Criterion | Golf | NBA | Movies | Tennis | NFL |
|-----------|------|-----|--------|--------|-----|
| **Sample Size** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Load Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **Narrative Richness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Temporal Coverage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Universal Themes** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Known to Work** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **High π Testing** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **TOTAL SCORE** | **32/35** | **29/35** | **27/35** | **24/35** | **24/35** |

---

## 🎯 RECOMMENDATION: **GOLF**

### Test Order:

**1. Golf (Comprehensive Test)**
- Tests ALL 55 transformers optimally
- Covers full π range (high narrativity)
- Excellent temporal coverage
- Fast enough for iteration
- **Best for validating the COMPLETE transformer library**

**2. NBA (Validation Test)**
- Confirms everything still works
- Lower π (different characteristics)
- Very fast (already optimized)
- **Best for regression testing**

**3. Movies (Diversity Test) - Optional**
- Different domain type entirely
- Pure narrative (not competitive)
- Tests universality claims
- **Best for proving cross-domain applicability**

---

## ⚡ Quick Efficiency Analysis

### Golf Testing Estimates:

**For 7,700 tournaments:**
- Fast transformers (<0.1s): 7,700 × 0.05s = 6 minutes
- Medium transformers (0.1-0.5s): 7,700 × 0.2s = 26 minutes
- Slow transformers (>0.5s): 7,700 × 1s = 128 minutes

**Total estimated time: 2-3 hours for ALL 55 transformers**

**With sampling (500 tournaments):**
- Total estimated time: 10-15 minutes for ALL 55 transformers

### NBA Testing Estimates:

**For 11,976 games:**
- Total estimated time: 3-4 hours for ALL 55 transformers

**With sampling (500 games):**
- Total estimated time: 10-15 minutes (we just did 34 transformers in 17 seconds!)

---

## 🚀 Recommended Test Strategy

### Phase 1: Quick Validation (500 samples)
**Dataset:** Golf (500 tournaments)  
**Time:** 15 minutes  
**Purpose:** Verify all 55 transformers work  
**Transformers:** All 55

### Phase 2: Full Performance Test (All data)
**Dataset:** Golf (7,700 tournaments)  
**Time:** 2-3 hours  
**Purpose:** Complete performance profile  
**Transformers:** All 55

### Phase 3: Cross-Domain Validation (500 samples each)
**Datasets:** NBA + Movies  
**Time:** 30 minutes  
**Purpose:** Prove universality  
**Transformers:** Top 20-30 performers from Phase 1

---

## 💡 Why Golf is IDEAL:

1. **Perfect Size:** Not too big (like MLB), not too small (like Startups)
2. **Richest Narratives:** Player names, courses, tournaments, historical context
3. **Full Temporal:** Seasons, careers, tournaments, rounds
4. **Highest π:** Best for testing narrative-driven transformers
5. **Known Success:** Where nominative richness breakthrough happened
6. **Fast Enough:** Can iterate quickly
7. **Diverse Enough:** Tests all transformer types

**Golf is the Goldilocks dataset - just right! ⛳**

---

## 📋 Test Script Features Needed

For Golf comprehensive test, we need to handle:
- ✅ Text narratives (player descriptions, tournament stories)
- ✅ Temporal sequences (round progression, career arcs)
- ✅ Nominative richness (player names, course names)
- ✅ Competitive context (rankings, matchups)
- ✅ Universal themes (redemption stories, underdogs)
- ✅ Structural patterns (4-round arc, pressure building)

Golf data structure should include:
- Tournament narratives
- Player names and stats
- Round-by-round scores (temporal)
- Course information (context)
- Historical context (careers, majors)

---

## ✅ FINAL ANSWER

**Most Efficient:** 🏆 **Golf with 500-sample validation**

**Why:**
- 15 minutes to test all 55 transformers
- Covers all transformer types optimally
- High π means narrative transformers shine
- Rich temporal context for temporal transformers
- Known to produce excellent results
- Fast iteration for debugging

**Command to run:**
```bash
python3 test_all_55_transformers_golf.py --sample_size 500
```

**Then scale up:**
```bash
python3 test_all_55_transformers_golf.py --sample_size 7700
```

**Golf is your answer! ⛳🏆**

