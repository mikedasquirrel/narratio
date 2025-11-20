# UPDATED REVALIDATION INSTRUCTIONS
## With Narrative Genome Clarification

**Date:** November 17, 2025  
**CRITICAL UPDATE:** Narrative = Complete Information Genome (Not Just Text)

---

## BEFORE YOU START: Understanding "Narrative"

**READ THIS FIRST:** `docs/NARRATIVE_DEFINITION.md` or `docs/QUICK_REFERENCE_NARRATIVE.md`

### Key Concept:
In our framework, **"narrative" does NOT mean literary text.**

**Narrative = The complete genome of information about an instance:**
- All structured data (numbers, categories, booleans)
- All relationships (vs, head-to-head, comparisons)
- All temporal context (trends, streaks, timing)
- All nominative features (names, prestige, brands)
- Optional: natural language text (synthesized if not available)

### Examples:

**NHL Game Narrative:**
```json
{
  "home_team": "Boston Bruins",
  "away_team": "Toronto Maple Leafs",
  "rest_advantage": 2,
  "is_rivalry": true,
  "home_goalie": "Ullmark",
  "betting_odds": {"home": -150, "away": 130},
  "cup_history_diff": -7,
  ... 50+ more fields
}
```
This structured data IS the narrative genome.

**The Universal Pipeline:**
1. Accepts this complete genome (structured or text)
2. Extracts features from ALL available information
3. Synthesizes English text if transformers need it
4. Returns 300+ features for prediction

---

## REVALIDATION COMMANDS

### General Template (All Domains):

```bash
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization

python3 narrative_optimization/universal_domain_processor.py \
  --domain [DOMAIN_NAME] \
  --sample_size [N] \
  --use_transformers \
  --save_results
```

**What this does:**
1. Loads domain's **complete information genome** (not just text!)
2. Each "narrative" includes ALL structured fields, relationships, context
3. Transformers extract features from the complete genome
4. Text synthesis happens automatically if needed
5. Returns: π (narrativity), Δ (agency), R², patterns
6. Saves: `narrative_optimization/results/domains/{domain}_results.json`

---

## PRIORITY DOMAINS

### 1. TENNIS
```bash
python3 narrative_optimization/universal_domain_processor.py --domain tennis --sample_size 5000 --use_transformers --save_results
```

**Narrative genome includes:**
- Player rankings, head-to-head history, surface type, tournament round
- Recent form, serve statistics, break point conversion
- Tournament prestige, betting odds, weather conditions
- Optional: Match description text (synthesized if not available)

**Report:** π, Δ, R², sample size, production readiness

---

### 2. GOLF
```bash
python3 narrative_optimization/universal_domain_processor.py --domain golf --sample_size 5000 --use_transformers --save_results
```

**Narrative genome includes:**
- Player world ranking, course history, recent form
- Weather conditions, tournament prestige, field strength
- Nominative features (30+ proper nouns per tournament)
- Hole-by-hole statistics, putting averages, scoring trends

**Previously:** 97.7% R² via nominative enrichment

---

### 3. UFC
```bash
python3 narrative_optimization/universal_domain_processor.py --domain ufc --sample_size 3000 --use_transformers --save_results
```

**Narrative genome includes:**
- Fighter records, weight class, reach advantages
- Fighting style matchups, knockout power, submission rates
- Recent performance, training camp quality, age factors
- Narrative hype, betting odds, title implications

**Expected:** High π (0.72) but low Δ (performance-dominated)

---

### 4. MLB
```bash
python3 narrative_optimization/universal_domain_processor.py --domain mlb --sample_size 5000 --use_transformers --save_results
```

**Narrative genome includes:**
- Team records, pitcher statistics, batting averages
- Home/away splits, weather, ballpark factors
- Recent streaks, playoff implications, roster health
- Manager decisions, bullpen quality, defensive metrics

---

### 5. MOVIES
```bash
python3 narrative_optimization/universal_domain_processor.py --domain movies --sample_size 2000 --use_transformers --save_results
```

**Narrative genome includes:**
- Director prestige, budget percentile, studio strength
- Genre, runtime, MPAA rating, release timing
- Cast prestige, screenplay quality, cinematographer
- Marketing spend, critical buzz, awards positioning
- PLUS: Plot summary text (one part of many features)

**Note:** Plot text is PART of genome, not the entire narrative

---

### 6. OSCARS
```bash
python3 narrative_optimization/universal_domain_processor.py --domain oscars --sample_size 500 --use_transformers --save_results
```

**Narrative genome includes:**
- All movie features above
- PLUS: Nomination timing, guild awards, critical consensus
- Campaign spend, industry politics, thematic resonance
- Competitive field strength, precursor wins

**Previously:** Perfect AUC=1.00 classification

---

### 7. STARTUPS
```bash
python3 narrative_optimization/universal_domain_processor.py --domain startups --sample_size 500 --use_transformers --save_results
```

**Narrative genome includes:**
- Founder credentials, team size, location
- Funding stage, traction metrics, growth rate
- Market size, competition, technology stack
- YC batch, previous exits, product demo quality
- PLUS: Pitch description text

**Previously:** r=0.980 (98% R²) - needs validation

---

### 8. CRYPTO
```bash
python3 narrative_optimization/universal_domain_processor.py --domain crypto --sample_size 3000 --use_transformers --save_results
```

**Narrative genome includes:**
- Coin name features, market cap, trading volume
- Team credentials, technology innovation, use case
- Community size, GitHub activity, partnerships
- Whitepaper quality, tokenomics, exchange listings
- PLUS: Project description text

**Previously:** AUC=0.925

---

## BATCH PROCESSING

### All Sports:
```bash
python3 -c "
import sys
sys.path.insert(0, 'narrative_optimization')
from universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor()
processor.process_batch(['tennis', 'golf', 'ufc', 'mlb', 'boxing'], sample_size=5000)
"
```

### All Entertainment:
```bash
python3 -c "
import sys
sys.path.insert(0, 'narrative_optimization')
from universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor()
processor.process_batch(['movies', 'oscars', 'music'], sample_size=3000)
"
```

### Everything:
```bash
python3 narrative_optimization/universal_domain_processor.py --all --max_per_domain 5000
```

---

## REPORTING FORMAT

After each domain processes, report:

```
Domain: [NAME]
Status: [COMPLETE / ERROR]

GENOME QUALITY:
- Structured fields: [count]
- Text available: [YES/NO]
- Relationships: [count]
- Temporal features: [YES/NO]

RESULTS:
- Narrativity (π): [X.XX]
- Agency (Δ): [X.XXX]
- Δ/π ratio: [X.XX] (>0.5 = passes threshold)
- R² or AUC: [X.X%]
- Sample size: [N]

PRODUCTION READY: [YES/NO/MARGINAL]

Results file: narrative_optimization/results/domains/{domain}_results.json

NOTES:
- [Genome completeness]
- [Key findings]
- [Any issues]
```

---

## CRITICAL REMINDERS

1. **Narrative ≠ Text Story**
   - It's the complete information genome
   - Structured data is primary, text is optional
   - More fields = better predictions

2. **Domain Registry Already Handles This**
   - Configurations load complete genomes
   - Custom extractors build structured narratives
   - Text synthesis automatic when needed

3. **Transformers Process Genomes**
   - Some work directly on structured fields
   - Some need text (synthesized if not present)
   - All features concatenated (300+)

4. **Better Genome = Better Results**
   - Include every relevant feature
   - Don't discard structured data
   - Text is bonus, not requirement

---

## DOCUMENTATION REFERENCES

- **Full Explanation:** `docs/NARRATIVE_DEFINITION.md`
- **Quick Reference:** `docs/QUICK_REFERENCE_NARRATIVE.md`
- **Domain Registry:** `narrative_optimization/domain_registry.py`
- **Universal Processor:** `narrative_optimization/universal_domain_processor.py`

---

**These updated instructions reflect the true meaning of "narrative" in our framework.**

Start with TENNIS - highest previous ROI (127%), good genome completeness.

