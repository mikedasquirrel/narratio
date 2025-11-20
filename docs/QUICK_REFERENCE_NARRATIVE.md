# QUICK REFERENCE: What is a "Narrative"?

**CRITICAL:** Read this before processing any domain.

---

## One-Sentence Definition

**Narrative = The complete genome of information that determines an outcome.**

---

## Common Misconception

❌ **WRONG:** "Narrative means a written English story or plot summary"

✅ **RIGHT:** "Narrative means ALL factual data, context, features, and relationships for an instance"

---

## Examples

### NHL Game Narrative:
```
NOT: "The exciting playoff matchup between..."
YES: {home_team, away_team, rest_days, goalie_quality, rivalry, 
     betting_odds, cup_history, record_differential, + 50 more fields}
```

### Startup Narrative:
```
NOT: "A revolutionary AI platform that..."
YES: {founders, funding_stage, traction, YC_batch, team_size, 
     market_size, growth_rate, + 30 more fields}
```

### Movie Narrative:
```
NOT: Just the plot summary
YES: {director_prestige, budget, genre, runtime, studio, 
     release_timing, plot_summary, + 40 more fields}
```

---

## Key Points

1. **Structured data IS narrative** - Numbers, categories, relationships all count
2. **Text is optional** - Can be synthesized from structured data if needed
3. **More info = better** - Include every relevant feature/fact
4. **Genome thinking** - What information determines the outcome?

---

## When Processing Domains

**Ask yourself:**
- "What is the complete information substrate for this instance?"
- NOT "What text describes this event?"

**Include:**
- ✅ All numbers, stats, metrics
- ✅ All categories, types, classes  
- ✅ All relationships, comparisons
- ✅ All temporal context, trends
- ✅ All names, brands, prestige signals
- ✅ Optional: natural language text

---

## Universal Pipeline Handles This

The processor automatically:
1. Accepts structured genome (JSON, dict, or text)
2. Extracts features from all available information
3. Synthesizes text if transformers need it
4. Concatenates all features

You provide the genome → pipeline does the rest.

---

**See full documentation:** `docs/NARRATIVE_DEFINITION.md`

