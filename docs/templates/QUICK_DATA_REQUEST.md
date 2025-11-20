# ⚡ Quick Data Request for Narrative Analysis

**TL;DR:** Send us text descriptions + outcomes, we'll discover the narrative formula that predicts success in your domain.

---

## What We Need (5-Minute Version)

### 1. Your Domain Name
Example: `esports_teams`, `wine_reviews`, `job_postings`, `startup_pitches`

### 2. Text Descriptions
Any of:
- Marketing copy
- Profiles/bios
- Product descriptions
- News articles
- Names (if name-focused domain)
- Self-descriptions
- Reviews

**Minimum:** 20-50 examples  
**Optimal:** 100+ examples

### 3. Outcomes
What happened? Examples:
- Winner/loser
- Rating/score
- Sales/performance
- Success/failure
- Preference/choice

**Must be:** Actual observed outcomes, not predictions

### 4. Context (Optional but Helpful)
- When did it happen?
- What category?
- Stakes level (low/medium/high)?
- Any other relevant info?

---

## Simple Template

```json
[
  {
    "id": "001",
    "text_description": "Your full text here (the more descriptive, the better)",
    "outcome": "winner/score/rating/etc",
    "context": "any helpful context"
  },
  {
    "id": "002",
    "text_description": "Another example...",
    "outcome": "...",
    "context": "..."
  }
]
```

---

## What You'll Get Back

1. **Prediction Model** - Forecast outcomes from narratives (R² or accuracy score)
2. **Feature Rankings** - Which narrative elements matter most
3. **Interactive Dashboard** - Explore your data visually
4. **Research Report** - Publication-ready statistical validation
5. **Domain Classification** - Where you fit in narrative taxonomy

**Timeline:** 2-4 weeks

---

## Real Examples

### Example 1: Sports Team
```json
{
  "id": "lakers_vs_celtics_game7",
  "text_description": "Lakers riding 7-game win streak, championship experience, 
                       facing historic rivals in elimination game. LeBron brings 
                       veteran leadership and clutch performance record.",
  "outcome": "lakers_won",
  "context": "2024 NBA Finals Game 7"
}
```

### Example 2: Product
```json
{
  "id": "iphone_15_pro",
  "text_description": "Elegant titanium design with cutting-edge camera system. 
                       Intuitive interface and premium build quality deliver 
                       flagship experience.",
  "outcome": 8.7,
  "context": "Smartphone review score (0-10)"
}
```

### Example 3: Name → Outcome
```json
{
  "id": "hurricane_katrina",
  "text_description": "Katrina",
  "outcome": 0.45,
  "context": "Evacuation rate (0-1)"
}
```

---

## Visibility Score (Helps Us Calibrate)

How obvious/measurable is your outcome?

- **90-100:** Sports scores, election results (perfectly measurable)
- **70-90:** Sales figures, ratings (clearly measurable)
- **50-70:** Survey results, tracked behaviors (measurable with effort)
- **30-50:** Purchase decisions, engagement (partially observable)
- **10-30:** Preferences, sentiment, compatibility (mostly hidden)

**Why it matters:** We predict effect size before analyzing:
- High visibility (90+) → Narrative matters less (obvious outcomes)
- Low visibility (30-) → Narrative matters MORE (uncertain outcomes)

---

## Quick Decision Tree

**Do you have head-to-head comparisons?**
- YES → Use comparison format (A vs B with winner)
- NO → Use observation format (description + outcome)

**Is outcome measurable?**
- YES → Perfect! Send actual measurements
- PARTIALLY → Send best available proxy
- NO → Let's discuss (might still work)

**Do you have 50+ examples?**
- YES → Excellent, proceed!
- 20-50 → Good, proceed with caution
- < 20 → Might be too small, but send anyway

**Is text human-written?**
- YES → Perfect!
- NO (just keywords/tags) → Might work, depends
- NO (just numbers) → Won't work for narrative analysis

---

## Send Your Data

**Email:** [your contact]

**Include:**
1. JSON file or CSV
2. Short description of domain
3. Visibility score estimate (0-100)
4. Your contact info

**Subject line:** "Narrative Analysis: [YOUR_DOMAIN]"

We'll reply within 24 hours!

---

## Questions Before Sending?

**Q: How much text do I need per example?**  
A: Minimum 10 words, optimal 50-200 words, no maximum.

**Q: Can I send messy/imperfect data?**  
A: Yes! We'll clean it. Just let us know what's messy.

**Q: What if I don't have outcomes for everything?**  
A: Send what you have. Partial data > no data.

**Q: Can outcomes be predictions instead of actual?**  
A: Prefer actual, but if predictions are all you have, we can work with it.

**Q: My domain is weird/niche, will it work?**  
A: Probably yes! We've analyzed everything from hurricane names to cryptocurrency. Try us.

**Q: Is my data confidential?**  
A: Yes, unless you want us to publish results. Your call.

---

## That's It!

Don't overthink it. Send us:
1. Text descriptions
2. Outcomes
3. Basic context

We'll handle the rest.

**"Better stories win — let's prove it in your domain."** 🎯


