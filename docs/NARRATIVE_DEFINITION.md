# NARRATIVE DEFINITION
## Critical Conceptual Clarification for the Universal Pipeline

**Date**: November 17, 2025  
**Status**: Core Framework Documentation  
**Importance**: CRITICAL - Affects all domain processing

---

## The Misconception

When people see "narrative" in our pipeline, they often assume it means:
- A written English text story
- A plot summary
- A literary narrative
- Natural language description

**This is INCORRECT.**

---

## What "Narrative" Actually Means

**Narrative = The Complete Genome of Information That Defines an Instance**

A "narrative" in our framework is:
- **ALL factual data** about an event/entity/outcome
- **The complete context** surrounding what happened
- **The story's genome** - every feature, fact, signal that played into the outcome
- **Structured or unstructured** - can be numbers, text, categories, relationships
- **The information substrate** from which the outcome emerged

### Examples Across Domains:

#### NHL Game "Narrative"
**NOT:** "The Bruins faced the Maple Leafs in an exciting rivalry matchup..."

**ACTUALLY:**
```json
{
  "home_team": "Boston Bruins",
  "away_team": "Toronto Maple Leafs",
  "home_record": 0.650,
  "away_record": 0.540,
  "rest_advantage": +2,
  "home_back_to_back": false,
  "away_back_to_back": true,
  "is_rivalry": true,
  "is_playoff": false,
  "home_goalie": "Linus Ullmark",
  "away_goalie": "Ilya Samsonov",
  "venue": "TD Garden",
  "betting_odds": {"home_moneyline": -150, "away_moneyline": +130},
  "record_differential": 0.110,
  "season_progress": 0.75,
  "temperature": null,
  "home_streak": 3,
  "away_streak": -2,
  "cup_history_home": 6,
  "cup_history_away": 13,
  "coach_prestige_home": 0.72,
  "coach_prestige_away": 0.68
}
```

**THIS is the narrative** - the complete information genome that determines the outcome.

We may **synthesize it into English** for transformer processing:
> "2024-25 regular season rivalry game: Boston Bruins (65.0% win rate, 6 Cups) host Toronto Maple Leafs (54.0%, 13 Cups) at TD Garden. Rest advantage: +2 days (Bruins rested, Leafs on back-to-back). Record differential +11.0%. Betting odds: BOS -150 vs TOR +130. Projected goalies: Ullmark vs Samsonov."

But the **genome is the narrative**, not the English text.

#### NBA Game "Narrative"
**NOT:** "LeBron James leads the Lakers against..."

**ACTUALLY:**
```json
{
  "home_team": "Los Angeles Lakers",
  "away_team": "Miami Heat",
  "home_wins": 42,
  "home_losses": 28,
  "away_wins": 38,
  "away_losses": 32,
  "home_last_10": "7-3",
  "away_last_10": "5-5",
  "spread": -4.5,
  "total": 218.5,
  "home_star_player": "LeBron James",
  "away_star_player": "Jimmy Butler",
  "star_matchup_history": "15-12 LeBron",
  "season_phase": "late",
  "home_playoff_position": 5,
  "away_playoff_position": 7,
  "days_rest_home": 1,
  "days_rest_away": 2
}
```

#### Startup "Narrative"
**NOT:** "A revolutionary AI platform that..."

**ACTUALLY:**
```json
{
  "product_description": "AI-powered customer service automation",
  "founders": ["John Smith (ex-Google)", "Jane Doe (Stanford PhD)"],
  "funding_stage": "Series A",
  "team_size": 12,
  "location": "San Francisco",
  "YC_batch": "W22",
  "founding_year": 2021,
  "technology_stack": ["Python", "TensorFlow", "React"],
  "market_size": "50B",
  "competitors": ["Zendesk", "Intercom"],
  "traction": "50K MRR",
  "growth_rate": 0.15,
  "customer_count": 23,
  "pitch_deck_quality": 0.82,
  "product_demo_exists": true,
  "founder_previous_exits": 1
}
```

#### Movie "Narrative"
**NOT:** The plot summary text

**ACTUALLY:**
```json
{
  "title": "The Godfather",
  "genre": ["Crime", "Drama"],
  "director": "Francis Ford Coppola",
  "director_prestige": 0.95,
  "budget": 6000000,
  "budget_percentile": 0.78,
  "runtime": 175,
  "mpaa_rating": "R",
  "release_month": 3,
  "release_year": 1972,
  "studio": "Paramount",
  "studio_prestige": 0.88,
  "lead_actor_prestige": 0.92,
  "screenplay_adapted": true,
  "source_material_prestige": 0.89,
  "cinematographer_prestige": 0.85,
  "awards_season_release": false,
  "plot_summary": "The aging patriarch of an organized crime...",
  "plot_complexity": 0.82,
  "character_count": 47,
  "named_character_count": 23
}
```

---

## Why This Matters

### 1. **Domain Flexibility**

If "narrative" = English text, then:
- ❌ Domains without text can't be processed
- ❌ Structured data domains are excluded
- ❌ Multi-modal information can't be used

If "narrative" = complete genome, then:
- ✅ ANY domain can be processed
- ✅ Structured data is native
- ✅ Text, numbers, categories, relationships all count
- ✅ We can synthesize text from structure when needed

### 2. **Feature Extraction**

Our transformers don't require English text input. They extract features from:
- **Structured fields** (numbers, categories, booleans)
- **Relationships** (team vs team, player vs player)
- **Temporal patterns** (streaks, trends, season progress)
- **Nominative features** (names, brands, prestige)
- **Synthesized text** (generated from structured data)
- **Natural text** (when available, like plot summaries)

### 3. **Universal Pipeline Design**

The Universal Domain Processor handles:
```python
# Input: Complete information genome (any format)
narrative_genome = {
    # Structured fields
    "team_a": "Boston Bruins",
    "team_b": "Toronto Maple Leafs",
    "spread": -2.5,
    
    # Temporal context
    "rest_days": [1, 0],
    "recent_form": [0.7, 0.4],
    
    # Nominative features
    "coach_prestige": [0.72, 0.68],
    "franchise_value": [5.2e9, 4.8e9],
    
    # Relationships
    "head_to_head_history": "15-12",
    "rivalry_intensity": 0.89,
    
    # Optional: Natural language
    "description": "Rivalry matchup with playoff implications..."
}

# Processor extracts features from ALL available information
# Text synthesis happens internally if needed for text-based transformers
```

---

## Implementation in Domain Registry

Our domain configurations already follow this principle:

```python
DomainConfig(
    name='nhl',
    data_path='data/domains/nhl_games_with_odds.json',
    
    # Can point to a text field OR structured data
    narrative_field='synthetic_narrative',  # Can be synthesized
    
    # Custom extractor builds narrative genome from structure
    custom_extractor=extract_nhl_data,
    
    # Outcome is part of the genome
    outcome_field='home_won'
)
```

The `extract_nhl_data` function:
1. Loads structured game data (JSON with many fields)
2. Builds complete information genome
3. Optionally synthesizes English text for text-based transformers
4. Returns genome representation

---

## Transformer Processing

### Text-Based Transformers
When a transformer requires natural language input:
```python
# If narrative is already text
if isinstance(narrative, str):
    features = transformer.extract(narrative)

# If narrative is structured
else:
    # Synthesize text representation
    text = synthesize_narrative_text(narrative)
    features = transformer.extract(text)
```

### Structure-Based Transformers
Many transformers work directly on structured genome:
```python
# Temporal transformer
temporal_features = extract_temporal_patterns(
    narrative['rest_days'],
    narrative['recent_form'],
    narrative['season_progress']
)

# Nominative transformer
nominative_features = extract_nominative_signals(
    narrative['team_names'],
    narrative['coach_prestige'],
    narrative['franchise_value']
)

# Relational transformer
relational_features = extract_relationships(
    narrative['head_to_head'],
    narrative['rivalry_intensity']
)
```

---

## Correct Terminology

### ✅ USE:
- "Narrative genome"
- "Information substrate"
- "Complete context"
- "Event features"
- "Instance data"
- "Narrative = all information about the instance"

### ❌ AVOID:
- "The narrative" (implying single text)
- "Written narrative" (too specific)
- "Story text" (subset of genome)
- "Description" (subset of genome)

### ✅ EXAMPLES:
- "The NHL game narrative includes rest advantage, goalie quality, rivalry status, and 50+ other structured features"
- "The startup narrative genome comprises founder credentials, traction metrics, market size, and pitch quality"
- "We extract 354 transformer features from each narrative genome"

---

## Domain Design Guidelines

When adding a new domain, think:

**❌ WRONG Question:**
"What text field contains the narrative?"

**✅ RIGHT Question:**
"What is the complete information genome that determines the outcome?"

### Domain Addition Checklist:

1. **Identify all relevant features**
   - Numbers (stats, metrics, measurements)
   - Categories (types, genres, classes)
   - Relationships (vs, head-to-head, comparisons)
   - Temporal (trends, streaks, timing)
   - Nominative (names, brands, prestige)
   - Context (location, conditions, circumstances)

2. **Structure the genome**
   - JSON with all fields
   - Each field typed appropriately
   - Relationships explicit
   - Temporal ordering preserved

3. **Text synthesis (optional)**
   - Only if text-based transformers needed
   - Generated from structured data
   - Descriptive, not creative
   - All key features mentioned

4. **Extractor function**
   - Loads raw data
   - Builds complete genome
   - Returns (narratives, outcomes)
   - narratives can be dict, JSON, or synthesized text

---

## Examples of Genome-First Thinking

### Example 1: Chess Games

**Beginner thinking:**
"The narrative is the game PGN notation."

**Genome thinking:**
"The narrative genome includes: player ratings, opening choice, time control, tournament type, player nationalities, historical head-to-head, position evaluations at key moments, critical decision points, time pressure situations, and final outcome context."

### Example 2: Supreme Court Cases

**Beginner thinking:**
"The narrative is the majority opinion text."

**Genome thinking:**
"The narrative genome includes: case type, circuit of origin, ideological composition of the court, oral argument quality scores, amicus brief count, parties involved, constitutional questions, precedent citations, opinion length, writing justice's ideology, vote split, year, and political context."

### Example 3: Weather Events

**Beginner thinking:**
"The narrative is the weather report description."

**Genome thinking:**
"The narrative genome includes: atmospheric pressure, temperature, humidity, wind patterns, historical averages, seasonal context, geographic location, terrain effects, satellite data, radar signatures, model predictions, and time-series of all measurements."

---

## Impact on Revalidation Instructions

When telling someone to "revalidate a domain through the universal pipeline," they should understand:

**NOT:**
"Extract the text narrative and run transformers on it"

**YES:**
"Load the complete information genome (all structured data, relationships, temporal context, nominative features) and process through transformers that extract features from this complete substrate. Text synthesis happens automatically if needed."

---

## Technical Implementation

### Current (Correct) Flow:
```python
# 1. Load domain data (any format)
raw_data = load_domain_json('nhl_games.json')

# 2. Extract narrative genome (can be structured)
for game in raw_data:
    narrative_genome = {
        'home_team': game['home'],
        'away_team': game['away'],
        'rest_advantage': game['rest_diff'],
        'is_rivalry': game['rivalry'],
        'betting_odds': game['odds'],
        # ... 50+ more fields
    }
    
    # 3. Universal processor handles genome
    features = processor.extract_features(narrative_genome)
    # Processor decides how to use genome:
    # - Some transformers work on structured fields directly
    # - Some transformers need text (synthesized if needed)
    # - All features concatenated
```

### What We DON'T Do:
```python
# ❌ WRONG: Require text upfront
narrative_text = game['description']  # May not exist!
features = processor.extract_features(narrative_text)  # Loses information!
```

---

## Summary

**Core Principle:**
> Narrative = Complete Information Genome, Not Literary Text

**Implications:**
1. ANY domain can be processed (structured or unstructured)
2. Text is optional (can be synthesized from structure)
3. Transformers extract from complete genome (not just text)
4. More information = better features = higher accuracy
5. Domain addition is about identifying the genome, not finding text

**When Adding Domains:**
- Think: "What information determines the outcome?"
- Not: "What text describes the event?"

**When Processing:**
- Use ALL available information
- Synthesize text if transformers need it
- Don't discard structured data for narrative text
- The genome IS the narrative

---

**This is fundamental to how the Universal Pipeline works.**

Every domain instruction, every transformer, every feature extraction - all assume narrative = genome, not text.

---

**Document Version**: 1.0  
**Last Updated**: November 17, 2025  
**Status**: Core Framework Definition

