# 🎯 Data Collection Template for Narrative Taxonomical Analysis

**Version:** 1.0  
**Date:** November 10, 2025  
**Purpose:** General-purpose template for collecting data on ANY domain for narrative analysis and formulization

---

## 🧬 What This System Does

We analyze **narratives** across any domain to discover **predictive patterns** in how things are named, described, and positioned. Our system has successfully analyzed:

- **Sports:** NBA games, MMA fighters → predicting winners from team/fighter narratives
- **Finance:** Cryptocurrency projects → predicting returns from project descriptions
- **Entertainment:** Movies → predicting box office from plot summaries
- **Mental Health:** Disorder names → predicting stigma from diagnostic labels
- **Natural Disasters:** Hurricane names → predicting evacuation rates from storm names
- **Products:** Tech comparisons → predicting preference from marketing language

**Core Finding:** "Better stories win" — narrative features predict outcomes better than demographics or statistics alone.

---

## 📊 What Data We Need From You

For your domain, we need data structured as **comparisons** (A vs B) or **observations** (text + outcome).

### Option 1: Comparison Format (Preferred for Competitive Domains)

When you have **head-to-head matchups** or **direct comparisons**:

```json
{
  "domain": "your_domain_name",
  "domain_type": "competition|evaluation|matching|prediction",
  "visibility": 0-100,
  "comparisons": [
    {
      "comparison_id": "unique_id_001",
      "text_a": "Full text description of option A",
      "text_b": "Full text description of option B",
      "context": {
        "stakes": "low|medium|high|championship",
        "timing": "early|mid|late|critical",
        "additional_context": "any other relevant context"
      },
      "outcome": {
        "winner": "A|B|tie",
        "margin": 0.0-1.0,
        "certainty": "actual|predicted|uncertain"
      },
      "metadata": {
        "date": "YYYY-MM-DD",
        "category": "subcategory if applicable",
        "weight": 1.0
      }
    }
  ]
}
```

**Examples:**
- **Sports:** "Lakers with championship momentum" vs "Celtics in elimination game" → Winner: Lakers (0.65 certainty)
- **Products:** "iPhone description" vs "Android description" → Winner: iPhone (sales data)
- **Dating:** Profile A vs Profile B → Winner: A (match success)

### Option 2: Observation Format (For Non-Competitive Domains)

When you have **individual observations** with outcomes:

```json
{
  "domain": "your_domain_name",
  "domain_type": "assessment|rating|outcome_prediction",
  "visibility": 0-100,
  "observations": [
    {
      "observation_id": "unique_id_001",
      "text": "Full text description or name to analyze",
      "features": {
        "feature_1": "value or measurement",
        "feature_2": "value or measurement"
      },
      "outcome": {
        "primary_outcome": "numeric or categorical",
        "secondary_outcomes": {
          "outcome_2": "value",
          "outcome_3": "value"
        }
      },
      "context": {
        "year": 2024,
        "category": "subcategory",
        "additional_context": "any relevant context"
      }
    }
  ]
}
```

**Examples:**
- **Hurricanes:** Name "Katrina" → Outcome: evacuation rate 0.45, casualties 1833
- **Mental Health:** "Schizophrenia" → Outcome: stigma score 8.2/10
- **Startups:** Company description → Outcome: funding success, valuation

---

## 🎯 Domain Specification

### Required Fields

**1. Domain Name** (lowercase, no spaces)
- Examples: `nba`, `crypto`, `hurricanes`, `dating_profiles`

**2. Domain Type** (select one)
- `competition` - Head-to-head matchups with winners
- `evaluation` - Rated/scored items
- `matching` - Compatibility/fit assessment
- `prediction` - Forecast future outcomes
- `assessment` - Diagnostic/classification

**3. Visibility Score** (0-100)
How observable/measurable is the outcome?

| Score | Description | Examples |
|-------|-------------|----------|
| **0-20** | Hidden outcomes | Dating compatibility, internal sentiment |
| **20-40** | Partially observable | Purchase decisions, engagement |
| **40-60** | Observable with effort | Survey results, tracked behaviors |
| **60-80** | Clearly measurable | Sales figures, click rates |
| **80-100** | Perfectly measurable | Sports scores, election results |

**Why visibility matters:** Our theory predicts effect size based on visibility:
```
Effect = 0.45 - 0.319(Visibility/100) + 0.15(GenreCongruence)
```
Higher visibility → narrative matters less (outcomes are obvious)  
Lower visibility → narrative matters MORE (outcomes are uncertain)

---

## 📝 Text Requirements

### What Makes Good "Text" Data

**GOOD TEXT** (Rich, narrative, descriptive):
```
"The Lakers bring championship pedigree with LeBron's experience in 
high-pressure situations, riding a 7-game win streak and facing their 
historic rivals in a must-win elimination game. The team shows momentum 
and confidence, having overcome adversity all season."
```

**LESS GOOD** (Sparse, statistical only):
```
"Lakers: 52-30 record, 112.3 PPG, 45.2% FG"
```

**IDEAL** (Both narrative + context):
```
"The Lakers (52-30) bring championship pedigree with LeBron's experience 
in high-pressure situations, averaging 112.3 PPG. They're riding a 7-game 
win streak and facing their historic rivals the Celtics in a must-win Game 7. 
The team shows momentum and confidence, having overcome adversity all season."
```

### Text Sources

Can be any of:
- **Natural language descriptions** (written by humans)
- **Marketing copy** (product descriptions, profiles)
- **News articles** (event coverage)
- **Self-descriptions** (bios, profiles, statements)
- **Names alone** (if domain is name-focused like hurricanes)
- **Technical specifications** (with context)
- **Mixed** (combination of above)

### Minimum Text Length

- **Name-only domains:** 1 word minimum (e.g., "Katrina")
- **Descriptive domains:** 20 words minimum
- **Optimal:** 100-300 words per text
- **Maximum:** No limit, but 500+ words may have diminishing returns

---

## 🧪 Outcome Requirements

### Types of Outcomes

**Binary** (Win/Loss):
```json
{"winner": "A", "confidence": 0.75}
```

**Continuous** (Scores, rates):
```json
{"evacuation_rate": 0.45, "casualties": 1833}
```

**Categorical** (Classifications):
```json
{"category": "high_performer", "rank": 3}
```

**Multiple** (Several outcomes):
```json
{
  "primary": {"winner": "A"},
  "secondary": {
    "margin": 12.5,
    "surprise_factor": 0.82,
    "public_interest": "high"
  }
}
```

### Ground Truth Requirements

**IDEAL:**
- Actual observed outcomes (not predictions)
- Objective measurements (not opinions)
- Large sample size (n > 50 observations)
- Temporal separation (outcome happens AFTER text creation)

**ACCEPTABLE:**
- Expert ratings (if systematic)
- Survey results (if validated)
- Proxy measures (if justified)
- Smaller samples (n ≥ 20)

**AVOID:**
- Circular outcomes (text describes outcome)
- Author bias (text creator knows outcome)
- Retrospective fitting (text modified after outcome)

---

## 📦 Data Format Examples by Domain

### Example 1: Sports (Competition)

```json
{
  "domain": "mma_fights",
  "domain_type": "competition",
  "visibility": 95,
  "comparisons": [
    {
      "comparison_id": "ufc_287_001",
      "text_a": "Conor McGregor brings striking dominance with knockout power, 
                 riding a comeback narrative after injury. He's motivated, 
                 aggressive, and seeking redemption in front of home crowd.",
      "text_b": "Khabib Nurmagomedov's undefeated grappling style and mental 
                 fortitude make him a relentless force. He's methodical, 
                 patient, and has never lost a professional fight.",
      "context": {
        "stakes": "championship",
        "timing": "title_fight",
        "weight_class": "lightweight",
        "location": "Las Vegas"
      },
      "outcome": {
        "winner": "B",
        "margin": 0.85,
        "method": "submission",
        "round": 4
      }
    }
  ]
}
```

### Example 2: Product Evaluation (Observation)

```json
{
  "domain": "smartphone_reviews",
  "domain_type": "evaluation",
  "visibility": 75,
  "observations": [
    {
      "observation_id": "iphone_15_pro",
      "text": "The iPhone 15 Pro combines elegant design with cutting-edge 
               technology. Its titanium frame feels premium and substantial. 
               The camera system captures stunning detail in any lighting. 
               The interface is intuitive and responsive. A true flagship 
               that delivers on its promises.",
      "features": {
        "price": 999,
        "brand": "Apple",
        "release_year": 2023
      },
      "outcome": {
        "primary_outcome": 8.7,
        "secondary_outcomes": {
          "sales_millions": 45.2,
          "satisfaction_rating": 4.6,
          "market_share": 0.18
        }
      }
    }
  ]
}
```

### Example 3: Names → Outcomes (Observation)

```json
{
  "domain": "pharmaceutical_brands",
  "domain_type": "assessment",
  "visibility": 45,
  "observations": [
    {
      "observation_id": "drug_001",
      "text": "Zoloft",
      "features": {
        "syllables": 2,
        "phonetic_hardness": 0.65,
        "memorability": 0.78,
        "therapeutic_class": "antidepressant"
      },
      "outcome": {
        "primary_outcome": 0.72,
        "secondary_outcomes": {
          "prescription_rate": 15.2,
          "patient_adherence": 0.68,
          "brand_recognition": 0.85
        }
      },
      "context": {
        "year_introduced": 1991,
        "patent_status": "expired",
        "generic_available": true
      }
    }
  ]
}
```

### Example 4: Profiles → Matches (Comparison)

```json
{
  "domain": "dating_profiles",
  "domain_type": "matching",
  "visibility": 25,
  "comparisons": [
    {
      "comparison_id": "match_001",
      "text_a": "I'm an adventurous soul who loves spontaneous road trips and 
                 trying new cuisines. I value deep conversations under the 
                 stars and believe in living life to the fullest. Looking for 
                 someone who embraces growth and isn't afraid of vulnerability.",
      "text_b": "Structured and goal-oriented professional who enjoys 
                 weekend hikes and documentary nights. I appreciate routine 
                 and planning ahead. Seeking a partner who values stability 
                 and has clear life direction.",
      "context": {
        "stakes": "medium",
        "platform": "dating_app",
        "user_age_a": 28,
        "user_age_b": 32
      },
      "outcome": {
        "winner": "complementary_match",
        "margin": 0.68,
        "certainty": "observed",
        "actual_outcome": "successful_relationship"
      }
    }
  ]
}
```

---

## 🎨 Domain Metadata

### Genre Classification

Help us understand your domain's "genre" (like a book genre):

**Character-Driven** (identity matters most):
- Sports, personal profiles, dating, resumes
- Expected effect: r = 0.4-0.6+

**Plot-Driven** (content/facts matter most):
- News, technical specs, data reports
- Expected effect: r = 0.1-0.3

**Style-Driven** (presentation matters most):
- Reviews, marketing, creative writing
- Expected effect: r = 0.3-0.5

**Ensemble-Driven** (relationships matter most):
- Team dynamics, network effects, group profiles
- Expected effect: r = 0.3-0.5

### Alpha Parameter (Balance)

Based on domain, estimate narrative vs. statistical balance:

```
α = 0.0-0.2  → Highly narrative (identity-driven)
α = 0.2-0.4  → Moderately narrative (mixed)
α = 0.4-0.6  → Balanced (hybrid)
α = 0.6-0.8  → Moderately statistical (content-driven)
α = 0.8-1.0  → Highly statistical (plot-driven)
```

**Examples:**
- NBA games: α ≈ 0.10 (empirically discovered)
- Wine reviews: α ≈ 0.15 (nominative-heavy)
- Dating profiles: α ≈ 0.25 (narrative-driven)
- News articles: α ≈ 0.70 (content-driven)

---

## 📏 Sample Size Requirements

### Minimum Viable

- **Comparison format:** 20+ comparisons
- **Observation format:** 30+ observations
- **Statistical power:** Sufficient for r ≥ 0.3 detection

### Recommended

- **Comparison format:** 50-100+ comparisons
- **Observation format:** 100-200+ observations
- **Statistical power:** Sufficient for r ≥ 0.2 detection

### Optimal

- **Comparison format:** 200+ comparisons
- **Observation format:** 500+ observations
- **Statistical power:** Sufficient for r ≥ 0.1 detection
- **Cross-validation:** Multiple subsamples

### Sample Diversity

Ensure variation in:
- **Outcome distribution:** Don't have 95% one outcome
- **Text lengths:** Mix of short and long
- **Context levels:** Multiple stake/timing combinations
- **Time periods:** Span multiple years if possible
- **Categories:** Cover subcategories if applicable

---

## 🔬 What We'll Discover

### 1. Narrative Formula for Your Domain

We'll extract 116+ features across 6 dimensions:

**Nominative Analysis** (naming patterns):
- Semantic fields (motion, cognition, emotion, etc.)
- Proper noun usage
- Categorization patterns
- Identity markers

**Self-Perception** (agency, confidence):
- First-person language
- Attribution patterns
- Growth vs. stasis orientation
- Self-confidence markers

**Narrative Potential** (growth, openness):
- Future orientation
- Possibility language
- Flexibility indicators
- Arc position

**Linguistic Patterns** (voice, style):
- Tense distribution
- Active/passive voice
- Sentiment
- Syntactic complexity

**Relational Value** (complementarity):
- Internal coherence
- Synergy potential
- Complementarity scores

**Ensemble Effects** (diversity, networks):
- Co-occurrence patterns
- Network centrality
- Thematic diversity

### 2. Predictive Model

We'll build models predicting your outcomes:

```
Outcome = f(
    Narrative_Quality,
    Context_Weight,
    Temporal_Factor,
    Domain_Specific_α
)
```

**Expected outputs:**
- Prediction accuracy (R² or classification accuracy)
- Feature importance rankings
- Narrative vs. statistical comparison
- Context weighting formula
- Temporal dynamics (if longitudinal data)

### 3. Domain Classification

Your domain will be classified in our taxonomy:

```
Narrative Life
├─ Kingdom: [Sports|Products|Profiles|Brands|Media]
│   └─ Phylum: [your_domain]
│       ├─ Optimal α parameter
│       ├─ Dominant transformers
│       ├─ Visibility score
│       ├─ Effect size prediction
│       └─ Genre classification
```

### 4. Cross-Domain Insights

We'll compare your domain to our existing 13 domains:
- DNA overlap (feature similarity)
- Transferable patterns
- Universal constants
- Domain-specific manifestations

### 5. Publication-Ready Results

You'll receive:
- Statistical validation (p-values, effect sizes, confidence intervals)
- Visualization dashboards
- Interactive web interface
- Academic paper framework
- Replication instructions

---

## 📤 How to Send Data

### Format Options

**Option 1: JSON file** (preferred)
```
your_domain_data.json
```

**Option 2: CSV files**
```
comparisons.csv  or  observations.csv
outcomes.csv
metadata.csv
```

**Option 3: API integration**
```
POST https://your-api.com/export
```

**Option 4: Database dump**
```
SQL, MongoDB export, etc.
```

### Include Documentation

**README.txt with:**
- Domain name and description
- Data source and collection method
- Outcome definitions
- Known biases or limitations
- Contact information
- License/usage rights

### Data Quality Checklist

✅ All required fields present  
✅ Text is actual human language (not just IDs)  
✅ Outcomes are ground truth (not circular)  
✅ Sample size sufficient (n ≥ 20)  
✅ No major data quality issues  
✅ Metadata documented  
✅ Usage rights clarified  

---

## 💡 Example Domains We'd Love to Analyze

### High Priority

- **Political campaigns:** Campaign messaging → election results
- **Job postings:** Job descriptions → application rates
- **Restaurant reviews:** Review text → Michelin stars
- **Movie trailers:** Trailer descriptions → box office
- **Scientific papers:** Abstract text → citation counts
- **Music descriptions:** Song descriptions → streaming numbers
- **Real estate:** Property listings → sale price/speed
- **Grants:** Grant proposals → funding success
- **Startups:** Pitch decks → funding rounds
- **Legal cases:** Case descriptions → verdicts

### Emerging Domains

- **NFT collections:** Project descriptions → floor price
- **Podcasts:** Episode descriptions → downloads
- **Substack:** Newsletter descriptions → subscriber growth
- **YouTube:** Video descriptions → view counts
- **Tweets/posts:** Tweet text → engagement
- **GitHub repos:** README quality → star count
- **Courses:** Course descriptions → enrollment
- **Apps:** App store descriptions → downloads

### Research Domains

- **Clinical trials:** Trial descriptions → success rates
- **Educational programs:** Program descriptions → outcomes
- **Therapy modalities:** Therapy descriptions → efficacy
- **Conservation programs:** Project descriptions → impact
- **Social movements:** Movement narratives → adoption
- **Organizational culture:** Culture descriptions → performance

---

## 🎯 Expected Timeline

Once we receive your data:

**Week 1: Data Processing**
- Ingest and validate data
- Build domain-specific transformers
- Extract narrative features

**Week 2: Analysis**
- Run experiments
- Discover optimal α
- Build predictive models
- Compare to visibility prediction

**Week 3: Validation**
- Cross-validation
- Statistical testing
- Hypothesis validation
- Robustness checks

**Week 4: Delivery**
- Generate visualizations
- Create web interface
- Write results documentation
- Prepare publication materials

**Total:** 4 weeks from data receipt to complete analysis

---

## 📞 Questions?

**Conceptual questions:**
- Read: `NARRATIVE_METHODOLOGY_PHILOSOPHY.md`
- Read: `UNIVERSAL_STRUCTURE_DOMAIN_SPECIFICS.md`

**Technical questions:**
- See examples in: `narrative_optimization/domains/`
- Check: `docs/guides/DOMAIN_INTEGRATION_GUIDE.md`

**Data format questions:**
- See: Hurricane example (`domains/hurricanes/`)
- See: Mental health example (`domains/mental_health/`)

---

## 🎊 Ready to Begin?

Send your data with this information:

```
DOMAIN: [your_domain_name]
TYPE: [competition|evaluation|matching|prediction|assessment]
VISIBILITY: [0-100]
GENRE: [character|plot|style|ensemble]
SAMPLE SIZE: [number of observations]
FORMAT: [JSON|CSV|API|other]
EXPECTED EFFECT: [your hypothesis about narrative importance]
CONTACT: [your email/info]
```

We'll confirm receipt within 24 hours and begin analysis!

---

**"Every domain has a narrative structure. Every narrative has predictive power. Every prediction reveals the deep architecture of stories."**

**Version:** 1.0  
**System:** Narrative Optimization 2.1  
**Framework:** Universal Structure, Domain-Specific Manifestation  
**Philosophy:** Better stories win — and we prove exactly how.

---


