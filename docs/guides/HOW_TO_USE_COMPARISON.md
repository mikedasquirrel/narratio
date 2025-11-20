# How to Use the AI-Enhanced Comparison System

**Live URL**: `http://127.0.0.1:5738/analyze/compare`

---

## Quick Start

###  Step 1: Navigate to the Page
Open your browser to: `http://127.0.0.1:5738/analyze/compare`

### Step 2: Enter Your Comparison

**Example: Baseball Teams**

**Text A**: `The Dodgers have legendary championship history and competitive excellence`

**Text B**: `The Phillies bring passionate fans and community pride to baseball`

**Your Question**: `who is more likely to win tonight?`

### Step 3: Run Analysis
Click **"run comprehensive analysis"** button

### Step 4: View Results
In ~6-9 seconds, you'll see:

#### 🤖 **AI Direct Answer Section** (Top)
```
"who is more likely to win tonight?"

Answer: Based on narrative analysis, the Dodgers show stronger achievement-oriented language 
with championship history emphasis, suggesting confidence and psychological momentum. 
Narrative potential score 1.17 vs 0.0 indicates growth-oriented framing. 
Slight advantage: Dodgers based on confidence signaling.

Reasoning: Championship history language creates psychological momentum; 
narrative potential indicates forward-oriented competitive framing
```

#### 📊 **Visual Analysis Dashboard**
1. **Radar Chart**: Shows all 6 dimensions at once
2. **Semantic Heatmap**: 10 semantic fields compared
3. **Linguistic Distribution**: Voice and temporal patterns
4. **Feature Importance**: Top discriminating variables

#### 📝 **AI Insights**
- Comparison type: "sports_teams"  
- Category: "baseball"
- Confidence: 80%
- 4-5 key insights about narrative differences
- Narrative theme for each text
- Implications and recommendations

#### 🔬 **Detailed Transformer Analysis**
Full breakdown of all 6 transformers with 100+ features

---

## What The AI Analyzes

### For "Dodgers" vs "Phillies"

**Nominative Analysis** identifies:
- **Motion language** (action, win, compete) — Dodgers 2x higher
- **Achievement semantic field** — Dodgers emphasize past victories
- **Community semantic field** — Phillies emphasize fans/support
- **Proper nouns** — Team names, cities, players

**Self-Perception** reveals:
- **Confidence markers** — Dodgers show higher agency patterns
- **Identity construction** — Different self-presentation styles
- **Attribution patterns** — Achievement vs community focus

**Narrative Potential** shows:
- **Future orientation** — Forward-looking competitive framing
- **Growth language** — Progressive vs static narratives
- **Momentum indicators** — Which team signals forward movement

**Linguistic Patterns** captures:
- **Temporal distribution** — Past (history) vs Present (passion)
- **Agency patterns** — Active competitive language usage
- **Complexity** — Sophistication of narrative construction

**Relational Value** measures:
- **Complementarity** — How different yet related the narratives are
- **Synergy** — Non-additive narrative effects

**Ensemble Effects** analyzes:
- **Network patterns** — Key term co-occurrences
- **Diversity** — Variety in narrative elements

---

## More Example Questions

### Sports
- "who will win tonight?"
- "which team has more momentum?"
- "who has better team spirit?"
- "which narrative signals confidence?"

### Products
- "which should I buy?"
- "which is better value?"
- "which is more innovative?"
- "which aligns with quality?"

### User Profiles
- "who should I choose?"
- "who is more authentic?"
- "who has more growth potential?"
- "who would be a better match?"

### Brands
- "which brand aligns with my values?"
- "which is more customer-focused?"
- "which prioritizes innovation?"
- "which has better mission?"

---

## What Makes This Powerful

### 1. Context-Aware AI
- Understands **your question**
- Frames analysis around your intent
- Provides **direct answers** backed by evidence

### 2. Comprehensive Coverage
- **100+ variables** analyzed automatically
- All relevant narrative dimensions covered
- Nothing missed

### 3. Intelligent Presumptions
- AI detects comparison type (sports, products, etc.)
- Identifies relevant variables for that domain
- Highlights what matters most

### 4. Visual Understanding
- Multiple chart types show patterns at a glance
- Interactive exploration
- Color-coded comparisons

### 5. Actionable Insights
- Not just data — actual insights
- Specific, non-obvious findings
- Practical implications

---

## Understanding the Results

### Similarity Score (0-100%)
- **0-30%**: Very different narratives
- **30-50%**: Moderately different
- **50-70%**: Similar with key differences
- **70-90%**: Very similar narratives
- **90-100%**: Nearly identical

### Difference Scores (per transformer)
Higher = More different on that dimension
- Helps identify WHERE narratives diverge

### Feature Values
Each feature is a 0-1 normalized score
- Higher = More of that characteristic
- Compare side-by-side to see differences

### Semantic Fields
10 domains of meaning:
- **Motion**: Action, movement, competition
- **Cognition**: Thinking, knowing, learning
- **Emotion**: Feeling, passion, connection
- **Perception**: Seeing, observing, noticing
- **Communication**: Talking, sharing, expressing
- **Creation**: Making, building, producing
- **Change**: Transforming, evolving, becoming
- **Possession**: Having, owning, controlling
- **Existence**: Being, living, staying
- **Social**: Meeting, connecting, relating

---

## Pro Tips

### 1. Be Specific in Questions
❌ "which is better?" (vague)
✅ "which will win tonight?" (specific)
✅ "which offers better value for money?" (clear criteria)

### 2. Provide Context in Text
Short inputs work but more text = richer analysis
- Good: "dodgers"
- Better: "The Dodgers are a championship baseball team"
- Best: "The Dodgers have won multiple World Series championships and maintain competitive excellence"

### 3. Use Quick Examples
Click example buttons to see the system in action
- Then modify for your actual comparison

### 4. Explore All Visualizations
- Radar chart: Overall pattern
- Heatmap: Specific semantic differences
- Pie charts: Voice and temporal distribution
- Bar chart: Most important features

### 5. Read AI Insights
- They're generated specifically for YOUR comparison
- Often reveal non-obvious patterns
- Grounded in actual feature data

---

## Technical Details

### Processing Time
- **Transformers**: ~2-3 seconds (6 transformers, 100+ features)
- **AI Analysis**: ~3-5 seconds (GPT-4 interpretation)
- **Visualization**: ~1 second (rendering charts)
- **Total**: ~6-9 seconds

### What's Happening Behind the Scenes
1. Both texts processed through 6 transformers
2. 100+ features extracted per text
3. Differences calculated across all dimensions
4. AI receives full feature analysis + your question
5. GPT-4 generates contextualized insights
6. Visualizations render from data
7. Results displayed with animations

### API Endpoint
```
POST /analyze/api/comprehensive_compare

Body:
{
  "text_a": "string",
  "text_b": "string",
  "question": "string (optional)"
}

Response:
{
  "success": true,
  "comparison": {
    "metadata": {...},
    "transformers": {...},
    "ai_insights": {...},
    "overall_similarity": 0.72,
    ...
  }
}
```

---

## Troubleshooting

### No AI Insights Showing
- Check browser console for errors (F12)
- Verify OpenAI API key is valid
- Check network tab for API response
- Look for error messages in AI summary card

### Visualizations Not Rendering
- Ensure Plotly loaded (check console)
- Verify API returned data (check network tab)
- Try refreshing page
- Check for JavaScript errors

### Slow Performance
- Normal: 6-9 seconds for full analysis
- AI calls take time (GPT-4 is thorough)
- Consider using for meaningful comparisons

### Error Messages
- Read the error carefully
- Check browser console
- Verify both texts are entered
- Try simpler inputs first

---

## Real-World Applications

### Sports Analysis
Compare teams, players, coaches
Question examples:
- "who has better momentum?"
- "which team is more confident?"
- "who has psychological advantage?"

### Product Comparison
Compare features, marketing, value propositions
Question examples:
- "which is better for professionals?"
- "which prioritizes innovation?"
- "which offers better value?"

### User Profiling
Compare dating/professional profiles
Question examples:
- "who is more authentic?"
- "who has more growth potential?"
- "who would be better match?"

### Brand Analysis
Compare mission statements, values, positioning
Question examples:
- "which brand aligns with sustainability?"
- "which is more customer-centric?"
- "which innovates more?"

### Content Analysis
Compare writing styles, approaches, tones
Question examples:
- "which is more engaging?"
- "which is more authoritative?"
- "which resonates better?"

---

## Key Features Explained

### Nominative Analysis
**What it measures**: How things are named and categorized
**Why it matters**: Reveals framing strategies and identity construction

### Self-Perception
**What it measures**: Self-referential patterns and agency
**Why it matters**: Shows confidence, authenticity, self-awareness

### Narrative Potential
**What it measures**: Openness, possibility, growth orientation
**Why it matters**: Predicts adaptability and future focus

### Linguistic Patterns
**What it measures**: Voice, temporality, complexity
**Why it matters**: Reveals how story is told, not just what

### Relational Value
**What it measures**: Complementarity and synergy
**Why it matters**: Shows potential for relationships/combinations

### Ensemble Effects
**What it measures**: Co-occurrence and network patterns
**Why it matters**: Reveals diversity and interconnectedness

---

## Remember

This system analyzes **narrative patterns**, not objective truth.

For "who will win tonight?":
- AI analyzes **narrative confidence signals**
- Identifies **psychological momentum markers**
- Examines **achievement vs passion framing**
- But **doesn't predict actual outcomes**

What it DOES tell you:
- Which narrative signals more confidence
- Which uses achievement-oriented language
- Which emphasizes history vs. present
- Which shows forward momentum

Use this for understanding **narrative strategies** and **psychological patterns**, not fortune-telling!

---

**System created**: November 2025  
**Powered by**: 6 Narrative Transformers + GPT-4 Intelligence  
**Framework**: Narrative Optimization Research Testbed

