# 📊 YOUR EXPERIMENT EXPLAINED - SIMPLE TERMS

## What You're Looking At

**Chart**: Cross-Validation Accuracy Distribution  
**Shows**: How well 3 different methods predicted news article categories  
**Data Used**: 400 news articles from 20newsgroups dataset  

---

## THE THREE METHODS TESTED

### **1. Statistical Baseline** (Left Box - ~69% accurate)

**What It Does**: Counts how often words appear  
**Method**: TF-IDF (Term Frequency-Inverse Document Frequency)

**In Plain English**:
- Looks at which words appear in each article
- Gives higher scores to important/distinctive words
- Ignores common words like "the", "and"
- Classifies based on word frequencies

**Example**:
```
Article about space: "NASA", "rocket", "orbit", "astronaut" appear frequently
Article about sports: "team", "game", "score", "player" appear frequently

Method: If article has "NASA", "rocket" → probably space topic
```

**Why It Won (69%)**: News topics ARE just word content. Space articles talk about space, sports articles talk about sports.

---

### **2. Semantic Narrative** (Middle Box - ~67% accurate)

**What It Does**: Tries to understand meaning, not just count words  
**Method**: Creates "embeddings" (meaning representations) + groups similar articles

**In Plain English**:
- Converts words to numbers that capture meaning
- "dog" and "cat" get similar numbers (both animals)
- Groups articles by similar meanings
- Classifies based on semantic patterns

**Example**:
```
"NASA launched rocket" and "SpaceX sent satellite"
→ Different words but similar MEANING
→ Both get classified as space topic
```

**Why It Did Okay (67%)**: Understanding meaning helps a bit, but for news topics, word content is enough.

---

### **3. Domain Narrative** (Right Box - ~52% accurate)

**What It Does**: Analyzes writing style, structure, and quality  
**Method**: Looks at how text is written (not what it says)

**In Plain English**:
- Measures sentence length variety
- Counts paragraph structure
- Analyzes writing complexity
- Detects stylistic patterns

**Example**:
```
Article A: Long sentences, complex structure, formal style
Article B: Short sentences, simple structure, casual style

Method: Tries to predict topic from style alone
```

**Why It Failed (52%)**: Writing style doesn't predict news topics well. A space article and sports article can have same writing style.

---

## THE ACTUAL DATA USED

### **20newsgroups Dataset**:

**What It Is**: Collection of news articles from internet newsgroups  
**How Many**: 400 training articles, 100 test articles  
**Categories**: 4 categories selected:
1. `alt.atheism` - Religious discussions
2. `comp.graphics` - Computer graphics
3. `sci.space` - Space and astronomy  
4. `talk.religion.misc` - Religious topics

**Real Example from YOUR data**:

```
Sample Article #1 (sci.space category):
"From: henry@zoo.toronto.edu (Henry Spencer)
Subject: Re: Vulcan? (was Stupid Shut Cost arguements)

In article <1993Apr20.132026.15781@oaonmsu.edu> sariaahSPAMMED@mortis.mis.udayton.edu 
(Sariah Amari) writes:
>>Actually, I've heard that it is just inside Mercury's orbit, but due to
>>intense solar radiation, we can't see it...

>>Therefore, Vulcan must exist. ;-)

>Of course it does. There are a lot of Vulcans there. Unfortunately,
>they all live underground..."

Length: 482 characters
Category: Space (sci.space)
```

**This is REAL text** that was classified using the 3 methods.

---

## WHAT THE BOX PLOT MEANS

### **The Boxes**:
Each box shows how consistent the method was across 5 different test runs (cross-validation folds).

**Statistical Baseline Box**:
- **Median line** (middle): 69% accuracy
- **Box range**: Most runs were 66-71%
- **Whiskers**: Best was 75%, worst was 64%
- **Interpretation**: Consistently good (tight box = reliable)

**Semantic Narrative Box**:
- **Median**: 67% accuracy
- **Box range**: 65-69%
- **Interpretation**: Slightly more consistent (tighter box) but lower scores

**Domain Narrative Box**:
- **Median**: 49% accuracy (barely better than guessing!)
- **Box range**: 44-64%
- **Huge spread**: Very inconsistent
- **Interpretation**: Doesn't work well for this task

---

## DATABASE SUBSTANTIATION (Where Data Came From)

### **Source**: sklearn's fetch_20newsgroups() function

**Loaded via**:
```python
from sklearn.datasets import fetch_20newsgroups

# This downloads real news articles
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=['alt.atheism', 'comp.graphics', 'sci.space', 'talk.religion.misc'],
    shuffle=True,
    random_state=42
)
```

**Database Details**:
- **Origin**: Actual Usenet newsgroup posts from 1990s
- **Size**: ~20,000 total articles (we used 500)
- **Format**: Plain text with headers, body
- **Labels**: True category for each article
- **Validation**: Widely-used benchmark dataset in NLP research

**Your Specific Data**:
- **Training**: 400 articles
- **Test**: 100 articles
- **Split**: 80/20 train/test
- **Categories**: 4 (balanced selection)

---

## IN SIMPLEST TERMS

**What was tested**: Can we predict which category (space, graphics, atheism, religion) a news article belongs to?

**Method 1 (Statistical)**: Count important words → 69% correct  
**Method 2 (Semantic)**: Understand meaning → 67% correct  
**Method 3 (Domain)**: Analyze writing style → 52% correct  

**Database**: 400 real news articles from 1990s internet (sci.space, comp.graphics, etc.)

**Result**: Word counting wins because news topics ARE word content.

**The boxes show**: How consistent each method was (tight box = reliable, wide box = inconsistent)

---

## WHERE TO SEE THE ACTUAL DATA

**View 20 real samples**:
Visit: `http://localhost:5738/data/explore/01_baseline_comparison`

You'll see the actual text that was analyzed, like:
```
Sample #0: "From: henry@zoo.toronto.edu..."
Category: sci.space
Length: 482 chars
Words: 85

[Full text displayed]
```

**This is the database** - real articles, real categories, real predictions.

---

**Your chart shows 3 methods tested on 400 real news articles. Statistical word counting won (69%) because news topics are word-based. The data is real, from a standard research dataset.** ✅
