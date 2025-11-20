# Domain Integration Guide
## Narrative Optimization Framework

**Version**: 2.0.0  
**Last Updated**: November 2025

---

## Table of Contents

1. [Introduction & Purpose](#introduction--purpose)
2. [Architecture Overview](#architecture-overview)
3. [Data Format Specifications](#data-format-specifications)
4. [Transformer Catalog](#transformer-catalog)
5. [LLM Integration Instructions](#llm-integration-instructions)
6. [Integration Workflow](#integration-workflow)
7. [API Specifications](#api-specifications)
8. [Schema Examples](#schema-examples)
9. [Configuration JSON Format](#configuration-json-format)
10. [Code Examples](#code-examples)
11. [Ekko Platform Example](#ekko-platform-example)
12. [Troubleshooting & FAQ](#troubleshooting--faq)
13. [Extension & Customization](#extension--customization)
14. [References](#references)

---

## Introduction & Purpose

### What This Framework Does

The **Narrative Optimization Framework** is a modular machine learning research testbed designed to test whether **narrative quality and structure in feature engineering predicts outcomes better than statistical approaches alone**. 

The core philosophy: **"Better Stories Win"** - narrative-driven feature engineering that encodes domain expertise outperforms generic statistical approaches while maintaining interpretability.

### Why Integrate External Projects

External projects can leverage this framework to:

- **Test narrative hypotheses** on domain-specific data
- **Compare multiple narrative approaches** systematically
- **Validate domain theories** with rigorous ML experiments
- **Generate interpretable insights** from complex data
- **Benefit from pre-built transformers** for common narrative patterns

### How Integration Works

1. **Format your data** according to one of three standard formats (Text/Features/Mixed)
2. **Select appropriate transformers** that match your domain narrative
3. **Build a pipeline** encoding your hypothesis
4. **Run experiments** with cross-validation
5. **Export results** back to your project

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Narrative Framework                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data → Transformers → Pipelines → Experiments → Results    │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐                      │
│  │ Transformers │────>│  Pipelines   │                      │
│  │  (Narrative) │     │  (Assembly)  │                      │
│  └──────────────┘     └──────┬───────┘                      │
│                              │                              │
│                              v                              │
│                       ┌──────────────┐                       │
│                       │ Experiments  │                       │
│                       │  (Testing)   │                       │
│                       └──────┬───────┘                       │
│                              │                              │
│                ┌─────────────┴─────────────┐                │
│                v                           v                │
│         ┌─────────────┐           ┌──────────────┐          │
│         │ Evaluation  │           │Visualization│          │
│         │  (Metrics)  │           │   (Plots)    │          │
│         └─────────────┘           └──────────────┘          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Available Narrative Transformers

The framework provides 100+ narrative transformers spanning different hypotheses.

**To see all available transformers**:
```bash
python -m narrative_optimization.tools.list_transformers
```

**For complete documentation**: See `/docs/TRANSFORMERS_AND_PIPELINES.md`

**Key transformer categories**:
- **Core**: Foundational transformers (Nominative, SelfPerception, NarrativePotential, Linguistic, Ensemble, Relational)
- **Semantic**: Embedding-based understanding
- **Temporal**: Time-based features
- **Structural**: Plot structure (Conflict, Suspense, Framing)
- **Statistical**: TF-IDF baseline
10. **Custom transformers** - Build your own

### Hypothesis Testing Framework

The framework supports testing 10+ hypotheses:

- **H1**: Story quality predicts better than demographics
- **H2**: Character role complementarity
- **H3**: Arc position compatibility
- **H4**: Ensemble diversity predicts openness
- **H5**: Omissions more predictive than inclusions
- **H6**: Context-dependent weights outperform static
- **H7**: Priming effects matter
- **H8**: Interpretability-performance tradeoff
- **H9**: Domain transfer
- **H10**: Narrative coherence correlates with robustness

---

## Data Format Specifications

The framework accepts three standard data formats. Choose the one that best matches your source data.

### Format A: Raw Text + Labels

**Best for**: Text classification, sentiment analysis, content categorization

**Structure**:

```python
{
    "data": {
        "texts": [
            "First document text...",
            "Second document text...",
            ...
        ],
        "labels": [0, 1, 2, 0, ...],  # Integer labels
        "label_names": ["category_a", "category_b", "category_c"]
    },
    "metadata": {
        "n_samples": 1000,
        "n_classes": 3,
        "description": "Domain description"
    }
}
```

**File Formats Accepted**:
- JSON (recommended)
- CSV with `text` and `label` columns
- Numpy arrays (`X_train.npy`, `y_train.npy`)
- Python lists

**Example CSV**:

```csv
text,label
"User bio text here...",0
"Another profile description...",1
"Third text sample...",0
```

### Format B: Pre-Extracted Features

**Best for**: Pre-computed embeddings, existing feature matrices, numerical data

**Structure**:

```python
{
    "features": {
        "X": [[0.1, 0.5, ...], [0.2, 0.3, ...], ...],  # Feature matrix
        "y": [0, 1, 0, ...],  # Labels
        "feature_names": ["feature_1", "feature_2", ...],  # Optional
        "feature_descriptions": {  # Optional
            "feature_1": "What this feature represents",
            ...
        }
    },
    "metadata": {
        "n_samples": 1000,
        "n_features": 50,
        "feature_type": "embeddings|statistical|mixed",
        "source": "How features were extracted"
    }
}
```

**File Formats Accepted**:
- CSV (features as columns)
- Numpy arrays (`.npy`)
- Pandas DataFrames (pickled)
- HDF5

**Feature Requirements**:
- Numerical values only
- No missing values (use imputation first)
- Consistent dimensions across samples
- Feature names if interpretability desired

### Format C: Mixed Domain Data

**Best for**: Rich domain data with text, features, and metadata

**Structure**:

```python
{
    "data": {
        "texts": ["Text 1", "Text 2", ...],  # Raw text
        "features": [[0.1, 0.5, ...], ...],  # Pre-computed features
        "metadata": [  # Per-sample metadata
            {
                "user_id": "123",
                "timestamp": "2024-01-01",
                "context": {"key": "value"}
            },
            ...
        ],
        "labels": [0, 1, ...]
    },
    "schema": {
        "text_fields": ["bio", "description"],  # Which text fields to use
        "feature_fields": ["embedding", "sentiment_score"],  # Feature fields
        "metadata_fields": ["user_id", "timestamp", "context"],
        "label_field": "outcome"
    },
    "metadata": {
        "domain": "ekko_matching|recommendation|other",
        "n_samples": 1000,
        "description": "Domain-specific description"
    }
}
```

**Use Cases**:
- User profiles with bios + interaction history
- Products with descriptions + numerical features
- Conversations with text + temporal metadata
- Any domain combining multiple data types

---

## Transformer Catalog

### 1. StatisticalTransformer

**Narrative ID**: `statistical_baseline`

**Story**: Pure frequency-based statistics with no domain narrative. Serves as baseline.

**Input**: Raw text (list of strings)

**Output**: TF-IDF sparse matrix (typically 1000-5000 features)

**Use Cases**:
- Baseline for comparison
- Simple text classification
- When interpretability of individual terms matters

**Parameters**:
```python
StatisticalTransformer(
    max_features=1000,  # Max TF-IDF features
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,  # Min document frequency
    max_df=0.8  # Max document frequency
)
```

**Best Domains**: Any text-based domain, news classification, document categorization

---

### 2. SemanticNarrativeTransformer

**Narrative ID**: `semantic_narrative`

**Story**: Understanding semantic meaning through embeddings and clustering provides better features than word frequencies.

**Input**: Raw text

**Output**: Dense feature matrix (~70-120 features) including:
- Semantic embeddings (LSA dimensions)
- Cluster membership (one-hot)
- Cluster distances
- Semantic coherence metrics

**Use Cases**:
- Semantic similarity tasks
- Topic-based classification
- Understanding conceptual structure
- Cross-domain transfer

**Parameters**:
```python
SemanticNarrativeTransformer(
    n_components=50,  # Semantic dimensions
    n_clusters=10,  # Semantic groups
    max_features=5000  # Initial vocabulary size
)
```

**Best Domains**: Product reviews, scientific abstracts, content recommendation, semantic search

---

### 3. DomainTextNarrativeTransformer

**Narrative ID**: `domain_text_narrative`

**Story**: Expert-crafted features capturing writing style, document structure, and topical coherence outperform generic approaches.

**Input**: Raw text

**Output**: ~30-45 features including:
- Topic distributions (LDA)
- Topic coherence metrics
- Writing style features (lexical diversity, sentence patterns)
- Document structure features (paragraph organization)

**Use Cases**:
- Quality assessment
- Author attribution
- Content type classification
- Educational level prediction

**Parameters**:
```python
DomainTextNarrativeTransformer(
    n_topics=20,  # Number of LDA topics
    style_features=True,  # Include style analysis
    structure_features=True  # Include structure analysis
)
```

**Best Domains**: Academic writing, blog posts, professional profiles, creative content

---

### 4. NominativeAnalysisTransformer

**Narrative ID**: `nominative_analysis`

**Story**: How entities are named and categorized reveals narrative structure and identity construction.

**Input**: Raw text

**Output**: ~25-35 features including:
- Semantic field distributions (10 fields: motion, cognition, emotion, etc.)
- Proper noun patterns
- Category marker usage
- Identity construction metrics
- Naming consistency

**Use Cases**:
- Identity analysis
- Categorization behavior
- Domain-specific vocabulary analysis
- Branding and positioning

**Parameters**:
```python
NominativeAnalysisTransformer(
    n_semantic_fields=10,  # Semantic fields to track
    track_proper_nouns=True,  # Extract proper nouns
    track_categories=True  # Track categorical language
)
```

**Best Domains**: User profiles, brand descriptions, self-descriptions, social media bios

---

### 5. SelfPerceptionTransformer

**Narrative ID**: `self_perception`

**Story**: How people talk about themselves—their self-perception as reflected in language—predicts outcomes.

**Input**: Raw text (ideally first-person narratives)

**Output**: ~20-25 features including:
- First-person narrative intensity
- Self-attribution patterns (positive/negative traits)
- Growth orientation
- Aspirational vs descriptive balance
- Agency patterns
- Identity coherence
- Self-complexity

**Use Cases**:
- User profiling
- Personal statements
- Dating profiles
- Job applications
- Mental health screening

**Parameters**:
```python
SelfPerceptionTransformer(
    track_attribution=True,  # Track trait attribution
    track_growth=True,  # Track growth language
    track_coherence=True  # Measure identity coherence
)
```

**Best Domains**: Dating apps, career platforms, social networks, personal blogging

---

### 6. RelationalValueTransformer

**Narrative ID**: `relational_value`

**Story**: Narrative elements create value through relationships and complementarity, not just additive presence.

**Input**: Raw text

**Output**: ~9 features including:
- Internal complementarity (diversity within document)
- Relational density (connections to corpus)
- Synergy scores (non-linear interactions)
- Value attribution (relationships vs individuals)
- Relational coherence

**Use Cases**:
- Matching systems
- Team composition
- Content recommendation
- Collaboration prediction

**Parameters**:
```python
RelationalValueTransformer(
    n_features=100,  # TF-IDF features for similarity
    complementarity_threshold=0.3,  # Similarity threshold
    synergy_window=3  # Window for synergy detection
)
```

**Best Domains**: Dating/matching platforms, team formation, content pairing, recommendation systems

---

### 7. EnsembleNarrativeTransformer

**Narrative ID**: `ensemble_narrative`

**Story**: Elements gain meaning through co-occurrence patterns and network relationships, not isolation.

**Input**: Raw text

**Output**: ~7-11 features including:
- Ensemble size
- Co-occurrence density
- Shannon diversity
- Network centrality metrics
- Ensemble coherence
- Ensemble reach

**Use Cases**:
- Diversity assessment
- Portfolio composition
- Tag/keyword analysis
- Social network analysis

**Parameters**:
```python
EnsembleNarrativeTransformer(
    n_top_terms=50,  # Terms to track
    min_cooccurrence=2,  # Min co-occurrence threshold
    network_metrics=True,  # Compute network centrality
    diversity_metrics=True  # Compute diversity indices
)
```

**Best Domains**: Social networks, content tagging, portfolio analysis, recommendation diversity

---

### 8. LinguisticPatternsTransformer

**Narrative ID**: `linguistic_patterns`

**Story**: How a story is told (voice, agency, temporality) matters as much as what is told.

**Input**: Raw text

**Output**: ~19-28 features including:
- Narrative voice (POV density and consistency)
- Temporal orientation (past/present/future)
- Agency patterns (active vs passive)
- Emotional trajectory
- Linguistic complexity
- Evolution metrics (if enabled)

**Use Cases**:
- Writing quality assessment
- Narrative analysis
- Temporal orientation prediction
- Author style analysis

**Parameters**:
```python
LinguisticPatternsTransformer(
    track_evolution=True,  # Track feature evolution
    n_segments=3  # Segments for evolution tracking
)
```

**Best Domains**: Creative writing, journalism, academic papers, historical narratives

---

### 9. NarrativePotentialTransformer

**Narrative ID**: `narrative_potential`

**Story**: Narratives rich in possibility, open to change, and future-oriented predict better outcomes than closed, static narratives.

**Input**: Raw text

**Output**: ~25-30 features including:
- Future orientation
- Possibility language (modals)
- Growth mindset indicators
- Narrative flexibility
- Developmental arc position
- Openness to alternatives
- Temporal breadth
- Narrative momentum

**Use Cases**:
- Growth potential assessment
- Career trajectory prediction
- Innovation potential
- Change readiness

**Parameters**:
```python
NarrativePotentialTransformer(
    track_modality=True,  # Track possibility language
    track_flexibility=True,  # Measure flexibility
    track_arc_position=True  # Infer developmental arc
)
```

**Best Domains**: Career platforms, personal development, innovation assessment, startup evaluation

---

## LLM Integration Instructions

### System Prompt for External Bots

Use this system prompt when integrating domain data into the Narrative Optimization framework:

```markdown
You are a Domain Integration Expert for the Narrative Optimization Framework.

ROLE: Help researchers import external data analysis into our narrative-driven ML testbed.

AVAILABLE TRANSFORMERS:
1. StatisticalTransformer - TF-IDF baseline (no narrative)
2. SemanticNarrativeTransformer - Embeddings + semantic clustering
3. DomainTextNarrativeTransformer - Style + structure + topics
4. NominativeAnalysisTransformer - Naming and categorization patterns
5. SelfPerceptionTransformer - Self-referential patterns and identity
6. RelationalValueTransformer - Complementarity and synergy
7. EnsembleNarrativeTransformer - Co-occurrence and network effects
8. LinguisticPatternsTransformer - Voice, agency, temporality
9. NarrativePotentialTransformer - Openness, possibility, growth

AVAILABLE HYPOTHESES:
- H1: Story quality predicts better than demographics
- H2: Character role complementarity
- H3: Arc position compatibility
- H4: Ensemble diversity predicts openness
- H5: Omissions more predictive than inclusions
- H6: Context-dependent weights outperform static
- H7: Priming effects matter
- H8: Interpretability-performance tradeoff
- H9: Domain transfer
- H10: Narrative coherence correlates with robustness

DATA FORMATS:
- Format A: Raw text + labels (for text classification)
- Format B: Pre-extracted features (numerical matrices)
- Format C: Mixed domain data (text + features + metadata)

INTEGRATION WORKFLOW:
1. ANALYZE the domain and data structure
2. MAP to appropriate data format (A/B/C)
3. SELECT relevant transformers based on domain narrative
4. LINK to testable hypotheses (H1-H10)
5. GENERATE configuration JSON
6. FORMAT data according to schema

OUTPUT FORMAT: JSON configuration + formatted data structure
```

### Prompt Templates

#### Domain Analysis Prompt

```
Analyze the following domain for narrative optimization:

Domain: {domain_name}
Description: {domain_description}
Sample Data: {data_sample}

Tasks:
1. Identify the domain narrative (what story does this data tell?)
2. Map to data format (A: text, B: features, C: mixed)
3. Recommend 2-4 appropriate transformers
4. Identify relevant hypotheses to test
5. Generate integration configuration

Output as JSON with structure:
{
    "domain_analysis": {...},
    "recommended_format": "A|B|C",
    "transformers": [{...}],
    "hypotheses": ["H1", "H4", ...],
    "configuration": {...}
}
```

#### Transformer Selection Prompt

```
Given domain characteristics, select optimal transformers:

Domain Type: {domain_type}
Data Available: {data_types}
Research Question: {research_question}
Predicted Outcome: {outcome_variable}

Available Transformers: [See catalog above]

Select transformers that:
1. Match the domain narrative
2. Use available data types
3. Test the research question
4. Maintain interpretability

Output: Ordered list with rationale for each
```

#### Pipeline Configuration Prompt

```
Generate a complete pipeline configuration:

Domain: {domain_name}
Transformers: {selected_transformers}
Hypothesis: {hypothesis}

Generate:
1. Pipeline name and description
2. Transformer instantiation with parameters
3. Step rationales
4. Expected outcome
5. Evaluation metrics

Output: Complete NarrativePipeline configuration as JSON
```

### Example LLM Interaction

**User**: "I have user dating profiles with text bios and match success labels. How do I integrate?"

**LLM Response**:

```json
{
    "domain_analysis": {
        "domain": "dating_matching",
        "narrative": "User self-presentation predicts matching success",
        "data_type": "text_with_labels"
    },
    "recommended_format": "A",
    "transformers": [
        {
            "name": "SelfPerceptionTransformer",
            "rationale": "Analyzes self-presentation patterns in bios",
            "params": {
                "track_attribution": true,
                "track_growth": true,
                "track_coherence": true
            }
        },
        {
            "name": "NarrativePotentialTransformer",
            "rationale": "Measures openness and possibility in profiles",
            "params": {
                "track_modality": true,
                "track_flexibility": true
            }
        },
        {
            "name": "DomainTextNarrativeTransformer",
            "rationale": "Captures writing quality and style",
            "params": {
                "n_topics": 15,
                "style_features": true
            }
        }
    ],
    "hypotheses": ["H1", "H4"],
    "configuration": {
        "narrative_name": "Self-Presentation Quality",
        "hypothesis": "Quality of self-presentation in dating profiles predicts match success",
        "expected_outcome": "Narrative features outperform statistical baseline",
        "metrics": ["accuracy", "f1_macro", "roc_auc"]
    }
}
```

---

## Integration Workflow

### Step-by-Step Process

#### Step 1: Describe Your Domain

Provide:
- Domain name and type
- Research question
- Sample data (5-10 examples)
- Outcome variable
- Available data types

**Example**:

```python
domain_description = {
    "name": "ekko_matching",
    "type": "user_matching",
    "question": "Does narrative quality in profiles predict match success?",
    "data": {
        "texts": ["Sample bio 1", "Sample bio 2", ...],
        "labels": [1, 0, ...]  # 1=successful match, 0=no match
    },
    "outcome": "match_success_binary"
}
```

#### Step 2: Get Configuration

Use LLM agent or manual selection:

```python
# Option A: LLM-guided (recommended)
configuration = llm_agent.analyze_domain(domain_description)

# Option B: Manual selection
configuration = {
    "format": "A",  # Raw text + labels
    "transformers": ["SelfPerceptionTransformer", "NarrativePotentialTransformer"],
    "hypothesis": "H1",
    "metrics": ["f1_macro", "accuracy"]
}
```

#### Step 3: Format Your Data

Transform data to match selected format:

```python
# Format A example
formatted_data = {
    "X_train": list_of_text_strings,
    "y_train": list_of_labels,
    "X_test": list_of_test_strings,
    "y_test": list_of_test_labels,
    "metadata": {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(set(y_train)),
        "class_names": ["no_match", "successful_match"]
    }
}
```

#### Step 4: Import & Test

Load into framework and run experiment:

```python
from narrative_optimization.src.pipelines.narrative_pipeline import NarrativePipeline
from narrative_optimization.src.experiments.experiment import NarrativeExperiment
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from sklearn.ensemble import RandomForestClassifier

# Build pipeline
pipeline = NarrativePipeline(
    narrative_name="Self-Presentation Quality",
    hypothesis="Quality of self-presentation predicts match success"
)
pipeline.add_step(
    "self_perception", 
    SelfPerceptionTransformer(), 
    "Capture self-presentation patterns"
)
pipeline.add_step(
    "narrative_potential",
    NarrativePotentialTransformer(),
    "Measure openness and possibility"
)
pipeline.add_step(
    "classifier",
    RandomForestClassifier(n_estimators=100, random_state=42),
    "Predict match success"
)

sklearn_pipeline = pipeline.build()

# Run experiment
experiment = NarrativeExperiment(
    experiment_id="ekko_match_quality",
    description="Test whether profile narrative quality predicts matching"
)
experiment.add_narrative(sklearn_pipeline, "H1", "Self-Presentation")
experiment.define_evaluation(
    metrics=['f1_macro', 'accuracy', 'roc_auc'],
    cv_strategy=StratifiedKFold(5)
)

results = experiment.run(X_train, y_train)
```

#### Step 5: Export Results

Retrieve findings back to your project:

```python
# Get results
analysis = experiment.analyze()
best_narrative = analysis['best_narratives']['f1_macro']['name']
performance = analysis['best_narratives']['f1_macro']['score']

# Export
results_export = {
    "experiment_id": "ekko_match_quality",
    "best_approach": best_narrative,
    "performance": {
        "f1_macro": performance,
        "accuracy": analysis['best_narratives']['accuracy']['score']
    },
    "insights": pipeline.get_narrative_report(),
    "validated_hypothesis": "H1"
}

# Save for external project
import json
with open("results_for_ekko.json", "w") as f:
    json.dump(results_export, f, indent=2)
```

---

## API Specifications

If implementing API endpoints for programmatic integration:

### POST /api/integrate/analyze

Analyze domain and recommend configuration.

**Request**:

```json
{
    "domain": {
        "name": "string",
        "description": "string",
        "data_sample": ["text1", "text2", ...],
        "labels_sample": [0, 1, ...]
    }
}
```

**Response**:

```json
{
    "analysis": {
        "domain_type": "text|features|mixed",
        "recommended_format": "A|B|C",
        "narrative_themes": ["self_perception", "growth", ...],
        "complexity": "simple|moderate|complex"
    },
    "configuration": {
        "transformers": [...],
        "hypotheses": ["H1", "H4"],
        "pipeline_config": {...}
    }
}
```

### POST /api/integrate/import

Upload and import formatted data.

**Request** (multipart/form-data):
- `data_file`: JSON/CSV file with formatted data
- `config`: JSON configuration
- `domain_name`: String

**Response**:

```json
{
    "import_id": "uuid",
    "status": "success|error",
    "data_summary": {
        "n_samples": 1000,
        "n_features": 50,
        "format": "A|B|C"
    },
    "next_steps": ["run_experiment", "build_pipeline"]
}
```

### GET /api/integrate/transformers

List available transformers with documentation.

**Response**:

```json
{
    "transformers": [
        {
            "name": "SelfPerceptionTransformer",
            "narrative_id": "self_perception",
            "description": "...",
            "input_type": "text",
            "output_features": 20,
            "parameters": {...},
            "use_cases": [...],
            "best_domains": [...]
        },
        ...
    ]
}
```

### GET /api/integrate/schemas

Get format specifications and examples.

**Response**:

```json
{
    "formats": {
        "A": {
            "name": "Raw Text + Labels",
            "schema": {...},
            "example": {...}
        },
        "B": {...},
        "C": {...}
    }
}
```

---

## Schema Examples

### Example 1: Text Classification Domain (20newsgroups-style)

**Domain**: News article classification

**Format**: A (Raw Text + Labels)

**Data Structure**:

```json
{
    "data": {
        "texts": [
            "Article about space exploration and NASA missions...",
            "Discussion of computer graphics rendering techniques...",
            "Religious philosophy and theological debates..."
        ],
        "labels": [2, 1, 3],
        "label_names": ["alt.atheism", "comp.graphics", "sci.space", "talk.religion"]
    },
    "metadata": {
        "n_train": 400,
        "n_test": 100,
        "n_classes": 4,
        "domain": "news_classification",
        "source": "20newsgroups"
    }
}
```

**Recommended Transformers**:
- `DomainTextNarrativeTransformer` (style and structure)
- `SemanticNarrativeTransformer` (topic understanding)
- `StatisticalTransformer` (baseline)

**Hypothesis**: H1 (narrative features outperform statistical baseline)

---

### Example 2: User Matching Domain (Ekko-style)

**Domain**: Dating/matching platform

**Format**: C (Mixed Domain Data)

**Data Structure**:

```json
{
    "data": {
        "texts": [
            "I love hiking and outdoor adventures. Looking for someone who shares my passion for nature...",
            "Creative professional seeking intellectual connection. Enjoy art, philosophy, deep conversations..."
        ],
        "features": [
            [0.23, 0.45, 0.67, 0.12],  # Pre-computed: [activity_level, openness, age_norm, education_level]
            [0.78, 0.89, 0.45, 0.91]
        ],
        "metadata": [
            {
                "user_id": "user_001",
                "age": 28,
                "interactions": 15,
                "match_history": [0, 1, 0, 1, 1]
            },
            {
                "user_id": "user_002",
                "age": 32,
                "interactions": 23,
                "match_history": [1, 1, 0, 1, 1, 0]
            }
        ],
        "labels": [1, 1]  # Match success
    },
    "schema": {
        "text_field": "bio",
        "feature_fields": ["activity_level", "openness", "age_norm", "education_level"],
        "metadata_fields": ["user_id", "age", "interactions", "match_history"],
        "label_field": "match_success"
    },
    "metadata": {
        "domain": "ekko_matching",
        "n_samples": 1000,
        "task": "binary_classification",
        "description": "Predict match success from user profiles"
    }
}
```

**Recommended Transformers**:
- `SelfPerceptionTransformer` (analyze self-presentation)
- `NarrativePotentialTransformer` (openness and possibility)
- `EnsembleNarrativeTransformer` (diversity in match history)
- `RelationalValueTransformer` (complementarity for matching)

**Hypotheses**: H2 (role complementarity), H4 (ensemble diversity)

---

### Example 3: Recommendation System Domain

**Domain**: Content recommendation

**Format**: B (Pre-Extracted Features)

**Data Structure**:

```json
{
    "features": {
        "X": [
            [0.12, 0.45, 0.67, 0.89, 0.23, ...],  # 50 features per item
            [0.34, 0.56, 0.78, 0.90, 0.12, ...],
            ...
        ],
        "y": [1, 0, 1, ...],  # User engagement (clicked/not clicked)
        "feature_names": [
            "content_embedding_0", "content_embedding_1", ...,
            "diversity_score", "novelty_score", "quality_score"
        ],
        "feature_descriptions": {
            "diversity_score": "Content diversity relative to user history",
            "novelty_score": "How novel/unexpected this recommendation is",
            "quality_score": "Overall content quality rating"
        }
    },
    "metadata": {
        "n_samples": 5000,
        "n_features": 50,
        "feature_type": "mixed_embeddings_and_derived",
        "source": "recommendation_engine_v2",
        "domain": "content_recommendation"
    }
}
```

**Recommended Transformers**:
- Use features directly (already extracted)
- Can apply `EnsembleNarrativeTransformer` concepts to diversity metrics
- Focus on pipeline assembly and experiment design

**Hypotheses**: H4 (diversity), H6 (context-dependent weights)

---

### Example 4: Conversation Analysis Domain

**Domain**: Chat/conversation quality

**Format**: C (Mixed)

**Data Structure**:

```json
{
    "data": {
        "texts": [
            "User: Hi, how are you?\nBot: I'm doing great! How can I help you today?\nUser: I need information about...",
            ...
        ],
        "features": [
            [5, 150, 3.2, 0.75, 0.45],  # [turn_count, total_words, avg_turn_length, sentiment, engagement]
            ...
        ],
        "metadata": [
            {
                "conversation_id": "conv_001",
                "duration_seconds": 245,
                "user_satisfaction": 4.5,
                "resolved": true
            },
            ...
        ],
        "labels": [1, 0, 1, ...]  # 1=successful, 0=unsuccessful
    },
    "schema": {
        "text_field": "conversation_transcript",
        "feature_fields": ["turn_count", "total_words", "avg_turn_length", "sentiment", "engagement"],
        "metadata_fields": ["conversation_id", "duration_seconds", "user_satisfaction", "resolved"],
        "label_field": "success"
    },
    "metadata": {
        "domain": "conversation_quality",
        "n_samples": 2000,
        "task": "quality_prediction"
    }
}
```

**Recommended Transformers**:
- `LinguisticPatternsTransformer` (voice and agency patterns)
- `RelationalValueTransformer` (conversational complementarity)
- `NarrativePotentialTransformer` (forward momentum in conversation)

**Hypotheses**: H3 (arc compatibility), H7 (priming effects)

---

## Configuration JSON Format

### Standard Configuration Structure

```json
{
    "domain_name": "ekko_matching",
    "domain_description": "Dating platform user matching based on profile narratives",
    
    "data_format": "C",
    
    "input_schema": {
        "format_type": "mixed",
        "features": {
            "text_fields": ["bio", "interests"],
            "numerical_fields": ["age_norm", "activity_level", "openness"],
            "metadata_fields": ["user_id", "match_history"]
        },
        "labels": {
            "field": "match_success",
            "type": "binary",
            "classes": ["no_match", "successful_match"]
        },
        "validation": {
            "required_fields": ["bio", "match_success"],
            "min_samples": 100,
            "allow_missing": false
        }
    },
    
    "recommended_transformers": [
        {
            "transformer": "SelfPerceptionTransformer",
            "params": {
                "track_attribution": true,
                "track_growth": true,
                "track_coherence": true
            },
            "rationale": "Analyze how users present themselves in profiles",
            "priority": 1
        },
        {
            "transformer": "NarrativePotentialTransformer",
            "params": {
                "track_modality": true,
                "track_flexibility": true,
                "track_arc_position": true
            },
            "rationale": "Measure openness and possibility in self-descriptions",
            "priority": 2
        },
        {
            "transformer": "EnsembleNarrativeTransformer",
            "params": {
                "n_top_terms": 50,
                "network_metrics": true,
                "diversity_metrics": true
            },
            "rationale": "Assess diversity in interests and past matches",
            "priority": 3
        }
    ],
    
    "hypotheses_to_test": [
        {
            "id": "H2",
            "name": "Character role complementarity",
            "relevance": "Test if complementary self-presentations predict match success"
        },
        {
            "id": "H4",
            "name": "Ensemble diversity predicts openness",
            "relevance": "Test if diverse match history correlates with success"
        }
    ],
    
    "pipeline_config": {
        "narrative_name": "Profile Quality & Complementarity",
        "hypothesis": "Narrative quality in self-presentation and role complementarity predict matching success",
        "expected_outcome": "Narrative features outperform demographic features alone",
        "domain_assumptions": [
            "Users authentically represent themselves in bios",
            "Match success reflects genuine compatibility",
            "Text quality correlates with relationship potential"
        ],
        "steps": [
            {
                "name": "self_perception",
                "transformer": "SelfPerceptionTransformer",
                "params": {"track_attribution": true},
                "rationale": "Extract self-presentation patterns"
            },
            {
                "name": "narrative_potential",
                "transformer": "NarrativePotentialTransformer",
                "params": {"track_modality": true},
                "rationale": "Measure openness to connection"
            },
            {
                "name": "classifier",
                "transformer": "RandomForestClassifier",
                "params": {"n_estimators": 100, "random_state": 42},
                "rationale": "Predict match success from narrative features"
            }
        ]
    },
    
    "experiment_config": {
        "experiment_id": "ekko_match_quality_01",
        "description": "Test whether profile narrative quality predicts matching success",
        "metrics": ["f1_macro", "accuracy", "roc_auc", "precision_macro", "recall_macro"],
        "primary_metric": "f1_macro",
        "cv_strategy": "StratifiedKFold",
        "cv_splits": 5,
        "random_seed": 42,
        "baselines": [
            {
                "name": "Demographics Only",
                "features": ["age_norm", "activity_level"],
                "rationale": "Test if narrative adds value beyond demographics"
            },
            {
                "name": "Statistical Baseline",
                "transformer": "StatisticalTransformer",
                "rationale": "TF-IDF baseline for text"
            }
        ]
    },
    
    "output_config": {
        "export_format": ["json", "markdown", "pickle"],
        "save_models": true,
        "save_visualizations": true,
        "interpretability_report": true,
        "destination": "./results/ekko_integration/"
    }
}
```

---

## Code Examples

### Example 1: Complete Integration from JSON Config

```python
import json
from narrative_optimization.src.pipelines.narrative_pipeline import NarrativePipeline
from narrative_optimization.src.experiments.experiment import NarrativeExperiment
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load configuration
with open("domain_config.json", "r") as f:
    config = json.load(f)

# Load data (Format C - Mixed)
with open("ekko_data.json", "r") as f:
    data = json.load(f)

X_train = np.array(data["data"]["texts"])
y_train = np.array(data["data"]["labels"])

# Build pipeline from config
pipeline = NarrativePipeline(
    narrative_name=config["pipeline_config"]["narrative_name"],
    hypothesis=config["pipeline_config"]["hypothesis"],
    expected_outcome=config["pipeline_config"]["expected_outcome"]
)

# Add transformers from config
for step in config["pipeline_config"]["steps"]:
    if step["transformer"] == "SelfPerceptionTransformer":
        transformer = SelfPerceptionTransformer(**step["params"])
    elif step["transformer"] == "NarrativePotentialTransformer":
        transformer = NarrativePotentialTransformer(**step["params"])
    elif step["transformer"] == "RandomForestClassifier":
        transformer = RandomForestClassifier(**step["params"])
    
    pipeline.add_step(step["name"], transformer, step["rationale"])

# Build sklearn pipeline
sklearn_pipeline = pipeline.build()

# Create experiment
experiment = NarrativeExperiment(
    experiment_id=config["experiment_config"]["experiment_id"],
    description=config["experiment_config"]["description"]
)

# Add narrative
experiment.add_narrative(
    sklearn_pipeline,
    config["hypotheses_to_test"][0]["id"],
    config["pipeline_config"]["narrative_name"]
)

# Define evaluation
experiment.define_evaluation(
    metrics=config["experiment_config"]["metrics"],
    cv_strategy=StratifiedKFold(
        n_splits=config["experiment_config"]["cv_splits"],
        random_state=config["experiment_config"]["random_seed"],
        shuffle=True
    )
)

# Run experiment
results = experiment.run(X_train, y_train)

# Analyze and export
analysis = experiment.analyze()

# Save results
output_dir = config["output_config"]["destination"]
experiment.save_results(output_dir)

print(f"Experiment complete. Results saved to {output_dir}")
print(f"Best F1 Score: {analysis['best_narratives']['f1_macro']['score']:.3f}")
```

### Example 2: Quick Start with Manual Configuration

```python
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Your data
texts = ["User bio 1", "User bio 2", ...]  # List of strings
labels = [0, 1, 0, ...]  # List of labels

# Quick pipeline
pipeline = Pipeline([
    ('narrative', SelfPerceptionTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='f1_macro')
print(f"F1 Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Fit and use
pipeline.fit(texts, labels)
predictions = pipeline.predict(new_texts)
```

### Example 3: Comparing Multiple Narratives

```python
from narrative_optimization.src.experiments.experiment import NarrativeExperiment
from narrative_optimization.src.pipelines.narrative_pipeline import NarrativePipeline
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.semantic import SemanticNarrativeTransformer
from narrative_optimization.src.transformers.domain_text import DomainTextNarrativeTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# Narrative 1: Statistical Baseline
baseline_pipeline = NarrativePipeline("Statistical Baseline", "TF-IDF only")
baseline_pipeline.add_step("tfidf", StatisticalTransformer(), "Baseline approach")
baseline_pipeline.add_step("classifier", LogisticRegression(max_iter=1000), "Classifier")
baseline = baseline_pipeline.build()

# Narrative 2: Semantic Understanding
semantic_pipeline = NarrativePipeline("Semantic Narrative", "Embeddings capture meaning")
semantic_pipeline.add_step("semantic", SemanticNarrativeTransformer(), "Semantic features")
semantic_pipeline.add_step("classifier", LogisticRegression(max_iter=1000), "Classifier")
semantic = semantic_pipeline.build()

# Narrative 3: Domain Expertise
domain_pipeline = NarrativePipeline("Domain Narrative", "Expert features")
domain_pipeline.add_step("domain", DomainTextNarrativeTransformer(), "Domain-specific features")
domain_pipeline.add_step("classifier", LogisticRegression(max_iter=1000), "Classifier")
domain = domain_pipeline.build()

# Create experiment
experiment = NarrativeExperiment(
    experiment_id="narrative_comparison",
    description="Compare three narrative approaches"
)

# Add all narratives
experiment.add_narrative(baseline, "H1", "Baseline")
experiment.add_narrative(semantic, "H1", "Semantic")
experiment.add_narrative(domain, "H1", "Domain")

# Evaluate
experiment.define_evaluation(
    metrics=['f1_macro', 'accuracy'],
    cv_strategy=StratifiedKFold(5)
)

results = experiment.run(X_train, y_train)
analysis = experiment.analyze()

# Compare
print("\nPerformance Comparison:")
for narrative_name, scores in analysis['narrative_scores'].items():
    print(f"{narrative_name}: F1={scores['f1_macro']['test_mean']:.3f}")
```

### Example 4: Custom Transformer Example

```python
from narrative_optimization.src.transformers.base import NarrativeTransformer
import numpy as np

class CustomDomainTransformer(NarrativeTransformer):
    """Custom transformer for your specific domain."""
    
    def __init__(self, custom_param=10):
        super().__init__(
            narrative_id="custom_domain",
            description="Custom narrative hypothesis for my domain"
        )
        self.custom_param = custom_param
    
    def fit(self, X, y=None):
        """Learn domain patterns."""
        # Your fitting logic here
        self.metadata['learned_value'] = 'something'
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform to domain features."""
        self._validate_fitted()
        
        features = []
        for text in X:
            # Extract your domain-specific features
            doc_features = [
                len(text.split()),  # Word count
                text.count('?'),  # Questions
                # ... your features
            ]
            features.append(doc_features)
        
        return np.array(features)
    
    def _generate_interpretation(self):
        return f"Custom domain transformer learned: {self.metadata['learned_value']}"

# Use your custom transformer
custom_pipeline = Pipeline([
    ('custom', CustomDomainTransformer(custom_param=20)),
    ('classifier', RandomForestClassifier())
])
```

---

## Ekko Platform Example

### Complete Integration Example

**Scenario**: Ekko is a dating platform where users create profiles with text bios. We want to test whether narrative quality in bios predicts matching success.

### Step 1: Ekko Data Structure

```python
# Ekko raw data
ekko_data = {
    "users": [
        {
            "user_id": "ekko_001",
            "bio": "I'm an adventurous person who loves hiking and exploring new places. Looking for someone who shares my passion for the outdoors and meaningful conversations.",
            "age": 28,
            "interests": ["hiking", "photography", "reading"],
            "match_history": [
                {"matched_with": "ekko_045", "success": True},
                {"matched_with": "ekko_103", "success": False},
                {"matched_with": "ekko_167", "success": True}
            ]
        },
        {
            "user_id": "ekko_002",
            "bio": "Creative soul seeking intellectual connection. I enjoy deep conversations about philosophy, art, and life's mysteries. Let's explore ideas together.",
            "age": 32,
            "interests": ["art", "philosophy", "music", "writing"],
            "match_history": [
                {"matched_with": "ekko_078", "success": True},
                {"matched_with": "ekko_134", "success": True}
            ]
        },
        # ... more users
    ]
}
```

### Step 2: Format for Framework (Format C - Mixed)

```python
import numpy as np
from collections import Counter

def format_ekko_data(ekko_data):
    """Convert Ekko data to Framework Format C."""
    
    texts = []
    features = []
    metadata = []
    labels = []
    
    for user in ekko_data["users"]:
        # Text: bio
        texts.append(user["bio"])
        
        # Features: numerical/categorical
        age_norm = user["age"] / 100.0  # Normalize age
        n_interests = len(user["interests"])
        n_matches = len(user["match_history"])
        success_rate = sum(m["success"] for m in user["match_history"]) / (n_matches + 1)
        
        features.append([age_norm, n_interests, n_matches, success_rate])
        
        # Metadata
        metadata.append({
            "user_id": user["user_id"],
            "interests": user["interests"],
            "match_history": user["match_history"]
        })
        
        # Label: overall match success (binary or continuous)
        # For binary: 1 if success_rate > 0.5, else 0
        labels.append(1 if success_rate > 0.5 else 0)
    
    return {
        "data": {
            "texts": texts,
            "features": np.array(features),
            "metadata": metadata,
            "labels": np.array(labels)
        },
        "schema": {
            "text_field": "bio",
            "feature_fields": ["age_norm", "n_interests", "n_matches", "success_rate"],
            "metadata_fields": ["user_id", "interests", "match_history"],
            "label_field": "match_success"
        },
        "metadata": {
            "domain": "ekko_matching",
            "n_samples": len(texts),
            "task": "binary_classification",
            "description": "Predict user match success from profile narratives"
        }
    }

formatted_data = format_ekko_data(ekko_data)
```

### Step 3: Select Transformers

```python
# Based on domain analysis
ekko_config = {
    "transformers": [
        {
            "name": "SelfPerceptionTransformer",
            "rationale": "Analyze how users present themselves (authenticity, self-awareness)",
            "hypothesis": "Users with authentic self-presentation have better matches"
        },
        {
            "name": "NarrativePotentialTransformer",
            "rationale": "Measure openness, flexibility, future orientation",
            "hypothesis": "Open, possibility-rich profiles attract better matches"
        },
        {
            "name": "RelationalValueTransformer",
            "rationale": "Assess complementarity potential",
            "hypothesis": "Profiles with high relational value create better matches"
        },
        {
            "name": "EnsembleNarrativeTransformer",
            "rationale": "Analyze interest diversity",
            "hypothesis": "Diverse interests correlate with match success"
        }
    ],
    "hypotheses": ["H2", "H4"],
    "baseline": "Demographics only (age, n_interests, success_rate)"
}
```

### Step 4: Build Ekko Pipeline

```python
from narrative_optimization.src.pipelines.narrative_pipeline import NarrativePipeline
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer
from narrative_optimization.src.transformers.narrative_potential import NarrativePotentialTransformer
from narrative_optimization.src.transformers.relational import RelationalValueTransformer
from sklearn.ensemble import GradientBoostingClassifier

# Create narrative pipeline
ekko_pipeline = NarrativePipeline(
    narrative_name="Ekko Profile Quality & Complementarity",
    hypothesis="Narrative quality in profiles (authenticity + openness + relational value) predicts match success",
    expected_outcome="Narrative features outperform demographics alone",
    domain_assumptions=[
        "Users authentically represent themselves in bios",
        "Bio quality reflects relationship readiness",
        "Narrative patterns reveal compatibility indicators"
    ]
)

# Add narrative transformers
ekko_pipeline.add_step(
    "self_perception",
    SelfPerceptionTransformer(track_attribution=True, track_growth=True, track_coherence=True),
    "Capture self-presentation authenticity and self-awareness"
)

ekko_pipeline.add_step(
    "narrative_potential",
    NarrativePotentialTransformer(track_modality=True, track_flexibility=True),
    "Measure openness to connection and possibility"
)

ekko_pipeline.add_step(
    "relational_value",
    RelationalValueTransformer(complementarity_threshold=0.3),
    "Assess complementarity and relational potential"
)

ekko_pipeline.add_step(
    "classifier",
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Predict match success from narrative features"
)

# Build
ekko_sklearn_pipeline = ekko_pipeline.build()
```

### Step 5: Run Ekko Experiment

```python
from narrative_optimization.src.experiments.experiment import NarrativeExperiment
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Extract data
X = formatted_data["data"]["texts"]
y = formatted_data["data"]["labels"]

# Create experiment
ekko_experiment = NarrativeExperiment(
    experiment_id="ekko_profile_quality_v1",
    description="Test whether profile narrative quality predicts match success on Ekko platform"
)

# Add Ekko narrative
ekko_experiment.add_narrative(
    ekko_sklearn_pipeline,
    hypothesis_id="H2+H4",
    narrative_name="Profile Quality & Complementarity"
)

# Add baseline (demographics only)
from sklearn.preprocessing import StandardScaler

baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# For baseline, we'd use just the numerical features
# (This requires a custom adapter, simplified here)

# Define evaluation
ekko_experiment.define_evaluation(
    metrics=['f1_macro', 'accuracy', 'roc_auc', 'precision_macro', 'recall_macro'],
    cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Run
print("Running Ekko matching experiment...")
results = ekko_experiment.run(X, y)

# Analyze
analysis = ekko_experiment.analyze()
```

### Step 6: Export Results to Ekko Project

```python
# Generate comprehensive results for Ekko team
ekko_results_export = {
    "experiment": {
        "id": "ekko_profile_quality_v1",
        "date": "2024-11-10",
        "hypothesis": "Narrative quality predicts match success",
        "validated": analysis['best_narratives']['f1_macro']['name'] == "Profile Quality & Complementarity"
    },
    "performance": {
        "f1_macro": analysis['best_narratives']['f1_macro']['score'],
        "accuracy": analysis['best_narratives']['accuracy']['score'],
        "roc_auc": analysis['best_narratives']['roc_auc']['score']
    },
    "insights": {
        "narrative_report": ekko_pipeline.get_narrative_report(),
        "top_features": "Self-perception authenticity, openness to connection, relational potential",
        "key_findings": [
            "Users with authentic self-perception show 15% higher match success",
            "Openness and possibility language correlates with compatibility",
            "Complementarity features predict long-term match quality"
        ]
    },
    "recommendations": [
        "Encourage users to write authentic, detailed bios",
        "Highlight openness and future orientation in prompts",
        "Use narrative features in matching algorithm",
        "Consider narrative quality score in profile ranking"
    ],
    "hypotheses_validated": [
        {"id": "H2", "name": "Role complementarity", "status": "partial_support"},
        {"id": "H4", "name": "Ensemble diversity", "status": "strong_support"}
    ]
}

# Save for Ekko project
import json
with open("ekko_narrative_results.json", "w") as f:
    json.dump(ekko_results_export, f, indent=2)

# Generate markdown report
markdown_report = f"""
# Ekko Profile Quality Experiment Results

## Experiment Overview
- **ID**: ekko_profile_quality_v1
- **Hypothesis**: Narrative quality in profiles predicts match success
- **Date**: 2024-11-10

## Performance
- **F1 Score**: {analysis['best_narratives']['f1_macro']['score']:.3f}
- **Accuracy**: {analysis['best_narratives']['accuracy']['score']:.3f}
- **ROC AUC**: {analysis['best_narratives']['roc_auc']['score']:.3f}

## Key Findings
1. Users with authentic self-perception show 15% higher match success
2. Openness and possibility language correlates with compatibility
3. Complementarity features predict long-term match quality

## Recommendations
1. Encourage users to write authentic, detailed bios
2. Highlight openness and future orientation in prompts
3. Use narrative features in matching algorithm
4. Consider narrative quality score in profile ranking

## Hypotheses Validated
- **H2 (Role complementarity)**: Partial support
- **H4 (Ensemble diversity)**: Strong support
"""

with open("ekko_narrative_results.md", "w") as f:
    f.write(markdown_report)

print("Results exported to ekko_narrative_results.json and ekko_narrative_results.md")
```

### Step 7: Use Results in Ekko

```python
# In Ekko project, load results
with open("ekko_narrative_results.json", "r") as f:
    results = json.load(f)

if results["performance"]["f1_macro"] > 0.70:
    print("✅ Narrative features validated. Implementing in production...")
    # Integrate narrative scoring into Ekko matching algorithm
else:
    print("⚠️ Narrative features underperformed. Refining approach...")
```

---

## Troubleshooting & FAQ

### Common Issues

#### Issue 1: Data Format Mismatch

**Error**: `ValueError: Expected list of strings, got array`

**Solution**:
- Ensure text data is list of strings, not numpy array
- Convert: `X = X.tolist()` if needed
- Check that each element is a string: `assert all(isinstance(x, str) for x in X)`

#### Issue 2: Missing Values

**Error**: `ValueError: Input contains NaN`

**Solution**:
- Impute or remove missing values before transformation
- For text: `texts = [t if t else "" for t in texts]`
- For features: Use sklearn's `SimpleImputer`

#### Issue 3: Insufficient Data

**Error**: `ValueError: n_splits=5 cannot be greater than n_samples=3`

**Solution**:
- Reduce CV splits: `StratifiedKFold(n_splits=2)`
- Collect more data
- Use simple train/test split instead of CV

#### Issue 4: Incompatible Transformers

**Error**: Transformer expects text but receives features

**Solution**:
- Check transformer input requirements
- Text transformers: `StatisticalTransformer`, `SemanticNarrativeTransformer`, `SelfPerceptionTransformer`, etc.
- Feature transformers: Accept numerical arrays
- Use proper pipeline ordering

#### Issue 5: Memory Issues

**Error**: `MemoryError` with large datasets

**Solution**:
- Reduce `max_features` in TF-IDF transformers
- Use sparse matrices where possible
- Batch process data
- Reduce `n_components` in semantic transformers

### FAQ

**Q: Can I use multiple text fields?**

A: Yes, concatenate them:

```python
texts = [f"{user['bio']} {user['interests']}" for user in data]
```

**Q: How do I handle non-English text?**

A: Transformers use English stop words by default. For other languages:
- Modify transformer stop words
- Use language-specific preprocessing
- Consider translation pre-step

**Q: Can I combine narrative and non-narrative features?**

A: Absolutely! Use `FeatureUnion`:

```python
from sklearn.pipeline import FeatureUnion

combined = FeatureUnion([
    ('narrative', SelfPerceptionTransformer()),
    ('demographics', StandardScaler())  # Your numerical features
])
```

**Q: How long does training take?**

A: Depends on:
- Data size: 1000 samples typically 1-5 minutes
- Transformers: Text transformers slower than feature-based
- CV splits: 5-fold takes ~5x longer than single split
- Classifier: RandomForest faster than neural networks

**Q: Can I save trained pipelines?**

A: Yes:

```python
import pickle

# Save
with open("trained_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Load
with open("trained_pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

predictions = loaded_pipeline.predict(new_data)
```

**Q: How do I interpret results?**

A: Use narrative reports:

```python
report = pipeline.get_narrative_report()
print(report)

# For fitted transformers
transformer = pipeline.named_steps['self_perception']
interpretation = transformer._generate_interpretation()
```

**Q: What if my domain is very different?**

A: Create custom transformers:
- Inherit from `NarrativeTransformer`
- Implement `fit()` and `transform()`
- Encode your domain narrative
- See Custom Transformer example above

---

## Extension & Customization

### Creating Custom Transformers

```python
from narrative_optimization.src.transformers.base import NarrativeTransformer
import numpy as np

class MyDomainTransformer(NarrativeTransformer):
    """
    Custom transformer for [your domain].
    
    Narrative Hypothesis: [Your hypothesis about what matters in this domain]
    """
    
    def __init__(self, param1=10, param2=True):
        super().__init__(
            narrative_id="my_domain",
            description="[What story this transformer tells]"
        )
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y=None):
        """Learn domain patterns from data."""
        # Your learning logic
        self.metadata['learned_patterns'] = self._analyze_corpus(X)
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform to domain-specific features."""
        self._validate_fitted()
        
        features = []
        for item in X:
            item_features = self._extract_features(item)
            features.append(item_features)
        
        return np.array(features)
    
    def _extract_features(self, item):
        """Extract features from single item."""
        # Your feature extraction logic
        return [feature1, feature2, feature3, ...]
    
    def _analyze_corpus(self, X):
        """Analyze corpus for patterns."""
        # Corpus-level analysis
        return {"pattern": "value"}
    
    def _generate_interpretation(self):
        """Explain what was learned."""
        return f"Learned: {self.metadata['learned_patterns']}"
```

### Extending Existing Transformers

```python
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer

class EnhancedSelfPerceptionTransformer(SelfPerceptionTransformer):
    """Extended version with domain-specific additions."""
    
    def __init__(self, additional_param=True, **kwargs):
        super().__init__(**kwargs)
        self.additional_param = additional_param
        self.narrative_id = "enhanced_self_perception"
    
    def transform(self, X):
        """Add features to base transform."""
        # Get base features
        base_features = super().transform(X)
        
        # Add custom features
        additional_features = self._extract_additional_features(X)
        
        # Combine
        return np.hstack([base_features, additional_features])
    
    def _extract_additional_features(self, X):
        """Your additional features."""
        # Custom logic
        return np.array([[...] for x in X])
```

### Adding New Hypotheses

Add to `docs/hypotheses.md`:

```markdown
## H11: [Your Hypothesis Name]

**Status**: 🔴 Untested

**Hypothesis**: [Clear statement of what you expect]

**Operationalization**:
- **IV**: [Independent variable]
- **DV**: [Dependent variable]

**Metric**: [How you'll measure success]

**Test Procedure**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Success Criteria**: [What constitutes validation]

**Domain**: [Where this applies]
```

### Modifying Pipeline Configurations

```python
# Create reusable pipeline configurations

def create_profile_quality_pipeline(domain="general"):
    """Factory for profile quality pipelines."""
    
    if domain == "dating":
        transformers = [
            ("self_perception", SelfPerceptionTransformer()),
            ("potential", NarrativePotentialTransformer()),
            ("relational", RelationalValueTransformer())
        ]
    elif domain == "professional":
        transformers = [
            ("self_perception", SelfPerceptionTransformer()),
            ("domain_text", DomainTextNarrativeTransformer()),
            ("linguistic", LinguisticPatternsTransformer())
        ]
    else:
        transformers = [
            ("semantic", SemanticNarrativeTransformer()),
            ("domain_text", DomainTextNarrativeTransformer())
        ]
    
    pipeline = NarrativePipeline(
        narrative_name=f"{domain.title()} Profile Quality",
        hypothesis=f"Narrative quality in {domain} profiles predicts success"
    )
    
    for name, transformer in transformers:
        pipeline.add_step(name, transformer, f"{name} features")
    
    return pipeline

# Use
dating_pipeline = create_profile_quality_pipeline("dating")
professional_pipeline = create_profile_quality_pipeline("professional")
```

---

## References

### Core Documentation

- **Architecture**: `narrative_optimization/docs/architecture.md`
- **Hypotheses**: `narrative_optimization/docs/hypotheses.md`
- **Findings**: `narrative_optimization/docs/findings.md`
- **Roadmap**: `narrative_optimization/docs/roadmap.md`

### Source Code

- **Base Transformer**: `src/transformers/base.py`
- **Pipelines**: `src/pipelines/narrative_pipeline.py`
- **Experiments**: `src/experiments/experiment.py`
- **Evaluation**: `src/evaluation/evaluator.py`

### Transformer Implementations

All transformers in `src/transformers/`:
- `statistical.py` - TF-IDF baseline
- `semantic.py` - Semantic embeddings
- `domain_text.py` - Domain-specific text features
- `nominative.py` - Naming analysis
- `self_perception.py` - Self-referential patterns
- `relational.py` - Relational value
- `ensemble.py` - Ensemble effects
- `linguistic_advanced.py` - Linguistic patterns
- `narrative_potential.py` - Narrative potential

### Example Experiments

- **Baseline Comparison**: `experiments/01_baseline_comparison/`
- **Ensemble Effects**: `experiments/02_ensemble_effects/`
- **Linguistic Patterns**: `experiments/03_linguistic_patterns/`
- **Self Perception**: `experiments/04_self_perception/`

### Quick Start

- **Notebook**: `notebooks/00_quick_start.ipynb`
- **CLI**: `python run_experiment.py --help`
- **README**: `narrative_optimization/README.md`

---

## Contact & Support

For questions, issues, or collaboration:

1. Review this guide thoroughly
2. Check existing experiments for examples
3. Read transformer documentation
4. Run quick_start notebook

**Remember**: This framework tests narrative hypotheses. The goal isn't just better models—it's understanding *why* narratives matter in machine learning.

---

**Document Version**: 2.0.0  
**Framework Version**: 2.0.0  
**Last Updated**: November 10, 2025  
**Maintained By**: Narrative Optimization Research Team


