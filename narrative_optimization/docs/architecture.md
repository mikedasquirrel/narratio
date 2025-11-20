# System Architecture

Architecture documentation for the narrative optimization research testbed.

## Overview

The framework is built on modular, composable components following sklearn's design patterns.

```
┌─────────────────────────────────────────────────────────────┐
│                    Narrative Testbed                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐     ┌──────────────┐                     │
│  │  Data Layer   │────>│ Transformers │                     │
│  │  (toy_data)   │     │  (Narrative) │                     │
│  └───────────────┘     └──────┬───────┘                     │
│                               │                              │
│                               v                              │
│                        ┌──────────────┐                      │
│                        │  Pipelines   │                      │
│                        │  (Assembly)  │                      │
│                        └──────┬───────┘                      │
│                               │                              │
│                               v                              │
│                        ┌──────────────┐                      │
│                        │ Experiments  │                      │
│                        │  (Testing)   │                      │
│                        └──────┬───────┘                      │
│                               │                              │
│                 ┌─────────────┴─────────────┐               │
│                 v                           v               │
│          ┌─────────────┐           ┌──────────────┐         │
│          │ Evaluation  │           │Visualization│         │
│          │  (Metrics)  │           │   (Plots)    │         │
│          └─────────────┘           └──────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. NarrativeTransformer (Base Class)

**Location**: `src/transformers/base.py`

**Purpose**: Foundation for all narrative-driven feature engineering.

**Key Methods**:
- `fit(X, y)`: Learn from data
- `transform(X)`: Apply transformation
- `get_narrative_report()`: Explain what narrative learned

**Design Pattern**: Template Method

**Inheritance**:
```
BaseEstimator, TransformerMixin (sklearn)
         ↑
         │
  NarrativeTransformer
         ↑
         ├── StatisticalTransformer
         ├── SemanticNarrativeTransformer
         └── DomainTextNarrativeTransformer
```

**Key Attributes**:
- `narrative_id`: Unique identifier
- `description`: Hypothesis description
- `metadata`: Learned information
- `is_fitted_`: Fit status

---

### 2. NarrativePipeline

**Location**: `src/pipelines/narrative_pipeline.py`

**Purpose**: Assemble transformers into coherent narratives with full metadata.

**Architecture**:
```python
NarrativePipeline
    │
    ├── steps: [(name, transformer, rationale), ...]
    ├── metadata: {hypothesis, expected_outcome, ...}
    └── pipeline: sklearn.Pipeline (built)
```

**Key Features**:
- Sequential step chaining
- Parallel branches (FeatureUnion)
- YAML serialization
- Narrative documentation

**Usage Pattern**:
```python
pipeline = NarrativePipeline(name, hypothesis)
pipeline.add_step(name, transformer, rationale)
pipeline.add_parallel_features(name, branches, rationale)
sklearn_pipeline = pipeline.build()
```

---

### 3. NarrativeExperiment

**Location**: `src/experiments/experiment.py`

**Purpose**: Systematically compare competing narrative hypotheses.

**Architecture**:
```python
NarrativeExperiment
    │
    ├── narratives: [(pipeline, hypothesis, name), ...]
    ├── metrics: [metric_names]
    ├── cv_strategy: CrossValidation
    └── results: {experiment_results}
```

**Workflow**:
1. Add competing narratives
2. Define evaluation metrics
3. Run cross-validation
4. Analyze results
5. Generate reports

**Output**:
- JSON results
- Pickle (full)
- Markdown report
- Visualizations

---

### 4. NarrativeEvaluator

**Location**: `src/evaluation/evaluator.py`

**Purpose**: Multi-objective evaluation beyond traditional metrics.

**Evaluation Dimensions**:

```
NarrativeEvaluator
    ├── Performance Metrics
    │   ├── Accuracy, F1, Precision, Recall
    │   ├── ROC-AUC
    │   └── Confusion Matrix
    │
    ├── Narrative Quality
    │   ├── Coherence Score
    │   ├── Interpretability Score
    │   └── Documentation Completeness
    │
    └── Meta-Metrics
        ├── Robustness (perturbation)
        ├── Generalization Gap
        └── Overall Quality Score
```

**Key Methods**:
- `evaluate_performance()`: Standard ML metrics
- `evaluate_narrative_coherence()`: Story consistency
- `evaluate_interpretability()`: Understandability
- `evaluate_robustness()`: Stability
- `comprehensive_evaluation()`: All dimensions

---

### 5. Visualization System

**Location**: `src/visualization/narrative_plots.py`

**Purpose**: Interactive and static visualizations.

**Plot Types**:
1. Performance comparison (bar charts)
2. CV score distributions (box plots)
3. Confusion matrices (heatmaps)
4. Narrative quality radar (radar chart)
5. Generalization gap (grouped bars)
6. Comprehensive summary (multi-panel)

**Technologies**:
- Matplotlib/Seaborn (static)
- Plotly (interactive)

---

## Data Flow

### Experiment Execution Flow

```
1. Load Data
   └─> ToyDataGenerator.quick_load_toy_data()

2. Build Pipelines
   └─> NarrativePipeline
       └─> Add transformers
       └─> Build sklearn Pipeline

3. Create Experiment
   └─> NarrativeExperiment
       └─> Add narratives
       └─> Define evaluation

4. Run Experiment
   └─> Cross-validation
       └─> For each narrative:
           ├─> Fit pipeline
           ├─> Predict
           └─> Score

5. Analyze Results
   └─> NarrativeExperiment.analyze()
       └─> Best by metric
       └─> Statistical comparisons

6. Visualize
   └─> NarrativePlotter
       └─> Generate plots

7. Report
   └─> Markdown generation
   └─> Save artifacts
```

---

## Design Patterns

### 1. Template Method
`NarrativeTransformer` defines skeleton, subclasses implement specifics.

### 2. Builder
`NarrativePipeline` builds complex sklearn Pipelines step by step.

### 3. Strategy
Different transformers = different narrative strategies.

### 4. Observer
Experiments track and log all transformations.

### 5. Facade
Simple interfaces hide complex sklearn machinery.

---

## Extension Points

### Adding New Transformer

```python
from src.transformers.base import NarrativeTransformer

class NewNarrative(NarrativeTransformer):
    def __init__(self, params):
        super().__init__(
            narrative_id="new_narrative",
            description="What story this tells"
        )
        self.params = params
    
    def fit(self, X, y=None):
        # Learn from data
        self.metadata['learned'] = something
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        self._validate_fitted()
        # Transform according to narrative
        return X_transformed
    
    def _generate_interpretation(self):
        return "What this narrative learned"
```

### Adding New Experiment

1. Create directory in `experiments/`
2. Add `run_experiment.py`
3. Import framework components
4. Build pipelines
5. Create and run experiment
6. Document findings

### Adding New Metric

1. Add to `NarrativeEvaluator`
2. Update config
3. Use in experiments

---

## Configuration Management

### Config Hierarchy

```
config/experiment_config.yaml (default)
    ↓
experiments/{exp_id}/config.yaml (override)
    ↓
Command-line arguments (override)
```

### Config Structure

```yaml
data:
  source: "20newsgroups"
  n_samples: 500

cross_validation:
  strategy: "StratifiedKFold"
  n_splits: 5

metrics:
  primary: "f1_macro"
  additional: ["accuracy", "precision_macro"]

transformers:
  # Transformer-specific configs
```

---

## Testing Strategy

### Unit Tests
- Each transformer independently
- Pipeline assembly
- Evaluation metrics

### Integration Tests
- End-to-end pipeline execution
- Experiment workflow
- Result generation

### Test Structure
```
tests/
├── test_transformers.py
├── test_pipelines.py
├── test_experiments.py
├── test_evaluation.py
└── test_visualization.py
```

---

## Performance Considerations

### Memory
- Transformers fit incrementally
- Sparse matrices where possible
- Results saved incrementally

### Speed
- Parallel cross-validation
- Cached data loading
- Efficient numpy operations

### Scalability
- Modular design allows swapping components
- Pipeline parallelization supported
- Can scale to larger datasets

---

## Error Handling

### Validation Layers

1. **Input Validation**
   - Data schema checks
   - Required fields
   
2. **Fit Validation**
   - Transformer fitted before transform
   - Compatible data shapes

3. **Configuration Validation**
   - Valid metric names
   - Compatible CV strategies

4. **Result Validation**
   - Schema compliance
   - Completeness checks

---

## Future Architecture Improvements

### Considered Enhancements

1. **MLflow Integration**
   - Automatic experiment tracking
   - Model registry

2. **Distributed Execution**
   - Ray/Dask for parallelization
   - Multi-machine experiments

3. **Web Dashboard**
   - Interactive result exploration
   - Real-time experiment monitoring

4. **Auto-ML Integration**
   - Automatic narrative discovery
   - Hyperparameter optimization

---

## Best Practices

### For Developers

1. **Inherit from base classes**: Don't reinvent
2. **Document narratives**: Explain the "why"
3. **Write tests**: Every component
4. **Type hints**: Throughout
5. **Consistent style**: Follow existing patterns

### For Researchers

1. **One hypothesis per experiment**: Focus
2. **Document failures**: Negative results matter
3. **Version everything**: Reproducibility
4. **Interpret results**: Don't just report numbers
5. **Update findings**: Keep documentation current

