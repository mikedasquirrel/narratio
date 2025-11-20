# Testable Hypotheses

This document tracks all hypotheses in the narrative optimization research project.

## Status Legend
- 🔴 **Untested**: Not yet evaluated
- 🟡 **In Progress**: Currently being tested
- 🟢 **Validated**: Evidence supports hypothesis
- ⚫ **Refuted**: Evidence contradicts hypothesis
- 🔵 **Partial**: Mixed or context-dependent results

---

## H1: Story Quality Predicts Better Than Demographics

**Status**: 🟡 In Progress (Experiment 01)

**Hypothesis**: Story quality (coherence + depth + authenticity in text features) predicts classification outcomes better than pure statistical word frequencies.

**Operationalization**:
- **IV (Independent Variable)**: Feature engineering approach
  - Statistical baseline (TF-IDF)
  - Semantic narrative (embeddings + clustering)
  - Domain narrative (style + structure + topics)
- **DV (Dependent Variable)**: Classification accuracy and F1 score

**Metric**: Compare cross-validated F1-macro scores across three approaches

**Test Procedure**:
1. Build three pipelines with identical classifiers
2. Run 5-fold stratified cross-validation on 20newsgroups
3. Compare test scores with statistical significance testing
4. Analyze effect sizes

**Success Criteria**: Domain narrative F1 > Semantic F1 > Statistical F1 with p < 0.05

**Experiment**: `01_baseline_comparison`

---

## H2: Character Role Complementarity

**Status**: 🔴 Untested

**Hypothesis**: Character role complementarity (e.g., protagonist + mentor) outperforms role similarity (protagonist + protagonist) in relationship matching contexts.

**Note**: This hypothesis is domain-specific to relationship/matching platforms. Will be tested after H1 validation establishes baseline narrative approach.

**Operationalization**:
- **IV**: Pairing type (complementary vs similar roles)
- **DV**: Relationship success metrics

**Metric**: Success rate by role pairing combinations

**Test Procedure**: 
1. Extract character roles from user narratives
2. Compute all pairings
3. Chi-square test for categorical associations
4. Effect size measurement

**Success Criteria**: Complementary pairings show significantly higher success rates

**Future Work**: Requires domain-specific data from matching platform

---

## H3: Arc Position Compatibility

**Status**: 🔴 Untested

**Hypothesis**: Arc position compatibility (e.g., trials + resolution phases) predicts better outcomes than same-arc pairings in narrative contexts.

**Domain**: Relationship/narrative matching

**Operationalization**:
- **IV**: Arc pairing type (complementary vs matching)
- **DV**: Conversation depth, relationship duration

**Metric**: Conversation depth (message count) by arc pairing

**Test Procedure**:
1. Classify narrative arcs (beginning/trials/resolution)
2. Analyze pairings
3. ANOVA across pairing types

**Success Criteria**: Specific arc combinations show superior outcomes

**Future Work**: Domain-specific implementation needed

---

## H4: Ensemble Diversity Predicts Openness

**Status**: 🔴 Untested

**Hypothesis**: Ensemble diversity (variety in past interactions/partners) correlates with openness and predicts match success.

**Domain**: Relationship/network contexts

**Operationalization**:
- **IV**: Ensemble diversity score (variety in past interactions)
- **DV**: Match success rate

**Metric**: Correlation between ensemble_diversity and success_rate

**Test Procedure**:
1. Compute diversity score (e.g., Simpson's diversity index)
2. Regression with controls
3. Interaction effects analysis

**Success Criteria**: Significant positive correlation (r > 0.3, p < 0.05)

---

## H5: Omissions Are More Predictive Than Inclusions

**Status**: 🔴 Untested

**Hypothesis**: What people choose NOT to include (omitted tags, unused features, avoided topics) is more predictive than what they include.

**Operationalization**:
- **IV**: Model using inclusion features vs omission features vs both
- **DV**: Prediction accuracy

**Metric**: Model performance comparison (F1 scores)

**Test Procedure**:
1. Build three models:
   - Inclusions only
   - Omissions only  
   - Combined
2. Compare cross-validated performance
3. Feature importance analysis (SHAP)

**Success Criteria**: Omission-only model ≥ inclusion-only model

**Implementation**: Requires tracking what options were available but not chosen

---

## H6: Context-Dependent Weights Outperform Static Weights

**Status**: 🔴 Untested

**Hypothesis**: Dynamic feature weights that adapt to context (e.g., different weights for different conversation types) outperform static global weights.

**Operationalization**:
- **IV**: Weighting strategy (static vs dynamic)
- **DV**: Prediction accuracy across contexts

**Metric**: Cross-validated accuracy with and without context-specific weights

**Test Procedure**:
1. Cluster contexts (e.g., conversation types)
2. Learn context-specific weights
3. Compare to global weights
4. Test on held-out data

**Success Criteria**: Dynamic weights improve accuracy by >5%

**Implementation**: Requires multi-task learning or mixture of experts approach

---

## H7: Priming Effects Matter

**Status**: 🔴 Untested

**Hypothesis**: Recent context (what user just viewed/experienced) affects next match success beyond base compatibility - frame disruption can improve outcomes.

**Operationalization**:
- **IV**: Priming condition (consistent vs disrupted frame)
- **DV**: Match success rate

**Metric**: Success rate with vs without frame disruption

**Test Procedure**:
1. Identify priming patterns
2. Quasi-experimental comparison
3. Matched pairs analysis

**Success Criteria**: Frame disruption shows measurable effect (p < 0.05)

**Note**: Requires temporal data and careful causal inference

---

## Additional Hypotheses for Future Testing

### H8: Interpretability-Performance Tradeoff

**Hypothesis**: Within narrative approaches, there exists an optimal complexity level that balances interpretability and performance.

**Status**: 🔴 Untested

### H9: Domain Transfer

**Hypothesis**: Narrative patterns learned in one domain (e.g., text classification) transfer to other domains (e.g., matching, prediction).

**Status**: 🔴 Untested

### H10: Narrative Coherence Correlates with Robustness

**Hypothesis**: Pipelines with higher narrative coherence scores are more robust to perturbations and distribution shifts.

**Status**: 🔴 Untested

---

## Hypothesis Testing Workflow

For each hypothesis:

1. **Define**
   - State clearly
   - Operationalize variables
   - Specify metrics

2. **Design**
   - Choose appropriate test
   - Define success criteria
   - Plan analysis

3. **Implement**
   - Build required transformers
   - Create experiment
   - Collect data

4. **Analyze**
   - Run statistical tests
   - Compute effect sizes
   - Check assumptions

5. **Document**
   - Update status
   - Record findings
   - Note limitations

6. **Iterate**
   - Refine hypothesis
   - Follow-up questions
   - New hypotheses

---

## Meta-Questions

These questions guide our overall research direction:

1. **What makes a narrative "better"?**
   - Coherence, interpretability, performance, robustness?
   - How do we weight these dimensions?

2. **When do narratives matter most?**
   - Simple vs complex tasks?
   - Small vs large datasets?
   - High vs low stakes decisions?

3. **Can narratives be learned or must they be designed?**
   - Meta-learning approaches?
   - Automatic narrative discovery?

4. **How domain-specific must narratives be?**
   - Generic principles vs specific features?
   - Transfer learning for narratives?

---

## Updating This Document

When testing a hypothesis:
- Update status
- Link to experiment
- Summarize findings
- Note effect sizes and confidence

When refuting a hypothesis:
- Explain why
- Propose alternatives
- Document what was learned

When validating a hypothesis:
- Document evidence
- Note boundary conditions
- Suggest extensions

