# Research Findings

This is a living document tracking discoveries, insights, and patterns from narrative optimization experiments.

## Current Status

**Project Phase**: Complete Validation  
**Last Updated**: November 9, 2025  
**Experiments Completed**: 9 (Full suite)

---

## Findings by Experiment

### Experiment 01: Baseline Comparison ✅

**Status**: Complete  
**Hypothesis**: H1 - Narrative features outperform statistical baselines

**Findings**: 
- Statistical Baseline: **69.0%** accuracy (TF-IDF)
- Semantic Narrative: 66.8% (embeddings + clustering)
- Domain Narrative: 52.0% (style + structure)

**Key Insights**: 
- Simple TF-IDF is very strong for topic classification
- Semantic nearly matches (only 2% behind)
- Domain features underperformed significantly

**Implications**: Statistical baseline is the bar to beat (69%)

### Experiments 02-07: Advanced Transformers ✅

**Status**: All complete (individual tests)

**Findings**:
- Ensemble: 27.8%
- Linguistic: 37.3% (best of advanced)
- Self-Perception: 33.5%
- Narrative Potential: 29.0%
- Relational: 29.3%
- Nominative: 30.5%

**Key Insights**:
- All advanced transformers underperform baseline when used alone
- Linguistic captures most signal among advanced (voice, agency, temporality)
- Transform capture specific narrative aspects, not full content

**Implications**: Advanced transformers designed for **augmentation**, not replacement

### Experiments 08-09: Combinations & Multi-Modal ✅

**Status**: Complete

**Findings**:
- Stat + Linguistic: 63.8% (best combo)
- Stat + Ensemble: 61.8%
- Stat + Self-Perception: 61.0%
- Multi-Modal (all): 59.8%

**Key Insights**:
- Combinations don't beat baseline on generic classification
- More features ≠ better (multi-modal worst)
- Simple concatenation strategy insufficient

**Implications**: Need better integration OR domain-specific testing

---

## Theoretical Insights

### What We've Learned So Far

_This section will be populated as experiments run._

**About Feature Engineering**:
- TBD

**About Narrative Coherence**:
- TBD

**About Interpretability**:
- TBD

**About Domain Specificity**:
- TBD

---

## Patterns and Principles

_As experiments accumulate, patterns will emerge that guide future work._

### Emerging Principles

1. **Principle Name**: Description (once discovered)

### Counter-Intuitive Results

_Results that surprised us or contradicted expectations._

### Replicated Results

_Findings that hold across multiple experiments or datasets._

---

## Questions Raised

### Answered Questions

_Questions that experiments have resolved._

### Open Questions

_New questions raised by our findings._

1. Which narrative approach works best for text classification?
2. How much does narrative coherence matter vs raw performance?
3. Can narrative quality be quantified objectively?
4. Do narratives transfer across domains?

---

## Failed Approaches

_Documenting what didn't work is as valuable as what did._

### Approach Name

**What We Tried**: Description  
**Why It Failed**: Explanation  
**What We Learned**: Insights  
**Alternative Approaches**: What to try instead

---

## Unexpected Discoveries

_Serendipitous findings that weren't part of original hypotheses._

---

## Practical Recommendations

_Actionable insights for practitioners._

### When to Use Narrative Approaches

- TBD based on findings

### When Statistical Baselines Suffice

- TBD based on findings

### Feature Engineering Best Practices

- TBD based on findings

---

## Methodological Notes

### What Worked Well

- TBD

### What to Improve

- TBD

### Lessons for Future Experiments

- TBD

---

## Update Log

### [Date] - Initial Setup
- Framework implemented
- Baseline experiment designed
- Awaiting first results

---

## How to Update This Document

After each experiment:

1. **Add experiment section** with findings
2. **Update theoretical insights** with new learnings
3. **Record patterns** that emerge
4. **Document failures** honestly
5. **Note new questions** raised
6. **Update recommendations** based on evidence

Be honest about:
- What worked
- What didn't
- Why we think results occurred
- Limitations and caveats
- Confidence levels

Remember: Negative results are valuable results. Document them thoroughly.

