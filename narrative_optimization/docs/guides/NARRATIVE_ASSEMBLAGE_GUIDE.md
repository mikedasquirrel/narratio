# Narrative Assemblage Guide

## Overview

This guide explains how transformers work together in the narrative optimization framework, focusing on the **narrative assemblage** - how different transformers combine to extract comprehensive narrative features.

## Key Concepts

### Transformer Categories

Transformers are organized into categories based on what they measure:

1. **Nominative Transformers** (9 transformers)
   - Extract name-based features
   - Character names, author names, phonetic patterns
   - Social status, namespace ecology

2. **Phonetic Transformers** (1 transformer)
   - Phonetic patterns in names
   - Sound symbolism, euphony

3. **Structural Transformers** (3 transformers)
   - Plot structure, conflict, tension
   - Narrative arc, pacing

4. **Semantic Transformers** (5 transformers)
   - Emotional resonance, authenticity
   - Cultural context, expertise

5. **Statistical Transformers** (1 transformer)
   - TF-IDF baseline
   - Content-based features

### How Transformers Work Together

#### 1. Complementary Transformers

Transformers that capture different aspects:
- **Nominative + Phonetic**: Names and their sounds
- **Structural + Semantic**: Plot and emotion
- **Statistical + Narrative**: Content and story quality

**Example**: `NominativeAnalysisTransformer` + `PhoneticTransformer`
- Nominative extracts character names
- Phonetic analyzes sound patterns
- Together: Complete name analysis

#### 2. Synergistic Transformers

Transformers that amplify each other:
- **Ensemble + Relational**: Character relationships and co-occurrence
- **Multi-Scale + Temporal**: Scale and time evolution

**Example**: `EnsembleNarrativeTransformer` + `RelationalValueTransformer`
- Ensemble finds co-occurrence patterns
- Relational measures complementarity
- Together: Stronger relationship prediction

#### 3. Redundant Transformers

Transformers with overlapping information:
- Some transformers may capture similar patterns
- Useful for robustness but may not add much

**Example**: Multiple nominative transformers may overlap
- `UniversalNominativeTransformer` and `HierarchicalNominativeTransformer`
- Both extract nominative patterns but at different levels

#### 4. Antagonistic Transformers

Transformers that conflict:
- Rare but possible
- May indicate domain-specific patterns

## Transformer Selection Logic

### Based on Domain Narrativity (π)

**Low π (< 0.3)**: Constrained domains
- Focus: Statistical, structural transformers
- Less: Nominative, semantic transformers

**Medium π (0.3-0.7)**: Balanced domains
- Use: Mix of all transformer types
- Optimize: Based on domain characteristics

**High π (> 0.7)**: Open domains
- Focus: Nominative, semantic, ensemble transformers
- Less: Statistical baseline

### Based on Domain Type

**Novels** (π = 0.72):
- **Critical**: All nominative/phonetic transformers
- **Important**: Structural (conflict, suspense), semantic (emotional, cultural)
- **Useful**: Ensemble, relational, multi-scale

**Nonfiction** (π = 0.61):
- **Critical**: Expertise, authenticity, framing
- **Important**: Nominative (author names), information theory
- **Useful**: Statistical, temporal

**Combined** (π = 0.66):
- **Critical**: All transformers
- **Important**: Book type flag as feature
- **Useful**: Cross-domain validation

## Multi-Scale Analysis

Features can be analyzed at different scales:

### Nano Scale (Sentences)
- Individual sentence-level features
- Best for: Linguistic patterns, phonetic analysis
- Transformers: `LinguisticPatternsTransformer`, `PhoneticTransformer`

### Micro Scale (Paragraphs)
- Paragraph-level features
- Best for: Scene analysis, character interactions
- Transformers: `EnsembleNarrativeTransformer`, `RelationalValueTransformer`

### Meso Scale (Sections/Chapters)
- Section-level features
- Best for: Plot structure, narrative arc
- Transformers: `ConflictTensionTransformer`, `TemporalEvolutionTransformer`

### Macro Scale (Full Text)
- Book-level features
- Best for: Overall narrative quality
- Transformers: All transformers (aggregated)

## Feature Attribution

### Methods

1. **Ablation Study**
   - Remove one transformer at a time
   - Measure impact on performance
   - Identifies critical transformers

2. **Permutation Importance**
   - Permute features from one transformer
   - Measure importance
   - More robust than ablation

3. **SHAP Values** (if available)
   - Shapley Additive Explanations
   - Fair feature attribution
   - Shows individual feature contributions

### Interpreting Results

**High Impact Transformers**:
- Critical for domain
- Should always be included
- Example: Nominative transformers for novels

**Low Impact Transformers**:
- May be redundant
- Or domain-specific
- Consider removing for efficiency

## Best Practices

### 1. Start with Core Transformers

Always include:
- `StatisticalTransformer` (baseline)
- `NominativeAnalysisTransformer` (core nominative)
- `PhoneticTransformer` (phonetic patterns)
- Domain-appropriate transformers

### 2. Add Domain-Specific Transformers

Based on domain type:
- **Novels**: Structural, semantic transformers
- **Nonfiction**: Expertise, authenticity transformers
- **Sports**: Ensemble, relational transformers

### 3. Test Transformer Combinations

Use interaction analysis to find:
- Synergistic pairs
- Complementary pairs
- Redundant pairs

### 4. Optimize for Scale

Different scales may need different transformers:
- Nano: Linguistic, phonetic
- Micro: Ensemble, relational
- Meso: Structural, temporal
- Macro: All (aggregated)

### 5. Validate Cross-Domain

Test transformers across domains:
- Universal transformers work everywhere
- Domain-specific transformers may not transfer
- Use cross-domain validation

## Example: Novels Domain

### Recommended Transformer Set

**Core** (always include):
1. `StatisticalTransformer`
2. `NominativeAnalysisTransformer`
3. `PhoneticTransformer`
4. `EnsembleNarrativeTransformer`

**Novels-Specific** (highly recommended):
5. `UniversalNominativeTransformer`
6. `HierarchicalNominativeTransformer`
7. `ConflictTensionTransformer`
8. `SuspenseMysteryTransformer`
9. `EmotionalResonanceTransformer`
10. `CulturalContextTransformer`

**Additional** (as needed):
11. `RelationalValueTransformer`
12. `MultiScaleTransformer`
13. `GravitationalFeaturesTransformer`
14. All other nominative transformers

### Expected Performance

- **R²**: 0.3-0.6 (depending on outcome)
- **Nominative contribution**: 30-50% of importance
- **Best scale**: Meso (chapters/sections)

## Troubleshooting

### Low Performance

1. **Check transformer selection**
   - Include domain-appropriate transformers
   - Remove redundant transformers

2. **Check feature extraction**
   - Ensure nominatives are extracted correctly
   - Verify text quality

3. **Check scale**
   - Try different scales
   - Aggregate across scales

### High Redundancy

1. **Identify redundant pairs**
   - Use correlation analysis
   - Remove one from each pair

2. **Focus on complementary transformers**
   - Different aspects
   - Different scales

### Domain Transfer Issues

1. **Use universal transformers**
   - Nominative, phonetic, statistical
   - Avoid domain-specific transformers

2. **Validate cross-domain**
   - Test on multiple domains
   - Identify universal patterns

## Conclusion

Narrative assemblage is about finding the right combination of transformers for your domain. Start with core transformers, add domain-specific ones, test combinations, and optimize based on results.

The key is understanding:
- What each transformer measures
- How transformers interact
- What works for your domain
- How to optimize the combination

