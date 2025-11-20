# Domain Addition Template

**Follow this template when adding any new domain to the system**

---

## The Core Question

When adding a domain, ask:

> **"What are the stories of this domain? Do they match familiar stories from structurally similar domains? Do these stories unfold at predicted frequency (accounting for observation/reporting bias)? What patterns emerge in the trends towards story realization?"**

---

## Step-by-Step Process

### 1. Prepare Your Data

Create `data/domains/YOUR_DOMAIN_NAME.json`:

```json
{
  "texts": [
    "Narrative text 1...",
    "Narrative text 2..."
  ],
  "outcomes": [1, 0, 1, 0],
  "names": ["Entity 1", "Entity 2"],
  "timestamps": [1234567890, 1234567891],
  "metadata": {
    "domain_type": "individual_sport|team_sport|prestige|natural|etc",
    "competitive": true,
    "temporal_structure": "event|season|tournament"
  }
}
```

**Data Requirements**:
- Minimum 50 samples (100+ recommended)
- Binary or continuous outcomes
- Rich narrative text (not just entity names)
- Timestamps if available (for temporal analysis)

---

### 2. Run Master Integration

```bash
python MASTER_INTEGRATION.py your_domain data/domains/your_domain.json \
  --pi 0.7 \
  --type individual_expertise
```

**Estimate Domain π (Narrativity)**:
- 0.0-0.3: Highly constrained (lottery, aviation)
- 0.3-0.6: Moderate (team sports, markets)
- 0.6-0.8: High (individual sports, expertise)
- 0.8-1.0: Very high (entertainment, identity)

**Domain Types**:
- `individual_sport`: Tennis, golf, chess
- `team_sport`: NBA, NFL, MLB
- `prestige`: Oscars, WWE, awards
- `expertise`: Chess, startups, research
- `natural`: Hurricanes, earthquakes
- `social`: Mental health, housing, immigration
- `market`: Crypto, stocks, startups

---

### 3. Review Integration Results

Check `narrative_optimization/domains/your_domain/`:

**`integration_results.json`**:
```json
{
  "universal_patterns": {
    "universal_underdog": {"frequency": 0.23, ...},
    "universal_comeback": {"frequency": 0.18, ...}
  },
  "similar_domains": [
    ["tennis", 0.75],
    ["golf", 0.68]
  ],
  "domain_patterns": {
    "your_domain_pattern_1": {...}
  },
  "frequency_analysis": {
    "predicted_frequency": 0.30,
    "observed_frequency": 0.28,
    "meets_expectations": true
  }
}
```

**`ANALYSIS_REPORT.md`**: Human-readable summary

---

### 4. Validate Results

**Check Story Frequency**:
- Does observed frequency match predictions?
- If not, why? (Observation bias? Domain constraints?)

**Identify Familiar Stories**:
- Which universal patterns appear?
- Are they similar to structurally similar domains?

**Look for Unique Patterns**:
- What's unique to this domain?
- Do these patterns make domain sense?

---

### 5. Refine (Optional but Recommended)

#### 5a. Add Domain-Specific Archetype Transformer

Create `src/transformers/archetypes/your_domain_archetype.py`:

```python
from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig

class YourDomainArchetypeTransformer(DomainArchetypeTransformer):
    def __init__(self):
        config = DomainConfig('your_domain')
        super().__init__(config)
        
        # Domain-specific context
        self.key_events = ['championship', 'playoff', 'final']
        self.key_entities = ['star_player', 'underdog_team']
        
    def _extract_archetype_features(self, X) -> np.ndarray:
        base_features = super()._extract_archetype_features(X)
        
        # Add domain-specific boosts
        enhanced_features = []
        for i, text in enumerate(X):
            boost = 1.0
            
            # Key event boost
            if any(event in text.lower() for event in self.key_events):
                boost *= 1.3
            
            enhanced = base_features[i] * boost
            enhanced_features.append(enhanced)
        
        return np.array(enhanced_features)
```

#### 5b. Add Domain Configuration

Add to `src/config/domain_archetypes.py`:

```python
'your_domain': {
    'archetype_patterns': {
        'key_pattern_1': ['keyword1', 'keyword2', 'keyword3'],
        'key_pattern_2': ['keyword4', 'keyword5'],
        # ... more patterns
    },
    'nominative_richness_requirement': 20,  # How many named entities needed
    'archetype_weights': {
        'key_pattern_1': 0.35,
        'key_pattern_2': 0.25,
        # ... weights sum to 1.0
    },
    'pi': 0.70,
    'theta_range': (0.40, 0.50),
    'lambda_range': (0.60, 0.70),
    'prestige_domain': False
}
```

#### 5c. Re-run with Custom Configuration

```bash
python MASTER_INTEGRATION.py your_domain data/domains/your_domain.json
```

---

### 6. Document Findings

Create `domains/your_domain/README.md`:

```markdown
# Your Domain Analysis

**π**: 0.70  
**Type**: Individual Expertise  
**Status**: ✅ Integrated

## Key Findings

### Universal Patterns
- Underdog story: 23% frequency
- Comeback narrative: 18% frequency

### Domain-Specific Patterns
- Pattern 1: Description
- Pattern 2: Description

### Story Frequency
- Predicted: 30%
- Observed: 28%
- Assessment: Meets expectations

### Structurally Similar To
1. Tennis (75% similar)
2. Golf (68% similar)

## Emerging Trends
- Trend 1
- Trend 2
```

---

## Checklist

Before marking domain as complete:

- [ ] Data has 50+ samples
- [ ] Narratives are rich (not just names)
- [ ] Integration ran successfully
- [ ] Universal patterns identified
- [ ] Similar domains found
- [ ] Story frequency analyzed
- [ ] Results make domain sense
- [ ] Documentation created
- [ ] (Optional) Custom archetype transformer added
- [ ] (Optional) Domain config added
- [ ] Results validated on held-out data

---

## Common Issues

### Issue: Low pattern discovery
**Solution**: Check if narratives are detailed enough. System needs descriptive text, not just entity names.

### Issue: No similar domains found
**Solution**: This is okay for truly unique domains. Focus on universal patterns.

### Issue: Story frequency mismatch
**Solution**: Consider observation bias. Are certain story types more likely to be reported/recorded?

### Issue: Negative Д (narrative agency)
**Solution**: Check domain type. High-skill domains may need prestige equation. Review θ and λ estimates.

---

## Example: Adding "Chess" Domain

```bash
# 1. Prepare data
# data/domains/chess_games.json with games, outcomes

# 2. Run integration
python MASTER_INTEGRATION.py chess data/domains/chess_games.json \
  --pi 0.78 \
  --type individual_expertise

# 3. Review results
cat narrative_optimization/domains/chess/ANALYSIS_REPORT.md

# 4. Found similar to: tennis (72%), golf (68%)
# Universal patterns: underdog (21%), pressure (25%)

# 5. Add custom transformer
# src/transformers/archetypes/chess_archetype.py

# 6. Add config
# src/config/domain_archetypes.py: 'chess' entry

# 7. Re-run
python MASTER_INTEGRATION.py chess data/domains/chess_games.json

# 8. Validate: R² improved from 65% to 84%
# ✓ Complete
```

---

## Next Steps After Integration

1. **Test predictions**: Validate on new data
2. **Refine patterns**: Use active learning to focus on uncertain patterns
3. **Cross-domain analysis**: Compare to similar domains
4. **Temporal analysis**: Track pattern evolution over time
5. **Causal analysis**: Identify causal vs correlational patterns

---

**Remember**: The goal is to discover what stories exist in this domain, how they relate to familiar stories, and whether they unfold as predicted given domain constraints.

