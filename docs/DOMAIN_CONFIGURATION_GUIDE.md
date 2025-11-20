# Domain Configuration Guide

Complete guide to configuring story domains with custom transformers and settings.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Domain Registry](#domain-registry)
3. [Configuration Levels](#configuration-levels)
4. [Custom Transformers](#custom-transformers)
5. [Examples](#examples)
6. [Reference](#reference)

---

## Quick Start

### Add a New Domain (2 Minutes)

**Step 1**: Add entry to `narrative_optimization/domain_registry.py`:

```python
'chess': DomainConfig(
    name='chess',
    data_path='data/domains/chess_games.json',
    narrative_field='game_narrative',
    outcome_field='won',
    estimated_pi=0.40,
    description='Chess game narratives',
    outcome_type='binary'
)
```

**Step 2**: Process it:

```bash
# Via web interface
http://localhost:5000/process/domain-processor

# Or via Python
from narrative_optimization.universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor()
processor.process_domain('chess', sample_size=1000)
```

**Done!** The system automatically:
- Selects appropriate transformers based on π
- Configures timeout based on sample size
- Handles all feature extraction
- Saves results

---

## Domain Registry

### Registry Location

```
narrative_optimization/domain_registry.py
```

Central registry for ALL domains. One file to see everything.

### Domain Config Structure

```python
DomainConfig(
    name='domain_name',           # Unique identifier
    data_path='path/to/data.json', # Data file location
    narrative_field='text_field',  # Field with narrative text
    outcome_field='outcome_field', # Field with outcomes
    estimated_pi=0.5,              # Narrativity (0-1)
    description='Description',     # Human-readable
    outcome_type='continuous',     # continuous|binary
    min_narrative_length=50        # Minimum text length
)
```

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | str | Domain identifier | `'tennis'` |
| `data_path` | str | Path to JSON data | `'data/domains/tennis.json'` |
| `narrative_field` | str or list | Text field(s) | `'narrative'` or `['text', 'description']` |
| `outcome_field` | str | Outcome field | `'won'` or `'success_score'` |
| `estimated_pi` | float | Narrativity | `0.75` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | str | `''` | Human-readable description |
| `outcome_type` | str | `'continuous'` | `'binary'` or `'continuous'` |
| `min_narrative_length` | int | `50` | Minimum characters |
| `custom_extractor` | callable | `None` | Custom data loader |

---

## Configuration Levels

### Level 1: Basic Registration (Simplest)

Just register domain - system handles everything automatically.

```python
'poker': DomainConfig(
    name='poker',
    data_path='data/domains/poker_hands.json',
    narrative_field='hand_description',
    outcome_field='won',
    estimated_pi=0.83
)
```

**System automatically**:
- Selects 40-50 transformers based on π=0.83
- Includes sports transformers (if applicable)
- Configures timeout (~20min for 1000 samples)
- Handles all errors gracefully

### Level 2: Domain Config YAML (Recommended)

Create detailed configuration file for domain-specific settings.

**File**: `narrative_optimization/domains/{domain}/config.yaml`

```yaml
domain: poker
type: sports
narrativity:
  structural: 0.85    # High variability
  temporal: 0.70      # Game unfolds over time
  agency: 0.90        # Player skill matters
  interpretive: 0.80  # Reading opponents
  format: 0.60        # Structured but flexible

data:
  text_fields:
    - hand_description
    - player_notes
  outcome_field: won
  context_features:
    - pot_size
    - position
    - stack_size

outcome_type: binary

# Custom transformer augmentation
transformer_augmentation:
  - StrategicThinkingTransformer
  - RiskAssessmentTransformer
  - BluffDetectionTransformer

# Timeout configuration
timeout:
  base_minutes: 20
  per_1000_samples: 15
```

**Load Config**:

```python
from src.pipelines.domain_config import DomainConfig

config = DomainConfig.from_yaml('domains/poker/config.yaml')
```

### Level 3: Custom Transformers (Advanced)

Build domain-specific transformers for specialized features.

**Create Transformer**:

**File**: `narrative_optimization/src/transformers/poker/strategic_thinking.py`

```python
from transformers.base_transformer import BaseTransformer
import numpy as np

class StrategicThinkingTransformer(BaseTransformer):
    """
    Extracts strategic thinking patterns from poker narratives.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "StrategicThinkingTransformer"
    
    def fit(self, texts, y=None):
        """Fit transformer to training data."""
        # Learn patterns from training data
        return self
    
    def transform(self, texts):
        """Extract strategic thinking features."""
        features = []
        
        for text in texts:
            # Extract domain-specific features
            feature_vector = {
                'planning_depth': self._analyze_planning(text),
                'risk_awareness': self._analyze_risk(text),
                'opponent_modeling': self._analyze_opponents(text),
                'pot_odds_consideration': self._analyze_math(text),
                'position_awareness': self._analyze_position(text)
            }
            features.append(feature_vector)
        
        return np.array(features)
    
    def _analyze_planning(self, text):
        """Analyze planning depth in narrative."""
        # Implementation here
        pass
```

**Register Transformer**:

Add to `TRANSFORMER_CATALOG.json` or dynamically import.

---

## Custom Transformers

### When to Create Custom Transformers

✅ **Good Reasons**:
- Domain has unique features not captured by universal transformers
- Specialized domain knowledge can improve predictions
- Existing transformers miss important patterns
- You have domain-specific hypotheses to test

❌ **Don't Create If**:
- Universal transformers already capture the pattern
- Feature would apply to many domains (make it universal instead)
- You don't have domain expertise to validate features

### Custom Transformer Template

```python
from transformers.base_transformer import BaseTransformer
import numpy as np
from typing import List

class CustomDomainTransformer(BaseTransformer):
    """
    {Description of what this transformer extracts}
    
    Domain: {domain_name}
    Author: {your_name}
    Date: {date}
    """
    
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.name = "CustomDomainTransformer"
        self.param1 = param1
        self.param2 = param2
        
        # Features this transformer produces
        self.feature_names_ = [
            'feature_1',
            'feature_2',
            'feature_3'
        ]
    
    def fit(self, texts: List[str], y=None):
        """
        Learn patterns from training data.
        
        Parameters
        ----------
        texts : list of str
            Training narratives
        y : array-like, optional
            Training outcomes
        """
        # Fit logic here
        # Extract vocabularies, thresholds, etc.
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Extract features from narratives.
        
        Parameters
        ----------
        texts : list of str
            Narratives to transform
        
        Returns
        -------
        features : ndarray, shape (n_samples, n_features)
            Extracted features
        """
        features = []
        
        for text in texts:
            # Extract features
            feature_vector = [
                self._extract_feature1(text),
                self._extract_feature2(text),
                self._extract_feature3(text)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_feature1(self, text: str) -> float:
        """Extract specific feature."""
        # Implementation
        pass
```

### Registering Custom Transformers

**Option 1: Add to Domain Config**

```yaml
# domains/poker/config.yaml
transformer_augmentation:
  - StrategicThinkingTransformer
  - RiskAssessmentTransformer
```

**Option 2: Add to Transformer Selector**

```python
# src/transformers/transformer_selector.py

DOMAIN_SPECIFIC_TRANSFORMERS = {
    'poker': [
        'StrategicThinkingTransformer',
        'RiskAssessmentTransformer'
    ]
}
```

**Option 3: Dynamic Registration**

```python
from transformer_factory import register_transformer

register_transformer(
    name='StrategicThinkingTransformer',
    transformer_class=StrategicThinkingTransformer,
    domains=['poker'],
    pi_range=(0.7, 1.0)
)
```

---

## Examples

### Example 1: Sports Domain (NBA)

```python
# domain_registry.py
'nba': DomainConfig(
    name='nba',
    data_path='data/domains/nba_games.json',
    narrative_field=['pregame_narrative', 'description'],
    outcome_field='won',
    estimated_pi=0.49,
    description='NBA game narratives - team sport',
    outcome_type='binary'
)
```

**Automatic Configuration**:
- π=0.49 → Medium narrativity
- Type detected: Sports → Team sport
- Transformers selected: 45 total
  - Core: 7 universal
  - π-based: 5 medium-pi
  - Sports-specific: 5 (CompetitiveContext, TemporalMomentum, etc.)
  - Multi-scale: 3 (hierarchical game structure)
  - Renovation: 15 (temporal, cognitive, linguistic)

### Example 2: Entertainment Domain (Movies)

```python
# domain_registry.py
'movies': DomainConfig(
    name='movies',
    data_path='data/domains/imdb_movies.json',
    narrative_field='plot_summary',
    outcome_field='success_score',
    estimated_pi=0.65,
    description='Movie plot summaries',
    outcome_type='continuous',
    min_narrative_length=200
)
```

**Automatic Configuration**:
- π=0.65 → Medium-high narrativity
- Type: Entertainment
- Transformers: 48 total
  - Entertainment-specific: 6 (EmotionalResonance, VisualMultimodal, etc.)
  - High-pi features: Phonetic, Social Status, Cultural Context
  - Archetypal patterns: Hero journey, plot structures

### Example 3: Custom Business Domain (Startups)

```yaml
# domains/startups/config.yaml
domain: startups
type: business
narrativity:
  structural: 0.80
  temporal: 0.85
  agency: 0.90
  interpretive: 0.75
  format: 0.60

data:
  text_fields:
    - pitch_description
    - product_story
    - founder_narrative
  outcome_field: funded
  context_features:
    - team_size
    - industry
    - funding_stage

outcome_type: binary

transformer_augmentation:
  - OriginStoryTransformer
  - VisionClarityTransformer
  - MarketTimingTransformer
  - FounderCredibilityTransformer

timeout:
  base_minutes: 25
```

---

## Reference

### Narrativity Components (π)

| Component | Range | Description | Example (Low) | Example (High) |
|-----------|-------|-------------|---------------|----------------|
| `structural` | 0-1 | Branching paths possible | Coin flip (0.0) | Novel plot (0.9) |
| `temporal` | 0-1 | Unfolds over time | Instant outcome (0.1) | Career arc (0.95) |
| `agency` | 0-1 | Actor choice matters | Lottery (0.0) | Startup (0.95) |
| `interpretive` | 0-1 | Subjective judgment | Math problem (0.1) | Art critique (0.95) |
| `format` | 0-1 | Medium flexibility | Fixed format (0.2) | Open narrative (0.9) |

**π Formula**:
```
π = 0.30×structural + 0.20×temporal + 0.25×agency + 
    0.15×interpretive + 0.10×format
```

### Transformer Selection Rules

**π < 0.3** (Low Narrativity):
- Quantitative features emphasized
- Fundamental constraints
- Information theory measures

**π 0.3-0.7** (Medium Narrativity):
- Balanced approach
- Framing and optics
- Emotional resonance

**π > 0.7** (High Narrativity):
- Full nominative suite
- Cultural context
- Social status patterns

**Domain Type Additions**:
- **Sports**: Competitive context, temporal momentum, matchup advantage
- **Entertainment**: Emotional resonance, character complexity, visual multimodal
- **Business**: Namespace ecology, anticipatory communication, discoverability

### Data Format Requirements

**JSON Structure**:

```json
{
  "items": [
    {
      "narrative": "The narrative text goes here...",
      "outcome": 1,
      "context_var1": 123,
      "context_var2": "value"
    }
  ]
}
```

Or flat list:

```json
[
  {
    "text": "Narrative here...",
    "success": 0.75
  }
]
```

**Field Types**:
- `narrative_field`: String (any text)
- `outcome_field`: 
  - Binary: 0/1 or true/false
  - Continuous: Any number
- `context_fields`: Any type (optional)

---

## Best Practices

### Configuration
1. Start with basic registration
2. Add YAML config only if needed
3. Create custom transformers last

### Narrativity Estimation
1. Use existing domains as reference
2. Consider all 5 components
3. Round to 0.05 precision
4. Validate with small sample

### Custom Transformers
1. Validate features make domain sense
2. Test on held-out data
3. Compare to universal transformers
4. Document feature interpretation

### Data Quality
1. Minimum 500 samples for testing
2. Minimum 2000 for production
3. Clean narrative text (remove noise)
4. Validate outcomes are correct

---

## Next Steps

- **Process domains**: See `DOMAIN_PROCESSING_GUIDE.md`
- **Use API**: See `API_REFERENCE.md`
- **Quick commands**: See `QUICK_START.md`

---

**Last Updated**: November 2025  
**Version**: 2.0

