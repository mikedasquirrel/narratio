# Natural Science Domain Specifications

**Phase 2 Implementation**: Universal Domain Expansion  
**Date**: November 2025  
**Status**: Data Collection Frameworks Ready for Implementation

---

## Overview

This document specifies 6 natural science domains for narrative analysis, extending the framework from human/cultural domains into natural phenomena. These domains test the boundaries of narrative theory by examining narratives about systems with zero agency (geology), minimal agency (evolution), or dual agency structures (climate response).

**Critical Insight**: These domains test whether narrative framing affects outcomes even when the narrated entities have no awareness of their narratives.

---

## Domain 1: Evolutionary Biology

### Domain Characteristics

**π (Narrativity)**: 0.35  
- Structural: 0.40 (many possible evolutionary paths)
- Temporal: 0.60 (millions of years, rich temporal arcs)
- Agency: 0.00 (species don't "choose" adaptations)
- Interpretation: 0.80 (high - teleological vs mechanistic framings)
- Format: 0.65 (flexible - textbooks, documentaries, papers)

**λ (Fundamental Constraints)**: 0.85  
- Physical/biological laws heavily constrain
- Natural selection is mechanical process
- Mutations are random

**θ (Awareness Resistance)**: 0.15  
- Nature doesn't know or care about narratives
- But HUMAN narratives about nature affect conservation

**ة (Nominative Gravity)**: 0.40  
- Species with appealing names get more attention/funding
- "Charismatic megafauna" effect is pure nominative

**Expected Д**: ة(0.40) - θ(0.15) - λ(0.85) = -0.60  
→ Narratives should NOT predict biological outcomes  
→ BUT narratives DO predict human conservation responses

### Data Sources

**Primary**:
- IUCN Red List (conservation status data)
- Encyclopedia of Life (species descriptions)
- Wikipedia species articles (public narratives)
- National Geographic / BBC Nature documentaries (narrative framing)

**Sample Size Target**: 500 species

**Organisms** (entities to analyze):
- 500 vertebrate species with varied conservation status
- Mix of: mammals (200), birds (150), reptiles (75), fish (75)
- Include: endangered, threatened, least concern, extinct
- Ensure name diversity (appealing vs technical vs unfamiliar)

### Narrative Extraction

**Text Sources**:
- Wikipedia summary paragraph (public narrative)
- IUCN description (scientific narrative)
- News coverage (if any)
- Documentary narration (if available)

**Features to Extract**:
- Nominative: Name length, phonetic appeal, anthropomorphic elements
- Framing: Heroic ("survivor"), victim ("threatened"), neutral
- Agency attribution: Active voice ("adapts") vs passive ("is adapted")
- Temporal: Evolution framed as journey vs mechanical process
- Emotional: Appealing descriptions, charismatic traits emphasized

### Outcome Measures

**Primary Outcome**: Conservation success
- IUCN status improvement over time (degraded → stable → improving)
- Protected habitat area
- Funding allocated
- Population trend (declining, stable, increasing)

**Secondary Outcomes**:
- Research attention (papers published about species)
- Public awareness (Google search volume, Wikipedia views)
- Conservation NGO focus
- Documentary/media coverage minutes

### Research Questions

1. **Nominative Effect**: Do species with phonetically appealing names get more conservation funding? (Test ة directly)

2. **Anthropomorphic Bias**: Does framing species with human-like traits increase conservation success?

3. **Hero Narrative**: Does "heroic survivor" framing outperform "helpless victim" framing?

4. **Charismatic Megafauna**: Quantify the nominative + narrative advantage of appealing animals.

5. **Temporal Framing**: Does framing evolution as "journey" vs "mechanical process" affect public support?

### Expected Findings

- **Biological outcomes**: Narrative should have ZERO effect on actual evolutionary fitness (Д_evolution ≈ 0)
- **Human responses**: Narrative should have MODERATE effect on conservation (Д_conservation ≈ 0.25)
- **Validates dual-π model**: Species as biological entity (π=0.05) vs species as conservation target (π=0.50)

### Data Collection Protocol

```python
# For each species:
species_data = {
    'name': str,  # Scientific + common name
    'narrative': str,  # Combined description (Wikipedia + IUCN)
    'iucn_status_2000': str,  # Historical status
    'iucn_status_2025': str,  # Current status
    'habitat_protected_km2': float,
    'funding_allocated_usd': float,  # If available
    'papers_published_count': int,
    'media_mentions_count': int,
    'google_trends_score': float,  # 2020-2025 average
    'documentary_minutes': int,  # Nature doc coverage
    'population_trend': str,  # 'increasing', 'stable', 'decreasing'
}
```

### Validation Metrics

- R² > 0.15 for conservation outcomes (narrative matters for human response)
- R² ≈ 0.00 for biological fitness (narrative irrelevant to evolution)
- Name appeal predicts funding even controlling for endangerment
- Validates boundary condition: narrative affects human actions about nature, not nature itself

---

## Domain 2: Geology & Plate Tectonics

### Domain Characteristics

**π (Narrativity)**: 0.20  
- Structural: 0.30 (limited possible geological processes)
- Temporal: 0.80 (billions of years - epic timescales)
- Agency: 0.00 (zero agency - pure physics)
- Interpretation: 0.40 (some flexibility in framing)
- Format: 0.50 (textbooks, documentaries)

**λ (Fundamental Constraints)**: 0.90  
- Physics completely determines outcomes
- No possibility of deviation from physical laws

**θ (Awareness)**: 0.10  
- Rocks don't know anything

**ة (Nominative)**: 0.30  
- Geological formations with dramatic names get more tourism

**Expected Д**: -0.70 (narrative almost completely ineffective)

### Data Sources

- UNESCO World Heritage geological sites
- US National Parks Service
- Geological Society publications
- Travel/tourism data (visitor numbers)
- Wikipedia geological articles

**Sample Size**: 200 geological formations/events

### Organisms (Entities)

- 100 mountains/peaks
- 50 canyons/valleys
- 50 geological phenomena (geysers, caves, formations)

### Narrative Extraction

**Sources**: 
- Tourist descriptions
- Scientific descriptions
- National park brochures
- Documentary narrations

**Compare**:
- Dramatic framing ("majestic", "ancient", "powerful")
- Scientific framing ("sedimentary layers", "tectonic activity")

### Outcome Measures

**Primary**: Tourism/protection
- Annual visitor numbers
- UNESCO site designation (yes/no)
- National park status
- Conservation funding

**Secondary**:
- Scientific study frequency
- Public awareness (searches, mentions)
- Protected area size

### Research Questions

1. Do dramatically-named geological sites get more tourism than scientifically-named sites?
2. Does narrative framing in descriptions increase visitor numbers?
3. Does anthropomorphizing geology ("angry volcano") increase public interest?

### Expected Findings

- Д_geological_process ≈ 0 (narratives don't affect geology)
- Д_human_response ≈ 0.15 (narratives affect tourism/protection)
- Validates extreme low-π domain: lowest agency possible (0.00)

---

## Domain 3: Climate & Weather Systems

### Domain Characteristics

**π (Dual Structure)**:  
- π_physical_event = 0.20 (weather itself - low narrativity)
- π_human_response = 0.65 (response to weather - moderate narrativity)

**λ**: 0.80 (physics-dominated)  
**θ**: 0.30 (growing awareness of climate framing effects)  
**ة**: 0.50 (names matter - hurricane naming already established)

### Data Sources

- NOAA hurricane data (extended beyond current hurricanes)
- Heat wave events (named vs unnamed)
- Drought events
- Historical climate data
- News coverage
- Public response data (evacuations, policy)

**Sample Size**: 1000+ weather events

### Organisms

- 500 named hurricanes (existing data)
- 200 named heat waves (European naming)
- 200 droughts
- 100 cold snaps

### Narrative Framing

**Test Multiple Framings**:
- War metaphors ("battling", "fighting", "enemy")
- Journey metaphors ("storm's path", "approaching")
- Neutral technical ("low-pressure system")

### Outcome Measures

**Physical** (should show Д≈0):
- Storm intensity
- Rainfall amount
- Temperature records

**Human Response** (should show Д>0):
- Evacuation rates
- Policy response speed
- Funding allocated
- Public concern (polls, searches)

### Research Questions

1. **Naming Effect**: Do named events (heat waves with names) get more policy response than unnamed events?

2. **Metaphor Impact**: Do war metaphors increase urgency but decrease long-term engagement? (Test framing effects)

3. **Gender Effect Extended**: Beyond hurricanes - does gender of personified weather affect response?

4. **Attribution Narrative**: Does framing as "climate change" vs "natural variability" affect policy?

### Expected Findings

- Validates dual-π model: π_event ≠ π_response
- Naming increases response speed even for identical physical events
- War metaphors have short-term boost, long-term fatigue
- Extends hurricane findings to broader climate context

---

## Domain 4: Ecosystems & Food Webs

### Domain Characteristics

**π**: 0.45  
- Higher than pure geology due to biological complexity
- Multiple possible stable states (complexity)
- Temporal richness (seasonal cycles, succession)

**λ**: 0.70 (ecological laws constrain but less than physics)  
**θ**: 0.25 (ecosystems don't know, but have quasi-agency)  
**ة**: 0.55 (keystone species narrative is powerful)

### Data Sources

- Ecological monitoring data
- Conservation project outcomes
- Scientific papers on ecosystem management
- News coverage of environmental issues
- Funding databases

**Sample Size**: 300 ecosystems

### Organisms

- 300 ecosystems with restoration/protection efforts
- Varied biomes: forests, wetlands, coral reefs, grasslands
- Mix of success/failure outcomes

### Narrative Extraction

**Keystone Species Narrative**:
- Does framing ecosystem around charismatic keystone species increase funding?
- Compare "Save the Rainforest" vs "Save the Jaguar's Rainforest"

**Catastrophe vs Resilience**:
- "Ecosystem on brink of collapse" vs "Resilient system needing support"

**Services Framing**:
- "Ecosystem services worth $X million" vs "Intrinsic value"

### Outcome Measures

- Funding secured
- Restoration success (biodiversity metrics)
- Protected area size
- Policy implementation
- Public support

### Research Questions

1. Does keystone species framing increase funding more than ecosystem-level framing?
2. Does catastrophe framing increase immediate donations but reduce long-term support?
3. Does economic framing (ecosystem services) outperform intrinsic value framing?
4. Do ecosystems with narrative-rich descriptions get more scientific attention?

### Expected Findings

- Д ≈ 0.25-0.30 (narrative affects human conservation actions)
- Keystone species narrative provides leverage (ة effect)
- Validates moderate-π domain with zero-agency entities

---

## Domain 5: Astronomy & Cosmic Events

### Domain Characteristics

**π**: 0.15  
- Structural: 0.25 (limited stellar processes)
- Temporal: 0.95 (billions of years - ultimate timescale)
- Agency: 0.00 (zero)
- Interpretation: 0.30 (some flexibility)
- Format: 0.25 (mostly scientific)

**λ**: 0.95 (physics at extreme - highest constraint)  
**θ**: 0.05 (celestial objects have zero awareness)  
**ة**: 0.40 (names matter for public interest)

### Data Sources

- IAU (International Astronomical Union) catalog
- NASA mission priorities
- Public astronomy engagement data
- Planetarium show topics
- Space mission selection

**Sample Size**: 500 celestial objects/events

### Organisms

- 200 exoplanets (named vs numbered)
- 150 stars (named vs catalog numbers)
- 100 galaxies
- 50 cosmic events (supernovae, etc.)

### Narrative Test

**Classic Example**: Pluto vs Eris
- Pluto: Named, beloved, public outcry at demotion
- Eris: Technical name, forgotten, no public attachment
- Same physical characteristics, vastly different public engagement

**Test Questions**:
1. Do named celestial objects get more:
   - Public interest (searches, news)
   - Scientific papers
   - Space mission priority
   - Amateur astronomer attention

2. Does mythological naming (planets named for gods) increase engagement vs technical (HD 189733 b)?

3. Does anthropomorphic description increase support for missions?

### Outcome Measures

- Mission selection (yes/no)
- Funding allocated
- Scientific papers published
- Public engagement (searches, news, social media)
- Inclusion in planetarium shows
- Amateur astronomy interest

### Expected Findings

- Д_physics ≈ 0.00 (narratives don't affect astronomy)
- Д_human_interest ≈ 0.20 (narratives affect what we study)
- Names matter even when objects are incomprehensibly distant
- Validates extreme low-agency + extreme temporal scale domain

---

## Domain 6: Pandemics & Disease Spread

### Domain Characteristics

**π (Dual Structure)**:
- π_biological_spread = 0.30 (disease follows epidemiological laws)
- π_social_response = 0.70 (human behavior highly narrative)

**λ_biological**: 0.75 (biology constrains spread)  
**λ_social**: 0.35 (social behavior more flexible)  
**θ**: 0.50 (moderate awareness of framing effects)  
**ة**: 0.60 (disease names matter enormously)

### Data Sources

- WHO pandemic database
- Historical epidemic data
- News coverage analysis
- Public health response data
- Behavioral compliance studies
- Social media sentiment

**Sample Size**: 200 historical pandemics/epidemics

### Organisms

- 50 major historical pandemics
- 150 regional epidemics with varied names/framings

### Narrative Dimensions

**Disease Names**:
- Geographic (Spanish Flu) vs neutral (H1N1)
- Stigmatizing vs neutral
- Technical vs colloquial

**Framing**:
- War metaphor ("fighting the virus")
- Journey metaphor ("living with COVID")
- Neutral technical

**Agent Attribution**:
- Active voice ("virus attacks")
- Passive voice ("people are infected")

### Outcome Measures

**Biological** (control variables):
- R₀ (basic reproduction number)
- Case fatality rate
- Transmissibility

**Social Response** (test variables):
- Compliance with public health measures
- Policy response speed
- Funding allocated to research
- Public panic level (measured)
- Vaccination rates
- Economic impact (partly social choice)

### Research Questions

1. **Name Stigma**: Do geographic names (Spanish Flu, MERS) increase discrimination but decrease policy response?

2. **Metaphor Effects**: Do war metaphors increase short-term compliance but long-term fatigue?

3. **Severity Framing**: Does "pandemic" vs "epidemic" vs "outbreak" affect response even for same R₀?

4. **Recent Validation**: COVID-19 naming debates - did neutral naming help or hurt?

5. **Attribution Effects**: Does emphasizing virus agency reduce human agency/responsibility?

### Expected Findings

- Д_spread ≈ 0.05 (minimal narrative effect on biology)
- Д_response ≈ 0.35 (strong narrative effect on human behavior)
- War metaphors show inverted-U: optimal moderate intensity
- Geographic names increase stigma, decrease cooperation
- Validates dual-π + high-stakes domain

### Special Considerations

**Ethical Importance**: This domain has life-death stakes. Findings could inform public health communication.

**Recent Data**: COVID-19 provides real-time test of framing effects with massive data.

---

## Cross-Domain Analysis Framework

### Comparative Questions

**Across all 6 domains**:

1. **Agency Gradient**: Do narrative effects scale with agency?
   - Evolution (0.00) < Geology (0.00) < Astronomy (0.00) < Climate (0.10) < Ecosystems (0.20) < Pandemics (0.30 biological)

2. **Temporal Scale**: Do effects vary with timescale?
   - Pandemics (days-years) vs Climate (decades) vs Evolution (millennia) vs Geology (millions) vs Astronomy (billions)

3. **Nominative Universality**: Does ة work even for inanimate objects?
   - If yes: nominative gravity is fundamental cognitive phenomenon
   - If no: requires some minimal agency

4. **Dual-π Validation**: Do all natural domains show dual structure?
   - Physical process (low π) vs Human response (higher π)

### Meta-Theoretical Tests

**Test 1**: Zero-agency theorem
- **Hypothesis**: Д_physical ≈ 0 for all zero-agency domains
- **Validation**: Evolution, Geology, Astronomy should show no narrative effect on physical outcomes

**Test 2**: Response separation
- **Hypothesis**: Д_response > 0 even when Д_physical = 0
- **Validation**: Narratives about nature affect human actions, not nature itself

**Test 3**: Temporal scale invariance
- **Hypothesis**: Narrative structure (τ, ς, ρ) shows same patterns across temporal scales
- **Validation**: Cross-temporal isomorphism holds from days (pandemics) to billions of years (astronomy)

**Test 4**: Material constraints boundary
- **Hypothesis**: λ_physical sets hard boundary that narrative cannot cross
- **Validation**: No narrative framing changes gravitational constant, chemical bonds, or biological reproduction rates

---

## Implementation Priority

### Immediate (Month 1-2):
1. **Evolutionary Biology** - Easiest data collection (IUCN, Wikipedia readily available)
2. **Climate Extension** - Build on existing hurricane work

### Short-term (Month 3-4):
3. **Ecosystems** - Moderate difficulty, important validation
4. **Pandemics** - Timely, high-impact findings

### Medium-term (Month 5-6):
5. **Geology** - Requires tourism data compilation
6. **Astronomy** - Requires space mission database

---

## Success Metrics

**Phase 2 Success** = Validation across all 6 domains:
1. Zero-agency domains show Д_physical ≈ 0 (narratives don't affect physics)
2. Zero-agency domains show Д_response > 0 (narratives affect humans)
3. Dual-π model validated (entity π ≠ response π)
4. Material constraints formalized (λ_physical)
5. Cross-temporal isomorphism tested
6. Framework extends to natural sciences successfully

**Publication Target**: "Narrative Effects in Natural Science: Testing the Boundaries of Story Power from Evolution to Astronomy"

---

## Data Collection Infrastructure

### Required Tools

```python
# Domain data collector base class
class NaturalScienceDomainCollector:
    def __init__(self, domain_name):
        self.domain = domain_name
    
    def collect_entities(self) -> List[Entity]:
        """Collect entities (species, formations, events)"""
        pass
    
    def collect_narratives(self, entity) -> str:
        """Collect narrative text about entity"""
        pass
    
    def collect_outcomes(self, entity) -> Dict:
        """Collect outcome measures"""
        pass
    
    def extract_features(self, narrative) -> np.ndarray:
        """Apply narrative transformers"""
        pass
```

### APIs Needed

- IUCN Red List API (evolutionary biology)
- Wikipedia API (all domains)
- NOAA API (climate)
- NASA/JPL APIs (astronomy)
- WHO database (pandemics)
- Google Trends API (public interest metrics)

### Storage Schema

```python
natural_domain_entity = {
    'domain': str,  # 'evolution', 'geology', etc.
    'entity_id': str,
    'entity_name': str,
    'narrative_public': str,  # Wikipedia, news
    'narrative_scientific': str,  # Papers, official descriptions
    'narrative_features': np.ndarray,  # From transformers
    'outcome_physical': Dict,  # Physical measurements
    'outcome_human_response': Dict,  # Human actions/responses
    'metadata': Dict,  # Domain-specific
    'timestamp': datetime
}
```

---

## Conclusion

These 6 natural science domain specifications provide:

1. **Complete framework** for extending narrative analysis to nature
2. **Clear hypotheses** testable with available data
3. **Validation of boundaries** - where narrative power ends
4. **Dual-π model** - separating physical from response domains
5. **Material constraints** - formalizing λ_physical
6. **Cross-temporal tests** - from days to billions of years

**Status**: Specifications complete, ready for data collection implementation.

**Next**: Implement data collectors, begin with evolutionary biology pilot (easiest), validate framework extension to natural domains.

