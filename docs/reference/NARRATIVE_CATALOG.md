# Narrative Catalog
## Pure Narrative Elements: Universal Patterns, Characters, Structures & Tropes

**Purpose**: Document the fundamental narrative building blocks that exist across all domains  
**Status**: Living document - add new patterns as discovered  
**Last Updated**: November 14, 2025

---

## Fundamental Principle: Hierarchical Structural Constraints

**CRITICAL INSIGHT**: Narrative patterns exist in nested hierarchies

### Layer 1: Universal Patterns (Exist Everywhere)
- Underdog, Hero, Rivalry, Momentum, Championship
- These are abstract concepts

### Layer 2: Domain Structure (Constrains Possibilities)
- **3-minute song** CANNOT have novel-level character depth (time doesn't allow it)
- **17-game NFL season** CANNOT have 82-game arc development (structure prevents it)
- **5-player NBA team** CANNOT have 22-player ensemble narratives (too few people)
- **300-page novel** CAN have depth impossible in 120-minute film (space allows it)

### Layer 3: Sub-Domain Types (Within-Domain Variation)

**Not all films are the same** - they follow genre/type patterns:
- **Thriller films**: Tension buildup, twist at 75%, revelation at 90%
- **Romance films**: Meet-cute at 10%, conflict at 50%, reunion at 85%
- **Action films**: Set pieces every 20 minutes, climax vehicle chase
- **All films**: 120 minutes, but internal patterns differ

**Not all NFL games are the same** - they follow context patterns:
- **Division games**: Rivalry intensity, extra physicality, coaching chess match
- **Primetime games**: Amplified narratives, national stakes, performance pressure
- **Playoff games**: Single elimination, legacy stakes, maximum pressure
- **All NFL**: 4 quarters, but narrative patterns differ

**Not all NBA games are the same** - they follow type patterns:
- **Rivalry games** (Lakers-Celtics): Historic weight, intensity, star matchups
- **Playoff games**: Series narrative, adjustment patterns, desperation
- **Regular season**: Arc building, rest management, experimental lineups
- **All NBA**: 48 minutes, but patterns rhyme by type

**Not all novels are the same** - they follow genre patterns:
- **Mystery novels**: Clue placement, red herrings, reveal timing
- **Fantasy novels**: World-building front-loaded, quest structure
- **Literary fiction**: Character depth over plot, internal resolution
- **All novels**: Pages, but narrative structure varies by genre

### Layer 4: Individual Instances
- Specific game, specific film, specific novel
- Unique but constrained by layers 1-3

### The Rhyming Principle

**"All [type X] rhyme with each other"**:
- All thriller films rhyme (similar beat sheets)
- All division games rhyme (similar intensity patterns)
- All playoff elimination games rhyme (similar pressure)
- All revenge plots rhyme (similar arc)

**Structural isomorphism at sub-domain level**:
- NFL Division Game ≈ NBA Rivalry Game (both: historic conflict, extra intensity)
- NFL Playoff Game ≈ Tennis Grand Slam (both: high stakes, elimination)
- Thriller Film at 75% ≈ NFL Week 13 (both: late-stage intensity peak)
- Romance Film climax ≈ Championship Game (both: culmination, resolution)

---

## How to Use This Catalog

This catalog documents **universal narrative elements** - the archetypal patterns that exist BEFORE they manifest in specific domains.

But remember: **Domain structure determines which patterns can actually appear and how strongly**

**Example**:
- **Universal Pattern**: "Underdog" (exists as concept)
- **NBA Manifestation**: 8-seed vs 1-seed, record gap > 20 wins (82 games allow ranking clarity)
- **NFL Manifestation**: +10 point underdog (17 games = bigger spreads)
- **Song Manifestation**: Unknown artist vs established star (3 minutes = no arc, just contrast)
- **Novel Manifestation**: Protagonist's journey from weak to strong (300 pages allow full development)

**Structure determines DEPTH and TYPE of underdog narrative possible**

---

## I. Character Archetypes

### The Hero
**Definition**: Central protagonist driving the narrative  
**Characteristics**:
- Agency (makes decisions)
- Goal-oriented
- Faces challenges
- Undergoes transformation

**Variations**:
- **Reluctant Hero**: Forced into action (origin story)
- **Chosen One**: Destined for greatness
- **Everyman Hero**: Relatable, ordinary becoming extraordinary
- **Anti-Hero**: Flawed protagonist, morally ambiguous

**Transformer Coverage**: `character_complexity.py`, `self_perception.py`

---

### The Underdog
**Definition**: Competitor facing overwhelming odds  
**Characteristics**:
- Lower resources/status/skill (perception)
- Fighting against favorite
- Narrative sympathy
- Potential for upset

**Markers**:
- Explicit handicap (seeding, ranking, size)
- Historical disadvantage
- Resource disparity
- David vs Goliath framing

**Transformer Coverage**: `competitive_context.py`, `reputation_prestige.py`

**Domain Examples**:
- Sports: 8-seed vs 1-seed, +7 spread underdog
- Startups: Bootstrapped vs VC-backed
- Movies: Indie vs blockbuster
- Wine: New region vs established terroir

---

### The Mentor
**Definition**: Wise guide who enables hero's journey  
**Characteristics**:
- Experience and wisdom
- Teaching/guidance role
- Often sacrificial
- Enables transformation

**Variations**:
- **Master Teacher**: Formal training (coach, professor)
- **Sage Advisor**: Wisdom without training
- **Fallen Mentor**: Cautionary tale
- **Peer Mentor**: Equal who guides

**Transformer Coverage**: `origin_story.py`, `community_network.py`

---

### The Rival
**Definition**: Matched opponent who elevates the hero  
**Characteristics**:
- Comparable skill/status
- Historical conflict
- Mutual drive for excellence
- Narrative tension

**Rivalry Types**:
- **Historic Rivalry**: Long-standing conflict (Lakers-Celtics)
- **Emerging Rivalry**: Recent competition building
- **Friendly Rivalry**: Respectful competition
- **Bitter Rivalry**: Personal animosity

**Transformer Coverage**: `competitive_context.py`, `conflict_tension.py`

---

### The Trickster
**Definition**: Disruptive force using unconventional methods  
**Characteristics**:
- Rule-bending
- Unpredictable
- Strategic misdirection
- Challenges status quo

**Transformer Coverage**: `narrative_potential.py`, `strategic_positioning.py`

---

### The Collective/Ensemble
**Definition**: Group as unified character  
**Characteristics**:
- Chemistry/cohesion
- Complementary roles
- Shared identity
- Greater than sum of parts

**Ensemble Types**:
- **Team**: Sports team, company, band
- **Family**: Related unit
- **Alliance**: Temporary cooperation
- **Movement**: Ideological collective

**Transformer Coverage**: `community_network.py`, `ensemble_meta.py`

---

## II. Narrative Structures

### The Hero's Journey (Monomyth)
**Definition**: Universal narrative pattern of transformation  

**Stages**:
1. **Ordinary World**: Starting state
2. **Call to Adventure**: Challenge appears
3. **Refusal**: Initial resistance
4. **Meeting Mentor**: Guidance acquired
5. **Crossing Threshold**: Point of no return
6. **Tests/Allies/Enemies**: Challenges faced
7. **Approach**: Preparation for ordeal
8. **Ordeal**: Greatest challenge
9. **Reward**: Victory/transformation
10. **Return**: Back to ordinary world transformed
11. **Resurrection**: Final test
12. **Return with Elixir**: Wisdom shared

**Transformer Coverage**: `origin_story.py`, `temporal_momentum.py`, `character_complexity.py`

---

### The Redemption Arc
**Definition**: Fall from grace followed by recovery  

**Pattern**:
- Past success/high status
- Fall (failure, scandal, injury)
- Struggle in darkness
- Recognition/acceptance
- Climb back
- Redemption moment
- New wisdom/humility

**Markers**:
- "Comeback" language
- Past glory references
- Adversity overcome
- Second chance framing

**Transformer Coverage**: `temporal_momentum.py`, `character_complexity.py`, `reputation_prestige.py`

---

### The Rise (Ascension)
**Definition**: Steady climb from obscurity to prominence  

**Pattern**:
- Humble beginnings
- Early struggles
- Breakthrough moment
- Accelerating success
- Peak achievement

**Variations**:
- **Overnight Success**: Sudden rise (often with hidden preparation)
- **Steady Climb**: Gradual progression
- **Dark Horse**: Unexpected contender
- **Prodigy**: Early exceptional talent

**Transformer Coverage**: `temporal_momentum.py`, `narrative_potential.py`, `origin_story.py`

---

### The Tragedy
**Definition**: Fall from high position through flaw or fate  

**Pattern**:
- High status/success
- Fatal flaw or external force
- Poor decision/hubris
- Inevitable decline
- Catastrophic fall
- Recognition too late

**Transformer Coverage**: `conflict_tension.py`, `character_complexity.py`

---

### The Quest
**Definition**: Journey toward specific goal/object  

**Characteristics**:
- Clear objective
- Obstacles to overcome
- Companions/allies
- Tests of worthiness
- Achievement or noble failure

**Quest Types**:
- **Championship Quest**: Win the title
- **Discovery Quest**: Find knowledge/truth
- **Rescue Quest**: Save someone/something
- **Transformation Quest**: Change self/world

**Transformer Coverage**: `origin_story.py`, `conflict_tension.py`, `narrative_potential.py`

---

## III. Narrative Tropes

### Momentum
**Definition**: Forward-moving narrative energy creating expectation  

**Characteristics**:
- Win streaks (performance)
- Hot hand (perception)
- "On a roll" language
- Temporal acceleration
- Expectation of continuation

**Types**:
- **Performance Momentum**: Actual results improving
- **Narrative Momentum**: Story gaining energy
- **Temporal Momentum**: Time-based progression

**Mathematical Model**: Exponential decay (γ coefficient)
- NBA: γ = 0.948 (half-life ≈ 13 games)
- Different decay rates by domain

**Transformer Coverage**: `temporal_momentum_enhanced.py`

---

### The Upset
**Definition**: Unexpected victory defying predictions  

**Characteristics**:
- Underdog wins
- Disrupts expectations
- Creates new narrative
- Memorable/shocking

**Conditions**:
- Clear favorite established
- Prediction failure
- Significant gap (skill/resources)
- Narrative reversal

**Transformer Coverage**: `competitive_context.py`, `anticipatory_commitment.py`

---

### The Rivalry
**Definition**: Sustained competitive relationship  

**Characteristics**:
- Multiple encounters
- Historical context
- Mutual recognition
- Elevated stakes when matched
- Fan/cultural investment

**Rivalry Levels**:
- **Personal**: Individual animosity
- **Institutional**: Organizations/teams
- **Regional**: Geographic competition
- **Ideological**: Competing philosophies

**Transformer Coverage**: `competitive_context.py`, `reputation_prestige.py`

---

### The Comeback
**Definition**: Recovery from deficit or disadvantage  

**Types**:
- **Within-Event**: Overcome deficit in single contest
- **Career Comeback**: Return from injury/scandal/retirement
- **Seasonal Comeback**: Recover from poor start
- **Legacy Comeback**: Reputation rehabilitation

**Markers**:
- Deficit specified
- Turning point identified
- Mounting tension
- Dramatic resolution

**Transformer Coverage**: `temporal_momentum.py`, `conflict_tension.py`

---

### The Choke
**Definition**: Failure under high-pressure situation  

**Characteristics**:
- High expectations
- Pressure moment
- Unexpected poor performance
- Often from favorite
- Psychological narrative

**Transformer Coverage**: `conflict_tension.py`, `self_perception.py`

---

### The Cinderella Story
**Definition**: Unexpected underdog success  

**Characteristics**:
- No/low expectations
- Sustained unexpected success
- Captures imagination
- Eventually ends (usually)

**Markers**:
- "Cinderella" explicit reference
- Fairy tale language
- Magical/dreamlike framing
- Public emotional investment

**Transformer Coverage**: `competitive_context.py`, `temporal_momentum.py`, `narrative_potential.py`

---

### The Dynasty
**Definition**: Sustained dominance over extended period  

**Characteristics**:
- Multiple championships/victories
- Institutional excellence
- Generational success
- Becomes target/standard

**Dynasty Phases**:
1. **Building**: Assembling pieces
2. **Breakthrough**: First championship
3. **Reign**: Multiple victories
4. **Twilight**: Aging/decline
5. **Fall**: Dynasty ends

**Transformer Coverage**: `reputation_prestige.py`, `temporal_momentum.py`, `community_network.py`

---

### The Curse/Jinx
**Definition**: Perceived pattern of failure/bad luck  

**Characteristics**:
- Historical pattern
- Superstitious explanation
- Self-fulfilling prophecy
- Breaking curse = major narrative

**Types**:
- **Historical Curse**: Long drought (Curse of the Bambino)
- **Venue Curse**: Specific location
- **Matchup Curse**: Can't beat specific opponent
- **Personal Curse**: Individual's repeated failure

**Transformer Coverage**: `temporal_momentum.py`, `nominative.py` (if name-based)

---

## IV. Cast Structures

### Solo Protagonist
**Definition**: Single central character driving narrative  

**Characteristics**:
- Individual agency
- Personal journey
- Internal conflict prominent
- All pressure on one person

**Sports Examples**: Tennis, golf, boxing, individual track/field  
**Business Examples**: Solo founder, freelancer, author  

**Transformer Coverage**: `character_complexity.py`, `self_perception.py`

---

### Duo/Partnership
**Definition**: Two central characters in relationship  

**Types**:
- **Equals**: Balanced partnership
- **Leader-Follower**: Hierarchical
- **Rivals-Turned-Allies**: Former competitors
- **Mentor-Protégé**: Teaching relationship

**Dynamics**:
- Complementary skills
- Trust/chemistry
- Potential for conflict
- Synergy effects

**Transformer Coverage**: `community_network.py`, `ensemble_meta.py`

---

### Team/Ensemble
**Definition**: Multiple characters with defined roles  

**Characteristics**:
- Role differentiation
- Chemistry crucial
- Leadership structure
- Collective identity

**Ensemble Types**:
- **Star + Supporting**: One lead, others support
- **Balanced Ensemble**: Equal importance
- **Rotating Stars**: Different leaders by context
- **Collective**: No individual standout

**Transformer Coverage**: `community_network.py`, `ensemble_meta.py`, `reputation_prestige.py`

---

### Organization/Institution
**Definition**: Entity as character (company, team, movement)  

**Characteristics**:
- Institutional identity
- Legacy/history
- Culture/values
- Transcends individuals

**Transformer Coverage**: `reputation_prestige.py`, `origin_story.py`, `community_network.py`

---

## V. Setting/Context Archetypes

### The Championship
**Definition**: Highest-stakes competitive event  

**Characteristics**:
- Winner-take-all
- Culmination of season/journey
- Legacy-defining
- Maximum pressure
- Historical significance

**Variations**:
- **Finals**: Best-of-series championship
- **Title Game**: Single championship contest
- **Tournament Final**: Bracket culmination
- **Season Finale**: End of season stakes

**Context Weight**: Typically 2.5x - 3.0x baseline  
**Transformer Coverage**: `conflict_tension.py`, `anticipatory_commitment.py`

---

### The Origin Story
**Definition**: Beginning/emergence narrative  

**Characteristics**:
- Humble beginnings
- Foundational moment
- Character formation
- Sets trajectory

**Origin Types**:
- **Birth/Creation**: How it began
- **Discovery**: Talent recognized
- **Transformation**: Became who they are
- **Calling**: Found purpose

**Transformer Coverage**: `origin_story.py`

---

### The Crossroads
**Definition**: Decision point determining future path  

**Characteristics**:
- Multiple options
- Irreversible choice
- High stakes
- Narrative branching

**Transformer Coverage**: `anticipatory_commitment.py`, `conflict_tension.py`

---

### The Proving Ground
**Definition**: Test demonstrating worthiness  

**Characteristics**:
- Skepticism exists
- Must prove doubters wrong
- High-pressure situation
- Validation moment

**Examples**:
- Rookie's first game
- Debut performance
- Make-or-break opportunity
- "Prove it" scenario

**Transformer Coverage**: `conflict_tension.py`, `reputation_prestige.py`

---

### The Last Stand
**Definition**: Final desperate attempt  

**Characteristics**:
- Backs against wall
- Do-or-die situation
- Legacy on line
- Often futile but heroic

**Transformer Coverage**: `conflict_tension.py`, `temporal_momentum.py`

---

## VI. Temporal Patterns

### The Early Promise
**Definition**: Strong start creating expectations  

**Pattern**: High early performance → expectations rise → pressure increases

**Narrative Arc**:
- Breakout performance
- "Next big thing" labels
- Heightened expectations
- Can sustain or collapse

**Transformer Coverage**: `temporal_momentum.py`, `anticipatory_commitment.py`

---

### The Late Surge
**Definition**: Improvement toward end of period  

**Pattern**: Mediocre start → steady improvement → strong finish

**Narrative Power**:
- Recency bias amplifies
- "Peaking at right time"
- Momentum into next phase
- Overvalued by markets

**Finding**: NBA/NFL late-season signal strengthens (55% → 67%)  
**Transformer Coverage**: `temporal_momentum_enhanced.py`

---

### The Slump
**Definition**: Extended period of poor performance  

**Characteristics**:
- Multiple failures
- Psychological component
- "Can't buy a win"
- Eventually rebounds or spirals

**Transformer Coverage**: `temporal_momentum.py`, `self_perception.py`

---

### The Plateau
**Definition**: Sustained level performance without growth  

**Narrative Tension**: Expectation of progress but stagnation

**Transformer Coverage**: `temporal_momentum.py`, `narrative_potential.py`

---

## VII. Psychological/Emotional Patterns

### Confidence
**Definition**: Belief in ability to succeed  

**Markers**:
- Assertive language
- Risk-taking
- Past success references
- "We will" vs "we hope"

**Confidence Types**:
- **Earned**: Based on proven ability
- **Projected**: Performance confidence signals
- **Overconfidence**: Hubris
- **False Confidence**: Bravado masking doubt

**Transformer Coverage**: `self_perception.py`, `character_complexity.py`

---

### Pressure
**Definition**: Psychological weight of high stakes  

**Sources**:
- Expectations
- Stakes magnitude
- Historical weight
- Public scrutiny

**Effects**:
- Can elevate (rises to occasion)
- Can crush (chokes)
- Varies by individual/context

**Transformer Coverage**: `conflict_tension.py`, `anticipatory_commitment.py`

---

### Redemption Motivation
**Definition**: Drive to recover from failure/prove doubters wrong  

**Markers**:
- Past failure referenced
- "Prove them wrong" language
- Extra intensity
- Narrative of recovery

**Transformer Coverage**: `character_complexity.py`, `self_perception.py`, `temporal_momentum.py`

---

### Legacy Consciousness
**Definition**: Awareness of historical significance  

**Markers**:
- Career/historical context
- "For the legacy"
- Comparison to legends
- Moment magnitude awareness

**Transformer Coverage**: `reputation_prestige.py`, `anticipatory_commitment.py`

---

## VIII. Innovation & Disruption Patterns

### The Disruptor
**Definition**: Challenges established norms/methods  

**Characteristics**:
- Novel approach
- Threatens status quo
- Initial resistance
- Potential paradigm shift

**Transformer Coverage**: `narrative_potential.py`, `competitive_context.py`

---

### The Innovator
**Definition**: Creates new possibilities  

**Markers**:
- "First" language
- New category/method
- Future-oriented
- Groundbreaking framing

**Transformer Coverage**: `narrative_potential.py`, `origin_story.py`

---

### The Evolution
**Definition**: Gradual transformation over time  

**Pattern**: Old form → adaptation → new form → acceptance

**Transformer Coverage**: `temporal_momentum.py`, `narrative_potential.py`

---

## IX. Social/Cultural Patterns

### The Movement
**Definition**: Collective narrative of change  

**Characteristics**:
- Shared identity
- Ideological component
- Growing momentum
- Cultural significance

**Transformer Coverage**: `community_network.py`, `narrative_potential.py`

---

### The Phenomenon
**Definition**: Cultural event transcending original context  

**Characteristics**:
- Widespread attention
- Crosses boundaries
- Cultural conversation
- Zeitgeist moment

**Transformer Coverage**: `community_network.py`, `reputation_prestige.py`

---

### The Icon
**Definition**: Symbolic representation beyond self  

**Characteristics**:
- Transcends domain
- Cultural significance
- Represents values/ideas
- Enduring influence

**Transformer Coverage**: `reputation_prestige.py`, `character_complexity.py`

---

## X. Meta-Narrative Patterns

### The Narrative Inversion
**Definition**: Subversion of expected narrative  

**Examples**:
- Favorite as underdog framing
- Villain as hero
- Success as failure (expectations)

**Transformer Coverage**: `competitive_context.py`, `anticipatory_commitment.py`

---

### The Self-Aware Narrative
**Definition**: Explicit acknowledgment of narrative construction  

**Characteristics**:
- "Story" language used
- Narrative consciousness
- Meta-commentary
- WWE-style kayfabe

**Transformer Coverage**: Meta-level analysis

---

### The Mythologization
**Definition**: Real events elevated to mythic status  

**Process**: Real event → storytelling → embellishment → legend → myth

**Transformer Coverage**: `reputation_prestige.py`, `temporal_momentum.py`

---

## XI. Cross-Domain Constants

### Mathematical Ratios Discovered

#### 1.338 (NBA Constant)
**Found in**: NBA game analysis  
**Context**: Team-level calculation  
**Status**: Testing in other competitive domains  
**Hypothesis**: May appear in similar team competition structures

#### φ (Golden Ratio: 1.618)
**Status**: Not yet definitively found  
**Expected in**: Aesthetic/subjective domains  
**Looking for**: Optimal narrative proportions

#### γ (Decay Coefficients)
**NBA Momentum**: 0.948 (half-life ≈ 13 games)  
**Expected variation**: By domain and temporal scale  
**Hypothesis**: Each domain has characteristic decay rate

---

## How to Add to This Catalog

### When You Discover a New Pattern:

1. **Identify Category** (Character, Structure, Trope, etc.)
2. **Define Clearly** (What is it? What are characteristics?)
3. **Note Variations** (Different types/manifestations)
4. **Identify Markers** (How to detect it?)
5. **Map to Transformers** (Which extract this pattern?)
6. **Give Examples** (Cross-domain if possible)

### Template for New Entry:

```markdown
### [Pattern Name]
**Definition**: [Clear definition]

**Characteristics**:
- [Key trait 1]
- [Key trait 2]
- [Key trait 3]

**Variations**:
- **[Type 1]**: [Description]
- **[Type 2]**: [Description]

**Markers**: [How to detect in text]

**Transformer Coverage**: [Which transformers extract this]

**Domain Examples**: [Cross-domain manifestations]
```

---

## Research Questions

### Open Questions About Universal Patterns:

1. **Are there character archetypes we're missing?**
   - Beyond the classic hero/mentor/rival structure
   - Culture-specific archetypes that are actually universal

2. **What mathematical ratios are truly universal?**
   - Is 1.338 NBA-specific or competition-universal?
   - Where does φ appear?
   - What other constants exist?

3. **How do patterns combine?**
   - Hero + Underdog = ?
   - Rivalry + Championship = higher weight?
   - Systematic interaction effects

4. **What patterns are culture-dependent vs universal?**
   - Western narrative structures vs other cultures
   - Which cross all boundaries?

5. **How do patterns evolve over time?**
   - Do narrative tropes change?
   - Historical emergence of patterns
   - Future patterns?

---

## Connection to Transformers

### How This Catalog Feeds the System:

**Each transformer extracts specific patterns from this catalog:**

- `character_complexity.py` → Hero, Character Archetypes, Psychological Patterns
- `temporal_momentum_enhanced.py` → Temporal Patterns, Momentum, Decay
- `competitive_context.py` → Rivalry, Underdog, Championship Context
- `origin_story.py` → Origin Story, Hero's Journey, Quest
- `conflict_tension.py` → Pressure, Choke, Stakes, Proving Ground
- `narrative_potential.py` → Innovation, Disruption, Future-Orientation
- `reputation_prestige.py` → Icon, Dynasty, Legacy
- `community_network.py` → Ensemble, Movement, Collective
- `self_perception.py` → Confidence, Identity, Redemption Motivation
- `anticipatory_commitment.py` → Crossroads, Championship, Pressure

**This catalog defines WHAT patterns exist.**  
**Transformers define HOW to extract them.**  
**Domain structure defines WHICH CAN MANIFEST.**  
**Domain analysis defines WHERE and HOW MUCH they matter.**

### Domain Structure → Feature Importance

**Small team (5 players - NBA)**:
- Character complexity transformers matter MORE (can focus on individuals)
- Ensemble transformers matter LESS (only 5 people)
- Star player nominative features dominate

**Large team (22 players - NFL)**:
- Character complexity transformers matter LESS (too many to track)
- Ensemble transformers matter MORE (coordination crucial)
- QB nominative features dominate (touches every play)
- Coordinator names matter (call plays)

**Long narrative (300-page novel)**:
- Temporal momentum matters MORE (room for arc development)
- Origin story matters MORE (space to establish)
- Character complexity matters MORE (depth possible)

**Short narrative (3-minute song)**:
- Temporal momentum matters LESS (no time for development)
- Nominative features matter MORE (artist name = identity)
- Hook/repetition matters MORE (structural requirement)

**The same 47 transformers apply everywhere, but domain structure determines their RELATIVE IMPORTANCE**

---

## Usage in Analysis

### When Analyzing a New Domain:

1. **Review this catalog** for relevant patterns
2. **Map domain-specific manifestations** to universal patterns
3. **Confirm which transformers apply** to this domain
4. **Identify domain-specific variations** to add back to catalog
5. **Discover new patterns** not yet cataloged

### The Feedback Loop:

```
Catalog → Transformers → Domain Analysis → New Patterns Discovered → Update Catalog
```

This is a living, growing knowledge base of narrative fundamentals.

---

---

## XII. Instance-Level Concepts (November 2025 Update)

### Story Instance vs Story Domain

**Critical Distinction**:

**Story Domain** (formerly just "domain"):
- The genus/category: Golf, Supreme Court, Novels, NBA
- Has domain archetype (Ξ): What makes a great story HERE
- Has base narrativity (π_base): Typical openness
- Defines the "rules of the game"

**Story Instance** (formerly "organism"):
- The individual narrative: Tiger's 2019 Masters, Brown v. Board, Moby Dick
- Has genome (ж): Complete DNA of THIS story
- Has story quality (ю): How good THIS story is
- Has outcome (❊): Did THIS story succeed
- Has instance-specific π (π_effective): Can vary by complexity!

**Key Breakthrough**: π is NOT domain-constant. It varies by instance complexity within domain.

### Instance-Level Patterns

#### Dynamic Narrativity

**Pattern**: π_effective varies within domain by instance complexity

**Formula**: `π_effective = π_base + β × complexity`

**Example** (Supreme Court):
- Unanimous cases: π_effective ≈ 0.30 (evidence dominates)
- Split 5-4 cases: π_effective ≈ 0.70 (narrative decides)
- Domain average: π_base ≈ 0.52

**Implication**: Same domain, different narrative physics based on instance characteristics.

#### The Blind Narratio (Β)

**Definition**: Emergent equilibrium ratio between deterministic and free will forces

**Formula**: `Β = (ة + λ) / (θ + agency)`

Where:
- ة: Nominative gravity (name-based determinism)
- λ: Fundamental constraints (physics/training barriers)
- θ: Awareness resistance (conscious override)
- agency: Free will/narrator choice

**Properties**:
- Domain-specific (Basketball ≠ Supreme Court)
- Discoverable but not predictable
- Stable in long run (short-term variance)
- May vary by instance complexity
- Dual existence proof: BOTH determinism AND free will operate

**Example Values**:
- Golf: Β ≈ 0.73 (moderate determinism)
- Boxing: Β ≈ ? (high θ suppression suggests low Β)
- Supreme Court: Β varies by case complexity

#### Imperative Gravity (ф_imperative)

**Pattern**: Instances are pulled toward structurally similar domains for learning

**Formula**: `ф_imperative = (μ × domain_similarity) / domain_distance²`

**Example**:
- Complex Supreme Court case (π_eff = 0.70) pulled toward:
  - Oscars (π = 0.88, prestige dynamics)
  - Tennis (π = 0.75, individual mastery)
  - NOT toward Aviation (π = 0.05, too different)

**Function**: Cross-domain pattern transfer and learning

#### Awareness Amplification vs Resistance

**Two Types of Awareness**:

1. **θ_resistance** (awareness of determinism):
   - Suppresses narrative effects
   - Example: Boxing fighters know narrative shouldn't matter
   - High θ → low narrative impact

2. **θ_amplification** (awareness of potential):
   - Amplifies realization of narrative potential
   - Example: WWE performers explicitly play into narrative
   - High awareness of moment → amplified outcome

**Formula**: `outcome = base_prediction × (1 + θ_amp × potential × consciousness)`

**Key Insight**: Not all awareness is the same. Awareness OF narrative suppresses; awareness OF potential amplifies.

---

**Status**: Active Development  
**Patterns Cataloged**: 60+ universal + instance-level concepts
**Categories**: 12 (added Instance-Level)
**Transformer Coverage**: Expanding (new transformers for awareness amplification, imperative gravity)
**Last Major Addition**: Instance-Level Concepts (Nov 17, 2025)

**Add to this catalog as you discover new universal narrative patterns. This is the theoretical foundation for all domain analysis.**

