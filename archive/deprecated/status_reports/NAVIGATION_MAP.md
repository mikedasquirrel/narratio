# Complete Navigation Map
## November 20, 2025

## Two Independent Sites

### Research Framework Site (base.html navigation)

**Navbar Links:**
- Home (`/`)
- Formulas (`/formulas`)
- Transformers (`/transformers/analysis`)
- Explorations (`/findings`)
- Betting System → (`/betting`) [green link]

**Homepage (`/`):**
- Clean landing page
- Dual CTA: Betting System | Research Framework
- Framework Tools grid (6 cards):
  - Formulas & Variables → `/formulas`
  - Transformers → `/transformers/analysis`
  - Explorations → `/findings`
  - Narrative Analyzer → `/analyze`
  - Domain Processor → `/process`
  - Framework Story → `/framework-story`

**Footer:**
- "research framework | 8 validated domains | betting system →"

---

### Betting System Site (betting_base.html navigation)

**Navbar Links:**
- Home (`/betting`)
- Dashboard (`/betting/dashboard`)
- Live (`/betting/live`)
- Sports ▾
  - NHL (69.4% win, 32.5% ROI) → `/nhl`
  - NFL (66.7% win, 27.3% ROI) → `/nfl`
  - NBA (54.5% win, 7.6% ROI) → `/nba`
- Analysis ▾
  - Validation Results → `/nhl/validation`
  - Betting Patterns → `/nhl/patterns`
  - Performance Tracking → `/betting/performance`
- Resources ▾
  - Methodology → `/betting/methodology`
  - Risk Management → `/betting/risk-management`
  - Education → `/betting/education`
  - Investor Materials → `/investor`

**Homepage (`/betting`):**
- Production betting landing
- 3 validated systems showcase
- Quick access cards (6):
  - Betting Dashboard
  - Live Opportunities
  - Methodology
  - Validation Results
  - Risk Management
  - Investor Materials

**Footer:**
- "Narrative Betting System — Production-validated sports betting | NHL • NFL • NBA"

---

## Complete Route List

### Core Pages (Research Site)
```
/                           → Home (research landing)
/formulas                   → Pure theory, no data
/transformers/analysis      → 79 transformers + 6 supervised
/transformers/catalog       → Transformer catalog view
/findings                   → Narrative determinism explorations
/analyze                    → Narrative analyzer tool
/process                    → Domain processor tool
/framework-story            → Complete theory journey
/variables                  → Variable explanations
/narrativity                → Narrativity spectrum
/discoveries                → Key findings summary (old style)
```

### Betting System (Betting Site)
```
/betting                    → Betting landing page
/betting/dashboard          → Portfolio dashboard
/betting/live               → Live opportunities
/betting/methodology        → Technical docs
/betting/performance        → Performance tracking
/betting/risk-management    → Risk protocols
/betting/education          → Betting education

/nhl                        → NHL system dashboard
/nhl/validation             → NHL validation results
/nhl/patterns               → NHL betting patterns

/nfl                        → NFL system (via blueprint)
/nba                        → NBA system (via blueprint)

/investor                   → Investor materials
/investor/dashboard         → Interactive dashboard
/investor/proposal          → Investment proposal
/investor/presentation      → Investor presentation
/investor/validation        → Technical validation
```

### Result Pages (Research Site)
```
/nba-results                → NBA validation
/nfl-results                → NFL validation
/movie-results              → Movies analysis
/imdb-results               → IMDB analysis
/golf-results               → Golf analysis
/hurricanes-results         → Hurricanes dual π
/startups-results           → Startups (marginal)
```

### API Endpoints
```
/api/domains/all            → All domains JSON
/api/domains/phase7         → Phase 7 force data
/api/domains/<domain>/features  → Domain features
```

### Export Pages (Standalone - No Nav)
```
/export/nhl-validation      → Standalone NHL validation
/export/nhl-patterns        → Standalone NHL patterns
/export/nhl-performance     → Standalone NHL performance
```

---

## Navigation Rules

### Research Site (base.html)
**Use for:**
- `/` (home)
- `/formulas`
- `/transformers/*`
- `/findings`
- `/analyze`
- `/process`
- `/framework-story`
- `/variables`
- `/narrativity`
- `/discoveries`
- All result pages (`/*-results`)

**Characteristics:**
- Simple 4-link navbar
- Focus on theory and exploration
- No betting performance data
- Link to betting system (green)

### Betting Site (betting_base.html)
**Use for:**
- `/betting/*`
- `/nhl/*` (except results)
- `/nfl/*` (except results)
- `/nba/*` (except results)
- `/investor/*`

**Characteristics:**
- Full betting navigation
- Sports, Analysis, Resources dropdowns
- Performance metrics visible
- No research/framework links

---

## Testing Checklist

### Research Site Navigation
- [ ] `/` loads with clean design
- [ ] Navbar shows: Home, Formulas, Transformers, Explorations, Betting →
- [ ] `/formulas` has no domain data (pure theory)
- [ ] `/findings` shows explorations page
- [ ] `/transformers/analysis` shows supervised info
- [ ] All framework tool links work from homepage
- [ ] Footer link to betting works

### Betting Site Navigation
- [ ] `/betting` loads betting landing
- [ ] Navbar shows: Home, Dashboard, Live, Sports ▾, Analysis ▾, Resources ▾
- [ ] All dropdown links work
- [ ] `/nhl` uses betting navigation
- [ ] `/betting/dashboard` uses betting navigation
- [ ] `/investor` uses betting navigation
- [ ] No research links visible
- [ ] Footer shows betting message

### Cross-Navigation
- [ ] Can go from research → betting via navbar
- [ ] Can go from research → betting via homepage
- [ ] Betting site doesn't link back to research
- [ ] URLs are clean and logical
- [ ] No broken links

---

## File Structure

### Navigation Templates
- `templates/base.html` - Research site navigation
- `templates/betting_base.html` - Betting site navigation

### Homepage Templates
- `templates/home_clean.html` - Research homepage (/)
- `templates/betting_home.html` - Betting homepage (/betting)

### Content Pages
All betting templates extend `betting_base.html`:
- betting_dashboard.html
- live_betting_dashboard.html
- betting_methodology.html
- betting_performance.html
- betting_education.html
- betting_risk_management.html
- nhl_validation.html
- nhl_patterns.html
- nhl_unified.html
- investor_landing.html

All research templates extend `base.html`:
- formulas.html
- explorations.html
- transformer_analysis.html
- transformer_catalog.html
- [all result pages]

---

## Summary

**Clean Separation Achieved:**
- Research site: Theory, formulas, explorations
- Betting site: Production systems, performance, investor materials
- Independent navigation experiences
- Can share either URL without exposing the other content
- Professional presentation for each audience

