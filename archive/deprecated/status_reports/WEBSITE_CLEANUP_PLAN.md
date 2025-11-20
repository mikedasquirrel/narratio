# Website Cleanup Plan

## Pages to Consider Removing/Updating

Based on analysis of routes and data availability:

### ❌ Remove - Incomplete/No Data

1. **Bible** (`/bible`) - Experimental, no complete analysis
   - Route: `bible_bp`
   - Data: None found
   - Action: Comment out registration

2. **Conspiracies** (`/conspiracies`) - Experimental, no validated data
   - Route: `conspiracies_bp`
   - Data: None found
   - Action: Comment out registration

3. **Dinosaurs** (`/dinosaurs`) - Incomplete analysis
   - Route: `dinosaurs_bp`
   - Data: Limited
   - Action: Comment out unless has current analysis

4. **Hurricanes** (`/hurricanes`) - Incomplete analysis
   - Route: `hurricanes_bp`
   - Data: Limited
   - Action: Comment out unless has current analysis

5. **Poker** (`/poker`) - Incomplete
   - Route: `poker_bp`
   - Data: Unknown
   - Action: Comment out

6. **Ships** (`/ships`) - Experimental domain
   - Route: Likely has one
   - Data: Unknown
   - Action: Comment out

7. **Temporal Linguistics** (`/temporal-linguistics`) - Experimental
   - Route: `temporal_linguistics_bp`
   - Data: Experimental
   - Action: Comment out

8. **WWE** (`/wwe`) - Incomplete
   - Route: `wwe_domain_bp`
   - Data: Unknown
   - Action: Comment out

9. **Novels** (`/novels`) - Incomplete
   - Route: `novels_bp`
   - Data: None found
   - Action: Comment out

10. **Experiments** - Development only
    - Route: `experiments_bp`
    - Action: Already commented out (good)

11. **Interactive Viz** - Generic visualizations
    - Route: Potentially `interactive_viz` 
    - Action: Check if needed

### ⚠️ Update - Has Data but May Need Refresh

1. **NBA** - Has data, keep (production betting system)
2. **NFL** - Has data, keep (production betting system)
3. **NHL** - Has data, keep (production betting system)
4. **MLB** - Has data, keep
5. **Tennis** - Has data, keep
6. **Golf** - Has data, keep
7. **UFC** - Check if current

### ✅ Keep - Complete & Production Ready

1. **Home** - Main landing page
2. **Analysis** / **Narrative Analyzer** - Core functionality
3. **NBA Betting** - Production system
4. **NFL Betting** - Production system
5. **NHL Betting** - Production system
6. **MLB Dashboard** - Has complete data
7. **Movies/IMDB** - Core domain
8. **Oscars** - Core domain
9. **Mental Health** - Complete analysis
10. **Startups** - Core domain
11. **Crypto** - Has analysis
12. **Tennis** - Complete data
13. **Golf** - Complete data
14. **Housing** - Complete analysis
15. **Supreme Court** - Recent addition, breakthrough findings
16. **Cross Domain** - Core analysis page
17. **Narrativity Spectrum** - Core theory page
18. **Formulas** - Core discoveries
19. **Variables** - Theory page
20. **Free Will** - Theory/framework page
21. **Framework Story** - Core documentation
22. **Project Overview** - Documentation
23. **Betting Dashboard** - Production betting hub
24. **Live Betting** - Production API
25. **Domain Processor** - New production tool

### Meta/Utility Pages (Keep)
- **Help** - User documentation
- **Findings** - Research summaries
- **Insights Dashboard** - Cross-domain insights
- **Meta Evaluation** - System evaluation
- **Meta Analysis** - Cross-domain patterns

## Recommended Actions

### Phase 1: Comment Out Incomplete Pages

```python
# In app.py, comment out these lines:

# bible_bp = safe_import('bible', 'bible_bp')
# if bible_bp:
#     app.register_blueprint(bible_bp, url_prefix='/bible')

# conspiracies_bp = safe_import('conspiracies', 'conspiracies_bp')
# if conspiracies_bp:
#     app.register_blueprint(conspiracies_bp, url_prefix='/conspiracies')

# poker_bp = safe_import('poker', 'poker_bp')
# if poker_bp:
#     app.register_blueprint(poker_bp)

# hurricanes_bp = safe_import('hurricanes', 'hurricanes_bp')
# if hurricanes_bp:
#     app.register_blueprint(hurricanes_bp)

# dinosaurs_bp = safe_import('dinosaurs', 'dinosaurs_bp')
# if dinosaurs_bp:
#     app.register_blueprint(dinosaurs_bp)

# novels_bp = safe_import('novels', 'novels_bp')
# if novels_bp:
#     app.register_blueprint(novels_bp)

# wwe_domain_bp = safe_import('wwe_domain', 'wwe_domain_bp')
# if wwe_domain_bp:
#     app.register_blueprint(wwe_domain_bp)

# temporal_linguistics_bp = safe_import('temporal_linguistics', 'temporal_linguistics_bp')
# if temporal_linguistics_bp:
#     app.register_blueprint(temporal_linguistics_bp)
```

### Phase 2: Update Navigation (if exists)

Remove links to commented-out pages from navigation templates.

### Phase 3: Add Note to Home Page

Add section: "Domain Processing Pipeline" pointing to new `/process/domain-processor`

## Result

**Before**: ~40 routes (many incomplete)
**After**: ~25-30 routes (all complete and current)

**Benefits**:
- Cleaner, more professional interface
- No broken/incomplete pages
- Easier to navigate
- Clear what's production-ready

## Test Plan

1. Comment out incomplete blueprints
2. Restart Flask app
3. Test major pages still work:
   - Home
   - NBA/NFL/NHL betting
   - Narrative analyzer
   - Domain processor
4. Verify no 404s or errors
5. Check navigation is clean


