# Project Structure

This document describes the organized file structure of the Narrative Optimization Framework codebase.

## Root Directory

The root directory contains only essential project files:

```
novelization/
├── app.py                      # Main Flask application entry point
├── README.md                   # Project overview and documentation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── PROJECT_STRUCTURE.md        # This file
```

## Core Directories

### `/analysis/`
Analysis scripts and results for production backtesting and validation.

```
analysis/
├── cross_league_pattern_validation.py    # Cross-sport validation
├── EXECUTIVE_SUMMARY_BACKTEST.md         # Backtest results summary
├── RECENT_SEASON_BACKTEST_REPORT.md      # Recent season analysis
├── production_backtest_results.json      # Backtest data
├── multi_market_validation_status.json   # Validation metrics
└── visualizations/                       # Charts and graphs
    ├── roi_comparison.png
    ├── win_rate_comparison.png
    └── summary_dashboard.png
```

### `/archive/`
Historical code, deprecated implementations, and backup files.

```
archive/
├── backups/                              # Dated backup files
│   ├── transformer_backup_20251116_135130/
│   ├── transformer_backup_20251116_141256/
│   ├── transformer_backup_DEFINITIVE_20251116_141305/
│   └── app_*.py                          # Application backups
├── nba_exploration/                      # NBA experimental work
│   ├── analysis/
│   ├── domains/
│   └── experiments/
└── old_test_scripts/                     # Deprecated test files
```

### `/configs/`
Configuration files for different environments and domains.

### `/data/`
All data files organized by domain and type.

```
data/
├── domains/                              # Domain-specific datasets
│   ├── nfl_real_games.json
│   ├── nba_real_games.json
│   └── [other domain data files]
├── live/                                 # Real-time data feeds
├── predictions/                          # Model predictions
├── patterns/                             # Discovered patterns
├── paper_trading/                        # Simulated trading data
├── logs/                                 # Data collection logs
└── results/                              # Analysis results
```

**Specialized Data Directories:**
- `bible/` - Biblical narrative analysis
- `college_basketball/` - NCAA basketball data
- `conspiracy/` - Conspiracy theory analysis
- `literary_corpus/` - Literary works
- `march_madness/` - Tournament data
- `ml-latest-small/` - MovieLens dataset
- `MovieSummaries/` - Movie analysis corpus
- `nba_enriched/` - Enhanced NBA data

### `/data_collection/`
Scripts for collecting and building datasets from various sources.

```
data_collection/
├── nba_data_builder.py
├── nba_raw_player_data_collector.py
├── nhl_data_builder.py
├── nhl_data_builder_full_history.py
├── literary_corpus_collector.py
└── SOURCE_COLLECTION_SPECIFICATIONS.md
```

### `/docs/`
Complete project documentation organized by category.

```
docs/
├── README.md                             # Documentation index
│
├── guides/                               # User and developer guides
│   ├── LAUNCH_GUIDE.md
│   ├── NAVIGATION_INDEX.md
│   ├── README_MASTER.md
│   ├── START_HERE_FRAMEWORK_2_0.md
│   ├── WHAT_TO_DO_NEXT.md
│   ├── START_HERE_NBA_BETTING.md
│   ├── START_HERE_NHL.md
│   ├── START_HERE_BETTING_ENHANCEMENTS.md
│   ├── FINAL_DELIVERABLES_FRAMEWORK_2_0.md
│   ├── EASY_COMMANDS.md
│   ├── NEXT_STEPS_QUICK_REFERENCE.md
│   ├── WEBSITE_ACCESS_GUIDE.md
│   ├── ADD_NEW_DOMAIN_TEMPLATE.md
│   ├── DOMAIN_INTEGRATION_GUIDE.md
│   ├── COMPARISON_SYSTEM_GUIDE.md
│   ├── HOW_TO_USE_COMPARISON.md
│   └── README_FLASK.md
│
├── implementation/                       # Implementation summaries
│   ├── IMPLEMENTATION_COMPLETE_NOV_17_2025.md
│   ├── MASTER_SUMMARY_FRAMEWORK_2_0.md
│   ├── COMPLETE_FRAMEWORK_GUIDE.md
│   ├── COMPLETE_IMPLEMENTATION_SUMMARY_NOV_17_2025.md
│   ├── FRAMEWORK_IMPLEMENTATION_COMPLETE.md
│   ├── IMPLEMENTATION_PROGRESS.md
│   ├── BETTING_ENHANCEMENTS_IMPLEMENTATION_SUMMARY.md
│   ├── EXECUTIVE_SUMMARY_TODAYS_WORK.md
│   ├── PROJECT_OVERVIEW_COMPLETE.md
│   ├── SESSION_SUMMARY_EXTENDED.md
│   ├── TODAYS_WORK_SUMMARY.md
│   └── COMPLETE_VALIDATION_SUMMARY.txt
│
├── reference/                            # Technical reference docs
│   ├── DEFINITIVE_CANONICAL_LIST.md
│   ├── DOMAIN_DEVELOPMENT_STAGES.md
│   ├── DOMAIN_SPECTRUM_ANALYSIS.md
│   ├── DOMAIN_STATUS.md
│   ├── DOMAIN_TESTING_ANALYSIS.md
│   ├── FORMAL_VARIABLE_SYSTEM.md
│   ├── IMPERATIVE_GRAVITY_NETWORK.md
│   ├── NARRATIVE_CATALOG.md
│   ├── THEORETICAL_FRAMEWORK.md
│   ├── TRANSFORMER_SPEED_OPTIMIZATION_GUIDE.md
│   ├── UPDATED_CANONICAL_TRANSFORMERS.md
│   ├── README_TRANSFORMER_ANALYSIS.md
│   ├── BLIND_NARRATIO_RESULTS.md
│   ├── CLEANUP_GUIDE_STREAMLINE_SITE.md
│   ├── DEVELOPMENT.md
│   ├── EXECUTION_PROGRESS_REPORT.md
│   └── TRANSFORMER_EFFECTIVENESS_ANALYSIS.md
│
├── betting_systems/                      # Betting system documentation
│   ├── CLEAN_BETTING_STRUCTURE.md
│   ├── EXECUTIVE_SUMMARY_BETTING_ENHANCEMENTS.md
│   ├── NBA_BETTING_OPTIMIZED_COMPLETE.md
│   ├── NBA_BETTING_SYSTEM_COMPLETE.md
│   ├── NBA_BETTING_SYSTEM_README.md
│   ├── NFL_LIVE_BETTING_SYSTEM.md
│   ├── NHL_BETTING_STRATEGY_GUIDE.md
│   ├── NHL_BETTING_SYSTEM.md
│   └── TODAYS_BETS_NOVEMBER_16.md
│
├── domain_specific/                      # Domain analysis documentation
│   ├── MLB_COMPLETE_FINAL.txt
│   ├── MLB_IMPLEMENTATION_COMPLETE.txt
│   ├── MLB_TRANSFORMER_OPTIMIZATION_COMPLETE.md
│   ├── MOVIE_ANALYSIS_README.md
│   ├── MOVIE_ANALYSIS_RESULTS.md
│   ├── NBA_COMPLETE_ANALYSIS_SUMMARY.md
│   ├── NBA_FINAL_PRODUCTION_SYSTEM.md
│   ├── NBA_PROPS_EXPANSION_COMPLETE.md
│   ├── NBA_WORK_AUDIT_AND_CLEANUP.md
│   ├── NCAA_DATA_COLLECTION_STATUS.md
│   ├── NFL_ANALYSIS_EXECUTION_SUMMARY.md
│   ├── NHL_COMPLETE_SUCCESS.md
│   ├── NHL_DEPLOYMENT_COMPLETE.txt
│   ├── NHL_EXECUTIVE_SUMMARY.md
│   ├── NHL_FINAL_ANALYSIS.md
│   ├── NHL_IMPLEMENTATION_SUMMARY.md
│   ├── NHL_MASTER_SUMMARY.md
│   ├── NHL_NBA_NFL_COMPARISON.md
│   ├── NHL_QUICK_REFERENCE.txt
│   ├── NHL_README.md
│   ├── NHL_ROADMAP_TO_PRODUCTION.md
│   ├── NHL_SYSTEM_DEPLOYED.md
│   ├── NHL_SYSTEM_STATUS.txt
│   ├── NHL_TRANSFORMER_DISCOVERY.md
│   └── NHL_VALIDATED_COMPLETE.md
│
├── theory/                               # Theoretical framework docs
│   ├── NARRATIVE_FRAMEWORK.md
│   ├── NARRATIVITY_SPECTRUM_THEORY.md
│   ├── UNIVERSAL_NARRATIVE_THEORY.md
│   ├── TEMPORAL_DYNAMICS_THEORY.md
│   ├── three_force_model.md
│   ├── THREE_FORCE_SUMMARY.md
│   ├── prestige_equation.md
│   ├── formal_variables.md
│   └── [more theory files]
│
├── domains/                              # Domain-specific guides
│   ├── README.md
│   ├── MASTER_DOMAIN_FINDINGS.md
│   ├── EXAMPLE_DOMAINS_WE_WANT.md
│   ├── _TEMPLATE.md
│   ├── golf.md
│   ├── boxing.md
│   └── hurricanes.md
│
├── templates/                            # Documentation templates
│   ├── NEW_DOMAIN_ONBOARDING.md
│   ├── DATA_COLLECTION_PROMPT_TEMPLATE.md
│   ├── QUICK_DATA_REQUEST.md
│   └── TEMPORAL_LINGUISTICS_DATA_COLLECTION.md
│
├── strategic/                            # Strategic planning
│   ├── NEXT_STEPS_STRATEGIC.md
│   └── ACCESSIBLE_DATASETS_ANALYSIS.md
│
├── technical/                            # Technical documentation
│   ├── PYTHONANYWHERE_DEPLOY.md
│   ├── TRANSFORMER_CATALOG.md
│   └── TRANSFORMER_FRAMEWORK_ASSESSMENT.md
│
├── papers/                               # Research papers
│   └── HOUSING_NOMINATIVE_PAPER.md
│
├── archive/                              # Historical documentation
│   └── [archived documentation files]
│
├── analysis/                             # Analysis documentation
└── validation/                           # Validation documentation
```

### `/domain_schemas/`
JSON schema definitions for different domain types.

```
domain_schemas/
├── ekko_domain_schema.json
├── feature_domain_schema.json
├── hurricane_domain_schema.json
├── mixed_domain_schema.json
├── text_domain_schema.json
└── structural_definitions/
```

### `/logs/`
All log files from various operations and processes.

```
logs/
├── bets/                                 # Betting logs
│   └── bet_log_20251116.jsonl
├── betting/                              # Betting system logs
│   └── cron.log
├── nhl/                                  # NHL-specific logs
│   ├── full_history_collection.log
│   ├── predictions_*.log
│   └── tracker_*.log
└── [various log files]
```

### `/narrative_optimization/`
Core narrative analysis framework and transformers.

```
narrative_optimization/
├── src/
│   └── transformers/                     # Transformer implementations
│       ├── nominative.py                 # Universal transformers
│       ├── temporal_momentum_enhanced.py
│       ├── character_complexity.py
│       ├── [42+ more universal transformers]
│       └── sports/                       # Domain-specific transformers
│           ├── nfl_performance.py
│           ├── nba_performance.py
│           └── tennis_performance.py
├── domains/                              # Domain analysis implementations
│   ├── nfl/
│   ├── nba/
│   ├── tennis/
│   └── [20+ other domains]
└── experiments/                          # Experimental results
```

### `/nba_data_repo/`
NBA dataset repository (external data source).

### `/results/`
Analysis results, predictions, and model outputs.

```
results/
├── comprehensive_transformer_results.json
├── discovered_player_patterns.json
├── golf_comprehensive_results.json
├── golf_comprehensive_results.csv
├── movie_transformer_results.json
├── nba_comprehensive_ALL_55_results.csv
├── nba_comprehensive_ALL_55_results.json
├── nba_full_pipeline_results.json
├── pattern_validation_results.json
├── transformer_comparison_results.json
├── transformer_performance_analysis.csv
├── transformer_performance_analysis.json
├── movie_analysis_output.txt
└── COMPLETE_VALIDATION_SUMMARY.txt
```

### `/routes/`
Flask route handlers for the web application (55 route files).

```
routes/
├── nfl.py                                # NFL analysis routes
├── nba.py                                # NBA analysis routes
├── nhl.py                                # NHL analysis routes
├── startups.py                           # Startup analysis routes
└── [48+ other domain route files]
```

### `/scripts/`
Operational scripts organized by function.

```
scripts/
├── analysis/                             # Analysis scripts
│   ├── analyze_transformer_performance_simple.py
│   ├── analyze_transformer_performance.py
│   ├── backtest_production_quality.py
│   ├── check_progress.py
│   ├── discover_player_patterns.py
│   └── validate_player_patterns.py
│
├── testing/                              # Test scripts
│   ├── test_ALL_55_transformers_GOLF.py
│   ├── test_ALL_55_transformers_NBA_COMPREHENSIVE.py
│   ├── test_ALL_transformers_comprehensive.py
│   └── run_all_transformers_movies.py
│
├── utilities/                            # Utility scripts
│   ├── cleanup_DEFINITIVE.py
│   ├── cleanup_transformers_UPDATED.py
│   ├── cleanup_transformers.py
│   ├── fix_transformer_input_shapes.py
│   └── view_results.py
│
├── data_generation/                      # Data generation scripts
│   ├── build_player_data_from_pbp.py
│   ├── generate_all_predictions.py
│   ├── generate_sample_bets.py
│   ├── get_real_predictions.py
│   ├── GET_TODAYS_GAMES.py
│   ├── quick_train_and_predict.py
│   ├── rebuild_nfl_current.py
│   └── simple_daily_picks.py
│
├── nfl_analysis/                         # NFL-specific analysis
│   ├── phase1_validate_data.py
│   ├── phase2_nfl_transformer.py
│   ├── phase3_nominative_analysis.py
│   └── [more phase files]
│
├── [Shell scripts]
│   ├── CLEANUP_AND_RESTART.sh
│   ├── cleanup_nba_obsolete.sh
│   ├── INSTALL_CRON.sh
│   ├── monitor_until_complete.sh
│   ├── run_movie_analysis_complete.sh
│   ├── START_ENHANCED_SYSTEM.sh
│   ├── nba_automated_daily.sh
│   ├── nhl_automated_daily.sh
│   ├── nhl_watch_collection.sh
│   ├── test_all_enhancements.sh
│   └── validate_all_systems.sh
│
└── [Domain-specific scripts]
    ├── nba_*.py                          # NBA betting/prediction
    ├── nhl_*.py                          # NHL betting/prediction
    ├── nfl_*.py                          # NFL live betting
    ├── automated_bet_placer.py
    ├── live_game_monitor.py
    ├── paper_trading_system.py
    └── [more specialized scripts]
```

### `/static/`
Web application static assets.

```
static/
├── css/                                  # Stylesheets
├── js/                                   # JavaScript files
└── images/                               # Image assets
```

### `/templates/`
Flask HTML templates (127 template files).

```
templates/
├── index.html
├── domains/
├── betting/
└── [domain-specific templates]
```

### `/utils/`
Shared utility modules.

```
utils/
├── betting_edge_calculator.py
└── phase7_data_loader.py
```

## File Organization Principles

### 1. Root Simplicity
- Only essential project files in root directory
- Maximum ~10-15 files at root level
- Clear entry points (app.py, README.md)

### 2. Logical Grouping
- **Analysis** - Results and validation
- **Archive** - Historical/deprecated code
- **Data** - All datasets organized by domain
- **Docs** - Complete documentation tree
- **Scripts** - Organized by function (analysis, testing, utilities, data generation)
- **Logs** - All log files centralized

### 3. Documentation Structure
- **guides/** - How-to documentation
- **implementation/** - Project history and summaries
- **reference/** - Technical specifications
- **theory/** - Theoretical framework
- **betting_systems/** - Betting-specific docs
- **domain_specific/** - Per-domain analysis

### 4. Script Organization
- **analysis/** - Analytical operations
- **testing/** - Test suites
- **utilities/** - Helper scripts
- **data_generation/** - Data creation and prediction
- **[domain]_analysis/** - Domain-specific workflows

### 5. Archive Policy
- Backup files go to `archive/backups/`
- Deprecated code goes to `archive/old_*_scripts/`
- Experimental work goes to `archive/[domain]_exploration/`

## Navigation Quick Reference

**Starting Points:**
- Project overview: `README.md`
- Getting started: `docs/guides/START_HERE_FRAMEWORK_2_0.md`
- Web interface: `docs/guides/WEBSITE_ACCESS_GUIDE.md`
- Add new domain: `docs/templates/NEW_DOMAIN_ONBOARDING.md`

**Key Technical Docs:**
- Variables: `docs/reference/FORMAL_VARIABLE_SYSTEM.md`
- Transformers: `docs/technical/TRANSFORMER_CATALOG.md`
- Domains: `docs/reference/DOMAIN_STATUS.md`
- Theory: `docs/theory/NARRATIVE_FRAMEWORK.md`

**Betting Systems:**
- NHL: `docs/betting_systems/NHL_BETTING_SYSTEM.md`
- NFL: `docs/betting_systems/NFL_LIVE_BETTING_SYSTEM.md`
- NBA: `docs/betting_systems/NBA_BETTING_SYSTEM_COMPLETE.md`
- Backtest: `analysis/EXECUTIVE_SUMMARY_BACKTEST.md`

**Development:**
- Developer guide: `docs/DEVELOPER_GUIDE.md`
- Deployment: `docs/technical/PYTHONANYWHERE_DEPLOY.md`
- Caching: `docs/CACHING_GUIDE.md`

## Maintenance Notes

**What Goes Where:**

- **Log files** → `/logs/` (organized by domain/type)
- **Result files** → `/results/`
- **Documentation** → `/docs/` (categorized)
- **Scripts** → `/scripts/` (by function)
- **Backups** → `/archive/backups/`
- **Old code** → `/archive/`
- **Data files** → `/data/` (by domain)

**Cleanup Guidelines:**

1. Keep root directory minimal
2. Organize by function, not chronology
3. Archive rather than delete
4. Document structure changes
5. Use descriptive directory names
6. Maintain consistent organization

---

**Last Updated:** November 17, 2025  
**Organization Version:** 1.0  
**Status:** Clean and production-ready

