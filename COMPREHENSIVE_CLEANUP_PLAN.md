# Comprehensive Project Cleanup Plan

**Date**: November 20, 2025  
**Goal**: Remove redundant, confusing, and outdated files/folders to leave only the real deal

---

## Major Issues Identified

### 1. **Duplicate/Confusing Root-Level Status Files** (20+ files)
Multiple overlapping status/summary files in root directory creating confusion:

**KEEP (Most Current)**:
- `README.md` - Main project overview
- `CURRENT_STATE_SUMMARY.md` - Current state (if recent)

**ARCHIVE** (Redundant/Outdated):
- `CLEANUP_SUMMARY.md` - Old cleanup notes
- `CROSS_DOMAIN_INTEGRATION_PLAN.md` - Old plan
- `CROSS_DOMAIN_STATUS_NOV17.md` - Dated status
- `DOMAIN_REGISTRY_STATUS.md` - Superseded by registry system
- `GENOME_PROCESSOR_FIX_NOV17.md` - One-off fix notes
- `GENOME_RERUN_PROGRESS.md` - Old progress notes
- `HOMEPAGE_UPDATE_NOV17.md` - Old update notes
- `MUTEX_ANALYSIS.md` - Old analysis
- `NARRATIVE_ENHANCEMENT_SUMMARY.md` - Old summary
- `NAVIGATION_MAP.md` - Superseded by docs
- `NHL_TEMPORAL_COMPLETE_SUMMARY.md` - Domain-specific, should be in docs
- `NHL_TEMPORAL_FRAMEWORK.md` - Domain-specific, should be in docs
- `ODDS_API_INTEGRATION_SUMMARY.md` - Old integration notes
- `PROJECT_STRUCTURE.md` - Duplicate (exists in narrative_optimization/)
- `REVALIDATION_INSTRUCTIONS_UPDATED.md` - Old instructions
- `SITE_REORGANIZATION_COMPLETE.md` - Old reorganization notes
- `STATUS_UPDATE_GENOME_FIX.md` - Old status
- `VALIDATED_DOMAINS.md` - Superseded by registry
- `VARIABLES_FORMULAS_UPDATE_NOV17.md` - Old update notes
- `WEBSITE_CLEANUP_PLAN.md` - Old plan
- `WEBSITE_UPDATE_NOV17_2025.md` - Old update notes
- `BOT_UPDATE_COMMAND.txt` - Old command notes

### 2. **Duplicate README/Structure Files**
- Root `/README.md` ✅ KEEP (main)
- `/narrative_optimization/README.md` - Check if redundant
- `/docs/README.md` - Check if redundant
- `/PROJECT_STRUCTURE.md` - Duplicate of narrative_optimization version
- `/narrative_optimization/PROJECT_STRUCTURE.md` - Keep one

### 3. **Multiple "COMPLETE" Implementation Docs** (10+ files in `/docs/implementation/`)
All dated Nov 17, 2025 - likely redundant:
- `BETTING_ENHANCEMENTS_IMPLEMENTATION_SUMMARY.md`
- `COMPLETE_FRAMEWORK_GUIDE.md`
- `COMPLETE_IMPLEMENTATION_SUMMARY_NOV_17_2025.md`
- `COMPLETE_VALIDATION_SUMMARY.txt`
- `EXECUTIVE_SUMMARY_TODAYS_WORK.md`
- `FRAMEWORK_IMPLEMENTATION_COMPLETE.md`
- `IMPLEMENTATION_COMPLETE_NOV_17_2025.md`
- `IMPLEMENTATION_PROGRESS.md`
- `MASTER_SUMMARY_FRAMEWORK_2_0.md`
- `PROJECT_OVERVIEW_COMPLETE.md`
- `SESSION_SUMMARY_EXTENDED.md`
- `TODAYS_WORK_SUMMARY.md`

**Action**: Consolidate into ONE current implementation guide or archive all as historical

### 4. **Multiple "START_HERE" Guides** (5+ files in `/docs/guides/`)
Confusing for new users:
- `START_HERE_BETTING_ENHANCEMENTS.md`
- `START_HERE_FRAMEWORK_2_0.md`
- `START_HERE_NBA_BETTING.md`
- `START_HERE_NHL.md`
- `WHAT_TO_DO_NEXT.md`
- `NEXT_STEPS_QUICK_REFERENCE.md`

**Action**: Consolidate into `/docs/BOT_ONBOARDING.md` (already created) and domain-specific guides

### 5. **Duplicate CLEANUP_SUMMARY Files**
- `/CLEANUP_SUMMARY.md` (root)
- `/docs/CLEANUP_SUMMARY.md`

**Action**: Keep one or archive both

### 6. **Redundant NHL Documentation** (15+ files in `/docs/domain_specific/`)
Multiple overlapping NHL docs:
- `nhl_checkpoint_playbook.md`
- `NHL_COMPLETE_SUCCESS.md`
- `NHL_DEPLOYMENT_COMPLETE.txt`
- `NHL_EXECUTIVE_SUMMARY.md`
- `NHL_FINAL_ANALYSIS.md`
- `NHL_IMPLEMENTATION_SUMMARY.md`
- `NHL_MASTER_SUMMARY.md`
- `NHL_NBA_NFL_COMPARISON.md`
- `NHL_QUICK_REFERENCE.txt`
- `NHL_README.md`
- `NHL_ROADMAP_TO_PRODUCTION.md`
- `NHL_SYSTEM_DEPLOYED.md`
- `NHL_SYSTEM_STATUS.txt`
- `NHL_VALIDATED_COMPLETE.md`

**Action**: Consolidate into ONE current NHL guide

### 7. **Redundant NBA Documentation** (5+ files)
- `NBA_BETTING_OPTIMIZED_COMPLETE.md`
- `NBA_BETTING_SYSTEM_COMPLETE.md`
- `NBA_BETTING_SYSTEM_README.md`
- `NBA_COMPLETE_ANALYSIS_SUMMARY.md`
- `NBA_FINAL_PRODUCTION_SYSTEM.md` ✅ KEEP (most current)
- `NBA_PROPS_EXPANSION_COMPLETE.md`
- `NBA_WORK_AUDIT_AND_CLEANUP.md`

**Action**: Keep NBA_FINAL_PRODUCTION_SYSTEM.md, archive rest

### 8. **Confusing Files in narrative_optimization Root**
- `=1.24.0` - Looks like pip output, DELETE
- `=1.3.0` - Empty file, DELETE
- `2025_season_validation.json` - Move to appropriate data folder
- `BATCH_EXECUTION_STATUS.json` - Old status, archive
- `TRANSFORMER_CATALOG.json` - Superseded by registry
- Multiple test/run scripts in root (should be in scripts/)

### 9. **Duplicate Config Folders**
- `/config/` (2 files)
- `/configs/` (empty)
- `/narrative_optimization/config/` (2 files)

**Action**: Consolidate into one location

### 10. **Multiple Requirements Files**
- `/requirements.txt` ✅ KEEP (main)
- `/narrative_optimization/requirements.txt` - Check if different

### 11. **Test Files in Wrong Locations**
Root-level test files should be in `/tests/`:
- `quick_test.py`
- `test_imports.py`
- `test_processor.py`
- `test_simple.py`

### 12. **Empty/Near-Empty Folders**
- `/configs/` - Empty
- `/stubs/` - 1 file
- `/services/` - 1 file

---

## Cleanup Actions

### Phase 1: Archive Root-Level Status Files
Move to `/archive/deprecated/status_reports/`:
- All dated status/summary files from root
- All "COMPLETE" notes from root
- All "UPDATE" notes from root

### Phase 2: Consolidate Documentation
- Merge duplicate NHL docs into ONE current guide
- Merge duplicate NBA docs into ONE current guide  
- Consolidate "START_HERE" guides
- Remove duplicate CLEANUP_SUMMARY files

### Phase 3: Clean narrative_optimization Root
- Delete pip output files (`=1.24.0`, `=1.3.0`)
- Move test scripts to proper locations
- Archive old JSON status files
- Remove superseded TRANSFORMER_CATALOG.json

### Phase 4: Consolidate Config
- Merge `/config/` and `/narrative_optimization/config/`
- Delete empty `/configs/`

### Phase 5: Move Test Files
- Move root test files to `/tests/`

### Phase 6: Clean Implementation Docs
- Consolidate 12 "COMPLETE" docs in `/docs/implementation/` into ONE or archive all

---

## Expected Result

**Clear Structure**:
```
/
├── README.md (main entry point)
├── docs/
│   ├── BOT_ONBOARDING.md (quick start)
│   ├── TRANSFORMERS_AND_PIPELINES.md (transformer guide)
│   ├── DEVELOPER_GUIDE.md (architecture)
│   ├── ONBOARDING_HANDBOOK.md (domain onboarding)
│   ├── domain_specific/
│   │   ├── NBA_PRODUCTION_SYSTEM.md (ONE NBA guide)
│   │   └── NHL_PRODUCTION_SYSTEM.md (ONE NHL guide)
│   └── implementation/
│       └── CURRENT_IMPLEMENTATION.md (ONE implementation guide)
├── narrative_optimization/ (clean, organized)
├── scripts/ (all scripts here)
├── tests/ (all tests here)
├── config/ (ONE config location)
└── archive/
    └── deprecated/
        ├── status_reports/ (old status files)
        └── docs/ (old documentation)
```

**Benefits**:
1. New bots find current info immediately
2. No confusion about which file is current
3. Clear hierarchy
4. Historical info preserved but isolated

---

## Execution Order

1. Create archive directories
2. Move root-level status files
3. Consolidate NHL/NBA docs
4. Clean narrative_optimization root
5. Move test files
6. Consolidate config
7. Update main README with clear navigation

**Estimated Files to Archive/Delete**: 50-70 files
**Estimated Time**: 30-45 minutes

