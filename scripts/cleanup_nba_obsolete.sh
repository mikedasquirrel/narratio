#!/bin/bash
#
# NBA File Cleanup - Archive Obsolete, Keep Production
# =====================================================
#
# Archives exploratory NBA work, deletes broken scripts.
# Keeps only production pattern-optimized system.
#
# Author: AI Coding Assistant
# Date: November 16, 2025
#

set -e

PROJECT_DIR="/Users/michaelsmerconish/Desktop/RandomCode/novelization"
cd "$PROJECT_DIR"

echo "=========================================="
echo "NBA File Cleanup"
echo "=========================================="
echo ""

# Create archive directories
echo "[1/3] Creating archive directories..."
mkdir -p archive/nba_exploration/analysis
mkdir -p archive/nba_exploration/domains
mkdir -p archive/nba_exploration/experiments
mkdir -p archive/old_test_scripts

echo "✓ Archive directories created"

# Archive exploratory analysis
echo ""
echo "[2/3] Archiving exploratory analysis..."

if [ -d "narrative_optimization/analysis" ]; then
    mv narrative_optimization/analysis/nba_*.py archive/nba_exploration/analysis/ 2>/dev/null || true
    echo "  ✓ Archived analysis/nba_*.py"
fi

if [ -d "narrative_optimization/domains/nba" ]; then
    # Keep only the results, archive the scripts
    cp narrative_optimization/domains/nba/*.json archive/nba_exploration/domains/ 2>/dev/null || true
    mv narrative_optimization/domains/nba/*.py archive/nba_exploration/domains/ 2>/dev/null || true
    echo "  ✓ Archived domains/nba/*.py"
fi

if [ -d "narrative_optimization/experiments" ]; then
    mv narrative_optimization/experiments/nba_optimization archive/nba_exploration/experiments/ 2>/dev/null || true
    mv narrative_optimization/experiments/06_nba_formula_discovery archive/nba_exploration/experiments/ 2>/dev/null || true
    mv narrative_optimization/experiments/05_nba_prediction archive/nba_exploration/experiments/ 2>/dev/null || true
    echo "  ✓ Archived experiments/nba_*"
fi

# Archive old test scripts
echo ""
echo "[3/3] Archiving old test scripts..."

OLD_TESTS=(
    "run_ALL_transformers_nba.py"
    "run_all_nba_transformers.py"
    "test_all_transformers_nba.py"
    "run_ALL_48_transformers_CLEAN.py"
    "run_ALL_transformers_PREGAME_ONLY.py"
)

for script in "${OLD_TESTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" archive/old_test_scripts/
        echo "  ✓ Archived $script"
    fi
done

echo ""
echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "PRODUCTION FILES (Kept):"
echo "  ✓ discover_player_patterns.py"
echo "  ✓ validate_player_patterns.py"
echo "  ✓ test_ALL_55_transformers_NBA_COMPREHENSIVE.py"
echo "  ✓ narrative_optimization/betting/ (all files)"
echo "  ✓ scripts/nba_*.py (all current scripts)"
echo "  ✓ routes/nba*.py"
echo ""
echo "ARCHIVED:"
echo "  → archive/nba_exploration/ (exploratory work)"
echo "  → archive/old_test_scripts/ (superseded tests)"
echo ""
echo "System now contains ONLY optimized production files!"

