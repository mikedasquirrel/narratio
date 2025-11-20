#!/bin/bash
# Test All Betting System Enhancements
# =====================================
# Runs all test suites to validate complete system
#
# Author: AI Coding Assistant
# Date: November 16, 2025

echo "================================================================================"
echo "TESTING ALL BETTING SYSTEM ENHANCEMENTS"
echo "================================================================================"
echo ""

cd /Users/michaelsmerconish/Desktop/RandomCode/novelization

# Track results
PASSED=0
FAILED=0

# Test function
test_component() {
    local name=$1
    local command=$2
    
    echo "Testing: $name"
    echo "--------------------------------------------------------------------------------"
    
    if eval "$command" > /dev/null 2>&1; then
        echo "✓ PASS: $name"
        ((PASSED++))
    else
        echo "✗ FAIL: $name"
        ((FAILED++))
    fi
    echo ""
}

# 1. Model Performance
echo "1. MODEL PERFORMANCE IMPROVEMENTS"
echo "================================================================================"
test_component "Cross-Domain Features" "python3 narrative_optimization/feature_engineering/cross_domain_features.py"
test_component "NBA Advanced Ensemble" "python3 narrative_optimization/betting/nba_advanced_ensemble.py"
test_component "NFL Advanced Ensemble" "python3 narrative_optimization/betting/nfl_advanced_ensemble.py"
test_component "Unified Sports Model" "python3 narrative_optimization/betting/unified_sports_model.py"
echo ""

# 2. Pattern Analysis
echo "2. PATTERN ANALYSIS ENHANCEMENTS"
echo "================================================================================"
test_component "Higher-Order Patterns" "python3 narrative_optimization/patterns/higher_order_discovery.py"
test_component "Dynamic Weighting" "python3 narrative_optimization/patterns/dynamic_pattern_weighting.py"
test_component "Contextual Analyzer" "python3 narrative_optimization/patterns/contextual_pattern_analyzer.py"
test_component "Cross-League Validation" "python3 analysis/cross_league_pattern_validation.py"
echo ""

# 3. Risk Management
echo "3. RISK MANAGEMENT"
echo "================================================================================"
test_component "Kelly Criterion" "python3 narrative_optimization/betting/kelly_criterion.py"
test_component "Bankroll Simulator" "python3 narrative_optimization/betting/bankroll_simulator.py"
echo ""

# 4. Live Systems
echo "4. LIVE BETTING INFRASTRUCTURE"
echo "================================================================================"
test_component "Live Odds Fetcher" "python3 scripts/live_odds_fetcher.py"
test_component "Paper Trading" "python3 scripts/paper_trading_system.py"
echo ""

# Summary
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "System is ready for backtesting and deployment."
else
    echo "⚠️  Some tests failed. Review output above."
fi

echo ""
echo "================================================================================"
echo "NEXT STEPS:"
echo "  1. Run: python3 scripts/comprehensive_backtest.py"
echo "  2. Get API key: https://the-odds-api.com/"
echo "  3. Start paper trading for 2-4 weeks"
echo "  4. Deploy after validation"
echo "================================================================================"

