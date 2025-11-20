#!/bin/bash
# Start Enhanced Betting System
# ==============================
# Launches your Flask app with all new betting enhancements
#
# Usage: bash START_ENHANCED_SYSTEM.sh

echo "================================================================================"
echo "🎯 STARTING ENHANCED BETTING SYSTEM"
echo "================================================================================"
echo ""

cd /Users/michaelsmerconish/Desktop/RandomCode/novelization

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✓ Python 3 available"

# Check dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import numpy, pandas, sklearn, flask" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Core dependencies installed"
else
    echo "⚠️  Installing missing dependencies..."
    pip3 install numpy pandas scikit-learn flask flask-cors
fi

# Check optional dependencies
echo ""
echo "Checking optional dependencies..."
python3 -c "import xgboost" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ XGBoost installed (optimal performance)"
else
    echo "ℹ️  XGBoost not installed (optional, install with: pip3 install xgboost)"
fi

# Check for API key
echo ""
if [ -z "$THE_ODDS_API_KEY" ]; then
    echo "ℹ️  No Odds API key found (live odds will use mock data)"
    echo "   Get free key at: https://the-odds-api.com/"
    echo "   Then set: export THE_ODDS_API_KEY='your_key'"
else
    echo "✓ Odds API key configured"
fi

echo ""
echo "================================================================================"
echo "🚀 LAUNCHING FLASK APP"
echo "================================================================================"
echo ""
echo "Your enhanced betting system will be available at:"
echo ""
echo "  🏠 Main App:          http://localhost:5738/"
echo "  🎯 Live Dashboard:    http://localhost:5738/betting/live"
echo "  🏀 NBA Betting:       http://localhost:5738/nba/betting"
echo "  🏈 NFL Betting:       http://localhost:5738/nfl/betting"
echo ""
echo "  📊 API Health:        http://localhost:5738/api/live/health"
echo "  🎲 Opportunities:     http://localhost:5738/api/live/opportunities"
echo "  💰 Domains:           http://localhost:5738/domains"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "================================================================================"
echo ""

# Start Flask app
python3 app.py

