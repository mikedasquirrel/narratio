#!/bin/bash
# Automated Site Cleanup and Restart
# ===================================
# Switches to streamlined betting-focused app automatically

cd /Users/michaelsmerconish/Desktop/RandomCode/novelization

echo "================================================================================"
echo "🧹 CLEANING UP SITE - SWITCHING TO STREAMLINED APP"
echo "================================================================================"
echo ""

# 1. Backup current app
echo "1. Backing up current app.py..."
cp app.py app_FULL_VERSION_BACKUP_$(date +%Y%m%d_%H%M%S).py
echo "   ✓ Backup created"

# 2. Switch to streamlined app
echo ""
echo "2. Switching to streamlined betting-focused app..."
cp app_betting_focused.py app.py
echo "   ✓ Streamlined app activated"

# 3. Kill any existing Flask process
echo ""
echo "3. Stopping any existing Flask processes..."
pkill -f "python3 app.py" 2>/dev/null || true
sleep 2
echo "   ✓ Old processes stopped"

# 4. Start streamlined app
echo ""
echo "4. Starting streamlined betting system..."
echo ""
echo "================================================================================"
echo "🎯 YOUR CLEAN BETTING SITE IS STARTING"
echo "================================================================================"
echo ""
echo "Visit:"
echo "  🏠 Home: http://localhost:5738/"
echo "  🎯 Live Dashboard: http://localhost:5738/betting/live"
echo "  🏀 NBA Betting: http://localhost:5738/nba/betting"
echo "  🏈 NFL Betting: http://localhost:5738/nfl/betting"
echo "  📊 API Health: http://localhost:5738/api/live/health"
echo ""
echo "Changes:"
echo "  ✓ Removed 65+ redundant pages"
echo "  ✓ Kept all betting features"
echo "  ✓ Added new live dashboard"
echo "  ✓ All enhancements active"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================================================"
echo ""

# Start Flask
python3 app.py

