#!/bin/bash
#
# NBA Betting Automation - Cron Job Installer
# ============================================
#
# This script sets up the cron job for automated daily predictions.
# Run once to install, then system runs automatically every day.
#
# Author: AI Coding Assistant
# Date: November 16, 2025
#

echo ""
echo "================================================================================"
echo "NBA BETTING - CRON JOB INSTALLER"
echo "================================================================================"
echo ""

PROJECT_DIR="/Users/michaelsmerconish/Desktop/RandomCode/novelization"
SCRIPT_PATH="$PROJECT_DIR/scripts/nba_automated_daily.sh"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT_PATH"
echo "✓ Made script executable"

# Create log directory
mkdir -p "$PROJECT_DIR/logs/betting"
echo "✓ Created log directory"

# Generate crontab entry
CRON_ENTRY="0 9 * * * $SCRIPT_PATH >> $PROJECT_DIR/logs/betting/cron.log 2>&1"

echo ""
echo "Cron job entry to add:"
echo "─────────────────────────────────────────────────────────────────────────────"
echo "$CRON_ENTRY"
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "This will run daily at 9:00 AM EST"
echo ""

# Check if crontab exists
if crontab -l 2>/dev/null | grep -q "nba_automated_daily.sh"; then
    echo "⚠️  NBA cron job already exists"
    echo ""
    read -p "Replace existing entry? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 1
    fi
    
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "nba_automated_daily.sh" | crontab -
    echo "✓ Removed old entry"
fi

# Add new entry
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo ""
echo "✅ CRON JOB INSTALLED!"
echo ""
echo "The system will now:"
echo "  1. Run automatically every day at 9:00 AM EST"
echo "  2. Fetch today's games and odds"
echo "  3. Generate betting predictions"
echo "  4. Save to data/predictions/"
echo "  5. Update dashboard automatically"
echo ""
echo "View logs at: $PROJECT_DIR/logs/betting/cron.log"
echo "View dashboard at: http://127.0.0.1:5738/nba/betting/live"
echo ""
echo "To remove cron job later:"
echo "  crontab -e"
echo "  (delete the line with nba_automated_daily.sh)"
echo ""
echo "================================================================================"
echo "INSTALLATION COMPLETE"
echo "================================================================================"
echo ""

