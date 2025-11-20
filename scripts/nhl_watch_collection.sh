#!/bin/bash
#
# Watch NHL Historical Collection Progress
#
# Monitors the background collection process
#

echo "🏒 NHL HISTORICAL COLLECTION MONITOR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

LOG_FILE="/tmp/nhl_full_collection.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found. Collection may not be running."
    echo "Start with: python3 data_collection/nhl_data_builder_full_history.py &"
    exit 1
fi

echo "📊 Current Progress:"
echo "───────────────────────────────────────────────────────────────"

# Get latest progress
tail -20 "$LOG_FILE" | grep -E "Week|games|✓|✅" | tail -10

echo ""
echo "───────────────────────────────────────────────────────────────"

# Check if still running
if ps aux | grep -q "[n]hl_data_builder_full_history"; then
    echo "Status: 🟢 RUNNING"
else
    echo "Status: 🔴 COMPLETED or STOPPED"
fi

echo ""
echo "Full log: tail -f /tmp/nhl_full_collection.log"
echo "═══════════════════════════════════════════════════════════════"

