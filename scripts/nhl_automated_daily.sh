#!/bin/bash
#
# NHL Automated Daily Runner
#
# Runs daily betting predictions automatically:
# 1. Fetches live odds
# 2. Generates predictions
# 3. Tracks performance
# 4. Sends alerts (optional)
#
# Add to crontab:
# 0 9 * * * /path/to/nhl_automated_daily.sh >> /path/to/logs/nhl_daily.log 2>&1
#
# Author: Narrative Integration System
# Date: November 16, 2025

# Configuration
PROJECT_DIR="/Users/michaelsmerconish/Desktop/RandomCode/novelization"
LOG_DIR="$PROJECT_DIR/logs/nhl"
DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "NHL AUTOMATED DAILY RUNNER"
echo "================================================================================"
echo "Date: $DATE"
echo "Time: $(date +%H:%M:%S)"
echo ""

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Step 1: Fetch live odds
echo "📡 Step 1: Fetching live odds..."
python3 scripts/nhl_fetch_live_odds.py
if [ $? -eq 0 ]; then
    echo "   ✓ Odds fetched successfully"
else
    echo "   ⚠️  Odds fetch failed (using cached/mock data)"
fi
echo ""

# Step 2: Generate predictions
echo "🎯 Step 2: Generating predictions..."
python3 scripts/nhl_daily_predictions.py > "$LOG_DIR/predictions_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Predictions generated"
    
    # Display predictions
    echo ""
    echo "📊 TODAY'S RECOMMENDATIONS:"
    python3 -c "
import json, sys
from pathlib import Path
pred_file = Path('data/predictions').glob('nhl_predictions_*.json')
latest = max(pred_file, key=lambda x: x.stat().st_mtime, default=None)
if latest:
    with open(latest) as f:
        data = json.load(f)
        preds = [p for p in data.get('predictions', []) if p.get('recommendation')]
        if preds:
            for i, p in enumerate(preds, 1):
                rec = p['recommendation']
                game = p['game']
                print(f\"{i}. {game['away_team']} @ {game['home_team']}\")
                print(f\"   Bet: {rec['bet']} ({rec['unit_size']}u)\")
                print(f\"   Expected: {rec['expected_win_rate']:.1f}% win, {rec['expected_roi']:.1f}% ROI\")
                print(f\"   Pattern: {rec['pattern']}\")
                print()
        else:
            print('No high-confidence picks today')
" 2>&1
else
    echo "   ❌ Prediction generation failed"
fi
echo ""

# Step 3: Track performance
echo "📈 Step 3: Updating performance tracker..."
python3 scripts/nhl_performance_tracker.py > "$LOG_DIR/tracker_$TIMESTAMP.log" 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Performance tracked"
else
    echo "   ⚠️  No results to track yet"
fi
echo ""

# Step 4: Optional - Send alerts (if configured)
if [ -n "$NHL_ALERT_WEBHOOK" ]; then
    echo "📬 Step 4: Sending alerts..."
    # TODO: Implement Slack/email alerts
    echo "   ⏸️  Alerts not configured"
else
    echo "📬 Step 4: Alerts not configured (set NHL_ALERT_WEBHOOK)"
fi
echo ""

echo "================================================================================"
echo "✅ Daily run complete at $(date +%H:%M:%S)"
echo "================================================================================"
echo ""

# Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null

