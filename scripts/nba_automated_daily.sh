#!/bin/bash
#
# NBA Automated Daily Betting Predictions
# =========================================
#
# Runs daily at 9:00 AM EST to generate betting predictions.
# Fetches today's games, odds, and generates high-confidence picks.
#
# Setup cron job:
#   crontab -e
#   0 9 * * * /path/to/nba_automated_daily.sh >> /path/to/logs/nba_daily.log 2>&1
#
# Author: AI Coding Assistant
# Date: November 16, 2025
#

# Configuration
PROJECT_DIR="/Users/michaelsmerconish/Desktop/RandomCode/novelization"
PYTHON="python3"
LOG_DIR="$PROJECT_DIR/logs/betting"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Log file for today
DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/nba_daily_$DATE.log"

echo "========================================" >> "$LOG_FILE"
echo "NBA Daily Predictions - $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

cd "$PROJECT_DIR" || exit 1

# Step 1: Fetch today's games and odds
echo "" >> "$LOG_FILE"
echo "[1/2] Fetching today's games and odds..." >> "$LOG_FILE"
$PYTHON scripts/nba_fetch_today.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Data fetch complete" >> "$LOG_FILE"
else
    echo "✗ Data fetch failed" >> "$LOG_FILE"
    exit 1
fi

# Step 2: Generate predictions
echo "" >> "$LOG_FILE"
echo "[2/2] Generating predictions..." >> "$LOG_FILE"
$PYTHON scripts/nba_daily_predictions.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Predictions complete" >> "$LOG_FILE"
else
    echo "✗ Predictions failed" >> "$LOG_FILE"
    exit 1
fi

# Summary
echo "" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Daily run complete - $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Optional: Send notification (uncomment to enable)
# EMAIL="your@email.com"
# PRED_FILE="$PROJECT_DIR/data/predictions/nba_daily_$DATE.json"
# if [ -f "$PRED_FILE" ]; then
#     N_BETS=$(jq '.n_high_confidence_bets' "$PRED_FILE")
#     if [ "$N_BETS" -gt 0 ]; then
#         echo "NBA: $N_BETS high-confidence bets today" | mail -s "NBA Betting Alert" "$EMAIL"
#     fi
# fi

echo "✓ Automated run complete"
exit 0

