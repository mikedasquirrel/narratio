#!/bin/bash
# Monitor transformer progress until complete

echo "Monitoring transformer progress..."
echo "Press Ctrl+C to stop monitoring (analysis will continue in background)"
echo ""

while true; do
    clear
    python3 check_progress.py
    
    # Check if complete
    if grep -q '"status": "complete"' movie_transformer_progress.json 2>/dev/null; then
        echo ""
        echo "✓ ANALYSIS COMPLETE!"
        echo ""
        echo "Results saved to: movie_transformer_results.json"
        break
    fi
    
    echo ""
    echo "Checking again in 30 seconds..."
    sleep 30
done

