#!/bin/bash
# Complete Movie Analysis Pipeline
# Merges datasets then runs all transformers

echo "======================================================================"
echo "COMPLETE MOVIE ANALYSIS PIPELINE"
echo "======================================================================"
echo ""

# Step 1: Merge datasets
echo "STEP 1: Merging movie datasets..."
echo "----------------------------------------------------------------------"
python3 scripts/merge_movie_datasets.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Dataset merge failed. Aborting."
    exit 1
fi

echo ""
echo "STEP 2: Running all transformers..."
echo "----------------------------------------------------------------------"
python3 run_all_transformers_movies.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Transformer analysis failed."
    exit 1
fi

echo ""
echo "======================================================================"
echo "ANALYSIS COMPLETE"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - data/domains/movies_merged_complete.json (merged dataset)"
echo "  - movie_transformer_results.json (transformer results)"
echo "  - movie_transformer_progress.log (detailed log)"
echo "  - movie_transformer_progress.json (progress tracker)"
echo ""
echo "View results:"
echo "  cat movie_transformer_results.json | python3 -m json.tool | less"
echo ""

