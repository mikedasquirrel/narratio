#!/bin/bash
# Setup script for Free Will vs Determinism Narrative Analysis

echo "=========================================="
echo "Free Will Analysis Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install sentence-transformers>=2.2.0 torch>=2.0.0 spacy>=3.7.0 scipy>=1.11.0 networkx>=3.1
echo ""

# Download spaCy models
echo "Downloading spaCy models..."
echo "  - en_core_web_sm (small, fast model)..."
python3 -m spacy download en_core_web_sm

echo ""
echo "Optional: Download larger transformer model for better accuracy:"
echo "  python3 -m spacy download en_core_web_trf"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    print('✓ sentence-transformers installed')
except ImportError:
    print('✗ sentence-transformers NOT installed')

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print('✓ spaCy model loaded')
except:
    print('✗ spaCy model NOT loaded')

try:
    import networkx
    print('✓ networkx installed')
except ImportError:
    print('✗ networkx NOT installed')

try:
    import scipy
    print('✓ scipy installed')
except ImportError:
    print('✗ scipy NOT installed')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run example: python narrative_optimization/examples/free_will_analysis_example.py"
echo "  2. Read docs: narrative_optimization/docs/FREE_WILL_ANALYSIS.md"
echo ""

