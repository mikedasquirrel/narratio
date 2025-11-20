#!/bin/bash
# Quick fix script for bus errors caused by corrupted numpy/scikit-learn binaries

set -e

echo "=========================================="
echo "Fixing Bus Error - Reinstalling Packages"
echo "=========================================="
echo

# Get the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
VENV_PATH="$PROJECT_ROOT/.venv"

echo "Project root: $PROJECT_ROOT"
echo "Virtual env: $VENV_PATH"
echo

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Please create it first: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check Python version
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"
echo

# Uninstall problematic packages
echo "Step 1: Uninstalling problematic packages..."
pip uninstall -y numpy scikit-learn pandas scipy 2>/dev/null || true

# Clear pip cache
echo "Step 2: Clearing pip cache..."
pip cache purge 2>/dev/null || true

# Upgrade pip
echo "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet

# Reinstall packages with no cache
echo "Step 4: Reinstalling packages (this may take a few minutes)..."
pip install --no-cache-dir numpy>=1.24.0
pip install --no-cache-dir scikit-learn>=1.3.0
pip install --no-cache-dir pandas>=2.0.0

# Verify installation
echo
echo "Step 5: Verifying installation..."
python3 -c "import numpy; print(f'✓ NumPy {numpy.__version__} OK')" || { echo "✗ NumPy import failed"; exit 1; }
python3 -c "import sklearn; print(f'✓ scikit-learn {sklearn.__version__} OK')" || { echo "✗ sklearn import failed"; exit 1; }
python3 -c "import pandas; print(f'✓ Pandas {pandas.__version__} OK')" || { echo "✗ Pandas import failed"; exit 1; }

echo
echo "=========================================="
echo "✓ Fix completed successfully!"
echo "=========================================="
echo
echo "You can now run:"
echo "  python3 scripts/diagnose_imports.py"
echo "  python3 scripts/test_minimal.py"
echo

