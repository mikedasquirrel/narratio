#!/bin/bash

# Production Setup Script
# Sets up the narrative optimization system for production deployment

echo "=================================="
echo "PRODUCTION SETUP"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "\nPython version: $python_version"

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize system
echo -e "\nInitializing system..."
python INITIALIZE_SYSTEM.py

# Run health check
echo -e "\nRunning health check..."
python scripts/health_check.py

# Integrate existing domains
echo -e "\nIntegrating existing domains..."
python tools/integrate_existing_domains.py

# Create necessary directories
echo -e "\nCreating directories..."
mkdir -p narrative_optimization/domains
mkdir -p data/domains
mkdir -p exports
mkdir -p logs

# Set permissions
echo -e "\nSetting permissions..."
chmod +x scripts/*.py
chmod +x tools/*.py

# Generate initial reports
echo -e "\nGenerating initial reports..."
python scripts/automated_report.py

echo -e "\n=================================="
echo "PRODUCTION SETUP COMPLETE"
echo "=================================="
echo -e "\nNext steps:"
echo "  1. Start API: python api/api_server.py"
echo "  2. Run demo: python DEMO_COMPLETE_SYSTEM.py"
echo "  3. Add domains: python MASTER_INTEGRATION.py DOMAIN data/domains/DOMAIN.json"
echo "  4. Monitor: python scripts/monitor_performance.py"

