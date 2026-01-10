#!/bin/bash
# Quick setup script for BoC project

set -e

echo "=========================================="
echo "BoC Project Setup"
echo "=========================================="

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download datasets:"
echo "   python scripts/download_datasets.py --dataset coco --output-dir ./data"
echo ""
echo "2. Start training:"
echo "   python main.py train --dataset coco --data-root ./data --config base"
echo ""
echo "To activate the environment later, run:"
echo "   source .venv/bin/activate"
