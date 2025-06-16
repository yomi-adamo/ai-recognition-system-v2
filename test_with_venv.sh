#!/bin/bash

# Test script that activates virtual environment first
echo "🚀 Testing Phase 2 & Phase 3 with Virtual Environment"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found. Please create it first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if activation was successful
if [ "$VIRTUAL_ENV" == "" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"
echo ""

# Run the main test
echo "🧪 Running Phase 2 & Phase 3 tests..."
python test_phase2_phase3.py

# Deactivate virtual environment
deactivate
echo ""
echo "✅ Test complete - virtual environment deactivated"