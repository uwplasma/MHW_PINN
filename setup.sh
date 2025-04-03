#!/bin/bash

echo "Setting up the PINN for the 2D MHW system..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p data  # Only if you have data files

# Set up virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "No requirements.txt found. Please install dependencies manually."
fi

echo "Setup complete!"
