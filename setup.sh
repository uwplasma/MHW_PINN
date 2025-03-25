#!/bin/bash

# Clone the GitHub repository
git clone https://github.com/yourusername/MHW_PINN.git

# Change to the project directory
cd MHW_PINN

# Create necessary directories
mkdir data models loss training visualizations

# Create necessary files
touch models/mhw_network.py loss/physics_loss.py training/grid_setup.py training/grid_visual.py training/train.py results/animate.py requirements.txt README.md

chmod +x setup.sh

./setup.sh
