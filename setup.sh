#!/bin/bash

# Clone the GitHub repository
git clone https://github.com/yourusername/MHW_PINN.git

# Change to the project directory
cd MHW_PINN

# Create necessary directories
mkdir data models loss training visualizations

# Create necessary files
touch models/mhw_network.py loss/physics_loss.py training/train.py visualizations/loss_plot.py visualizations/animate.py requirements.txt README.md

chmod +x setup.sh

./setup.sh
