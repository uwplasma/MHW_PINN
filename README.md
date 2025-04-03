# Physics-Informed Neural Network for the 2D Modified Hasegawa-Wakatani System

## Overview
This repository contains a Physics-Informed Neural Network (PINN) implementation to solve the 2D Modified Hasegawa-Wakatani (MHW) system, which describes plasma turbulence in stellarators. The model enforces periodic boundary conditions and focuses on solving for the non-zonal components of the system.

## Project Structure
```
├── checkpoints/        # Stores model checkpoints
├── logs/              # Training logs
├── data/              # (Optional) Data files
├── model.py           # Defines the MHWNetwork class
├── physics_utils.py   # Contains physics-based computations (gradients, Poisson solver, etc.)
├── grid_utils.py      # Grid setup utilities
├── physics_loss.py    # PINN loss function based on MHW equations
├── grid_visual.py     # Visualization utilities for grid data
├── train.py           # Training script for the PINN
├── loss_converge.py   # Plots the total loss over the training steps
├── predicted_state.py # Plots and animates the predicted state for each grid after across each training step
├── troubleshoot.py    # Plots to visualize results and compare to expectations - are we seeing the behavior we expect, and if not, what areas of the code need to be targeted
├── requirements.txt   # Dependencies
├── setup.sh           # Setup script for environment
└── README.md          # This file
```

## Installation and Setup
### 1. Clone the repository
```bash
git clone <https://github.com/katiejoslyn/MHW_PINN>
cd <MHW_PINN>
```

### 2. Run the setup script
```bash
bash setup.sh
```
This script:
- Creates necessary directories (`checkpoints/`, `logs/`, `data/` if needed)
- Sets up a Python virtual environment (`venv/`)
- Installs dependencies from `requirements.txt`

### 3. Activate the virtual environment
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

## Model Details
The `MHWNetwork` class (defined in `model.py`) is a fully connected neural network with:
- **8 hidden layers** with **20 neurons each**, using `tanh` activations
- **Three output layers** for solving `phi`, `zeta`, and `n`
- Inputs are `(x, y, t)`, capturing spatial and temporal dependencies

## Training the Model
To start training:
```bash
python train.py
```
