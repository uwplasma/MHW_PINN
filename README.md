# Physics-Informed Neural Network for the 2D Modified Hasegawa-Wakatani System

## Overview
This repository aims to create a Physics-Informed Neural Network (PINN) to solve the 2D Modified Hasegawa-Wakatani (MHW) system of equations to understand how PINNS and AI can be used to aid in the prediction of plasma turbulence, speed up numerical simulations, and/or be incorporated into stellarator optimization studies.

## Project Structure
```
├── MHW_PINN_pt.ipynb/ # Jupyter Notebook code using PyTorch rather than TensorFlow
├── MHW_PINN_tf.ipynb/ # Original google colab code using TensorFlow
├── checkpoints/       # Stores model checkpoints
├── logs/              # Training logs
├── data/              # (Optional) Data files
├── model.py           # Defines the MHWNetwork class using both TensorFlow and PyTorch
├── physics_utils.py   # Contains physics-based computations (gradients, Poisson solver, etc.)
├── grid_utils.py      # Grid setup utilities using both TensorFlow and PyTorch
├── physics_loss.py    # PINN loss function based on MHW equations using both TensorFlow and PyTorch
├── grid_visual.py     # Visualization utilities for pre-trained grid data using both TensorFlow and PyTorch
├── train.py           # Training script for the PINN using both TensorFlow and PyTorch
├── loss_converge.py   # Plots the total loss over the training steps (same for TensorFlow and Pytorch, only requires matplotlib.pyplot)
├── predicted_state.py # Plots and animates the predicted state for each grid post-training across each training step for both TensorFlow and PyTorch
├── troubleshoot.py    # Plots to visualize results and compare to expectations - are we seeing the behavior we expect, and if not, what areas of the code need to be targeted (currently only included in the Google Colab code - i.e., the code utilizing TensorFlow, however none of these definitions require the use of TensorFlow so should work for both codes. 
├── requirements.txt   # Dependencies, including those for both TensorFlow and PyTorch
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
