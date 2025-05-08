from grid_utils import create_grid
from physics_utils import non_zonal_component
from model import MHWNetwork  # Assuming you have a model definition in model.py

# Define grid dimensions
Nx = 128  # Number of grid points in the x-direction
Ny = 128  # Number of grid points in the y-direction
T = 33    # Number of time steps
batch_size = 30  # Fixed batch size

# Create a meshgrid for 2D spatial domain and time (spatial: [-1, 1], time: [0, 1])
inputs, grid_x, grid_y, grid_t = create_grid(Nx=Nx, Ny=Ny, T=T, batch_size=batch_size)

# Initialize the model
model = MHWNetwork(num_hidden_layers=8, num_neurons=20)

# Forward pass through the model (batch processing)
phi_output, zeta_output, n_output = model(inputs)  # Outputs: (batch_size, Nx, Ny, T)

# Ensure outputs match expected shape
phi_grid = phi_output  # Shape: (batch_size, Nx, Ny, T)
zeta_grid = zeta_output  # Shape: (batch_size, Nx, Ny, T)
n_grid = n_output  # Shape: (batch_size, Nx, Ny, T)

# Compute non-zonal components before passing to loss function
tilde_phi = non_zonal_component(phi_grid, axis=2)  # Filtering along y-axis
tilde_n = non_zonal_component(n_grid, axis=2)  # Filtering along y-axis
