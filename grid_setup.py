import numpy as np
import tensorflow as tf

def initialize_grid(Nx, Ny, T, batch_size):
    """
    Initializes a 2D spatial-temporal grid for the MHW system.

    Parameters:
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        T (int): Number of time steps.
        batch_size (int): Size of the batch for training.

    Returns:
        inputs (tf.Tensor): Tensor of shape (batch_size, Nx, Ny, T, 3) representing the grid.
        grid_x, grid_y, grid_t (tf.Tensor): Individual tensors for x, y, and t coordinates.
    """
    grids_xy_t = np.meshgrid(np.linspace(-1, 1, Nx), 
                             np.linspace(-1, 1, Ny), 
                             np.linspace(0, 1, T), 
                             indexing='ij')

    # Convert to tensors
    grid_x, grid_y, grid_t = [tf.convert_to_tensor(t, dtype=tf.float32) for t in grids_xy_t]

    # Stack to create a single tensor of shape (Nx, Ny, T, 3)
    inputs = tf.stack([grid_x, grid_y, grid_t], axis=-1)

    # Expand dimensions to match batch format
    inputs = tf.expand_dims(inputs, axis=0)

    # Duplicate inputs to create a batch
    inputs = tf.tile(inputs, [batch_size, 1, 1, 1, 1])

    return inputs, grid_x, grid_y, grid_t
