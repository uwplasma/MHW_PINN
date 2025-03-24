import numpy as np
import tensorflow as tf

def initialize_grid(Nx, Ny, T, batch_size):
    """
    Creates a 3D space-time grid and prepares it for batch processing.

    Args:
        Nx, Ny: Grid points in x and y directions
        T: Number of time steps
        batch_size: Number of samples per batch

    Returns:
        inputs: Tensor of shape (batch_size, Nx, Ny, T, 3)
        grid_x, grid_y, grid_t: Coordinate tensors
    """
    grids_xy_t = np.meshgrid(
        np.linspace(-1, 1, Nx), 
        np.linspace(-1, 1, Ny), 
        np.linspace(0, 1, T), 
        indexing='ij'
    )

    grid_x, grid_y, grid_t = [tf.convert_to_tensor(t, dtype=tf.float32) for t in grids_xy_t]

    inputs = tf.stack([grid_x, grid_y, grid_t], axis=-1)  # Shape: (Nx, Ny, T, 3)
    inputs = tf.expand_dims(inputs, axis=0)  # Shape: (1, Nx, Ny, T, 3)
    inputs = tf.tile(inputs, [batch_size, 1, 1, 1, 1])  # Shape: (batch_size, Nx, Ny, T, 3)

    return inputs, grid_x, grid_y, grid_t
