import tensorflow as tf
import numpy as np

# For both tensorflow and torch:
def calculate_grid_spacing(Nx, Ny):
    """
    Calculate uniform grid spacing for a domain spanning [-1, 1] in x and y directions.
    """
    x_spacing = 2.0 / (Nx - 1)
    y_spacing = 2.0 / (Ny - 1)
    return x_spacing, y_spacing

# using tensorflow:
def create_grid(Nx=128, Ny=128, T=33, batch_size=30):
    """
    Creates a 3D space-time grid and prepares it for batch processing.
    """
    grids_xy_t = np.meshgrid(
        np.linspace(-1, 1, Nx),
        np.linspace(-1, 1, Ny),
        np.linspace(0, 1, T),
        indexing='ij'
    )
    grid_x, grid_y, grid_t = [tf.convert_to_tensor(t, dtype=tf.float32) for t in grids_xy_t]
    inputs = tf.stack([grid_x, grid_y, grid_t], axis=-1)
    inputs = tf.expand_dims(inputs, axis=0)
    inputs = tf.tile(inputs, [batch_size, 1, 1, 1, 1])
    return inputs, grid_x, grid_y, grid_t

# using torch:
def create_grid(Nx=128, Ny=128, T=33, batch_size=30):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    t = np.linspace(0, 1, T)
    grids_xy_t = np.meshgrid(x, y, t, indexing='ij')
    grid_x, grid_y, grid_t = [torch.tensor(g, dtype=torch.float32) for g in grids_xy_t]

    inputs = torch.stack([grid_x, grid_y, grid_t], dim=-1)  # (Nx, Ny, T, 3)
    inputs = inputs.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # (batch, Nx, Ny, T, 3)

    dx, dy = calculate_grid_spacing(Nx, Ny)
    return inputs, grid_x, grid_y, grid_t, dx, dy

