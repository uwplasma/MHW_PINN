import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML
import tensorflow as tf

# Ensure grid setup is correctly imported
from training.grid_setup import initialize_grid
from models.mhw_network import MHWNetwork

# Define grid parameters
Nx, Ny, T, batch_size = 128, 128, 33, 30

# Initialize grid
inputs, grid_x, grid_y, grid_t = initialize_grid(Nx, Ny, T, batch_size)

# Load trained model
model = tf.keras.models.load_model('pinn_model.h5')

# Get model predictions
phi_pred, zeta_pred, n_pred = model(inputs)

# Convert to NumPy for visualization
phi_grid = phi_pred.numpy()
zeta_grid = zeta_pred.numpy()
n_grid = n_pred.numpy()

# Visualization function for static timesteps
def show_state_at_timestep(a, t_index, batch_index=0, title="Field"):
    a = np.squeeze(a)  # Remove singleton dimensions
    if a.ndim == 4:
        if batch_index >= a.shape[0] or t_index >= a.shape[3]:
            print(f"Error: Index out of range.")
            return
        a_at_t = a[batch_index, :, :, t_index]  # (Nx, Ny)
    elif a.ndim == 3:
        if t_index >= a.shape[2]:
            print(f"Error: Timestep {t_index} is out of range.")
            return
        a_at_t = a[:, :, t_index]  # (Nx, Ny)
    else:
        print("Invalid shape for visualization.")
        return

    plt.figure(figsize=(8, 6))
    im = plt.imshow(a_at_t, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(im)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{title} at Timestep {t_index} (Batch {batch_index})")
    plt.show()

# Static field visualization
t_index = 32
batch_index = 0
show_state_at_timestep(phi_grid, t_index, batch_index, "Phi Field")
show_state_at_timestep(zeta_grid, t_index, batch_index, "Zeta Field")
show_state_at_timestep(n_grid, t_index, batch_index, "n Field")

# Animation function for time evolution
def animate_field(a, title, batch_index=0):
    a = np.squeeze(a)
    if a.ndim == 4:
        if batch_index >= a.shape[0]:
            print("Error: Batch index out of range.")
            return None
        a = a[batch_index, :, :, :]
    if a.ndim != 3:
        print("Invalid shape for animation.")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(a[:, :, 0], origin='lower', cmap='inferno', aspect='auto')
    ax.set_title(f"{title} at Timestep 0")
    plt.colorbar(im)

    def update(frame):
        im.set_array(a[:, :, frame])
        ax.set_title(f"{title} at Timestep {frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=a.shape[2], interval=200)
    return ani

# Generate animations
phi_animation = animate_field(phi_grid, "Phi Field Evolution", batch_index)
zeta_animation = animate_field(zeta_grid, "Zeta Field Evolution", batch_index)
n_animation = animate_field(n_grid, "n Field Evolution", batch_index)

# Display animations in Jupyter Notebook
display(HTML(phi_animation.to_html5_video()))
display(HTML(zeta_animation.to_html5_video()))
display(HTML(n_animation.to_html5_video()))
