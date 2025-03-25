import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML

def show_state_at_timestep(a, t_index, batch_index=0, title="Field"):
    """
    Display a 2D spatial field at a given timestep.
    
    Parameters:
    - a: 4D numpy array (batch_size, Nx, Ny, T) or 3D (Nx, Ny, T)
    - t_index: Timestep to visualize
    - batch_index: Index of the batch sample to visualize (default=0)
    - title: Title for the plot
    """
    a = np.squeeze(a)  # Remove singleton dimensions
    
    if a.ndim == 4:
        if batch_index >= a.shape[0]:
            print(f"Error: Batch index {batch_index} is out of range. Max batch index is {a.shape[0] - 1}")
            return
        if t_index >= a.shape[3]:
            print(f"Error: Timestep {t_index} is out of range. The number of timesteps is {a.shape[3]}")
            return
        a_at_t = a[batch_index, :, :, t_index]
    
    elif a.ndim == 3:
        if t_index >= a.shape[2]:
            print(f"Error: Timestep {t_index} is out of range. The number of timesteps is {a.shape[2]}")
            return
        a_at_t = a[:, :, t_index]
    
    else:
        print("Expected a 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) input for plotting.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(a_at_t, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(im)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{title} at Timestep {t_index} (Batch {batch_index})")
    plt.tight_layout()
    plt.show()

def animate_field(a, title, batch_index=0):
    """
    Create an animation for the time evolution of a field.
    
    Parameters:
    - a: 4D numpy array (batch_size, Nx, Ny, T) or 3D (Nx, Ny, T)
    - title: Title for the animation
    - batch_index: Index of the batch sample to visualize (default=0)
    
    Returns:
    - ani: The animation object (to be displayed in Jupyter)
    """
    a = np.squeeze(a)
    
    if a.ndim == 4:
        if batch_index >= a.shape[0]:
            print(f"Error: Batch index {batch_index} is out of range. Max batch index is {a.shape[0] - 1}")
            return None
        a = a[batch_index, :, :, :]
    
    if a.ndim != 3:
        print("Expected a 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) input for animation.")
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

# Example usage (ensure you replace these with the actual arrays from your simulations)
t_index = 32  # Choose the timestep
batch_index = 0  # Choose which batch sample to visualize

show_state_at_timestep(tilde_phi_grid, t_index, batch_index, "Non-zonal Phi Field")
show_state_at_timestep(tilde_n_grid, t_index, batch_index, "Non-zonal n Field")

phi_animation = animate_field(tilde_phi_grid, "Non-zonal Phi Field Evolution", batch_index)
n_animation = animate_field(tilde_n_grid, "Non-zonal n Field Evolution", batch_index)

display(HTML(phi_animation.to_html5_video()))
display(HTML(n_animation.to_html5_video()))
