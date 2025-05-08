## pre-training grid visualization

# google colab code 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML

def animate_field(a, title, batch_index=0):
    """
    Create an animation for the time evolution of a field.

    Parameters:
    - a (np.ndarray): 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) array representing the field.
    - title (str): Title for the animation.
    - batch_index (int, optional): Index of the batch sample to visualize (default=0).

    Returns:
    - ani (animation.FuncAnimation): The animation object.
    """
    a = np.squeeze(a)  # Remove singleton dimensions
    print(f"Debug: Shape of input array a: {a.shape}")  # Debugging info

    if a.ndim == 4:
        if batch_index >= a.shape[0]:
            print(f"Error: Batch index {batch_index} is out of range. Max batch index is {a.shape[0] - 1}")
            return None
        a = a[batch_index, :, :, :]  # Extract the batch

    if a.ndim != 3:
        print("Error: Expected a 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) array for animation.")
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

def show_animation_in_colab(ani):
    """ Display the animation in Google Colab """
    video_html = ani.to_html5_video()
    display(HTML(video_html))

# Example Usage: Generate animations for standard and non-zonal fields
batch_index = 0

# Assuming phi_grid, tilde_phi, n_grid, tilde_n are already available as the field outputs
phi_animation = animate_field(phi_grid, "Phi Field Evolution", batch_index)
tilde_phi_animation = animate_field(tilde_phi, "Non-Zonal Phi Field Evolution", batch_index)


n_animation = animate_field(n_grid, "n Field Evolution", batch_index)
tilde_n_animation = animate_field(tilde_n, "Non-Zonal n Field Evolution", batch_index)

# zeta grid
zeta_animation = animate_field(zeta_grid, "Zeta Field Evolution", batch_index)

# Display animations in Google Colab
if phi_animation:
    show_animation_in_colab(phi_animation)
if tilde_phi_animation:
    show_animation_in_colab(tilde_phi_animation)

if n_animation:
    show_animation_in_colab(n_animation)
if tilde_n_animation:
    show_animation_in_colab(tilde_n_animation)

# using torch:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display

def animate_field(a, title, batch_index=0):
    """
    Create an animation for the time evolution of a field.

    Parameters:
    - a (np.ndarray or torch.Tensor): 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) array representing the field.
    - title (str): Title for the animation.
    - batch_index (int, optional): Index of the batch sample to visualize (default=0).

    Returns:
    - ani (animation.FuncAnimation): The animation object.
    """
    # If 'a' is a PyTorch tensor, detach it and convert to a NumPy array
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()  # Detach tensor, move to CPU if necessary, and convert to NumPy

    a = np.squeeze(a)  # Remove singleton dimensions
    print(f"Debug: Shape of input array a: {a.shape}")  # Debugging info

    if a.ndim == 4:
        if batch_index >= a.shape[0]:
            print(f"Error: Batch index {batch_index} is out of range. Max batch index is {a.shape[0] - 1}")
            return None
        a = a[batch_index, :, :, :]  # Extract the batch

    if a.ndim != 3:
        print("Error: Expected a 3D (Nx, Ny, T) or 4D (batch_size, Nx, Ny, T) array for animation.")
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

# Example Usage: Generate animations for standard and non-zonal fields
batch_index = 0

# Assuming phi_grid, tilde_phi, n_grid, tilde_n, and zeta_grid are already available as the field outputs
phi_animation = animate_field(phi_grid, "Phi Field Evolution", batch_index)
tilde_phi_animation = animate_field(tilde_phi, "Non-Zonal Phi Field Evolution", batch_index)
n_animation = animate_field(n_grid, "n Field Evolution", batch_index)
tilde_n_animation = animate_field(tilde_n, "Non-Zonal n Field Evolution", batch_index)
zeta_animation = animate_field(zeta_grid, "Zeta Field Evolution", batch_index)

from IPython.display import HTML

# Display animations in Jupyter Notebook
if phi_animation:
    display(HTML(phi_animation.to_jshtml()))
if tilde_phi_animation:
    display(HTML(tilde_phi_animation.to_jshtml()))
if n_animation:
    display(HTML(n_animation.to_jshtml()))
if tilde_n_animation:
    display(HTML(tilde_n_animation.to_jshtml()))
if zeta_animation:
    display(HTML(zeta_animation.to_jshtml()))
