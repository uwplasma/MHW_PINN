# using tensorflow:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_predicted_state(model, t_index, title, batch_size=30):
    """
    Visualizes the predicted 2D field at a given timestep using batch processing.

    Parameters:
    - model: Trained PINN model.
    - t_index: Timestep index for visualization.
    - title: Which field to plot ('Phi', 'Zeta', or 'n').
    - batch_size: Number of points to process per batch (default is 30 to match training).
    """
    # Generate a spatial grid
    x_vals = np.linspace(-1, 1, 128)  # Grid in x
    y_vals = np.linspace(-1, 1, 128)  # Grid in y
    X, Y = np.meshgrid(x_vals, y_vals)  # 2D meshgrid

    # Create input tensor (all spatial points at timestep t_index)
    t_fixed = np.full_like(X, fill_value=t_index / 100)  # Normalize time
    inputs = np.stack([X, Y, t_fixed], axis=-1)  # Shape (128, 128, 3)

    # Reshape inputs to (Nx * Ny, 3) for model prediction
    inputs_tensor = tf.convert_to_tensor(inputs.reshape(-1, 3), dtype=tf.float32)

    # Predict in batches
    num_points = inputs_tensor.shape[0]  # Total spatial points
    predictions = []  # Store batch results

    for i in range(0, num_points, batch_size):
        batch_input = inputs_tensor[i:i + batch_size]  # Select batch
        batch_input = tf.expand_dims(batch_input, axis=0)  # Add batch dimension
        phi_batch, zeta_batch, n_batch = model(batch_input)  # Model prediction

        # Store results (convert tensors to NumPy arrays)
        predictions.append((phi_batch.numpy(), zeta_batch.numpy(), n_batch.numpy()))

    # Concatenate all batch results
    phi_all = np.concatenate([p[0] for p in predictions], axis=1).flatten()
    zeta_all = np.concatenate([p[1] for p in predictions], axis=1).flatten()
    n_all = np.concatenate([p[2] for p in predictions], axis=1).flatten()

    # Reshape predictions to grid shape (128, 128)
    phi_grid = phi_all.reshape(128, 128)
    zeta_grid = zeta_all.reshape(128, 128)
    n_grid = n_all.reshape(128, 128)

    # Select the appropriate field
    field_map = {"Phi": phi_grid, "Zeta": zeta_grid, "n": n_grid}
    field = field_map.get(title, phi_grid)  # Default to Phi

    # Plot the field
    plt.figure(figsize=(8, 6))
    plt.imshow(field, origin="upper", cmap="inferno", aspect="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} at Timestep {t_index}")
    plt.show()

# Example: Visualize model predictions at timestep 32
t_index = 32
show_predicted_state(model, t_index, "Phi", batch_size=30)
show_predicted_state(model, t_index, "Zeta", batch_size=30)
show_predicted_state(model, t_index, "n", batch_size=30)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from IPython.display import display, HTML

def animate_predicted_state(model, title, batch_size=30):
    """
    Animates the predicted 2D field over time from the trained model with batch processing.

    Parameters:
    - model: Trained neural network model
    - title: The name of the field to animate ("Phi", "Zeta", or "n")
    - batch_size: Number of points to process per batch

    Returns:
    - ani: The animation object
    """
    # Generate spatial grid
    x_vals = np.linspace(-1, 1, 128)  # 128x128 grid in x
    y_vals = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Number of timesteps
    T = 33
    predicted_fields = np.zeros((128, 128, T))  # Store predicted fields

    for t_index in range(T):
        t_fixed = np.full_like(X, fill_value=t_index / 1000)  # Normalize time
        inputs = np.stack([X, Y, t_fixed], axis=-1)  # Shape (128, 128, 3)
        inputs_tensor = tf.convert_to_tensor(inputs.reshape(-1, 3), dtype=tf.float32)

        # Predict in batches
        num_points = inputs_tensor.shape[0]  # Total spatial points
        predictions = []

        for i in range(0, num_points, batch_size):
            batch_input = inputs_tensor[i:i + batch_size]  # Select batch
            batch_input = tf.expand_dims(batch_input, axis=0)  # Add batch dim
            phi_batch, zeta_batch, n_batch = model(batch_input)  # Model prediction

            # Store batch results
            predictions.append((phi_batch.numpy(), zeta_batch.numpy(), n_batch.numpy()))

        # Concatenate batch results
        phi_all = np.concatenate([p[0] for p in predictions], axis=1).flatten()
        zeta_all = np.concatenate([p[1] for p in predictions], axis=1).flatten()
        n_all = np.concatenate([p[2] for p in predictions], axis=1).flatten()

        # Reshape to (128, 128)
        phi_grid = phi_all.reshape(128, 128)
        zeta_grid = zeta_all.reshape(128, 128)
        n_grid = n_all.reshape(128, 128)

        # Store the selected field
        field_map = {"Phi": phi_grid, "Zeta": zeta_grid, "n": n_grid}
        predicted_fields[:, :, t_index] = field_map.get(title, phi_grid)  # Default to Phi

    # Create the animation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(predicted_fields[:, :, 0], origin='lower', cmap='inferno', aspect='auto')
    ax.set_title(f"{title} at Timestep 0")
    plt.colorbar(im)

    def update(frame):
        im.set_array(predicted_fields[:, :, frame])
        ax.set_title(f"{title} at Timestep {frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=200)

    return ani

# Generate animations for each field
phi_animation = animate_predicted_state(model, "Phi", batch_size=30)
zeta_animation = animate_predicted_state(model, "Zeta", batch_size=30)
n_animation = animate_predicted_state(model, "n", batch_size=30)

# Display animations in Google Colab
display(HTML(phi_animation.to_html5_video()))
display(HTML(zeta_animation.to_html5_video()))
display(HTML(n_animation.to_html5_video()))

# using torch:

import numpy as np
import torch
import matplotlib.pyplot as plt

def show_predicted_state(model, t_index, title, batch_size=30, device='cpu'):
    """
    Visualizes the predicted 2D field at a given timestep using batch processing in PyTorch.

    Parameters:
    - model: Trained PyTorch PINN model.
    - t_index: Timestep index for visualization.
    - title: Which field to plot ('Phi', 'Zeta', or 'n').
    - batch_size: Number of points to process per batch (default is 30 to match training).
    - device: Device on which the model is running ('cpu' or 'cuda').
    """
    model.eval()  # Set model to evaluation mode

    # Generate a spatial grid
    x_vals = np.linspace(-1, 1, 128)
    y_vals = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Create input tensor with fixed time
    t_fixed = np.full_like(X, fill_value=t_index / 33.0)  # Normalize time (match your training setup)
    inputs = np.stack([X, Y, t_fixed], axis=-1)  # Shape: (128, 128, 3)

    # Reshape to (Nx * Ny, 3)
    inputs_tensor = torch.tensor(inputs.reshape(-1, 3), dtype=torch.float32).to(device)

    num_points = inputs_tensor.shape[0]
    predictions_phi, predictions_zeta, predictions_n = [], [], []

    with torch.no_grad():
        for i in range(0, num_points, batch_size):
            batch_input = inputs_tensor[i:i + batch_size]
            batch_input = batch_input.unsqueeze(0)  # Add batch dim: (1, batch_size, 3)

            phi_batch, zeta_batch, n_batch = model(batch_input)  # Forward pass

            predictions_phi.append(phi_batch.squeeze(0).cpu())
            predictions_zeta.append(zeta_batch.squeeze(0).cpu())
            predictions_n.append(n_batch.squeeze(0).cpu())

    # Concatenate and reshape predictions
    phi_grid = torch.cat(predictions_phi, dim=0).numpy().reshape(128, 128)
    zeta_grid = torch.cat(predictions_zeta, dim=0).numpy().reshape(128, 128)
    n_grid = torch.cat(predictions_n, dim=0).numpy().reshape(128, 128)

    field_map = {"Phi": phi_grid, "Zeta": zeta_grid, "n": n_grid}
    field = field_map.get(title, phi_grid)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(field, origin="upper", cmap="inferno", aspect="auto")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{title} at Timestep {t_index}")
    plt.show()

# Example usage (assuming your model is loaded and on the right device):
t_index = 32
show_predicted_state(model, t_index, "Phi", batch_size=30)  # or 'cpu'
show_predicted_state(model, t_index, "Zeta", batch_size=30)
show_predicted_state(model, t_index, "n", batch_size=30)
