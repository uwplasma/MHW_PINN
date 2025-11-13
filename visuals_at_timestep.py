import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import MHWNetwork 
from keras.utils import custom_object_scope
from tensorflow.keras.models import load_model

#This format should get rid of the weird DType Policy error
#with custom_object_scope({'MHWNetwork': MHWNetwork, 'DTypePolicy': tf.keras.mixed_precision.Policy}):
#    model = load_model('pinn_model_final.h5')
#I may not need custon objects here but I still don't know what's up
#model = load_model('pinn_model_final_n100_10_14_2025.h5', custom_objects={'MHWNetwork' : MHWNetwork})
model = load_model('pinn_model_final_n100_11_6_2025.keras')

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
    t_fixed = np.full_like(X, fill_value=t_index / 33)  # Normalize time (adjust based on your time grid)
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
    phi_all = np.concatenate([p[0] for p in predictions], axis=0).flatten()
    zeta_all = np.concatenate([p[1] for p in predictions], axis=0).flatten()
    n_all = np.concatenate([p[2] for p in predictions], axis=0).flatten()

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
    plt.savefig('pinn_11_13_2025'+title)

# Example: Visualize model predictions at timestep 32
t_index = 32
show_predicted_state(model, t_index, "Phi", batch_size=32)
show_predicted_state(model, t_index, "Zeta", batch_size=32)
show_predicted_state(model, t_index, "n", batch_size=32)
