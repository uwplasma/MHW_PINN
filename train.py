import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.mhw_network import MHWNetwork
from loss.physics_loss import MHW_physics_loss
from training.grid_setup import initialize_grid  # Import grid setup function

# Define grid dimensions
Nx, Ny, T, batch_size = 128, 128, 33, 30

# Initialize grid
inputs, grid_x, grid_y, grid_t = initialize_grid(Nx, Ny, T, batch_size)

# Initialize the model
model = MHWNetwork(num_hidden_layers=8, num_neurons=20)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Training loop
ITERS = 10000
start = time.time()
loss_history = []

for optim_step in range(ITERS + 1):
    with tf.GradientTape() as tape:
        phi_pred, zeta_pred, n_pred = model(inputs)  # Forward pass
        loss_zeta, loss_n = MHW_physics_loss(phi_pred, zeta_pred, n_pred, grid_x, grid_y, grid_t, Nx, Ny)
        total_loss = loss_zeta + loss_n
        loss_history.append(total_loss.numpy())
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if optim_step < 3 or optim_step % 1000 == 0:
        print(f"Step {optim_step}, Loss: {total_loss.numpy():.6f}")

end = time.time()
print(f"Runtime: {end - start:.2f}s")

# Save trained model
model.save('pinn_model.h5')

# Plot loss convergence
plt.figure(figsize=(8, 6))
plt.plot(range(len(loss_history)), loss_history, label="Total Loss", color="b")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss Convergence During Training")
plt.legend()
plt.grid(True)
plt.show()
