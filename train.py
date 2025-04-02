import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from models.mhw_network import MHWNetwork
from loss.physics_loss import MHW_physics_loss
from training.grid_setup import initialize_grid  # Import grid setup function

import time
import tensorflow as tf

# Create model instance
model = MHWNetwork()

# Optimizer with Exponential Decay
initial_LR = 0.005
decay_steps = 2000
decay_rate = 0.98
LR_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_LR,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_schedule)

# Fixed batch size
batch_size = 30

# Training parameters
ITERS = 10000
Nx, Ny = 128, 128  # Grid resolution
num_physics_points = Nx * Ny // 4  # Increase physics sample size

# Loss weights
lambda_bc = 1.0
lambda_ph = 10.0

# Start timer
start = time.time()
loss_history = []

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # Boundary conditions
        x_bc, y_bc, t_bc, zeta_bc, n_bc, phi_bc = open_boundary(Nx, Ny, batch_size)
        x_bc, y_bc, t_bc = [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in [x_bc, y_bc, t_bc]]
        phi_bc, zeta_bc, n_bc = [tf.convert_to_tensor(arr, dtype=tf.float32) for arr in [phi_bc, zeta_bc, n_bc]]
        phi_bc, zeta_bc, n_bc = [tf.expand_dims(arr, axis=-1) for arr in [phi_bc, zeta_bc, n_bc]]
        inputs_bc = tf.stack([x_bc, y_bc, t_bc], axis=-1)

        phi_pred, zeta_pred, n_pred = model(inputs_bc)
        loss_phi_bc = tf.reduce_mean(tf.square(phi_pred - phi_bc))
        loss_zeta_bc = tf.reduce_mean(tf.square(zeta_pred - zeta_bc))
        loss_n_bc = tf.reduce_mean(tf.square(n_pred - n_bc))
        loss_bnd = loss_phi_bc + loss_zeta_bc + loss_n_bc

        # Interior physics points
        x_ph = tf.random.uniform([batch_size, num_physics_points], -1, 1, dtype=tf.float32)
        y_ph = tf.random.uniform([batch_size, num_physics_points], -1, 1, dtype=tf.float32)
        t_ph = tf.random.uniform([batch_size, num_physics_points], 0, 1, dtype=tf.float32)
        inputs_ph = tf.stack([x_ph, y_ph, t_ph], axis=-1)

        phi_pred_inner, zeta_pred_inner, n_pred_inner = model(inputs_ph)
        loss_ph_phi, loss_ph_n = MHW_physics_loss(phi_pred_inner, zeta_pred_inner, n_pred_inner, x_ph, y_ph, t_ph, Nx, Ny)
        loss_ph = loss_ph_phi + loss_ph_n

        # Total loss
        total_loss = lambda_bc * loss_bnd + lambda_ph * tf.reduce_mean(loss_ph)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else grad for grad in gradients]
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

    return total_loss

for optim_step in range(ITERS + 1):
    # Run a single training step and get loss
    total_loss = train_step()

    # Log training progress
    if optim_step < 3 or optim_step % 1000 == 0:
        print(f"Step {optim_step}, Loss: {total_loss.numpy():.6f}")

# Save final model
model.save('pinn_model_final.h5')

# Save final model
model.save('pinn_model_final.h5')
print(f"Runtime: {time.time() - start:.2f}s")
