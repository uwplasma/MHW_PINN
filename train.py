# tensorflow:

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

# torch:

import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Create model instance
model = MHWNetwork()  # Assuming you have the equivalent PyTorch class defined


# Optimizer with Exponential Decay
initial_LR = 0.005
decay_steps = 2000
decay_rate = 0.95
optimizer = optim.Adam(model.parameters(), lr=initial_LR)
lr_scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=decay_rate
)



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

# Training loop
for optim_step in range(ITERS + 1):
    model.train()
    
    # Boundary conditions
    x_bc, y_bc, t_bc, zeta_bc, n_bc, phi_bc = open_boundary(Nx, Ny, batch_size)
    x_bc, y_bc, t_bc = [torch.tensor(arr, dtype=torch.float32) for arr in [x_bc, y_bc, t_bc]]
    phi_bc, zeta_bc, n_bc = [torch.tensor(arr, dtype=torch.float32) for arr in [phi_bc, zeta_bc, n_bc]]
    phi_bc, zeta_bc, n_bc = [arr.unsqueeze(-1) for arr in [phi_bc, zeta_bc, n_bc]]
    inputs_bc = torch.stack([x_bc, y_bc, t_bc], dim=-1)

    # Forward pass for boundary conditions
    phi_pred, zeta_pred, n_pred = model(inputs_bc)
    loss_phi_bc = F.mse_loss(phi_pred, phi_bc)
    loss_zeta_bc = F.mse_loss(zeta_pred, zeta_bc)
    loss_n_bc = F.mse_loss(n_pred, n_bc)
    loss_bnd = loss_phi_bc + loss_zeta_bc + loss_n_bc

    # Interior physics points
    x_ph = torch.rand([batch_size, num_physics_points], dtype=torch.float32) * 2 - 1
    y_ph = torch.rand([batch_size, num_physics_points], dtype=torch.float32) * 2 - 1
    t_ph = torch.rand([batch_size, num_physics_points], dtype=torch.float32)
    inputs_ph = torch.stack([x_ph, y_ph, t_ph], dim=-1)

    # Forward pass for physics points
    phi_pred_inner, zeta_pred_inner, n_pred_inner = model(inputs_ph)
    loss_ph_phi, loss_ph_n = MHW_physics_loss(phi_pred_inner, zeta_pred_inner, n_pred_inner, x_ph, y_ph, t_ph, Nx, Ny)
    loss_ph = loss_ph_phi + loss_ph_n

    # Total loss
    total_loss = lambda_bc * loss_bnd + lambda_ph * loss_ph.mean()

    # Zero gradients, backprop, optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Apply learning rate decay
    lr_scheduler.step()

    # Log training progress
    if optim_step < 3 or optim_step % 1000 == 0:
        print(f"Step {optim_step}, Loss: {total_loss.item():.6f}")
        loss_history.append(total_loss.item())  # Save loss for further analysis

    # Save model checkpoint every 1000 steps
    if optim_step % 1000 == 0:
        torch.save(model.state_dict(), f"pinn_model_checkpoint_{optim_step}.pt")

# Save the final model
torch.save(model.state_dict(), 'pinn_model_final.pt')

# Final runtime output
print(f"Runtime: {time.time() - start:.2f}s")
