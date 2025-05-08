# using tensorflow

from phi.tf.flow import *
import numpy as np

# Below, subclassing 'tf.keras.Model' allows access to built-in TensorFlow functionalities
 # (i.e. easy model saving, checkpointing, training workflows, and override methods [like 'train_step'] if custom loop training needed)
class MHWNetwork(tf.keras.Model):
  '''
  Defines a NN for solving the Modified (2D) Hasegawa-Wakatani (MHW) system of equations.
  Hidden layers = 8 fully connected layers, tanh activations, 20 units each.
  Output layers: phi, zeta, n - each returning a single value per point
  Input: concatenated tensor (x,y,t), reflecting the spatial and temporal dependencies of the plasma dynamics.
  '''
  def __init__(self, num_hidden_layers=8, num_neurons=20):
      super(MHWNetwork, self).__init__()
      self.hidden_layers = [tf.keras.layers.Dense(num_neurons, activation=tf.nn.tanh) for _ in range(num_hidden_layers)]
      self.phi_output = tf.keras.layers.Dense(1)  # Output layer for phi
      self.zeta_output = tf.keras.layers.Dense(1)  # Output layer for zeta
      self.n_output = tf.keras.layers.Dense(1)  # Output layer for n

  def call(self, inputs):
      """
      Forward pass for batch inputs: expects shape (batch_size, 3), where 3 corresponds to (x, y, t)
      """
      x = inputs  # Shape: (batch_size, 3)
      for layer in self.hidden_layers:
          x = layer(x)  # Each layer maintains batch dimension
      phi_output = self.phi_output(x)
      zeta_output = self.zeta_output(x)
      n_output = self.n_output(x)


      return phi_output, zeta_output, n_output

# Instantiate model
model = MHWNetwork(num_hidden_layers=8, num_neurons=20)

# using torch:

import torch
import torch.nn as nn
import torch.nn.functional as F

class MHWNetwork(nn.Module):
    '''
    Defines a NN for solving the Modified (2D) Hasegawa-Wakatani (MHW) system of equations.
    Hidden layers = 8 fully connected layers, tanh activations, 20 units each.
    Output layers: phi, zeta, n - each returning a single value per point
    Input: tensor of shape (batch_size, 3) corresponding to (x, y, t).
    '''
    def __init__(self, num_hidden_layers=8, num_neurons=20):
        super(MHWNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(3 if i == 0 else num_neurons, num_neurons) for i in range(num_hidden_layers)
        ])
        self.phi_output = nn.Linear(num_neurons, 1)
        self.zeta_output = nn.Linear(num_neurons, 1)
        self.n_output = nn.Linear(num_neurons, 1)

    def forward(self, inputs):
        """
        Forward pass for batch inputs: expects shape (batch_size, 3), where 3 corresponds to (x, y, t)
        """
        x = inputs
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        phi_output = self.phi_output(x)
        zeta_output = self.zeta_output(x)
        n_output = self.n_output(x)
        return phi_output, zeta_output, n_output

# Instantiate model
model = MHWNetwork(num_hidden_layers=8, num_neurons=20)
