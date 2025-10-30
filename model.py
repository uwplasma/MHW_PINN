# using tensorflow

from phi.tf.flow import *
import numpy as np
import tensorflow as tf
from tensorflow import keras 

# Below, subclassing 'tf.keras.Model' allows access to built-in TensorFlow functionalities
 # (i.e. easy model saving, checkpointing, training workflows, and override methods [like 'train_step'] if custom loop training needed)
 #Adding this @ line because someone said it in a github troubleshooting thread 
@keras.saving.register_keras_serializable()
class MHWNetwork(tf.keras.Model):
    
    '''
  Defines a NN for solving the Modified (2D) Hasegawa-Wakatani (MHW) system of equations.
  Hidden layers = 8 fully connected layers, tanh activations, 20 units each.
  Output layers: phi, zeta, n - each returning a single value per point
  Input: concatenated tensor (x,y,t), reflecting the spatial and temporal dependencies of the plasma dynamics.
    '''

#Changing Neurons to 100
#Trying to remove , **kwargs from init and super init
#Trying to see 20 neurons
    def __init__(self, num_hidden_layers=8, num_neurons=100, **kwargs):
      super(MHWNetwork, self).__init__(**kwargs)
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
    
    #get and from config are here because we're using a custom object. I don't know what they need but they should be here
    #It seems like if anything is not a normal object, it should be serialised in get and de in from
    #So what may need to be serialized
    def get_config(self):
        base_config = super().get_config()
        config = {
            #"sublayer": keras.saving.serialize_keras_object(self.sublayer),
        }
        return {**base_config}

    #@classmethod
    #def from_config(cls, config):
        #sublayer_config = config.pop("sublayer")
        #sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        #return cls(**config)
'''
# Instantiate model
model = MHWNetwork(num_hidden_layers=8, num_neurons=20)

# using torch:

import torch
import torch.nn as nn
import torch.nn.functional as F

class MHWNetwork(nn.Module):
    '''
'''
    Defines a NN for solving the Modified (2D) Hasegawa-Wakatani (MHW) system of equations.
    Hidden layers = 8 fully connected layers, tanh activations, 20 units each.
    Output layers: phi, zeta, n - each returning a single value per point
    Input: tensor of shape (batch_size, 3) corresponding to (x, y, t).
'''
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
'''