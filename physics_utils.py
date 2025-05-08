# using tensorflow:

def calculate_gradients(field, axis, epsilon):
    """ Compute finite differences for gradient using a fourth-order accurate scheme. """
    shifted_forward = tf.roll(field, shift=-1, axis=axis)
    shifted_backward = tf.roll(field, shift=1, axis=axis)
    shifted_forward_2 = tf.roll(field, shift=-2, axis=axis)
    shifted_backward_2 = tf.roll(field, shift=2, axis=axis)
    return (-shifted_forward_2 + 8 * shifted_forward - 8 * shifted_backward + shifted_backward_2) / (12 * epsilon)

def poisson_bracket(f, g, Nx, Ny, epsilon):
    """ Compute the Poisson bracket {f, g} = ∂f/∂x * ∂g/∂y - ∂f/∂y * ∂g/∂x """
    f_x = calculate_gradients(f, axis=0, epsilon=epsilon)
    f_y = calculate_gradients(f, axis=1, epsilon=epsilon)
    g_x = calculate_gradients(g, axis=0, epsilon=epsilon)
    g_y = calculate_gradients(g, axis=1, epsilon=epsilon)
    return f_x * g_y - f_y * g_x

def poisson_solver_fft(zeta, Nx, Ny):
    """ Solve ∇²φ = ζ using FFT. """
    batch_size = tf.shape(zeta)[0]
    kx = tf.fftshift(tf.linspace(-Nx//2, Nx//2 - 1, Nx)) * (np.pi / (Nx // 2))
    ky = tf.fftshift(tf.linspace(-Ny//2, Ny//2 - 1, Ny)) * (np.pi / (Ny // 2))
    KX, KY = tf.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2 = tf.expand_dims(K2, axis=0)
    K2 = tf.tile(K2, [batch_size, 1, 1])
    zeta_hat = tf.signal.fft2d(tf.cast(zeta, tf.complex64))
    K2 = tf.where(K2 == 0, tf.constant(1e-10, dtype=tf.float32), K2)
    phi_hat = zeta_hat / (-K2)
    return tf.math.real(tf.signal.ifft2d(phi_hat))

def grad4(field, x, y, epsilon_x=0.0157, epsilon_y=0.0157):
    """ Compute the fourth-order Laplacian with periodicity enforced. """
    field_x = calculate_gradients(field, axis=0, epsilon=epsilon_x)
    field_y = calculate_gradients(field, axis=1, epsilon=epsilon_y)
    field_xx = calculate_gradients(field_x, axis=0, epsilon=epsilon_x)
    field_yy = calculate_gradients(field_y, axis=1, epsilon=epsilon_y)
    return field_xx + field_yy

def zonal_average(field, axis=1):
    """ Compute the zonal average <f> by averaging over the poloidal (y) direction. """
    return tf.reduce_mean(field, axis=axis, keepdims=True)

def non_zonal_component(field, axis=1):
    """ Compute the nonzonal component of a field: f̃ = f - ⟨f⟩. """
    return field - zonal_average(field, axis=axis)

def boundary_tx(Nx, Ny, batch_size):
    """ Initializes spatial grid and random perturbations for vorticity (ζ) and density (n). """
    x = tf.linspace(-1.0, 1.0, Nx)
    y = tf.linspace(-1.0, 1.0, Ny)
    x_grid, y_grid = tf.meshgrid(x, y, indexing='ij')
    x_grid = tf.tile(tf.expand_dims(x_grid, axis=0), [batch_size, 1, 1])
    y_grid = tf.tile(tf.expand_dims(y_grid, axis=0), [batch_size, 1, 1])
    zeta = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.01)
    n = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.01)
    t = tf.zeros((batch_size, Nx, Ny), dtype=tf.float32)
    phi = poisson_solver_fft(zeta, Nx, Ny)
    return x_grid, y_grid, t, zeta, n, phi

def open_boundary(Nx, Ny, batch_size, epsilon=0.0157):
    """ Initializes boundary conditions for the MHW system. """
    x = tf.linspace(-1.0, 1.0, Nx)
    y = tf.linspace(-1.0, 1.0, Ny)
    x_boundary, y_boundary = tf.meshgrid(x, y, indexing='ij')
    x_boundary = tf.tile(tf.expand_dims(x_boundary, axis=0), [batch_size, 1, 1])
    y_boundary = tf.tile(tf.expand_dims(y_boundary, axis=0), [batch_size, 1, 1])
    zeta_boundary = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.1)
    n_boundary = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.1)
    t_boundary = tf.zeros((batch_size, Nx, Ny), dtype=tf.float32)
    phi_boundary = poisson_bracket(zeta_boundary, n_boundary, Nx, Ny, epsilon)
    return x_boundary, y_boundary, t_boundary, zeta_boundary, n_boundary, phi_boundary

# using torch:

import numpy as np
import torch

def calculate_gradients(field, axis, epsilon):
    shifted_forward = torch.roll(field, shifts=-1, dims=axis)
    shifted_backward = torch.roll(field, shifts=1, dims=axis)
    shifted_forward_2 = torch.roll(field, shifts=-2, dims=axis)
    shifted_backward_2 = torch.roll(field, shifts=2, dims=axis)
    return (-shifted_forward_2 + 8 * shifted_forward - 8 * shifted_backward + shifted_backward_2) / (12 * epsilon)

def poisson_bracket(f, g, dx, dy):
    f_x = calculate_gradients(f, axis=1, epsilon=dx)
    f_y = calculate_gradients(f, axis=2, epsilon=dy)
    g_x = calculate_gradients(g, axis=1, epsilon=dx)
    g_y = calculate_gradients(g, axis=2, epsilon=dy)
    return f_x * g_y - f_y * g_x

def poisson_solver_fft(zeta, Nx, Ny):
    batch_size = zeta.shape[0]
    kx = torch.fft.fftshift(torch.arange(-Nx // 2, Nx // 2, dtype=torch.float32)) * (np.pi / (Nx // 2))
    ky = torch.fft.fftshift(torch.arange(-Ny // 2, Ny // 2, dtype=torch.float32)) * (np.pi / (Ny // 2))
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2 = K2.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, Nx, Ny)
    zeta_hat = torch.fft.fft2(zeta)
    K2 = torch.where(K2 == 0, torch.tensor(1e-10), K2)
    phi_hat = zeta_hat / (-K2)
    return torch.fft.ifft2(phi_hat).real

def grad4(field, dx, dy):
    field_x = calculate_gradients(field, axis=1, epsilon=dx)
    field_y = calculate_gradients(field, axis=2, epsilon=dy)
    field_xx = calculate_gradients(field_x, axis=1, epsilon=dx)
    field_yy = calculate_gradients(field_y, axis=2, epsilon=dy)
    return field_xx + field_yy

def zonal_average(field, axis=1):
    return torch.mean(field, dim=axis, keepdim=True)

def non_zonal_component(field, axis=1):
    return field - zonal_average(field, axis=axis)

def boundary_tx(Nx, Ny, batch_size):
    x = torch.linspace(-1.0, 1.0, Nx)
    y = torch.linspace(-1.0, 1.0, Ny)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    x_grid = x_grid.unsqueeze(0).repeat(batch_size, 1, 1)
    y_grid = y_grid.unsqueeze(0).repeat(batch_size, 1, 1)
    zeta = torch.randn(batch_size, Nx, Ny) * 0.01
    n = torch.randn(batch_size, Nx, Ny) * 0.01
    t = torch.zeros((batch_size, Nx, Ny), dtype=torch.float32)
    phi = poisson_solver_fft(zeta, Nx, Ny)
    return x_grid, y_grid, t, zeta, n, phi

def open_boundary(Nx, Ny, batch_size):
    x = torch.linspace(-1.0, 1.0, Nx)
    y = torch.linspace(-1.0, 1.0, Ny)
    x_boundary, y_boundary = torch.meshgrid(x, y, indexing='ij')
    x_boundary = x_boundary.unsqueeze(0).repeat(batch_size, 1, 1)
    y_boundary = y_boundary.unsqueeze(0).repeat(batch_size, 1, 1)
    zeta_boundary = torch.randn(batch_size, Nx, Ny) * 0.1
    n_boundary = torch.randn(batch_size, Nx, Ny) * 0.1
    t_boundary = torch.zeros((batch_size, Nx, Ny), dtype=torch.float32)
    dx, dy = calculate_grid_spacing(Nx, Ny)
    phi_boundary = poisson_bracket(zeta_boundary, n_boundary, dx, dy)
    return x_boundary, y_boundary, t_boundary, zeta_boundary, n_boundary, phi_boundary
