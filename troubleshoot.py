from scipy.fftpack import fft2, ifft2, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def compute_vorticity_batch(phi_predictions, dx=1.0/128):
    """
    Computes vorticity ω = ∇²φ for an entire batch of timesteps using FFT-based Laplacian.
    """
    T, Nx, Ny = phi_predictions.shape  # Get shape (timesteps, Nx, Ny)
    kx = fftfreq(Nx, d=dx) * 2.0 * np.pi
    ky = fftfreq(Ny, d=dx) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # Compute Laplacian in Fourier space
    laplacian_operator = -(KX**2 + KY**2)

    # Compute FFT for all timesteps at once
    phi_hat = fft2(phi_predictions, axes=(1, 2))

    # Apply Laplacian in Fourier space
    laplacian_phi_hat = laplacian_operator * phi_hat

    # Compute vorticity using inverse FFT
    vorticity = np.real(ifft2(laplacian_phi_hat, axes=(1, 2)))

    return vorticity

# Compute vorticity for all timesteps at once
vorticity_predictions = compute_vorticity_batch(phi_predictions)

# Plot vorticity at a specific timestep
t_index = 32
plt.figure(figsize=(6, 5))
plt.imshow(vorticity_predictions[t_index], origin='upper', cmap="inferno", extent=[-1, 1, -1, 1])
plt.colorbar(label="Vorticity ω")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Vorticity Field at Timestep {t_index}")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def compute_velocity_field_batch(phi_predictions, dx=1.0/128):
    """
    Computes velocity field (u, v) for an entire batch of timesteps.
    """
    T, Nx, Ny = phi_predictions.shape  # Get shape (timesteps, Nx, Ny)
    dy = dx  # Assuming dx = dy

    # Compute velocity components: u = -∂φ/∂y, v = ∂φ/∂x
    u = -np.gradient(phi_predictions, axis=2) / dy  # Negative y-derivative
    v = np.gradient(phi_predictions, axis=1) / dx   # Positive x-derivative

    return u, v

def plot_velocity_field(phi_grid, u, v, dx=1.0/128):
    """
    Plots velocity streamlines overlaid on phi.
    """
    Nx, Ny = phi_grid.shape
    x_vals = np.linspace(-1, 1, Nx)
    y_vals = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, u, v, color='white', linewidth=0.5)
    plt.imshow(phi_grid, origin='upper', cmap="inferno", extent=[-1, 1, -1, 1])
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity Streamlines Overlaid on Phi")
    plt.show()

# Compute velocity fields for all timesteps
u_predictions, v_predictions = compute_velocity_field_batch(phi_predictions)

# Plot velocity field at a chosen timestep
t_index = 32
plot_velocity_field(phi_predictions[t_index], u_predictions[t_index], v_predictions[t_index])

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

def compute_power_spectrum_batch(phi_predictions):
    """
    Computes the 2D power spectrum for an entire batch of phi fields over time.
    """
    T, Nx, Ny = phi_predictions.shape  # Get shape (timesteps, Nx, Ny)

    # Compute power spectrum for all timesteps
    phi_hat = np.abs(fft2(phi_predictions, axes=(1, 2)))**2  # Compute power spectrum
    phi_hat = fftshift(phi_hat, axes=(1, 2))  # Center k=0

    return phi_hat  # Shape: (T, Nx, Ny)

def plot_power_spectrum(phi_hat_t):
    """
    Plots the 2D power spectrum for a single timestep.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(phi_hat_t + 1e-10), origin='lower', cmap='inferno', extent=[-64, 64, -64, 64])
    plt.colorbar(label="log10 Power")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.title("Power Spectrum of Phi")
    plt.show()

# Compute power spectrum for all timesteps
phi_power_spectrum = compute_power_spectrum_batch(phi_predictions)

# Plot power spectrum at a chosen timestep
t_index = 32
plot_power_spectrum(phi_power_spectrum[t_index])

import numpy as np
import matplotlib.pyplot as plt

def compute_cross_correlation_batch(phi_predictions, n_predictions):
    """
    Computes the cross-correlation between phi and n for all timesteps in a batch.
    """
    # Compute mean and std along the spatial dimensions (1,2) for each timestep
    phi_mean = np.mean(phi_predictions, axis=(1, 2), keepdims=True)
    n_mean = np.mean(n_predictions, axis=(1, 2), keepdims=True)

    phi_std = np.std(phi_predictions, axis=(1, 2), keepdims=True)
    n_std = np.std(n_predictions, axis=(1, 2), keepdims=True)

    # Compute cross-correlation for all timesteps at once
    correlation = np.mean((phi_predictions - phi_mean) * (n_predictions - n_mean), axis=(1, 2)) / (phi_std.squeeze() * n_std.squeeze())

    return correlation  # Shape: (T,)

# Compute correlation for all timesteps at once
correlations = compute_cross_correlation_batch(phi_predictions, n_predictions)

# Plot cross-correlation over time
plt.figure(figsize=(6, 4))
plt.plot(range(timesteps), correlations, marker='o', linestyle='-', color='b')
plt.xlabel("Time")
plt.ylabel("Cross-correlation ⟨φ, n⟩")
plt.title("Cross-Correlation Between Phi and n Over Time")
plt.grid()
plt.show()
