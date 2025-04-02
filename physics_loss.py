import tensorflow as tf

def MHW_physics_loss(phi, zeta, n, x, y, t, Nx, Ny, alpha=0.5, kappa=1.0, mu=1.0, epsilon=0.0157):
    """
    Physics-based loss function for the MHW System in 2D with non-zonal components.

    Parameters:
        phi, zeta, n: Tensor fields of shape (batch_size, Nx, Ny)
        x, y, t: Spatial and temporal coordinates
        Nx, Ny: Grid dimensions
        alpha, kappa, mu: Physical parameters
        epsilon: Small value for numerical stability

    Returns:
        loss_zeta, loss_n: Scalar loss values computed from residuals
    """

    from utils import non_zonal_component, poisson_bracket, grad4, calculate_gradients

    # Compute non-zonal components
    tilde_phi = non_zonal_component(phi, axis=1)  # (batch_size, Nx, Ny)
    tilde_n = non_zonal_component(n, axis=1)  # (batch_size, Nx, Ny)

    # Poisson brackets using non-zonal components
    zeta_PB = poisson_bracket(tilde_phi, zeta, Nx, Ny, epsilon=epsilon)  # (batch_size, Nx, Ny)
    n_PB = poisson_bracket(tilde_phi, tilde_n, Nx, Ny, epsilon=epsilon)  # (batch_size, Nx, Ny)

    # Fourth-order Laplacians (batch-wise computation)
    grad4_zeta = grad4(zeta, x, y, epsilon_x=epsilon, epsilon_y=epsilon)  # (batch_size, Nx, Ny)
    grad4_n = grad4(tilde_n, x, y, epsilon_x=epsilon, epsilon_y=epsilon)  # (batch_size, Nx, Ny)

    # Compute ∂tilde{φ}/∂y term (gradient along y-axis)
    grad_tilde_phi_y = calculate_gradients(tilde_phi, axis=1, epsilon=epsilon)  # (batch_size, Nx, Ny)

    # Compute residuals for the MHW equations
    loss_zeta_unnorm = zeta_PB - alpha * (tilde_phi - tilde_n) + mu * grad4_zeta
    loss_n_unnorm = n_PB - alpha * (tilde_phi - tilde_n) + kappa * grad_tilde_phi_y + mu * grad4_n

    # Compute L2 norm of residuals over the entire batch
    loss_zeta = tf.sqrt(tf.reduce_mean(tf.square(loss_zeta_unnorm)))
    loss_n = tf.sqrt(tf.reduce_mean(tf.square(loss_n_unnorm)))

    return loss_zeta, loss_n
