def calculate_grid_spacing(Nx, Ny):
    """
    Calculate uniform grid spacing for a domain spanning [-1, 1] in x and y directions.
    """
    x_spacing = 2.0 / (Nx - 1)  # Assuming the domain is [-1, 1]
    y_spacing = 2.0 / (Ny - 1)
    return x_spacing, y_spacing

def create_grid(Nx=128, Ny=128, T=33, batch_size=30):
    """
    Creates a 3D space-time grid and prepares it for batch processing.

    Args:
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        T (int): Number of time steps
        batch_size (int): Number of samples per batch

    Returns:
        inputs (tf.Tensor): Tensor of shape (batch_size, Nx, Ny, T, 3)
        grid_x, grid_y, grid_t (tf.Tensor): Coordinate tensors for visualization
    """
    # Create meshgrid for (x, y, t)
    grids_xy_t = np.meshgrid(
        np.linspace(-1, 1, Nx),  # X-coordinates
        np.linspace(-1, 1, Ny),  # Y-coordinates
        np.linspace(0, 1, T),    # Time steps
        indexing='ij'
    )

    # Convert to TensorFlow tensors
    grid_x, grid_y, grid_t = [tf.convert_to_tensor(t, dtype=tf.float32) for t in grids_xy_t]

    # Stack into single input tensor (Nx, Ny, T, 3)
    inputs = tf.stack([grid_x, grid_y, grid_t], axis=-1) 

    # Expand dimensions to match batch format
    inputs = tf.expand_dims(inputs, axis=0)  # (1, Nx, Ny, T, 3)

    # Duplicate for batch size
    inputs = tf.tile(inputs, [batch_size, 1, 1, 1, 1])  # (batch_size, Nx, Ny, T, 3)

    return inputs, grid_x, grid_y, grid_t


# Below, we also need to include some definitions to successfully compute the necessary calucations for the 2D (modified) Hasegawa-Wakatani Simulation,
# including computations for the gradient, a poisson bracket and fourth-order laplacian.
def calculate_gradients(field, axis, epsilon):
    """
    Compute finite differences for gradient of a field with respect to a given axis.
    Uses a second-order central difference scheme with periodic boundaries.
    """
    shifted_forward = tf.roll(field, shift=-1, axis=axis) # Moves field one step forward (f(x+e))
    shifted_backward = tf.roll(field, shift=1, axis=axis) # Moves field one step backward (f(x-e))
    shifted_forward_2 = tf.roll(field, shift=-2, axis=axis) # Moves field two steps forward (f(x+2e))
    shifted_backward_2 = tf.roll(field, shift=2, axis=axis) # Movies field two steps backward (f(x-2e)

    # Second-order central difference formula = fourth-order accurate approximation of the gradient,
    # which reduces numerical errors compared to a standart two-point shceme
    # (with a standard finite differnence calucation [(f(x+e)-f(x-e))/e], simply indexing the boundary on a period grid would yield an error,
    # so tf.roll ensures that values from the opposite side of the array are used when we shift elements)
    return (-shifted_forward_2 + 8 * shifted_forward - 8 * shifted_backward + shifted_backward_2) / (12 * epsilon)

def poisson_bracket(f, g, Nx, Ny, epsilon):
    """
    Compute the Poisson bracket {f, g} = ∂f/∂x * ∂g/∂y - ∂f/∂y * ∂g/∂x
    with periodicity enforced, using finite differences for derivatives.

    Parameters:
    f: Tensor, the first field
    g: Tensor, the second field
    Nx: int, number of grid points in the x-direction
    Ny: int, number of grid points in the y-direction
    epsilon: float, the grid spacing in both directions

    Returns:
    Poisson bracket result as a Tensor
    """
     # Calculate gradients for f and g along both x and y axes
    f_x = calculate_gradients(f, axis=0, epsilon=epsilon)
    f_y = calculate_gradients(f, axis=1, epsilon=epsilon)
    g_x = calculate_gradients(g, axis=0, epsilon=epsilon)
    g_y = calculate_gradients(g, axis=1, epsilon=epsilon)

    # Compute the Poisson bracket {f, g} = ∂f/∂x * ∂g/∂y - ∂f/∂y * ∂g/∂x
    poisson_bracket_result = f_x * g_y - f_y * g_x

    return poisson_bracket_result

def poisson_solver_fft(zeta, Nx, Ny):
    """
    Solve ∇²φ = ζ using FFT for batched inputs.

    Parameters:
        zeta: Tensor of shape (batch_size, Nx, Ny), representing vorticity.
        Nx, Ny: Grid resolution.

    Returns:
        phi: Tensor of shape (batch_size, Nx, Ny), the electrostatic potential.
    """
    batch_size = tf.shape(zeta)[0]  # Get batch size dynamically

    # Define wavenumbers
    kx = tf.fftshift(tf.linspace(-Nx//2, Nx//2 - 1, Nx)) * (np.pi / (Nx // 2))
    ky = tf.fftshift(tf.linspace(-Ny//2, Ny//2 - 1, Ny)) * (np.pi / (Ny // 2))

    # Create 2D meshgrid of wavenumbers
    KX, KY = tf.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2  # Squared wavenumbers

    # Expand K2 to match batch size
    K2 = tf.expand_dims(K2, axis=0)  # Shape: (1, Nx, Ny)
    K2 = tf.tile(K2, [batch_size, 1, 1])  # Shape: (batch_size, Nx, Ny)

    # Compute FFT of vorticity (ζ)
    zeta_hat = tf.signal.fft2d(tf.cast(zeta, tf.complex64))

    # Avoid division by zero
    K2 = tf.where(K2 == 0, tf.constant(1e-10, dtype=tf.float32), K2)

    # Solve for phi in Fourier space
    phi_hat = zeta_hat / (-K2)

    # Inverse FFT to get phi in real space
    phi = tf.math.real(tf.signal.ifft2d(phi_hat))

    return phi

def grad4(field, x, y, epsilon_x=0.0157, epsilon_y=0.0157):
    """
    Compute the fourth-order Laplacian with periodicity enforced.
    """
    # First derivative with respect to x and y
    field_x = calculate_gradients(field, axis=0, epsilon=epsilon_x)
    field_y = calculate_gradients(field, axis=1, epsilon=epsilon_y)

    # Second derivative (Laplacian) with respect to x and y
    field_xx = calculate_gradients(field_x, axis=0, epsilon=epsilon_x)
    field_yy = calculate_gradients(field_y, axis=1, epsilon=epsilon_y)

    return field_xx + field_yy

# We need to define zonal and non zonal averages for the MHW since when restricted to 2D, the HW system does not contain zonal flows [Hakim].
def zonal_average(field, axis=1):
  """
  Compute the zonal average <f> by averaging over the poloidal (y) direction.

  Args:
    field: 2D tensor shape (Nx, Ny) representing f(x,y)

  Returns:
    1D tensor of shape (Nx,) with the zonal mean <f>
  """
  return tf.reduce_mean(field, axis=axis, keepdims=True)

def non_zonal_component(field, axis=1):
  """
    Compute the nonzonal component of a field: f̃ = f - ⟨f⟩.

    Args:
        field: 2D tensor of shape (Nx, Ny).

    Returns:
        2D tensor with nonzonal part f̃.
    """
  return field - zonal_average(field, axis=axis)


def boundary_tx(Nx, Ny, batch_size):
  '''
  Initalizes spatial grid and random perturbations for the vorticity (ζ) and perturbed number density (n)
  in the MHW system
  '''

  x = tf.linspace(-1.0, 1.0, Nx)
  y = tf.linspace(-1.0, 1.0, Ny)
  x_grid, y_grid = tf.meshgrid(x, y, indexing='ij')

  # Expand to match batch size
  x_grid = tf.tile(tf.expand_dims(x_grid, axis=0), [batch_size, 1, 1])
  y_grid = tf.tile(tf.expand_dims(y_grid, axis=0), [batch_size, 1, 1])

  # Initialize random perturbations
  zeta = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.01)
  n = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.01)

  # Time tensor (batched)
  t = tf.zeros((batch_size, Nx, Ny), dtype=tf.float32)

  # Compute phi for each batch
  phi = poisson_solver_fft(zeta, Nx, Ny)

  return x_grid, y_grid, t, zeta, n, phi


def open_boundary(Nx, Ny, batch_size, epsilon=0.0157):  # Add epsilon as an argument (finite difference spacing)
    # Create grid points for x and y; 2D meshgrid
    # x and y range from [-1,1] defining a normalized computational domain
    x = tf.linspace(-1.0, 1.0, Nx)
    y = tf.linspace(-1.0, 1.0, Ny)
    x_boundary, y_boundary = tf.meshgrid(x, y, indexing='ij')

    # Expand to match batch size
    x_boundary = tf.tile(tf.expand_dims(x_boundary, axis=0), [batch_size, 1, 1])
    y_boundary = tf.tile(tf.expand_dims(y_boundary, axis=0), [batch_size, 1, 1])

    # Initialize boundary fields with random perturbations
    zeta_boundary = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.1)
    n_boundary = tf.random.normal(shape=(batch_size, Nx, Ny), mean=0.0, stddev=0.1)

    # Time tensor (batched)
    t_boundary = tf.zeros((batch_size, Nx, Ny), dtype=tf.float32)

    # Compute phi for each batch
    phi_boundary = poisson_bracket(zeta_boundary, n_boundary, Nx, Ny, epsilon)

    return x_boundary, y_boundary, t_boundary, zeta_boundary, n_boundary, phi_boundary

def MHW_physics_loss(phi, zeta, n, x, y, t, Nx, Ny, alpha=0.5, kappa=1.0, mu=1.0, epsilon=0.0157):
    """
    Physics-based loss function for the MHW System in 2D.
    Now supports batch processing.

    Parameters:
        phi, zeta, n: Tensor fields of shape (batch_size, Nx, Ny)
        x, y, t: Spatial and temporal coordinates
        Nx, Ny: Grid dimensions
        alpha, kappa, mu: Physical parameters
        epsilon: Small value for numerical stability

    Returns:
        loss_zeta, loss_n: Scalar loss values computed from residuals
    """

    # Poisson brackets (computed separately for each batch)
    zeta_PB = poisson_bracket(phi, zeta, Nx, Ny, epsilon=epsilon)  # Shape: (batch_size, Nx, Ny)
    n_PB = poisson_bracket(phi, n, Nx, Ny, epsilon=epsilon)  # Shape: (batch_size, Nx, Ny)

    # Fourth-order Laplacians (batch-wise computation)
    grad4_zeta = grad4(zeta, x, y, epsilon_x=epsilon, epsilon_y=epsilon)  # (batch_size, Nx, Ny)
    grad4_n = grad4(n, x, y, epsilon_x=epsilon, epsilon_y=epsilon)  # (batch_size, Nx, Ny)

    # Compute non-zonal components
    tilde_phi = non_zonal_component(phi, axis=1)  # Extract along y-direction (batch_size, Nx, Ny)
    tilde_n = non_zonal_component(n, axis=1)  # Extract along y-direction (batch_size, Nx, Ny)

    # Compute ∂φ/∂y term (gradient along y-axis)
    grad_phi_y = calculate_gradients(phi, axis=1, epsilon=epsilon)  # (batch_size, Nx, Ny)

    # Compute residuals for the MHW equations
    loss_zeta_unnorm = zeta_PB - alpha * (tilde_phi - tilde_n) + mu * grad4_zeta
    loss_n_unnorm = n_PB - alpha * (tilde_phi - tilde_n) + kappa * grad_phi_y + mu * grad4_n

    # Compute L2 norm of residuals over the entire batch
    loss_zeta = tf.sqrt(tf.reduce_mean(tf.square(loss_zeta_unnorm)))
    loss_n = tf.sqrt(tf.reduce_mean(tf.square(loss_n_unnorm)))

    return loss_zeta, loss_n
