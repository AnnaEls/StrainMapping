import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def build_gaussian_basis(
    H: int,
    W: int,
    sigmas: List[float],
    stride: int = 1,
    device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Build a fully vectorized multi-scale Gaussian basis over an image grid.

    For each spatial center (sampled on a regular grid with the given stride)
    and each sigma value, a 2D Gaussian of the form

        exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))

    is generated.

    Parameters
    ----------
    H : int
        Image height.
    W : int
        Image width.
    sigmas : list of float
        Gaussian standard deviations (in pixels).
    stride : int, optional
        Spacing between Gaussian centers in pixels (default: 1).
    device : str or torch.device, optional
        Device on which tensors are allocated.

    Returns
    -------
    basis : torch.Tensor
        Tensor of shape (N_basis, H, W), where
        N_basis = N_centers * len(sigmas).
    centers : torch.Tensor
        Tensor of shape (N_centers, 2) containing (y, x) center coordinates.
    sigmas : list of float
        Sigma values used to construct the basis.
    """

    # ------------------------------------------------------------
    # Image coordinate grid: (H, W)
    # ------------------------------------------------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    # ------------------------------------------------------------
    # Gaussian centers sampled on a grid
    # ------------------------------------------------------------
    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, W, stride, device=device)

    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([cy.flatten(), cx.flatten()], dim=1)  # (N_centers, 2)
    N_centers = centers.shape[0]

    # ------------------------------------------------------------
    # Reshape tensors for full broadcasting
    #
    # yy, xx          -> (1, 1, H, W)
    # centers (cy,cx) -> (N_centers, 1, 1, 1)
    # sigmas          -> (1, N_sigmas, 1, 1)
    # ------------------------------------------------------------
    yy = yy.view(1, 1, H, W)
    xx = xx.view(1, 1, H, W)

    cy = centers[:, 0].view(N_centers, 1, 1, 1)
    cx = centers[:, 1].view(N_centers, 1, 1, 1)

    sigmas_t = torch.tensor(sigmas, device=device).view(1, -1, 1, 1)
    sigma2 = 2.0 * sigmas_t ** 2

    # ------------------------------------------------------------
    # Squared distance and Gaussian evaluation
    # Resulting shape: (N_centers, N_sigmas, H, W)
    # ------------------------------------------------------------
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    basis = torch.exp(-dist2 / sigma2)

    # ------------------------------------------------------------
    # Flatten (center, sigma) dimensions
    # Final shape: (N_basis, H, W)
    # ------------------------------------------------------------
    basis = basis.reshape(-1, H, W)

    return basis, centers, sigmas


def build_ideal_image(
    w_raw: torch.nn.Parameter,
    basis: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct an image as a weighted sum of Gaussian basis functions.

    This function computes a linear combination of fixed Gaussian basis
    elements using learnable weights. Conceptually, it evaluates:

        I(y, x) = sum_k w_k * g_k(y, x)

    Parameters
    ----------
    w_raw : torch.nn.Parameter
        Learnable weights of shape (N_basis,).
        These coefficients control the contribution of each Gaussian basis
        function to the reconstructed image.
    basis : torch.Tensor
        Fixed Gaussian basis of shape (N_basis, H, W), where each slice
        basis[k] is a 2D Gaussian centered at a specific location and scale.

    Returns
    -------
    I_ideal : torch.Tensor
        Reconstructed image of shape (H, W).
    """

    # Efficient weighted sum over all basis elements
    # "n,nxy->xy" contracts the basis index n
    I_ideal = torch.einsum("n,nxy->xy", w_raw, basis)

    return I_ideal

def loss_fn(I_pred: torch.Tensor, I_target: torch.Tensor, w_raw: torch.nn.Parameter):
    """
    Compute the loss function.

    Parameters
    ----------
    I_pred : torch.Tensor

    I_target : torch.Tensor

    w_raw : torch.nn.Parameter
        Unconstrained weights

    Returns
    -------
    loss : torch.Tensor
    """

    loss = torch.sqrt(torch.mean((I_pred - I_target)**2))
    return loss


def train(I_target: torch.Tensor, 
          sigmas: List[float], 
          stride:int=1, 
          lr: float = 0.001, 
          n_iters: int = 100,
          plot:bool = False,
          plot_it:int=1):
  
  """
    Train a linear Gaussian basis image model by optimizing per-basis amplitudes.

    This function approximates a 2D target image as a weighted sum of fixed,
    spatially distributed Gaussian basis functions. The Gaussian centers and
    widths (sigmas) are fixed and determined by `build_gaussian_basis`, while
    the amplitudes of each basis function are learned via gradient-based
    optimization.

    The optimization minimizes a user-defined loss between the reconstructed
    image and the target image, optionally including regularization on the
    Gaussian amplitudes.

    Parameters
    ----------
    I_target : torch.Tensor
        2D target image of shape (H, W). The image is normalized internally
        by its maximum value and moved to the selected device (CPU or CUDA).

    sigmas : List[float]
        List of Gaussian standard deviations (in pixels) used to construct
        the multi-scale Gaussian basis. Each sigma generates a full grid
        of Gaussian functions across the image.

    stride : int, optional
        Spatial stride (in pixels) between neighboring Gaussian centers.
        Smaller values yield a denser basis and higher reconstruction
        fidelity at the cost of increased computation. Default is 1.

    lr : float, optional
        Learning rate for the Adam optimizer used to optimize Gaussian
        amplitudes. Default is 1e-3.

    n_iters : int, optional
        Number of optimization iterations. Default is 100.

    plot : bool, optional
        If True, intermediate reconstruction results and residuals are
        visualized during training. Default is False.

    plot_it : int, optional
        Plotting interval (in iterations) when `plot=True`. Default is 1.

    Returns
    -------
    basis : torch.Tensor
        Gaussian basis tensor of shape (N_basis, H, W), where each slice
        corresponds to a fixed Gaussian function.

    centers : torch.Tensor or np.ndarray
        Array of Gaussian center coordinates with shape (N_basis, 2),
        typically given as (x, y) pixel locations.

    sigmas : torch.Tensor or np.ndarray
        Per-basis Gaussian standard deviations corresponding to `basis`
        and `centers`.

    w_raw : torch.nn.Parameter
        Learnable amplitude parameters of shape (N_basis,). These weights
        define the contribution of each Gaussian basis function to the
        reconstructed image.

    See Also
    --------
    build_gaussian_basis : Constructs the fixed Gaussian basis functions.
    build_ideal_image : Reconstructs an image from Gaussian basis weights.
    loss_fn : Computes the reconstruction loss and optional regularization.
    """
  H, W = I_target.shape
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Build Gaussian basis 
  basis, centers, sigmas = build_gaussian_basis(
      H=H,
      W=W,
      sigmas=sigmas,
      stride=stride,
      device=device,
    )

  # Normalize target image
  I_target = I_target / I_target.max()
  I_target = torch.from_numpy(I_target).float().to(device)

  N_basis = basis.shape[0]
  w_raw = torch.nn.Parameter(0.01 * torch.randn(N_basis, device=basis.device))

  optimizer = torch.optim.Adam([w_raw], lr=lr)

  for it in range(n_iters):
      optimizer.zero_grad()

      I_pred = build_ideal_image(w_raw, basis)
      loss = loss_fn(I_pred, I_target, w_raw)

      loss.backward()
      optimizer.step()

      if plot and (it+1) % plot_it == 0:
        print(f"Iter {it+1:04d} | Loss {loss.item():.6f}")

        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1)
        plt.imshow(I_target.cpu(), cmap="gray")
        plt.title("Target")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(I_pred.detach().cpu(), cmap="gray")
        plt.title("Gaussian reconstruction")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow((I_pred - I_target).detach().cpu(), cmap="seismic")
        plt.title("Residual")
        plt.colorbar()
        plt.axis("off")

        plt.show() 
        
  return basis, centers, sigmas, w_raw 


@torch.no_grad()
def extract_gaussians_from_basis(
    w_raw: torch.nn.Parameter,
    centers: torch.Tensor,
    sigmas: list,
    H: int,
    W: int,
    peak_kernel: int = 5,
    threshold_rel: float = 0.5,
):
    """
    Extract Gaussian primitives from a learned multi-sigma Gaussian basis model.

    This function interprets a learned Gaussian basis expansion by aggregating
    per-center weights across multiple predefined sigma values, constructing
    a spatial atom-strength map, detecting salient local maxima via non-maximum
    suppression, and estimating effective Gaussian parameters for each detected
    peak.

     Parameters
    ----------
    w_raw : torch.nn.Parameter
        Raw learned basis weights of shape ``[N_sigma * N_centers]``.
        Weights are passed through ``softplus`` to ensure non-negativity.
        Gradients are not tracked inside this function.

    centers : torch.Tensor
        Tensor of shape ``[N_centers, 2]`` containing integer-valued
        ``(y, x)`` pixel coordinates corresponding to basis centers.
        Coordinates must satisfy ``0 <= y < H`` and ``0 <= x < W``.

    sigmas : list of float
        List of Gaussian standard deviations used in the basis.
        Length defines ``N_sigma`` and must be consistent with ``w_raw``.

    H : int
        Height of the image grid.

    W : int
        Width of the image grid.

    peak_kernel : int, optional
        Size of the square window used for non-maximum suppression.
        Must be an odd integer. Default is 5.

    threshold_rel : float, optional
        Relative threshold applied to the atom-strength map prior to
        peak detection. Peaks with value less than
        ``threshold_rel * atom_map.max()`` are suppressed.
        Default is 0.1.

    Returns
    -------
    gaussians : list of dict
        List of detected Gaussian primitives. Each dictionary contains:
            - ``x`` : float
                x-coordinate of the Gaussian center (column index).
            - ``y`` : float
                y-coordinate of the Gaussian center (row index).
            - ``sigma`` : float
                Effective Gaussian standard deviation, computed as a
                weight-averaged sigma across basis components.
            - ``amplitude`` : float
                Aggregate amplitude (sum of basis weights) at this center.

    atom_map : torch.Tensor
        Tensor of shape ``[H, W]`` containing the aggregated atom-strength
        map formed by summing basis weights across sigmas at each center
        location.

    peaks : torch.Tensor
        Boolean tensor of shape ``[H, W]`` indicating detected local maxima
        after thresholding and non-maximum suppression.

    """

    assert peak_kernel % 2 == 1, "peak_kernel must be odd"

    device = w_raw.device
    sigmas_t = torch.tensor(sigmas, device=device)

    # --- reshape weights ---
    w = F.softplus(w_raw)
    n_sigma = len(sigmas)
    n_centers = centers.shape[0]

    assert w.numel() == n_sigma * n_centers

    w_map = w.view(n_sigma, n_centers)          # [N_sigma, N_centers]
    w_center = w_map.sum(dim=0)                 # [N_centers]

    # --- build atom strength map ---
    ys = centers[:, 0].long()
    xs = centers[:, 1].long()

    atom_map = torch.zeros((H, W), device=device)
    atom_map[ys, xs] = w_center

    # --- threshold ---
    thr = threshold_rel * atom_map.max()
    mask = atom_map > thr

    # --- non-maximum suppression ---
    pooled = F.max_pool2d(
        atom_map[None, None],
        kernel_size=peak_kernel,
        stride=1,
        padding=peak_kernel // 2
    )[0, 0]

    peaks = (atom_map == pooled) & mask

    # --- center index lookup table ---
    index_map = -torch.ones((H, W), dtype=torch.long, device=device)
    index_map[ys, xs] = torch.arange(n_centers, device=device)

    # --- extract peak indices ---
    py, px = torch.where(peaks)
    idx = index_map[py, px]

    valid = idx >= 0
    if not valid.any():
        return [], atom_map, peaks

    py = py[valid]
    px = px[valid]
    idx = idx[valid]

    # --- batch Gaussian parameter estimation ---
    w_sigma = w_map[:, idx]                     # [N_sigma, N_peaks]
    amp = w_sigma.sum(dim=0)                    # [N_peaks]

    valid = amp > 0
    if not valid.any():
        return [], atom_map, peaks

    w_sigma = w_sigma[:, valid]
    amp = amp[valid]
    py = py[valid]
    px = px[valid]

    sigma = (w_sigma * sigmas_t[:, None]).sum(dim=0) / amp

    # --- convert to Python objects ---
    gaussians = [
        dict(
            x=float(x),
            y=float(y),
            sigma=float(s),
            amplitude=float(a),
        )
        for x, y, s, a in zip(px, py, sigma, amp)
    ]

    return gaussians, atom_map, peaks


def plot_gaussian_centers(
    image: np.ndarray | torch.Tensor,
    gaussians: list,
    title: str = "Detected Gaussians",
    plot_sigma: bool = False,
    sigma_scale: float = 0.5,     # multiply sigma for visibility
    edge_color: str = "cyan",
    line_width: float = 1.5,
):
    """
    Visualize detected Gaussian centers overlaid on an image.

    This function displays a grayscale image and overlays the locations of
    detected Gaussian primitives. Each Gaussian is represented by a marker
    at its center, and optionally by a circle whose radius corresponds to
    the estimated Gaussian standard deviation.

    Parameters
    ----------
    image : numpy.ndarray or torch.Tensor
        Input image of shape ``[H, W]`` or ``[H, W, C]``. If a torch tensor
        is provided, it is detached from the computation graph and moved
        to the CPU before visualization.

    gaussians : list of dict
        List of Gaussian descriptors. Each dictionary is expected to
        contain at least the following keys:
            - ``"x"`` : float
                x-coordinate (column index) of the Gaussian center.
            - ``"y"`` : float
                y-coordinate (row index) of the Gaussian center.
        If present, the optional key:
            - ``"sigma"`` : float
                Standard deviation of the Gaussian, used to draw a radius
                when ``plot_sigma=True``.

    title : str, optional
        Title of the plot. Default is ``"Detected Gaussians"``.

    plot_sigma : bool, optional
        If ``True``, draw a circular outline at each Gaussian center with
        radius proportional to its estimated standard deviation.
        Default is ``False``.

    sigma_scale : float, optional
        Scaling factor applied to the Gaussian standard deviation when
        drawing circles. This is intended for visualization only and
        does not affect the underlying data. Default is 0.5.

    edge_color : str, optional
        Color used for drawing Gaussian radius circles. Ignored if
        ``plot_sigma=False``. Default is ``"cyan"``.

    line_width : float, optional
        Line width used for drawing Gaussian radius circles.
        Default is 1.5.

    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    for g in gaussians:
        x = g["x"]
        y = g["y"]
        sigma = g.get("sigma", None)

        if sigma is None:
            continue

        r = sigma_scale * sigma

        if plot_sigma:
            circ = Circle(
                (x, y),
                radius=r,
                edgecolor=edge_color,
                facecolor="none",
                linewidth=line_width
            )
            ax.add_patch(circ)

        ax.scatter(x, y, s=1, facecolors="none", edgecolors="red", linewidths=1.5)

    plt.show()





