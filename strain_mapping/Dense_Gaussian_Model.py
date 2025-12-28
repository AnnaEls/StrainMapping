import torch
from typing import List, Tuple, Optional

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
