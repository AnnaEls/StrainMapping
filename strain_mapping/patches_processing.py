from typing import List, Tuple
import numpy as np

def extract_patches(
    image: np.ndarray,
    patch_size: int = 128,
    stride: int = 64
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract patches from a 2D image array.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W).
    patch_size : int, optional
        Height and width of each square patch, by default 128.
    stride : int, optional
        Stride between patch top-left corners, by default 64.

    Returns
    -------
    patches : List[np.ndarray]
        List of extracted image patches, each of shape (patch_size, patch_size).
    coords : List[Tuple[int, int]]
        List of (y, x) coordinates corresponding to the top-left corner
        of each extracted patch.
    """
    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    H, W = image.shape
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
            patches.append(patch)
            coords.append((y0, x0))

    return patches, coords
