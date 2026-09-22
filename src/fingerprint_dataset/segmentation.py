# segmentation.py
import numpy as np
from scipy import ndimage


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Pixel-level gradient magnitude using Sobel, smoothed with a Gaussian filter.
    """
    gx = ndimage.sobel(image.astype(np.float64), axis=1)
    gy = ndimage.sobel(image.astype(np.float64), axis=0)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    sigma = 13 / 3
    smoothed = ndimage.gaussian_filter(magnitude, sigma=sigma)
    return smoothed


def create_mask(smoothed_magnitude: np.ndarray, percentile: float = 95, threshold_ratio: float = 0.2) -> np.ndarray:
    """
    Adaptive threshold based on the image's own gradient magnitude distribution.
    """
    norm_threshold = np.percentile(smoothed_magnitude, percentile) * threshold_ratio
    return smoothed_magnitude > norm_threshold


def clean_mask(mask: np.ndarray, closing_iterations: int = 6, opening_iterations: int = 12) -> np.ndarray:
    """
    Morphological cleanup: closing to fill gaps and concavities,
    keep largest component, fill remaining holes, then opening
    to remove small blobs, and keep largest component again.
    """
    structure = ndimage.generate_binary_structure(2, 1)

    closed = ndimage.binary_closing(mask, structure=structure, iterations=closing_iterations)
    closed = _keep_largest_component(closed)
    filled = ndimage.binary_fill_holes(closed)

    opened = ndimage.binary_opening(filled, structure=structure, iterations=opening_iterations)
    opened = _keep_largest_component(opened)

    return opened


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num_features = ndimage.label(mask)
    if num_features <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove background pixels using a binary mask.
    """
    result = image.copy()
    result[~mask] = 0
    return result


def segment(image: np.ndarray) -> np.ndarray:
    """
    Full segmentation pipeline: gradient magnitude -> adaptive
    threshold -> morphological cleanup.
    """
    magnitude = gradient_magnitude(image)
    raw_mask = create_mask(magnitude)
    return clean_mask(raw_mask)