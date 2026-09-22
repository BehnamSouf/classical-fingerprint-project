# orientation.py
import numpy as np
from scipy import ndimage


def preprocess_image(image: np.ndarray, mask: np.ndarray | None = None,
                      percentile: float = 19, sigma_smooth: float = 1.25,
                      median_size: int = 5) -> np.ndarray:
    """
    Contrast-stretch the image based on its own intensity distribution,
    then smooth it (Gaussian + median filter) to reduce noise before
    gradient computation.
    """
    masked_pixels = image[mask] if mask is not None else image
    low = np.percentile(masked_pixels, percentile)
    high = np.percentile(masked_pixels, 100 - percentile)

    if high > low:
        image = np.clip((image.astype(np.float64) - low) * 255 / (high - low), 0, 255)

    image = ndimage.gaussian_filter(image, sigma=sigma_smooth)
    image = ndimage.median_filter(image, size=median_size)
    return image


def compute_gradients(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Horizontal and vertical image gradients via Sobel filters.
    Background gradients are zeroed out if a mask is provided, so
    that background noise does not influence the orientation estimate.
    """
    gx = ndimage.sobel(image.astype(np.float64), axis=1)
    gy = ndimage.sobel(image.astype(np.float64), axis=0)

    if mask is not None:
        gx = np.where(mask, gx, 0)
        gy = np.where(mask, gy, 0)

    return gx, gy


def compute_orientation_and_strength(gx: np.ndarray, gy: np.ndarray, sigma: float,
                                      mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel-level ridge orientation and coherence (strength), using the
    doubled-angle least-squares method with continuous Gaussian smoothing
    instead of block averaging.

    Orientation is in radians, range [0, pi).
    Strength is in [0, 1]: close to 1 where local gradients are strongly
    aligned (clear ridge), close to 0 where they are scattered (noise,
    singularities like core/delta).
    """
    gx2 = gx ** 2
    gy2 = gy ** 2
    gxy2 = -2 * gx * gy  # minus sign for a counter-clockwise angle convention

    sum_gx2 = ndimage.gaussian_filter(gx2, sigma=sigma)
    sum_gy2 = ndimage.gaussian_filter(gy2, sigma=sigma)
    d = ndimage.gaussian_filter(gxy2, sigma=sigma)
    n = sum_gx2 - sum_gy2

    orientations = (np.arctan2(d, n) + np.pi) / 2 % np.pi

    denom = sum_gx2 + sum_gy2
    strengths = np.divide(np.sqrt(n ** 2 + d ** 2), denom,
                           out=np.zeros_like(denom), where=denom != 0)

    if mask is not None:
        strengths = np.where(mask, strengths, 0)

    return orientations, strengths


def estimate_orientation(image: np.ndarray, mask: np.ndarray | None = None,
                          sigma: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Full orientation field estimation pipeline: preprocessing ->
    gradients -> orientation and strength.

    Returns (orientations, strengths), both pixel-level and the same
    shape as the input image.
    """
    preprocessed = preprocess_image(image, mask)
    gx, gy = compute_gradients(preprocessed, mask)
    return compute_orientation_and_strength(gx, gy, sigma, mask)