# src/fingerprint_dataset/enhancement.py
import math

import numpy as np
from scipy import ndimage


def _validate_bank_parameters(period_min: float, period_max: float,
                               periods_count: int, orientations_count: int) -> None:
    if not (np.isfinite(period_min) and np.isfinite(period_max) and period_min < period_max):
        raise ValueError("Expected 0 < period_min < period_max.")
    if not isinstance(periods_count, (int, np.integer)) or periods_count < 2:
        raise ValueError("periods_count must be an integer >= 2.")
    if not isinstance(orientations_count, (int, np.integer)) or orientations_count < 1:
        raise ValueError("orientations_count must be a positive integer.")


def gabor_kernel(
    period: float,
    ridge_orientation: float,
    gamma: float = 1.0,
) -> np.ndarray:
    sigma = 5.0 * period / 12.0
    half_size = math.ceil(3.0 * max(sigma, sigma / gamma))

    coordinates = np.arange(
        -half_size,
        half_size + 1,
        dtype=np.float64,
    )
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")

    # The carrier is perpendicular to the ridge direction.
    carrier_orientation = np.pi / 2.0 - ridge_orientation

    cosine = np.cos(carrier_orientation)
    sine = np.sin(carrier_orientation)

    x_theta = x * cosine + y * sine
    y_theta = -x * sine + y * cosine

    envelope = np.exp(
        -(x_theta**2 + gamma**2 * y_theta**2)
        / (2.0 * sigma**2)
    )
    carrier = np.cos(
        2.0 * np.pi * x_theta / period
    )

    kernel = envelope * carrier
    kernel -= kernel.mean()

    energy = np.linalg.norm(kernel)
    if not np.isfinite(energy) or energy <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a degenerate Gabor kernel.")

    return kernel / energy



def build_gabor_bank(period_min: float = 5.0, period_max: float = 20.0,
                      periods_count: int = 10, orientations_count: int = 10) -> list[np.ndarray]:
    """Build a period-major, orientation-minor filter bank."""
    _validate_bank_parameters(period_min, period_max, periods_count, orientations_count)
    periods = np.linspace(period_min, period_max, periods_count)
    orientations = np.arange(orientations_count) * np.pi / orientations_count
    return [gabor_kernel(period, orientation) for period in periods for orientation in orientations]


def discretize_periods(periods: np.ndarray, period_min: float, period_max: float,
                        periods_count: int) -> np.ndarray:
    clipped = np.clip(periods, period_min, period_max)
    normalized = (clipped - period_min) / (period_max - period_min)
    return np.rint(normalized * (periods_count - 1)).astype(np.int32)


def discretize_orientations(orientations: np.ndarray, orientations_count: int) -> np.ndarray:
    normalized = np.remainder(orientations, np.pi) / np.pi
    return np.rint(normalized * orientations_count).astype(np.int32) % orientations_count


def bank_indices(orientations: np.ndarray, periods: np.ndarray, period_min: float, period_max: float,
                  periods_count: int, orientations_count: int) -> np.ndarray:
    period_bins = discretize_periods(periods, period_min, period_max, periods_count)
    orientation_bins = discretize_orientations(orientations, orientations_count)
    return period_bins * orientations_count + orientation_bins


def enhance(image: np.ndarray, mask: np.ndarray, orientations: np.ndarray, periods: np.ndarray,
            strengths: np.ndarray | None = None,
            period_min: float = 5.0, period_max: float = 20.0,
            periods_count: int = 10, orientations_count: int = 10,
            orientation_offset: float = 0.0,
            coherence_low: float = 0.20, coherence_high: float = 0.45) -> np.ndarray:
    """
    Enhance a grayscale image using locally selected Gabor filters,
    softly blended with the original image based on orientation
    coherence (strength).

    Rationale: near singular points (core/delta), ridge orientation
    changes rapidly from pixel to pixel. Picking a single hard-selected
    Gabor filter per pixel in that region makes neighboring pixels use
    filters with very different phase, producing a spurious "pinwheel"
    artifact unrelated to the real ridge pattern. Blending toward the
    original image where coherence is low avoids applying a directional
    filter where the local direction is not reliable in the first place.

    If `strengths` is not provided, no coherence blending is applied
    (equivalent to full-strength Gabor everywhere valid).
    """
    _validate_bank_parameters(period_min, period_max, periods_count, orientations_count)

    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    orientations = np.asarray(orientations, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.float64)

    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2D grayscale array.")
    if any(array.shape != image.shape for array in (mask, orientations, periods)):
        raise ValueError("image, mask, orientations, and periods must have identical shapes.")

    valid = mask & np.isfinite(orientations) & np.isfinite(periods) & (periods > 0)

    if not valid.any():
        return image.astype(np.uint8)

    safe_orientations = np.where(valid, orientations, 0.0)
    safe_orientations = np.remainder(safe_orientations + orientation_offset, np.pi)
    safe_periods = np.where(valid, periods, period_min)

    indices = bank_indices(safe_orientations, safe_periods, period_min, period_max,
                            periods_count, orientations_count)
    bank = build_gabor_bank(period_min, period_max, periods_count, orientations_count)

    inverted = 255.0 - image
    filtered = np.zeros(image.shape, dtype=np.float64)

    for index in np.unique(indices[valid]):
        response = ndimage.convolve(inverted, bank[int(index)], mode="reflect")
        selected = valid & (indices == index)
        filtered[selected] = response[selected]

    # Normalize the Gabor response (percentile-based, robust to outliers)
    values = filtered[valid]
    low_p, high_p = np.percentile(values, [2.0, 98.0])
    tolerance = 1e-12 * max(1.0, abs(low_p), abs(high_p))
    if high_p - low_p <= tolerance:
        return image.astype(np.uint8)

    filtered = np.clip((filtered - low_p) / (high_p - low_p), 0.0, 1.0) * 255.0

    if strengths is not None:
        strengths = np.asarray(strengths, dtype=np.float64)
        if strengths.shape != image.shape:
            raise ValueError("strengths must have the same shape as image.")

        # alpha=0 -> keep original image, alpha=1 -> fully use Gabor
        # response. Ramps smoothly between coherence_low and
        # coherence_high instead of a hard cutoff, to avoid a visible
        # seam at the threshold.
        safe_strengths = np.where(valid, strengths, 0.0)
        alpha = np.clip((safe_strengths - coherence_low) / (coherence_high - coherence_low), 0.0, 1.0)
    else:
        alpha = valid.astype(np.float64)

    enhanced = alpha * filtered + (1.0 - alpha) * image
    enhanced[~mask] = 0.0

    return np.rint(np.clip(enhanced, 0, 255)).astype(np.uint8)