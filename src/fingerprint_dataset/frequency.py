# src/fingerprint_dataset/frequency.py
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------

WINDOW_SIZE = (43, 23)          # (height, width): tall across ridges, narrow along them
PERIOD_MIN = 5.0
PERIOD_MAX = 20.0

MIN_VALID_DISTANCES = 3
DISTANCE_TOLERANCE = 3.0
MIN_BACKGROUND_DISTANCE = 11.0

FILL_ITERATIONS = 20
SMOOTHING_SIGMA = 5.0


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

@dataclass
class FrequencyStats:
    """
    Per-stage counters explaining why sample points did or did not
    produce a valid period estimate. Useful for debugging why a large
    fraction of the fingerprint ends up without a usable frequency.
    """
    candidate_count: int = 0
    edge_skipped_count: int = 0
    period_failed_count: int = 0
    period_valid_count: int = 0
    remaining_invalid_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    return image


def _validate_window_size(window_size: tuple[int, int]) -> tuple[int, int]:
    height, width = int(window_size[0]), int(window_size[1])
    if height < 3 or width < 3:
        raise ValueError("window dimensions must be at least 3")
    return height, width


# ---------------------------------------------------------------------
# Window rotation
# ---------------------------------------------------------------------

def rotate_window(image: np.ndarray, center: tuple[int, int], angle: float,
                   window_size: tuple[int, int] = WINDOW_SIZE) -> np.ndarray:
    """
    Extract a fixed-size window around `center`, rotated so that the
    local ridge direction becomes horizontal.

    angle is the ridge orientation in radians, as produced by
    orientation.estimate_orientation (image coordinates: x right, y down).

    Convention: scipy.ndimage.rotate uses positive angles for
    counter-clockwise rotation of the array. Rotating the patch by
    -angle_degrees aligns a ridge that is tilted by `angle` so it
    becomes horizontal in the output. This sign is verified by
    test_frequency.py; if that test fails, this is the first place
    to check.
    """
    image = _validate_image(image)
    window_height, window_width = _validate_window_size(window_size)

    y, x = int(round(center[0])), int(round(center[1]))
    angle = float(angle) if np.isfinite(angle) else 0.0

    # Pad generously so the requested window is always fully available
    # after rotation, even near the image border.
    pad = int(np.ceil(np.hypot(window_height, window_width) / 2.0)) + 2
    padded = np.pad(image, pad, mode="reflect")

    py, px = y + pad, x + pad
    patch = padded[py - pad:py + pad + 1, px - pad:px + pad + 1]

    angle_degrees = -np.degrees(angle)
    rotated = ndimage.rotate(patch, angle_degrees, reshape=False,
                              order=1, mode="reflect")

    cy, cx = rotated.shape[0] // 2, rotated.shape[1] // 2
    start_y = cy - window_height // 2
    start_x = cx - window_width // 2

    return rotated[start_y:start_y + window_height, start_x:start_x + window_width]


# ---------------------------------------------------------------------
# 1D signature and extrema
# ---------------------------------------------------------------------

def x_signature(window: np.ndarray) -> np.ndarray:
    """
    Collapse a rotated window to a 1D signal by averaging each row.
    Ridges (dark) and valleys between them (bright) become the
    minima/maxima of this signal.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError("window must be 2D")
    return window.mean(axis=1)


def find_local_extrema(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find indices of local maxima (peaks) and minima (valleys) in a 1D
    signal, ignoring the first and last sample (which have no defined
    neighbor on one side).
    """
    if signal.size < 3:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    inner = signal[1:-1]
    is_peak = (inner > signal[:-2]) & (inner >= signal[2:])
    is_valley = (inner < signal[:-2]) & (inner <= signal[2:])

    peak_indices = np.where(is_peak)[0] + 1
    valley_indices = np.where(is_valley)[0] + 1
    return peak_indices, valley_indices


def _consistent_period(distances: np.ndarray, min_valid_distances: int,
                        distance_tolerance: float) -> float | None:
    """
    Given a set of candidate peak-to-peak (or valley-to-valley)
    distances, reject outliers by keeping only those close to the
    median, and return the median of the survivors as the period
    estimate. Returns None if too few distances remain.
    """
    distances = distances[distances > 0]
    if distances.size < min_valid_distances:
        return None

    median_distance = float(np.median(distances))
    close = distances[np.abs(distances - median_distance) <= distance_tolerance]

    if close.size < 2:
        return None

    return float(np.median(close))


# ---------------------------------------------------------------------
# Period estimation at a single point
# ---------------------------------------------------------------------

def estimate_period_at_point(window: np.ndarray, period_min: float = PERIOD_MIN,
                              period_max: float = PERIOD_MAX,
                              min_valid_distances: int = MIN_VALID_DISTANCES,
                              distance_tolerance: float = DISTANCE_TOLERANCE) -> float:
    """
    Estimate the local ridge period from one rotated window.

    Primary estimate: distances between consecutive peaks and between
    consecutive valleys, treated as separate samples of the same
    underlying period (this avoids treating a peak-to-valley gap,
    which is only half a period, as a full one).

    Fallback: if there are too few same-type extrema (e.g. a short or
    noisy window), consecutive peak/valley distances (mixed) are used
    and doubled, since a peak-to-valley distance is approximately half
    a ridge period.

    Returns NaN if no reliable estimate can be made.
    """
    signal = x_signature(window)
    if signal.size < 5:
        return np.nan

    signal = signal - signal.mean()  # remove DC offset, does not affect extrema locations
    peaks, valleys = find_local_extrema(signal)

    same_type_distances = []
    if peaks.size >= 2:
        same_type_distances.append(np.diff(peaks))
    if valleys.size >= 2:
        same_type_distances.append(np.diff(valleys))

    period = None
    if same_type_distances:
        period = _consistent_period(
            np.concatenate(same_type_distances), min_valid_distances, distance_tolerance
        )

    if period is None:
        all_extrema = np.sort(np.concatenate([peaks, valleys]))
        if all_extrema.size >= 2:
            fallback_period = _consistent_period(
                np.diff(all_extrema), min_valid_distances, distance_tolerance
            )
            if fallback_period is not None:
                period = fallback_period * 2.0

    if period is None or not (period_min <= period <= period_max):
        return np.nan

    return period


# ---------------------------------------------------------------------
# Filling missing values within the fingerprint domain
# ---------------------------------------------------------------------

def fill_missing(field: np.ndarray, valid: np.ndarray, domain: np.ndarray,
                  max_iterations: int = FILL_ITERATIONS) -> np.ndarray:
    """
    Propagate valid values outward to fill gaps, restricted to `domain`
    (the fingerprint mask). Background pixels never contribute to or
    receive filled values. Any point still unfilled after propagation
    (e.g. isolated pockets with no valid neighbor at all) gets the
    median of all valid values in the domain, so the result never
    contains gaps inside the fingerprint.
    """
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & domain & np.isfinite(field)

    filled = np.where(valid, field, 0.0)
    known = valid.copy()
    kernel = np.ones((3, 3))

    for _ in range(max_iterations):
        if known.all() or not domain[~known].any():
            break

        neighbor_sum = ndimage.convolve(np.where(known, filled, 0.0), kernel, mode="constant")
        neighbor_count = ndimage.convolve(known.astype(np.float64), kernel, mode="constant")

        can_fill = domain & ~known & (neighbor_count > 0)
        if not can_fill.any():
            break

        filled[can_fill] = neighbor_sum[can_fill] / neighbor_count[can_fill]
        known[can_fill] = True

    still_missing = domain & ~known
    if still_missing.any() and known.any():
        filled[still_missing] = np.median(filled[known])

    filled[~domain] = 0.0
    return filled


# ---------------------------------------------------------------------
# Masked smoothing (avoids background zeros leaking into the result)
# ---------------------------------------------------------------------

def masked_gaussian_smooth(field: np.ndarray, domain: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-blur `field`, weighted only by pixels inside `domain`, so
    that background zeros outside the fingerprint do not drag down
    values near the mask boundary (a plain gaussian_filter over the
    whole array would do exactly that).
    """
    if sigma <= 0:
        result = field.copy()
        result[~domain] = 0.0
        return result

    weighted = ndimage.gaussian_filter(np.where(domain, field, 0.0), sigma=sigma)
    weights = ndimage.gaussian_filter(domain.astype(np.float64), sigma=sigma)

    result = np.divide(weighted, weights, out=np.zeros_like(field), where=weights > 1e-8)
    result[~domain] = 0.0
    return result


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------

def estimate_frequency_field(image: np.ndarray, orientations: np.ndarray, mask: np.ndarray,
                              window_size: tuple[int, int] = WINDOW_SIZE,
                              period_min: float = PERIOD_MIN, period_max: float = PERIOD_MAX,
                              min_valid_distances: int = MIN_VALID_DISTANCES,
                              distance_tolerance: float = DISTANCE_TOLERANCE,
                              min_background_distance: float = MIN_BACKGROUND_DISTANCE,
                              step: int | None = None,
                              fill_iterations: int = FILL_ITERATIONS,
                              smoothing_sigma: float = SMOOTHING_SIGMA,
                              return_stats: bool = False):
    """
    Estimate the local ridge period field (in pixels; convert with
    period_to_frequency if cycles-per-pixel is needed).

    Sample points are laid out on a grid (spaced by `step`, defaulting
    to the window width so samples do not overlap heavily). At each
    sample point sufficiently far from the mask boundary, a rotated
    window is used to estimate the local period. Missing points are
    filled by propagation from valid neighbors, then the whole field
    is smoothed with a mask-aware Gaussian blur.

    Returns the period map, or (period_map, FrequencyStats) if
    return_stats is True.
    """
    image = _validate_image(image)
    orientations = np.asarray(orientations, dtype=np.float64)
    domain = np.asarray(mask, dtype=bool)

    if orientations.shape != image.shape or domain.shape != image.shape:
        raise ValueError("image, orientations, and mask must have the same shape")

    window_height, window_width = _validate_window_size(window_size)
    if step is None:
        step = max(4, window_width // 2)

    height, width = image.shape
    stats = FrequencyStats()

    periods = np.zeros((height, width))
    valid = np.zeros((height, width), dtype=bool)

    if not domain.any():
        result = np.zeros((height, width))
        return (result, stats) if return_stats else result

    distance_to_background = ndimage.distance_transform_edt(
        np.pad(domain, 1, mode="constant", constant_values=False)
    )[1:-1, 1:-1]

    half_h, half_w = window_height // 2, window_width // 2
    y_start, y_end = half_h, height - half_h
    x_start, x_end = half_w, width - half_w

    for y in range(y_start, y_end, step):
        for x in range(x_start, x_end, step):
            if not domain[y, x]:
                continue
            stats.candidate_count += 1

            if distance_to_background[y, x] < min_background_distance:
                stats.edge_skipped_count += 1
                continue

            angle = orientations[y, x]
            if not np.isfinite(angle):
                stats.period_failed_count += 1
                continue

            window = rotate_window(image, (y, x), angle, window_size)
            period = estimate_period_at_point(
                window, period_min, period_max, min_valid_distances, distance_tolerance
            )

            if not np.isfinite(period):
                stats.period_failed_count += 1
                continue

            periods[y, x] = period
            valid[y, x] = True
            stats.period_valid_count += 1

    periods = fill_missing(periods, valid, domain, fill_iterations)
    periods = masked_gaussian_smooth(periods, domain, smoothing_sigma)

    in_domain_valid = domain & (periods > 0)
    periods[in_domain_valid] = np.clip(periods[in_domain_valid], period_min, period_max)
    periods[~domain] = 0.0

    stats.remaining_invalid_count = int(np.count_nonzero(domain & (periods <= 0)))

    return (periods, stats) if return_stats else periods


# ---------------------------------------------------------------------
# Conversion utility
# ---------------------------------------------------------------------

def period_to_frequency(period_map: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Convert a period map (pixels between ridges) to a frequency map
    (cycles per pixel), i.e. 1 / period. Zero or invalid periods map
    to zero frequency.
    """
    period_map = np.asarray(period_map, dtype=np.float64)
    domain = np.ones_like(period_map, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    frequency_map = np.zeros_like(period_map)
    valid = domain & (period_map > 0)
    frequency_map[valid] = 1.0 / period_map[valid]
    return frequency_map