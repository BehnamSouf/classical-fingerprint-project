# src/fingerprint_dataset/matching.py
from __future__ import annotations

import numpy as np


def normalize_for_correlation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalize an image over its masked
    (foreground) region only. Used only as a display/consistency
    convenience; the actual matching score always re-normalizes on
    the overlap region at each shift (see correlation_score), since
    the overlap's mean/variance generally differs from the whole
    foreground's.
    """
    image = image.astype(np.float64)
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return np.zeros_like(image)

    values = image[mask]
    mean, std = values.mean(), values.std()

    if std < 1e-8:
        return np.zeros_like(image)

    normalized = np.zeros_like(image)
    normalized[mask] = (image[mask] - mean) / std
    return normalized


def crop_to_common_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop two arrays to their common (minimum) height and width.
    """
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def _overlap_regions(height: int, width: int, dy: int, dx: int):
    """
    For a shift (dy, dx) of image b relative to image a, return the
    slice bounds of the overlapping region in each image's own
    coordinate frame.
    """
    y0_a, y1_a = max(0, dy), min(height, height + dy)
    x0_a, x1_a = max(0, dx), min(width, width + dx)
    y0_b, y1_b = max(0, -dy), min(height, height - dy)
    x0_b, x1_b = max(0, -dx), min(width, width - dx)
    return (y0_a, y1_a, x0_a, x1_a), (y0_b, y1_b, x0_b, x1_b)


def _ncc_at_shift(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray,
                   dy: int, dx: int, min_overlap: int) -> float | None:
    """
    Compute normalized cross-correlation between image_a and image_b
    shifted by (dy, dx), using only the overlapping foreground region
    (mask_a & mask_b in that region). Both patches are re-centered
    (mean subtracted) on this specific overlap before computing NCC --
    the overlap's own mean/variance, not the whole image's, is what
    matters for a correct correlation coefficient.

    Returns None if the overlap is too small or degenerate (e.g. one
    patch has zero variance), signaling "this shift produced no valid
    comparison" rather than a real score of 0.
    """
    height, width = image_a.shape
    (y0_a, y1_a, x0_a, x1_a), (y0_b, y1_b, x0_b, x1_b) = _overlap_regions(height, width, dy, dx)

    if y1_a - y0_a < 10 or x1_a - x0_a < 10:
        return None

    patch_mask = mask_a[y0_a:y1_a, x0_a:x1_a] & mask_b[y0_b:y1_b, x0_b:x1_b]
    if patch_mask.sum() < min_overlap:
        return None

    a = image_a[y0_a:y1_a, x0_a:x1_a][patch_mask].astype(np.float64)
    b = image_b[y0_b:y1_b, x0_b:x1_b][patch_mask].astype(np.float64)

    a = a - a.mean()
    b = b - b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


def correlation_score(image_a: np.ndarray, mask_a: np.ndarray,
                       image_b: np.ndarray, mask_b: np.ndarray,
                       coarse_max_shift: int = 20, coarse_step: int = 4,
                       fine_radius: int = 4,
                       min_overlap: int = 500) -> float | None:
    """
    Similarity score between two (enhanced) fingerprint images, using
    normalized cross-correlation (NCC) searched over translation only
    (no rotation -- see module docstring / caller for that limitation).

    Two-stage search:
    1. Coarse grid over [-coarse_max_shift, coarse_max_shift] with
       step `coarse_step` (always including shift (0, 0) explicitly).
    2. Fine search with step 1 in a `fine_radius` window around the
       best coarse shift, since ridge spacing (~10px) means the true
       best alignment can fall between coarse grid points.

    Returns the best NCC score found (in [-1, 1]), or None if no shift
    produced a valid comparison (e.g. masks never overlap enough) --
    this is a genuine matching failure, distinct from a real score of 0,
    and callers must decide how to handle it (e.g. exclude from EER,
    or count as a forced non-match).
    """
    image_a, image_b = crop_to_common_size(image_a, image_b)
    mask_a, mask_b = crop_to_common_size(np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool))

    coarse_shifts = {(0, 0)}
    for dy in range(-coarse_max_shift, coarse_max_shift + 1, coarse_step):
        for dx in range(-coarse_max_shift, coarse_max_shift + 1, coarse_step):
            coarse_shifts.add((dy, dx))

    best_score = None
    best_shift = (0, 0)

    for dy, dx in coarse_shifts:
        score = _ncc_at_shift(image_a, mask_a, image_b, mask_b, dy, dx, min_overlap)
        if score is not None and (best_score is None or score > best_score):
            best_score, best_shift = score, (dy, dx)

    # Fine search around the best coarse shift.
    by, bx = best_shift
    for dy in range(by - fine_radius, by + fine_radius + 1):
        for dx in range(bx - fine_radius, bx + fine_radius + 1):
            score = _ncc_at_shift(image_a, mask_a, image_b, mask_b, dy, dx, min_overlap)
            if score is not None and (best_score is None or score > best_score):
                best_score = score

    return best_score


def compute_far_frr(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    curves. Thresholds are built from the actual unique scores present
    in the data (plus points just below the minimum and just above the
    maximum, to include the "accept all" / "reject all" endpoints),
    rather than an arbitrary fixed grid -- this makes every FAR/FRR
    step in the returned curve correspond to an achievable operating
    point, not an interpolated one.

    FAR(t) = fraction of impostor pairs with score >= t (wrongly accepted)
    FRR(t) = fraction of genuine pairs with score < t (wrongly rejected)

    Returns (thresholds, far, frr), sorted by threshold ascending.
    """
    genuine_scores = np.asarray(genuine_scores, dtype=np.float64)
    impostor_scores = np.asarray(impostor_scores, dtype=np.float64)

    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise ValueError("Both genuine_scores and impostor_scores must be non-empty.")

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    unique_scores = np.unique(all_scores)

    thresholds = np.concatenate([
        [unique_scores[0] - 1e-6],   # below every score: accept all -> FAR=1, FRR=0
        unique_scores,
        [unique_scores[-1] + 1e-6],  # above every score: reject all -> FAR=0, FRR=1
    ])

    far = np.array([(impostor_scores >= t).mean() for t in thresholds])
    frr = np.array([(genuine_scores < t).mean() for t in thresholds])

    return thresholds, far, frr


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute the Equal Error Rate: the operating point where FAR and
    FRR are closest (FAR and FRR curves are step functions from
    discrete data, so an exact crossing may not exist; this reports
    the closest achievable point and is therefore an approximation,
    not an interpolated exact crossing).

    Returns (eer, threshold_at_eer), where eer is a fraction (0.0-1.0).
    """
    thresholds, far, frr = compute_far_frr(genuine_scores, impostor_scores)

    diff = np.abs(far - frr)
    best_index = np.argmin(diff)

    eer = float((far[best_index] + frr[best_index]) / 2.0)
    threshold_at_eer = float(thresholds[best_index])

    return eer, threshold_at_eer