# scripts/diagnose_final_blur.py
import numpy as np
from scipy import ndimage

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import (
    preprocess_image,
    rotate_window,
    x_signature,
    peak_valley_distances,
    fill_missing,
)

ds = FVCDataset("data/FVC2002_DB2_B")
image = ds.samples[0].load()
mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)

preprocessed = preprocess_image(image)
padded_mask = np.pad(mask, 1, mode="constant", constant_values=False)
distance_to_background = ndimage.distance_transform_edt(padded_mask)[1:-1, 1:-1]

height, width = image.shape
step, border = 8, 25
min_background_distance = 11
window_size = (23, 43)
period_min, period_max = 5, 20
min_valid_distances = 4

periods = np.zeros((height, width))
valid = np.zeros((height, width), dtype=bool)

for y in range(border, height - border, step):
    for x in range(border, width - border, step):
        if distance_to_background[y, x] < min_background_distance:
            continue
        angle = orientations[y, x]
        window = rotate_window(preprocessed, (y, x), angle, window_size)
        signal = x_signature(window)
        distances = peak_valley_distances(signal, period_min, period_max)
        if distances.size < min_valid_distances:
            continue
        median_distance = np.median(distances)
        close_distances = distances[np.abs(distances - median_distance) <= 2]
        if close_distances.size > 0:
            periods[y, x] = close_distances.mean()
            valid[y, x] = True

periods = fill_missing(periods, valid, max_iterations=20)

frequencies_before_blur = np.divide(1.0, periods, out=np.zeros_like(periods), where=periods != 0)
invalid_before_blur = mask & (frequencies_before_blur <= 0)
print(f"Invalid BEFORE final gaussian_filter: {invalid_before_blur.sum()} "
      f"({100*invalid_before_blur.sum()/mask.sum():.1f}%)")

frequencies_after_blur = ndimage.gaussian_filter(frequencies_before_blur, sigma=5)
invalid_after_blur = mask & (frequencies_after_blur <= 0)
print(f"Invalid AFTER final gaussian_filter:  {invalid_after_blur.sum()} "
      f"({100*invalid_after_blur.sum()/mask.sum():.1f}%)")

# how many pixels are close to zero but not exactly zero, right after the blur?
near_zero = mask & (frequencies_after_blur > 0) & (frequencies_after_blur < 0.02)
print(f"Near-zero (0 < f < 0.02) after blur:  {near_zero.sum()} "
      f"({100*near_zero.sum()/mask.sum():.1f}%)")