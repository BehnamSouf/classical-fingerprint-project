# scripts/diagnose_fill_missing.py
import numpy as np
from PIL import Image as PILImage
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

print(f"Valid grid points before fill_missing: {valid.sum()}")

filled_periods = fill_missing(periods, valid, max_iterations=20)
still_invalid_after_fill = mask & (filled_periods == 0)

print(f"Mask foreground pixels: {mask.sum()}")
print(f"Still invalid (period==0) after fill_missing: {still_invalid_after_fill.sum()}")
print(f"Percentage: {100 * still_invalid_after_fill.sum() / mask.sum():.1f}%")

# Visualize: where are the still-invalid pixels located?
diagnostic = np.zeros((*image.shape, 3), dtype=np.uint8)
diagnostic[mask] = (220, 220, 220)
diagnostic[still_invalid_after_fill] = (255, 0, 0)

from pathlib import Path
Path("output").mkdir(exist_ok=True)
PILImage.fromarray(diagnostic).save("output/fill_missing_diagnostic.png")
print("saved output/fill_missing_diagnostic.png")