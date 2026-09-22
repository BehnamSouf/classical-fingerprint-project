# scripts/diagnose_v2.py
import numpy as np

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field

ds = FVCDataset("data/FVC2002_DB2_B")
image = ds.samples[0].load()
mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)

frequencies = estimate_frequency_field(image, orientations, mask)

invalid = mask & (frequencies <= 0)
print(f"Mask foreground pixels: {mask.sum()}")
print(f"Invalid frequency pixels (final function output): {invalid.sum()}")
print(f"Percentage: {100 * invalid.sum() / mask.sum():.1f}%")