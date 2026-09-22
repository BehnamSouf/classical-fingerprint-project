# scripts/visualize_enhancement.py
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field
from fingerprint_dataset.enhancement import enhance

ds = FVCDataset("data/FVC2002_DB2_B")
sample = ds.samples[0]
image = sample.load()

print("segmenting...")
mask = segment(image)

print("estimating orientation...")
orientations, strengths = estimate_orientation(image, mask)

print("estimating frequency...")
frequencies = estimate_frequency_field(image, orientations, mask)
periods = np.divide(1.0, frequencies, out=np.zeros_like(frequencies), where=frequencies != 0)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# --- diagnostic: find where enhancement invalidates foreground pixels ---
invalid_orientation = mask & ~np.isfinite(orientations)

invalid_frequency = mask & (
    ~np.isfinite(frequencies) | (frequencies <= 0)
)

invalid_period = mask & (
    ~np.isfinite(periods) | (periods <= 0)
)

valid = (
    mask
    & ~invalid_orientation
    & ~invalid_frequency
    & ~invalid_period
)

diagnostic = np.zeros((*image.shape, 3), dtype=np.uint8)
diagnostic[valid] = (220, 220, 220)
diagnostic[invalid_frequency | invalid_period] = (255, 0, 0)
diagnostic[invalid_orientation] = (0, 100, 255)

PILImage.fromarray(diagnostic).save(output_dir / "enh_diagnostic.png")
PILImage.fromarray((mask.astype(np.uint8) * 255)).save(output_dir / "enh_mask.png")

foreground_count = np.count_nonzero(mask)
print(f"Foreground pixels: {foreground_count}")
print(f"Valid pixels: {np.count_nonzero(valid)}")
print("Invalid frequency pixels:", np.count_nonzero(invalid_frequency))
print("Invalid orientation pixels:", np.count_nonzero(invalid_orientation))
print("Invalid period pixels:", np.count_nonzero(invalid_period))
# --- end diagnostic ---

print("enhancing...")
enhanced = enhance(image, mask, orientations, periods)

PILImage.fromarray(image).save(output_dir / "enh_original.png")
PILImage.fromarray(enhanced).save(output_dir / "enh_enhanced.png")
print("done")
