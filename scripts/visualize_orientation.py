# scripts/visualize_orientation.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation

ds = FVCDataset("data/FVC2002_DB2_B")
sample = ds.samples[0]
image = sample.load()

mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)

# Downsample the orientation field to a coarser grid, just for plotting
# clean, readable direction lines (drawing one line per pixel would be unreadable)
block_size = 16
fig, ax = plt.subplots(figsize=(6, 10))
ax.imshow(image, cmap="gray")

height, width = orientations.shape
for y in range(block_size // 2, height, block_size):
    for x in range(block_size // 2, width, block_size):
        if not mask[y, x]:
            continue
        angle = orientations[y, x]
        strength = strengths[y, x]
        length = (block_size / 2) * strength  # weaker regions get shorter lines
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color="red", linewidth=0.8)

Path("output").mkdir(exist_ok=True)
plt.savefig("output/orientation_field.png", dpi=150)
print("done -- check output/orientation_field.png")