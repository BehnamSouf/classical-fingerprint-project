# tests/test_segmentation_visual.py  (یا یه اسکریپت جدا مثل scripts/visualize_segmentation.py)
from pathlib import Path
from PIL import Image

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import (
    block_variance,
    create_block_mask,
    expand_block_mask,
    apply_mask,
)

ds = FVCDataset("data/FVC2002_DB2_B")
sample = ds.samples[0]
img = sample.load()

variances = block_variance(img, block_size=16)
block_mask = create_block_mask(variances)
pixel_mask = expand_block_mask(block_mask, block_size=16, image_shape=img.shape)
segmented = apply_mask(img, pixel_mask)

Path("output").mkdir(exist_ok=True)
Image.fromarray(img).save("output/original.png")
Image.fromarray(segmented).save("output/segmented.png")
print("done — output/original.png و output/segmented.png رو با چشم مقایسه کن")