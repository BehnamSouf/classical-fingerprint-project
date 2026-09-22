from pathlib import Path
from PIL import Image

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment, apply_mask

ds = FVCDataset("data/FVC2002_DB2_B")
sample = ds.samples[1]
img = sample.load()

mask = segment(img)
segmented = apply_mask(img, mask)

Path("output").mkdir(exist_ok=True)
Image.fromarray(img).save("output/original.png")
Image.fromarray(segmented).save("output/segmented.png")
print("done")