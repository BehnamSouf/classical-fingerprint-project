# scripts/diagnose_coherence.py
from pathlib import Path
from PIL import Image as PILImage

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field
from fingerprint_dataset.enhancement import enhance

ds = FVCDataset("data/FVC2002_DB2_B")
image = ds.samples[0].load()
mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)
periods = estimate_frequency_field(image, orientations, mask)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for low, high in [(0.0, 0.001), (0.20, 0.45), (0.35, 0.55), (0.50, 0.70)]:
    print(f"enhancing with coherence_low={low}, coherence_high={high}...")
    enhanced = enhance(
        image, mask, orientations, periods, strengths=strengths,
        coherence_low=low, coherence_high=high,
    )
    PILImage.fromarray(enhanced).save(output_dir / f"enh_coh_{low}_{high}.png")

print("done")