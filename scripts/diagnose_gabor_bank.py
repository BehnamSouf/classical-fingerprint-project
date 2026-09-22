# scripts/diagnose_gabor_bank.py
from pathlib import Path
from PIL import Image as PILImage

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field, period_to_frequency
from fingerprint_dataset.enhancement import enhance

ds = FVCDataset("data/FVC2002_DB2_B")
image = ds.samples[0].load()
mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)
periods = estimate_frequency_field(image, orientations, mask)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

for periods_count, orientations_count in [(10, 10), (20, 20), (40, 40)]:
    print(f"enhancing with periods_count={periods_count}, orientations_count={orientations_count}...")
    enhanced = enhance(
        image, mask, orientations, periods,
        periods_count=periods_count,
        orientations_count=orientations_count,
    )
    PILImage.fromarray(enhanced).save(
        output_dir / f"enh_bank_{periods_count}x{orientations_count}.png"
    )

print("done")