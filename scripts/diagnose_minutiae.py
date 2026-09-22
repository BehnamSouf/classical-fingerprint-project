# scripts/diagnose_minutiae.py
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field
from fingerprint_dataset.enhancement import enhance
from fingerprint_dataset.minutiae import (
    binarize_enhanced, skeletonize_ridges, prune_skeleton,
    remove_short_components, estimate_ridge_period, extract_minutiae,
    filter_minutiae, MinutiaeConfig, draw_minutiae,
)

ds = FVCDataset("data/FVC2002_DB2_B")
sample = ds.samples[0]
image = sample.load()
mask = segment(image)
orientations, strengths = estimate_orientation(image, mask)
periods = estimate_frequency_field(image, orientations, mask)
enhanced = enhance(image, mask, orientations, periods, strengths=strengths)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Step 1: binarization
binary, polarity = binarize_enhanced(enhanced, mask, polarity="auto", reference=image)
print(f"polarity chosen: {polarity}")
PILImage.fromarray((binary * 255).astype(np.uint8)).save(output_dir / "min_1_binary.png")

# Step 2: raw skeleton (before pruning)
skeleton = skeletonize_ridges(binary)
PILImage.fromarray((skeleton * 255).astype(np.uint8)).save(output_dir / "min_2_skeleton_raw.png")

# Step 3: pruned skeleton
period = estimate_ridge_period(periods, mask)
print(f"estimated period: {period:.1f} px")
cfg = MinutiaeConfig()
min_branch = int(round(np.clip(cfg.min_branch_period_factor * period, 4, 20)))
pruned = prune_skeleton(skeleton, min_branch, cfg.prune_iterations, cfg.remove_islands)
pruned = skeletonize_ridges(pruned)
pruned, n_frag = remove_short_components(pruned, int(round(cfg.min_component_period_factor * period)))
print(f"fragments removed: {n_frag}")
PILImage.fromarray((pruned * 255).astype(np.uint8)).save(output_dir / "min_3_skeleton_pruned.png")

# Step 4: raw minutiae (before filtering) - count only
track = int(round(np.clip(cfg.track_period_factor * period, cfg.min_track_length, cfg.max_track_length)))
margin = cfg.border_margin_factor * period
raw = extract_minutiae(pruned, mask, coherence=strengths, track_length=track,
                        min_arm_length=cfg.min_arm_length, border_margin=margin)
n_e = sum(m.type == "ending" for m in raw)
print(f"raw minutiae: {len(raw)} total ({n_e} endings, {len(raw)-n_e} bifurcations)")

stats = {}
filtered = filter_minutiae(raw, mask, period=period, config=cfg, stats=stats)
n_e2 = sum(m.type == "ending" for m in filtered)
print(f"after filter: {len(filtered)} total ({n_e2} endings, {len(filtered)-n_e2} bifurcations)")
print(f"removed by: {stats}")

draw_minutiae(image, filtered, skeleton=pruned, mask=mask,
              save_path=str(output_dir / "min_4_overlay.png"))
print("done")