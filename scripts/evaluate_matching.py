# scripts/evaluate_matching.py
import numpy as np

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field
from fingerprint_dataset.enhancement import enhance
from fingerprint_dataset.matching import correlation_score, compute_eer


def process_sample(sample):
    image = sample.load()
    mask = segment(image)
    orientations, strengths = estimate_orientation(image, mask)
    periods = estimate_frequency_field(image, orientations, mask)
    enhanced = enhance(image, mask, orientations, periods, strengths=strengths)
    return image, enhanced, mask


def score_pairs(pairs, processed, image_index: int):
    """image_index: 0 for raw image, 1 for enhanced image (see process_sample order)."""
    scores = []
    failed = 0
    for s1, s2 in pairs:
        img_a, mask_a = processed[(s1.finger_id, s1.impression_id)][image_index], processed[(s1.finger_id, s1.impression_id)][2]
        img_b, mask_b = processed[(s2.finger_id, s2.impression_id)][image_index], processed[(s2.finger_id, s2.impression_id)][2]
        score = correlation_score(img_a, mask_a, img_b, mask_b)
        if score is None:
            failed += 1
        else:
            scores.append(score)
    return np.array(scores), failed


ds = FVCDataset("data/FVC2002_DB2_B")

print(f"Processing all {len(ds)} images through the pipeline...")
processed = {}
for i, sample in enumerate(ds.samples):
    processed[(sample.finger_id, sample.impression_id)] = process_sample(sample)
    print(f"  {i + 1}/{len(ds)}", end="\r")
print()

import random
random.seed(42)

genuine_pairs = random.sample(ds.genuine_pairs(), 50)
impostor_pairs = random.sample(ds.impostor_pairs(), 50)

for label, image_index in [("RAW", 0), ("ENHANCED", 1)]:
    print(f"\n=== {label} ===")

    genuine_scores, genuine_failed = score_pairs(genuine_pairs, processed, image_index)
    impostor_scores, impostor_failed = score_pairs(impostor_pairs, processed, image_index)

    print(f"Genuine:  {len(genuine_scores)} scored, {genuine_failed} failed "
          f"(mean={genuine_scores.mean():.3f}, std={genuine_scores.std():.3f})")
    print(f"Impostor: {len(impostor_scores)} scored, {impostor_failed} failed "
          f"(mean={impostor_scores.mean():.3f}, std={impostor_scores.std():.3f})")

    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    print(f"EER = {eer * 100:.2f}% at threshold = {threshold:.4f}")