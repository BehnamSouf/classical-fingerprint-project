# scripts/evaluate_minutiae_matching.py
"""
Evaluate the minutiae-based matcher (matcher.py + minutiae.py) on the FVC
dataset, reporting FAR/FRR/EER -- the two metrics the job posting asked for.

Run:
    python scripts/evaluate_minutiae_matching.py

Optional: limit the number of impostor pairs for a quick smoke test:
    python scripts/evaluate_minutiae_matching.py --limit 50
"""
import argparse
import time

import numpy as np

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.frequency import estimate_frequency_field
from fingerprint_dataset.enhancement import enhance
from fingerprint_dataset.minutiae import extract_minutiae_from_enhanced, MinutiaeConfig
from fingerprint_dataset.matcher import MinutiaeMatcher, MatcherConfig, make_template, compute_eer


def process_sample(sample, minutiae_cfg: MinutiaeConfig):
    """
    Run the full classical pipeline on one sample and extract its minutiae.
    Returns a matcher.Template (minutiae + mask) plus the minutiae count,
    so the caller can flag samples with suspiciously few minutiae before
    they silently produce a zero score.
    """
    image = sample.load()
    mask = segment(image)
    orientations, strengths = estimate_orientation(image, mask)
    periods = estimate_frequency_field(image, orientations, mask)
    enhanced = enhance(image, mask, orientations, periods, strengths=strengths)

    result = extract_minutiae_from_enhanced(
        enhanced, mask, coherence=strengths, period_map=periods,
        reference=image, config=minutiae_cfg,
    )
    template = make_template(result.minutiae, mask)
    return template, len(result.minutiae)


def score_pairs(pairs, templates, matcher):
    scores = []
    zero_hyp_count = 0
    for s1, s2 in pairs:
        template_a = templates[(s1.finger_id, s1.impression_id)]
        template_b = templates[(s2.finger_id, s2.impression_id)]
        result = matcher.match(template_a, template_b)
        scores.append(result.score)
        if result.n_hypotheses == 0:
            zero_hyp_count += 1
    return np.array(scores), zero_hyp_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="limit number of impostor pairs (for a quick test)")
    args = parser.parse_args()

    ds = FVCDataset("data/FVC2002_DB2_B")
    minutiae_cfg = MinutiaeConfig()  # defaults; tune here if needed
    matcher = MinutiaeMatcher(MatcherConfig())

    print(f"Extracting minutiae for all {len(ds)} images...")
    templates = {}
    low_minutiae_samples = []
    t0 = time.time()
    for i, sample in enumerate(ds.samples):
        template, n_minutiae = process_sample(sample, minutiae_cfg)
        templates[(sample.finger_id, sample.impression_id)] = template
        if n_minutiae < matcher.config.min_minutiae:
            low_minutiae_samples.append((sample.finger_id, sample.impression_id, n_minutiae))
        print(f"  {i + 1}/{len(ds)} (finger={sample.finger_id}, "
              f"impression={sample.impression_id}, minutiae={n_minutiae})", end="\r")
    print(f"\nDone in {time.time() - t0:.1f}s")

    if low_minutiae_samples:
        print(f"\nWarning: {len(low_minutiae_samples)} samples have fewer than "
              f"{matcher.config.min_minutiae} minutiae (will always score 0):")
        for finger_id, impression_id, n in low_minutiae_samples:
            print(f"  finger={finger_id}, impression={impression_id}: {n} minutiae")

    genuine_pairs = ds.genuine_pairs()
    impostor_pairs = ds.impostor_pairs()
    if args.limit is not None:
        import random
        random.seed(42)
        genuine_pairs = random.sample(genuine_pairs, min(args.limit, len(genuine_pairs)))
        impostor_pairs = random.sample(impostor_pairs, min(args.limit, len(impostor_pairs)))

    print(f"\nScoring {len(genuine_pairs)} genuine pairs...")
    genuine_scores, genuine_zero_hyp = score_pairs(genuine_pairs, templates, matcher)

    print(f"Scoring {len(impostor_pairs)} impostor pairs...")
    impostor_scores, impostor_zero_hyp = score_pairs(impostor_pairs, templates, matcher)

    print(f"\nGenuine:  mean={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}, "
          f"zero-score={np.count_nonzero(genuine_scores == 0)}/{len(genuine_scores)}, "
          f"no-alignment-found={genuine_zero_hyp}")
    print(f"Impostor: mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}, "
          f"zero-score={np.count_nonzero(impostor_scores == 0)}/{len(impostor_scores)}, "
          f"no-alignment-found={impostor_zero_hyp}")

    eer, threshold = compute_eer(genuine_scores, impostor_scores)
    print(f"\nEER = {eer * 100:.2f}% at threshold = {threshold:.4f}")


if __name__ == "__main__":
    main()