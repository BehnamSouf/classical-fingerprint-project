# Classical Fingerprint Recognition Pipeline

A from-scratch implementation of a classical (non-deep-learning) fingerprint
recognition pipeline, covering segmentation, ridge orientation and frequency
estimation, Gabor-based enhancement, minutiae extraction, and minutiae-based
matching. Evaluated on FVC2002 with FAR/FRR/EER.

Built as a learning and portfolio project, with each stage independently
validated against synthetic test cases and cross-checked against the
published [pyfing]

## Pipeline
raw image
-> segmentation.py foreground / background mask
-> orientation.py per-pixel ridge orientation + coherence
-> frequency.py local ridge period (X-signature method)
-> enhancement.py Gabor-bank enhancement, guided by orientation/period
-> minutiae.py binarization -> skeleton -> pruning -> minutiae
-> matcher.py Hough-based rigid alignment + one-to-one pairing

`pipeline.py` ties all of this into two functions: `process_image()` (raw
image -> everything up to a matchable template) and `match()` /
`match_images()` (two templates or two raw images -> similarity score).

## Results

Evaluated on FVC2002 DB2_B (10 fingers x 8 impressions, 80 images):

| Metric | Value |
|---|---|
| EER | 6.79% |
| Genuine score (mean / std) | 0.284 / 0.160 |
| Impostor score (mean / std) | 0.024 / 0.018 |
| Pairs evaluated | 280 genuine, 500 impostor |

Reproduce with:
```bash
python scripts/evaluate_minutiae_matching.py --limit 500
```

## Module notes

- **segmentation.py** -- gradient-magnitude foreground detection with
  morphological cleanup (closing -> largest component -> hole filling ->
  opening), matching the GradMag/GMFS approach.
- **orientation.py** -- least-squares gradient orientation with doubled-angle
  averaging (Gaussian-smoothed, pixel-level), plus a coherence/strength map
  used later to blend enhancement near singular points (core/delta).
- **frequency.py** -- X-signature method: rotate a local window so ridges are
  horizontal, project to 1D, estimate the period from peak/valley spacing;
  gaps are filled by propagation from valid neighbors, then a mask-aware
  Gaussian smooth avoids background zeros leaking into the boundary.
- **enhancement.py** -- a precomputed bank of Gabor filters (period x
  orientation), with the filter carrier built perpendicular to the ridge
  direction, blended with the original image based on orientation coherence
  to avoid artifacts near singular points.
- **minutiae.py** -- binarization, skeletonization (scikit-image or a
  Zhang-Suen fallback), spur/fragment pruning, Crossing-Number minutiae
  detection with direction tracing, and false-minutiae filtering (duplicates,
  broken ridges, bridges, spurs).
- **matcher.py** -- Hough-voting for candidate rigid alignments, Hungarian
  one-to-one pairing refined by a Kabsch least-squares fit, with an
  overlap-normalized score.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Datasets (not included, see `.gitignore`): download the free "B" subset of
FVC2002 DB2 from the official [FVC site](http://bias.csr.unibo.it/fvc2002/)
and place it under `data/FVC2002_DB2_B/`.

## Tests

```bash
pytest tests/ -v
```

## Project structure
src/fingerprint_dataset/
dataset.py FVC dataset loader, genuine/impostor pair generation
segmentation.py
orientation.py
frequency.py
enhancement.py
minutiae.py
matcher.py
pipeline.py ties everything together
scripts/ visualization and evaluation scripts
tests/ pytest unit tests per module