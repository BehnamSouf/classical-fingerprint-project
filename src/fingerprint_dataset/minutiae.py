"""
minutiae.py -- minutiae extraction from an enhanced fingerprint image.

Pipeline
--------
    enhanced image + mask
        -> binarize_enhanced()        ridge / valley binary image
        -> skeletonize_ridges()       1-pixel-wide ridge skeleton
        -> prune_skeleton()           remove short spurs and tiny islands
        -> skeletonize_ridges()       re-thin (cleans 2-pixel junction remnants)
        -> extract_minutiae()         Crossing Number + direction tracing
        -> filter_minutiae()          false-minutiae removal
    (all of it in one call: extract_minutiae_from_enhanced())

Conventions (IMPORTANT -- read before writing the matcher)
----------------------------------------------------------
* Coordinates: x = column, y = row, origin at the top-left pixel, y grows
  downward (normal numpy image indexing: image[y, x]).
* Minutia angle: radians in [0, 2*pi), measured counter-clockwise *as seen on
  screen* (i.e. as in a normal math plot, y pointing up):
        angle = atan2(-(dy), dx)          with (dx, dy) in image coordinates
* Ending      : angle points from the ridge end *along the ridge body*, i.e.
                away from the tip, into the ridge.
* Bifurcation : angle is the bisector of the two diverging branches, i.e. it
                points from the junction along the branches (opposite to the
                stem). Ending and bifurcation therefore follow the same rule:
                "the direction in which the ridge(s) continue away from the
                point".
* The ridge orientation field (range [0, pi)) is NOT the minutia angle. The
  minutia angle has a full 2*pi range and is obtained by tracing the skeleton.
  If you need the ISO/IEC 19794-2 convention, check the sign/offset once and
  convert with a constant shift; the matcher only needs consistency between the
  two images being compared.
* Ridge polarity: FVC raw images have DARK ridges. The Gabor output of your
  enhancement may have ridges dark or bright depending on the kernel sign.
  binarize_enhanced(polarity="auto", reference=raw_image) decides it from the
  correlation with the raw image; otherwise pass "dark" / "bright" explicitly.

Dependencies: numpy, scipy. scikit-image is used for skeletonization when
available (a vectorised Zhang-Suen fallback is included). matplotlib is only
needed by draw_minutiae().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:  # optional, faster and better-behaved skeletonization
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:  # pragma: no cover - depends on the environment
    _sk_skeletonize = None

TWO_PI = 2.0 * math.pi

ENDING = "ending"
BIFURCATION = "bifurcation"

# 8-neighbourhood in circular order: N, NE, E, SE, S, SW, W, NW  as (dy, dx).
_NEIGH: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Minutia:
    """A single minutia. See the module docstring for conventions."""

    x: float
    y: float
    angle: float                 # radians in [0, 2*pi)
    type: str                    # ENDING or BIFURCATION
    quality: float = 1.0         # in [0, 1]

    def as_tuple(self) -> Tuple[float, float, float, int, float]:
        return (self.x, self.y, self.angle, 0 if self.type == ENDING else 1, self.quality)


@dataclass
class MinutiaeConfig:
    """All tunable parameters. Distances are expressed as multiples of the
    ridge period (median inter-ridge distance, in pixels) so that the module
    adapts to the image resolution."""

    # --- binarization ---
    polarity: str = "dark"                # "dark" | "bright" | "auto"
    binarize_sigma: float = 6.0           # Gaussian sigma of the local mean (px)
    threshold_offset: float = 0.0         # gray levels; >0 makes ridges thinner
    smooth_binary: bool = True            # 3x3 majority filter
    min_object_area: int = 30             # remove ridge blobs smaller than this
    min_hole_area: int = 30               # fill holes smaller than this

    # --- skeleton pruning ---
    min_branch_period_factor: float = 1.0  # spur length threshold = factor * period
    prune_iterations: int = 2
    remove_islands: bool = True            # also drop short isolated segments
    min_component_period_factor: float = 3.0  # drop isolated skeleton pieces shorter
                                              # than factor * period pixels (0 = off)

    # --- direction tracing ---
    track_period_factor: float = 1.1       # trace length = factor * period
    min_track_length: int = 6
    max_track_length: int = 14
    min_arm_length: int = 3                # shorter arms => unreliable minutia

    # --- filtering ---
    border_margin_factor: float = 1.5      # distance from mask border (x period)
    min_quality: float = 0.0
    dup_dist_factor: float = 0.35          # same-type duplicates
    break_dist_factor: float = 1.6         # two facing endings (broken ridge)
    break_angle_tol_deg: float = 40.0
    bridge_dist_factor: float = 0.6        # two close bifurcations
    spur_dist_factor: float = 0.7          # ending next to a bifurcation
    max_minutiae: Optional[int] = None     # keep only the best N (None = all)

    # --- period ---
    default_period: float = 9.0            # used when no period map is given
    min_period: float = 5.0
    max_period: float = 15.0


@dataclass
class MinutiaeResult:
    minutiae: List[Minutia]                # after filtering
    raw_minutiae: List[Minutia]            # before filtering
    binary: np.ndarray                     # bool, ridges = True
    skeleton: np.ndarray                   # bool, before pruning
    skeleton_pruned: np.ndarray            # bool, used for extraction
    polarity: str                          # polarity actually used
    period: float                          # ridge period used (px)
    config: MinutiaeConfig = field(default_factory=MinutiaeConfig)
    filter_stats: dict = field(default_factory=dict)   # removals per filter rule
    n_fragments_removed: int = 0           # isolated short skeleton pieces dropped


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def angle_from_vector(dx: float, dy: float) -> float:
    """Angle in [0, 2*pi) of an image-coordinate vector (dx, dy), measured
    counter-clockwise as seen on screen (y axis flipped)."""
    return math.atan2(-dy, dx) % TWO_PI


def angle_difference(a: float, b: float) -> float:
    """Signed smallest difference a - b, wrapped to (-pi, pi]."""
    d = (a - b + math.pi) % TWO_PI - math.pi
    return math.pi if d == -math.pi else d


def minutiae_to_array(minutiae: Sequence[Minutia]) -> np.ndarray:
    """(N, 5) float32 array with columns x, y, angle, type (0 ending,
    1 bifurcation), quality -- convenient input for a matcher."""
    if not minutiae:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray([m.as_tuple() for m in minutiae], dtype=np.float32)


def minutiae_from_array(arr: np.ndarray) -> List[Minutia]:
    """Inverse of minutiae_to_array()."""
    out = []
    for x, y, a, t, q in np.asarray(arr, dtype=float).reshape(-1, 5):
        out.append(Minutia(float(x), float(y), float(a) % TWO_PI,
                           ENDING if int(t) == 0 else BIFURCATION, float(q)))
    return out


def estimate_ridge_period(
    period_map: Optional[np.ndarray],
    mask: np.ndarray,
    default: float = 9.0,
    lo: float = 5.0,
    hi: float = 15.0,
) -> float:
    """Median ridge period (pixels) inside the mask. NaN / non-positive values
    are ignored. Falls back to `default` when nothing valid is available."""
    if period_map is None:
        return float(np.clip(default, lo, hi))
    p = np.asarray(period_map, dtype=np.float64)
    valid = np.isfinite(p) & (p > 0) & np.asarray(mask, dtype=bool)
    if not valid.any():
        return float(np.clip(default, lo, hi))
    return float(np.clip(np.median(p[valid]), lo, hi))


def _border_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest non-mask pixel. The
    image border counts as non-mask, so a mask covering the whole image still
    keeps minutiae away from the image edges."""
    m = np.pad(np.asarray(mask).astype(bool), 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_edt(m)[1:-1, 1:-1]


def _validate_inputs(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    if mask.shape != image.shape:
        raise ValueError(f"mask shape {mask.shape} != image shape {image.shape}")
    return image, mask


# ----------------------------------------------------------------------------
# 1. Binarization
# ----------------------------------------------------------------------------
def _masked_gaussian_mean(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian local mean computed only over masked pixels (normalized
    convolution), so the background does not bias the mean."""
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(values * m, sigma, mode="nearest")
    den = ndimage.gaussian_filter(m, sigma, mode="nearest")
    return num / np.maximum(den, 1e-6)


def _remove_small_components(binary: np.ndarray, min_area: int, connectivity8: bool) -> np.ndarray:
    if min_area <= 1 or not binary.any():
        return binary
    structure = np.ones((3, 3), dtype=bool) if connectivity8 else None
    labels, n = ndimage.label(binary, structure=structure)
    if n == 0:
        return binary
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def choose_polarity(
    enhanced: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    sigma: float = 6.0,
) -> str:
    """Decide whether ridges are dark or bright in `enhanced`, using a raw
    image whose ridges are dark (FVC convention) as reference: if the
    locally-normalised enhanced and raw images are positively correlated, the
    enhanced image has dark ridges too."""
    enh = np.asarray(enhanced, dtype=np.float32)
    ref = ndimage.gaussian_filter(np.asarray(reference, dtype=np.float32), 1.0)
    d_e = enh - _masked_gaussian_mean(enh, mask, sigma)
    d_r = ref - _masked_gaussian_mean(ref, mask, sigma)
    a, b = d_e[mask], d_r[mask]
    if a.size == 0:
        return "dark"
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 1e-9:
        return "dark"
    corr = float((a * b).sum() / denom)
    return "dark" if corr >= 0 else "bright"


def binarize_enhanced(
    enhanced: np.ndarray,
    mask: np.ndarray,
    polarity: str = "dark",
    reference: Optional[np.ndarray] = None,
    sigma: float = 6.0,
    threshold_offset: float = 0.0,
    smooth: bool = True,
    min_object_area: int = 30,
    min_hole_area: int = 30,
) -> Tuple[np.ndarray, str]:
    """Binarize an enhanced fingerprint with a masked local-mean threshold.

    Returns (ridges, polarity_used) where `ridges` is a bool array that is
    True on ridge pixels and False elsewhere (always False outside the mask).

    polarity: "dark"   -> ridges are darker than their local mean
              "bright" -> ridges are brighter than their local mean
              "auto"   -> decided from `reference` (raw image, dark ridges);
                          falls back to "dark" when no reference is given.
    """
    enhanced, mask = _validate_inputs(enhanced, mask)
    if polarity not in ("dark", "bright", "auto"):
        raise ValueError("polarity must be 'dark', 'bright' or 'auto'")
    if polarity == "auto":
        polarity = "dark" if reference is None else choose_polarity(enhanced, reference, mask, sigma)

    enh = enhanced.astype(np.float32)
    diff = enh - _masked_gaussian_mean(enh, mask, sigma)
    if polarity == "dark":
        ridges = diff < -threshold_offset
    else:
        ridges = diff > threshold_offset
    ridges &= mask

    if smooth:
        ridges = ndimage.uniform_filter(ridges.astype(np.float32), size=3, mode="constant") > 0.5
        ridges &= mask

    # remove tiny ridge blobs (8-connectivity)
    ridges = _remove_small_components(ridges, min_object_area, connectivity8=True)
    # fill tiny holes STRICTLY WITHIN THE MASK: hole candidates are background
    # pixels inside the mask only (4-connectivity, the dual of 8-connected
    # foreground). The outside-mask background is never part of a hole
    # component, so it cannot influence hole filling.
    bg_inside = (~ridges) & mask
    large_bg = _remove_small_components(bg_inside, min_hole_area, connectivity8=False)
    small_holes = bg_inside & ~large_bg
    ridges = (ridges | small_holes) & mask
    return ridges, polarity


# ----------------------------------------------------------------------------
# 2. Skeletonization
# ----------------------------------------------------------------------------
def _zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Vectorised Zhang-Suen thinning (fallback when scikit-image is missing)."""
    img = np.pad(binary.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            c = img[1:-1, 1:-1]
            b = (p2.astype(np.int16) + p3 + p4 + p5 + p6 + p7 + p8 + p9)
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            a = np.zeros_like(b)
            for k in range(8):
                a += ((seq[k] == 0) & (seq[k + 1] == 1))
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            delete = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if delete.any():
                img[1:-1, 1:-1][delete] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def skeletonize_ridges(binary: np.ndarray, method: str = "auto") -> np.ndarray:
    """Thin a binary ridge image to a 1-pixel-wide, 8-connected skeleton.

    method: "auto" (scikit-image if installed, else Zhang-Suen),
            "skimage", or "zhang_suen".
    """
    binary = np.asarray(binary).astype(bool)
    if method == "auto":
        method = "skimage" if _sk_skeletonize is not None else "zhang_suen"
    if method == "skimage":
        if _sk_skeletonize is None:
            raise RuntimeError("scikit-image is not installed")
        return np.asarray(_sk_skeletonize(binary), dtype=bool)
    if method == "zhang_suen":
        return _zhang_suen(binary)
    raise ValueError(f"unknown skeletonization method: {method}")


# ----------------------------------------------------------------------------
# 3. Crossing number
# ----------------------------------------------------------------------------
def _neighbor_stack(skel: np.ndarray) -> np.ndarray:
    """(8, H, W) int8 stack of neighbour values in circular order."""
    p = np.pad(skel.astype(np.int8), 1)
    h, w = skel.shape
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NEIGH])


def crossing_number(skeleton: np.ndarray) -> np.ndarray:
    """Crossing number of every skeleton pixel (0 elsewhere).

        CN = 1/2 * sum_i |P_i - P_(i+1)|   over the 8 circular neighbours

    For a thin skeleton CN equals the number of separate neighbour runs:
    CN=1 ending, CN=2 ridge interior, CN=3 bifurcation, CN>=4 crossing.
    """
    skel = np.asarray(skeleton).astype(bool)
    n = _neighbor_stack(skel)
    cn = np.abs(n - np.roll(n, -1, axis=0)).sum(axis=0) // 2
    cn = cn.astype(np.int8)
    cn[~skel] = 0
    return cn


# ----------------------------------------------------------------------------
# Skeleton tracing (works on a zero-padded uint8 array; padded coordinates)
# ----------------------------------------------------------------------------
def _cn_at(S: np.ndarray, y: int, x: int) -> int:
    v = [int(S[y + dy, x + dx]) for dy, dx in _NEIGH]
    return sum(abs(v[i] - v[(i + 1) % 8]) for i in range(8)) // 2


def _trace(
    S: np.ndarray,
    start: Tuple[int, int],
    visited: set,
    max_len: int,
    stop_at_junction: bool = True,
) -> Tuple[List[Tuple[int, int]], str]:
    """Follow the skeleton from `start` (which must be a skeleton pixel not in
    `visited`) for at most `max_len` pixels.

    Returns (path, reason), path including `start`. reason is one of:
      "length"   -> reached max_len
      "end"      -> no unvisited neighbour left (dead end)
      "junction" -> a junction (CN >= 3) is adjacent to the last pixel; it is
                    NOT included in path (only when stop_at_junction=True)
    """
    path = [start]
    visited = set(visited)
    visited.add(start)
    cy, cx = start
    prev_move = None
    while len(path) < max_len:
        cands = []
        for dy, dx in _NEIGH:
            ny, nx = cy + dy, cx + dx
            if S[ny, nx] and (ny, nx) not in visited:
                cands.append((ny, nx))
        if not cands:
            return path, "end"
        if stop_at_junction and any(_cn_at(S, c[0], c[1]) >= 3 for c in cands):
            return path, "junction"
        if len(cands) > 1:
            if prev_move is not None:
                pdy, pdx = prev_move
                cands.sort(key=lambda c: -((c[0] - cy) * pdy + (c[1] - cx) * pdx))
            else:
                cands.sort(key=lambda c: abs(c[0] - cy) + abs(c[1] - cx))
        nxt = cands[0]
        prev_move = (nxt[0] - cy, nxt[1] - cx)
        visited.add(nxt)
        path.append(nxt)
        cy, cx = nxt
    return path, "length"


def _neighbor_runs(S: np.ndarray, y: int, x: int) -> List[List[Tuple[int, int]]]:
    """Runs of consecutive skeleton neighbours (circular order) around (y, x).
    Each run is a list of (y, x) pixels; a run is one 'arm' leaving the point."""
    vals = [int(S[y + dy, x + dx]) for dy, dx in _NEIGH]
    if all(vals):
        return []
    # rotate so that the scan starts on a zero -> runs never wrap around
    start = vals.index(0)
    order = [(start + i) % 8 for i in range(8)]
    runs: List[List[Tuple[int, int]]] = []
    cur: List[Tuple[int, int]] = []
    for idx in order:
        if vals[idx]:
            dy, dx = _NEIGH[idx]
            cur.append((y + dy, x + dx))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


# ----------------------------------------------------------------------------
# 4. Skeleton pruning
# ----------------------------------------------------------------------------
def prune_skeleton(
    skeleton: np.ndarray,
    min_branch_length: int,
    iterations: int = 2,
    remove_islands: bool = True,
) -> np.ndarray:
    """Remove short spurs (branches shorter than `min_branch_length` pixels
    that end at a junction) and, optionally, short isolated segments.

    Long ridges are never shortened. Repeating the pass (`iterations`) removes
    spurs that only become visible after a neighbouring spur is gone."""
    skel = np.asarray(skeleton).astype(bool)
    S = np.pad(skel.astype(np.uint8), 1)
    min_len = max(int(min_branch_length), 1)

    for _ in range(max(int(iterations), 1)):
        cn = _neighbor_cn_padded(S)
        endpoints = np.argwhere((S == 1) & (cn == 1))
        removed_any = False
        for y, x in endpoints:
            y, x = int(y), int(x)
            if not S[y, x] or _cn_at(S, y, x) != 1:
                continue  # already removed / changed by an earlier deletion
            path, reason = _trace(S, (y, x), set(), min_len)
            short = len(path) < min_len
            if reason == "junction" and short:
                for py, px in path:
                    S[py, px] = 0
                removed_any = True
            elif reason == "end" and short and remove_islands:
                for py, px in path:
                    S[py, px] = 0
                removed_any = True
        if not removed_any:
            break
    return S[1:-1, 1:-1].astype(bool)


def remove_short_components(skeleton: np.ndarray, min_pixels: int) -> Tuple[np.ndarray, int]:
    """Remove 8-connected skeleton pieces with fewer than `min_pixels` pixels
    (isolated ridge fragments, which always produce two unreliable endings).
    Returns (cleaned skeleton, number of pieces removed)."""
    skel = np.asarray(skeleton).astype(bool)
    if min_pixels <= 1 or not skel.any():
        return skel, 0
    labels, n = ndimage.label(skel, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    small = sizes < min_pixels
    small[0] = False
    return skel & ~small[labels], int(small.sum())


def _neighbor_cn_padded(S: np.ndarray) -> np.ndarray:
    """Crossing number for an already zero-padded array (same padded shape)."""
    h, w = S.shape
    cn = np.zeros((h, w), dtype=np.int8)
    inner = crossing_number(S[1:-1, 1:-1].astype(bool))
    cn[1:-1, 1:-1] = inner
    return cn


# ----------------------------------------------------------------------------
# 5. Minutiae extraction
# ----------------------------------------------------------------------------
def _local_coherence(coherence: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if coherence is None:
        return np.ones(shape, dtype=np.float32)
    c = np.asarray(coherence, dtype=np.float32)
    if c.shape != shape:
        raise ValueError(f"coherence shape {c.shape} != image shape {shape}")
    c = np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0)
    return ndimage.uniform_filter(np.clip(c, 0.0, 1.0), size=5, mode="nearest")


def extract_minutiae(
    skeleton: np.ndarray,
    mask: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    track_length: int = 10,
    min_arm_length: int = 3,
    border_margin: float = 0.0,
) -> List[Minutia]:
    """Detect endings (CN=1) and bifurcations (CN=3) on a thin skeleton and
    compute their 2*pi-range direction by tracing the ridge arms.

    Only pixels at least `border_margin` pixels inside the mask are examined.
    `quality` = local mean coherence (1.0 if no coherence map is given) times a
    border factor that ramps from 0 to 1 over 2*border_margin pixels.
    """
    skel = np.asarray(skeleton).astype(bool)
    skel, mask = _validate_inputs(skel, mask)
    h, w = skel.shape

    dist = _border_distance(mask)
    qcoh = _local_coherence(coherence, skel.shape)
    cn = crossing_number(skel)
    S = np.pad(skel.astype(np.uint8), 1)

    ys, xs = np.nonzero(((cn == 1) | (cn == 3)) & mask & (dist >= max(border_margin, 0.0)))
    out: List[Minutia] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        py, px = y + 1, x + 1
        runs = _neighbor_runs(S, py, px)

        angles: List[float] = []
        ok = True
        for r, run in enumerate(runs):
            rep = run[len(run) // 2]
            # the point itself and the pixels of the OTHER arms are off limits,
            # so that arms cannot leak into each other through the junction
            visited = {(py, px)}
            for r2, other in enumerate(runs):
                if r2 != r:
                    visited.update(other)
            path, _ = _trace(S, rep, visited, track_length, stop_at_junction=False)
            if len(path) < min_arm_length:
                ok = False
                break
            ey, ex = path[-1]
            angles.append(angle_from_vector(ex - px, ey - py))
        if not ok or not angles:
            continue

        if cn[y, x] == 1:
            if len(angles) != 1:
                continue
            angle, mtype = angles[0], ENDING
        else:
            if len(angles) != 3:
                continue
            angle = _bifurcation_angle(angles)
            mtype = BIFURCATION

        bf = 1.0 if border_margin <= 0 else float(np.clip(dist[y, x] / (2.0 * border_margin), 0.0, 1.0))
        quality = float(np.clip(qcoh[y, x] * bf, 0.0, 1.0))
        out.append(Minutia(float(x), float(y), float(angle), mtype, quality))
    return out


def _bifurcation_angle(arm_angles: Sequence[float]) -> float:
    """Direction of a bifurcation from its three arm directions: the two arms
    with the smallest angular separation are the branches, the remaining one is
    the stem; the minutia angle is the bisector of the branches (opposite to
    the stem)."""
    a = list(arm_angles)
    best = None
    for i in range(3):
        for j in range(i + 1, 3):
            sep = abs(angle_difference(a[i], a[j]))
            if best is None or sep < best[0]:
                best = (sep, i, j)
    _, i, j = best
    k = 3 - i - j
    sx = math.cos(a[i]) + math.cos(a[j])
    sy = math.sin(a[i]) + math.sin(a[j])
    if math.hypot(sx, sy) < 1e-6:  # branches exactly opposite: use the stem
        return (a[k] + math.pi) % TWO_PI
    return math.atan2(sy, sx) % TWO_PI


# ----------------------------------------------------------------------------
# 6. False-minutiae removal
# ----------------------------------------------------------------------------
def _pairs(minutiae: Sequence[Minutia], alive: np.ndarray, radius: float):
    idx = np.flatnonzero(alive)
    if idx.size < 2:
        return []
    pts = np.array([[minutiae[i].x, minutiae[i].y] for i in idx])
    tree = cKDTree(pts)
    pairs = sorted(tree.query_pairs(radius))
    return [(int(idx[a]), int(idx[b])) for a, b in pairs]


def filter_minutiae(
    minutiae: Sequence[Minutia],
    mask: np.ndarray,
    period: float = 9.0,
    config: Optional[MinutiaeConfig] = None,
    stats: Optional[dict] = None,
) -> List[Minutia]:
    """Remove unreliable minutiae. Rules, applied in this order:

    1. outside the mask, or closer than border_margin to its border
    2. quality < min_quality
    3. duplicates: same type closer than dup_dist -> keep the best one
    4. broken ridge: two endings facing each other within break_dist
       (opposite directions and the segment joining them aligned with them)
       -> remove both
    5. bridge / lake: two bifurcations closer than bridge_dist -> remove both
    6. spur: an ending and a bifurcation closer than spur_dist -> remove both
    7. optional cap: keep the max_minutiae best by quality

    If a dict is passed as `stats`, it is filled with the number of minutiae
    removed by each rule (keys: border, quality, duplicate, broken_ridge,
    bridge, spur, cap) -- useful to see which rule is doing the damage.
    """
    cfg = config or MinutiaeConfig()
    if stats is not None:
        for key in ("border", "quality", "duplicate", "broken_ridge", "bridge", "spur", "cap"):
            stats[key] = 0
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    ms = list(minutiae)
    if not ms:
        return []

    # 1 + 2 -----------------------------------------------------------------
    dist = _border_distance(mask)
    margin = cfg.border_margin_factor * period
    keep = []
    for m in ms:
        xi, yi = int(round(m.x)), int(round(m.y))
        if not (0 <= xi < w and 0 <= yi < h):
            if stats is not None:
                stats["border"] += 1
            continue
        if dist[yi, xi] < margin:
            if stats is not None:
                stats["border"] += 1
            continue
        if m.quality < cfg.min_quality:
            if stats is not None:
                stats["quality"] += 1
            continue
        keep.append(m)
    ms = keep
    if not ms:
        return []

    alive = np.ones(len(ms), dtype=bool)

    # 3 duplicates ----------------------------------------------------------
    dup_dist = max(cfg.dup_dist_factor * period, 2.0)
    for i, j in _pairs(ms, alive, dup_dist):
        if alive[i] and alive[j] and ms[i].type == ms[j].type:
            loser = j if ms[i].quality >= ms[j].quality else i
            alive[loser] = False
            if stats is not None:
                stats["duplicate"] += 1

    # 4 broken ridges (ending - ending) --------------------------------------
    break_dist = cfg.break_dist_factor * period
    tol = math.radians(cfg.break_angle_tol_deg)
    for i, j in _pairs(ms, alive, break_dist):
        if not (alive[i] and alive[j]):
            continue
        a, b = ms[i], ms[j]
        if a.type != ENDING or b.type != ENDING:
            continue
        opposite = abs(abs(angle_difference(a.angle, b.angle)) - math.pi) < tol
        if not opposite:
            continue
        # a's ridge body lies along a.angle, so the gap towards b must lie
        # along a.angle + pi (and vice versa)
        ab = angle_from_vector(b.x - a.x, b.y - a.y)
        ba = angle_from_vector(a.x - b.x, a.y - b.y)
        aligned = (abs(angle_difference(ab, a.angle + math.pi)) < 1.2 * tol and
                   abs(angle_difference(ba, b.angle + math.pi)) < 1.2 * tol)
        if aligned:
            alive[i] = alive[j] = False
            if stats is not None:
                stats["broken_ridge"] += 2

    # 5 bridges / lakes (bifurcation - bifurcation) --------------------------
    for i, j in _pairs(ms, alive, cfg.bridge_dist_factor * period):
        if alive[i] and alive[j] and ms[i].type == BIFURCATION and ms[j].type == BIFURCATION:
            alive[i] = alive[j] = False
            if stats is not None:
                stats["bridge"] += 2

    # 6 spurs (ending - bifurcation) ------------------------------------------
    for i, j in _pairs(ms, alive, cfg.spur_dist_factor * period):
        if alive[i] and alive[j] and ms[i].type != ms[j].type:
            alive[i] = alive[j] = False
            if stats is not None:
                stats["spur"] += 2

    result = [m for m, a in zip(ms, alive) if a]

    # 7 cap ------------------------------------------------------------------
    if cfg.max_minutiae is not None and len(result) > cfg.max_minutiae:
        result.sort(key=lambda m: -m.quality)
        if stats is not None:
            stats["cap"] += len(result) - cfg.max_minutiae
        result = result[: cfg.max_minutiae]
    return result


# ----------------------------------------------------------------------------
# 7. Full pipeline
# ----------------------------------------------------------------------------
def extract_minutiae_from_enhanced(
    enhanced: np.ndarray,
    mask: np.ndarray,
    coherence: Optional[np.ndarray] = None,
    period_map: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    config: Optional[MinutiaeConfig] = None,
    skeleton_method: str = "auto",
) -> MinutiaeResult:
    """Enhanced image -> filtered minutiae (plus all intermediate images).

    enhanced   : 2-D image (uint8 or float) from the enhancement stage
    mask       : bool foreground mask from segmentation
    coherence  : optional [0, 1] orientation coherence / strength map
    period_map : optional ridge period map (pixels); NaN/<=0 are ignored
    reference  : optional raw image, only used when config.polarity == "auto"
    """
    cfg = config or MinutiaeConfig()
    enhanced, mask = _validate_inputs(enhanced, mask)

    period = estimate_ridge_period(period_map, mask, cfg.default_period,
                                   cfg.min_period, cfg.max_period)

    binary, polarity = binarize_enhanced(
        enhanced, mask, polarity=cfg.polarity, reference=reference,
        sigma=cfg.binarize_sigma, threshold_offset=cfg.threshold_offset,
        smooth=cfg.smooth_binary, min_object_area=cfg.min_object_area,
        min_hole_area=cfg.min_hole_area,
    )

    skeleton = skeletonize_ridges(binary, method=skeleton_method)

    min_branch = int(round(np.clip(cfg.min_branch_period_factor * period, 4, 20)))
    pruned = prune_skeleton(skeleton, min_branch, cfg.prune_iterations, cfg.remove_islands)
    pruned = skeletonize_ridges(pruned, method=skeleton_method)  # clean junctions
    n_frag = 0
    if cfg.min_component_period_factor > 0:
        pruned, n_frag = remove_short_components(
            pruned, int(round(cfg.min_component_period_factor * period)))

    track = int(round(np.clip(cfg.track_period_factor * period,
                              cfg.min_track_length, cfg.max_track_length)))
    margin = cfg.border_margin_factor * period
    raw = extract_minutiae(pruned, mask, coherence=coherence, track_length=track,
                           min_arm_length=cfg.min_arm_length, border_margin=margin)
    stats: dict = {}
    filtered = filter_minutiae(raw, mask, period=period, config=cfg, stats=stats)

    return MinutiaeResult(
        minutiae=filtered, raw_minutiae=raw, binary=binary, skeleton=skeleton,
        skeleton_pruned=pruned, polarity=polarity, period=period, config=cfg,
        filter_stats=stats, n_fragments_removed=n_frag,
    )


# ----------------------------------------------------------------------------
# 8. Visualization
# ----------------------------------------------------------------------------
def draw_minutiae(
    image: np.ndarray,
    minutiae: Sequence[Minutia],
    skeleton: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    direction_length: float = 12.0,
    removed: Sequence[Minutia] = (),
):
    """Overlay minutiae on the image: red = ending, blue = bifurcation, with a
    short line showing each minutia's direction. Optionally overlays the
    skeleton (green), the mask contour (yellow) and, via `removed`, the
    minutiae rejected by the filter (small orange crosses). Returns the figure."""
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / max(image.shape[1], 1) + 0.5))
    ax.imshow(image, cmap="gray", interpolation="nearest")
    if skeleton is not None:
        ov = np.zeros(skeleton.shape + (4,), dtype=np.float32)
        ov[skeleton] = (0.0, 0.9, 0.0, 0.55)
        ax.imshow(ov, interpolation="nearest")
    if mask is not None and np.asarray(mask).any():
        ax.contour(np.asarray(mask).astype(float), levels=[0.5], colors="yellow", linewidths=0.8)

    for m in removed:
        ax.plot(m.x, m.y, "x", color="orange", markersize=7, markeredgewidth=1.6)

    for m in minutiae:
        color = "red" if m.type == ENDING else "blue"
        ax.plot(m.x, m.y, "o", markerfacecolor="none", markeredgecolor=color,
                markersize=8, markeredgewidth=1.4)
        ax.plot([m.x, m.x + direction_length * math.cos(m.angle)],
                [m.y, m.y - direction_length * math.sin(m.angle)],
                "-", color=color, linewidth=1.2)
    n_e = sum(m.type == ENDING for m in minutiae)
    ax.set_title(title or f"minutiae: {n_e} endings (red), {len(minutiae) - n_e} bifurcations (blue)")
    ax.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


# ----------------------------------------------------------------------------
# Command line helper
# ----------------------------------------------------------------------------
def _load_gray(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def _parse_overrides(pairs: Sequence[str], cfg: MinutiaeConfig) -> None:
    """Apply KEY=VALUE overrides to a MinutiaeConfig (types are taken from the
    existing field value; use 'none' for Optional fields)."""
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, val = item.split("=", 1)
        key = key.strip()
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown parameter '{key}'. Valid: {', '.join(cfg.__dataclass_fields__)}")
        cur = getattr(cfg, key)
        if val.strip().lower() == "none":
            new = None
        elif isinstance(cur, bool):
            new = val.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            new = int(val)
        elif isinstance(cur, float) or cur is None:
            new = float(val)
            if cur is None and float(val).is_integer():
                new = int(val)
        else:
            new = val
        setattr(cfg, key, new)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the extraction on image files and save an overlay for visual
    inspection -- the first thing to check before building the matcher:

        python minutiae.py --enhanced enh.png --raw raw.tif --mask mask.png \\
                           --out minutiae_overlay.png --show-removed \\
                           --set break_dist_factor=1.8 --set bridge_dist_factor=0.4

    --mask is optional if the project package `fingerprint_dataset` is
    importable (its segment() is then used on --raw / --enhanced).
    Any MinutiaeConfig field can be overridden with --set KEY=VALUE.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Extract minutiae from an enhanced fingerprint.")
    ap.add_argument("--enhanced", required=True, help="enhanced image (ridges dark or bright)")
    ap.add_argument("--raw", help="raw image (dark ridges); used for polarity=auto and as overlay background")
    ap.add_argument("--mask", help="binary mask image (non-zero = fingerprint)")
    ap.add_argument("--polarity", default="auto", choices=["auto", "dark", "bright"])
    ap.add_argument("--period", type=float, default=None, help="ridge period in pixels (default 9)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override any MinutiaeConfig field (repeatable)")
    ap.add_argument("--show-removed", action="store_true",
                    help="draw minutiae rejected by the filter as orange crosses")
    ap.add_argument("--save-debug", metavar="DIR",
                    help="also save binary.png and skeleton.png (pruned) into DIR")
    ap.add_argument("--out", default="minutiae_overlay.png")
    args = ap.parse_args(argv)

    enhanced = _load_gray(args.enhanced)
    raw = _load_gray(args.raw) if args.raw else None

    if args.mask:
        mask = _load_gray(args.mask) > 0
    else:
        try:
            from fingerprint_dataset.segmentation import segment  # project module
        except Exception as exc:
            raise SystemExit(f"--mask not given and project segment() not importable: {exc}")
        mask = np.asarray(segment(raw if raw is not None else enhanced)).astype(bool)

    cfg = MinutiaeConfig(polarity=args.polarity)
    if args.period:
        cfg.default_period = args.period
    _parse_overrides(args.overrides, cfg)
    res = extract_minutiae_from_enhanced(enhanced, mask, reference=raw, config=cfg)

    def counts(ms):
        e = sum(m.type == ENDING for m in ms)
        return f"{len(ms)} ({e} endings, {len(ms) - e} bifurcations)"

    print(f"polarity used : {res.polarity}")
    print(f"ridge period  : {res.period:.1f} px")
    print(f"short pieces  : {res.n_fragments_removed} isolated skeleton fragments removed")
    print(f"raw minutiae  : {counts(res.raw_minutiae)}")
    print(f"after filter  : {counts(res.minutiae)}")
    removed_by = ", ".join(f"{k}={v}" for k, v in res.filter_stats.items() if v)
    print(f"removed by    : {removed_by or 'nothing'}")
    kept = {id(m) for m in res.minutiae}
    removed = [m for m in res.raw_minutiae if id(m) not in kept] if args.show_removed else ()
    draw_minutiae(raw if raw is not None else enhanced, res.minutiae,
                  skeleton=res.skeleton_pruned, mask=mask, save_path=args.out, removed=removed)
    print(f"overlay saved : {args.out}")
    if args.save_debug:
        import os
        from PIL import Image
        os.makedirs(args.save_debug, exist_ok=True)
        Image.fromarray((res.binary * 255).astype(np.uint8)).save(os.path.join(args.save_debug, "binary.png"))
        Image.fromarray((res.skeleton_pruned * 255).astype(np.uint8)).save(os.path.join(args.save_debug, "skeleton.png"))
        print(f"debug images  : {args.save_debug}/binary.png, skeleton.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())