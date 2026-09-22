"""
matcher.py -- minutiae-based fingerprint matcher (rigid alignment by Hough
voting, one-to-one pairing, overlap-normalised scores).

It consumes the output of minutiae.py and returns a similarity score for two
fingerprints, which is what the FAR / FRR / EER evaluation needs.

Algorithm
---------
1. Candidate alignments (Hough voting, Ratha et al. 1996): every compatible
   pair (minutia i of A, minutia j of B) implies a rigid transform
   (rotation = angle_B[j] - angle_A[i], translation from the positions).
   The transforms vote in a quantised (rotation, tx, ty) accumulator; the
   strongest local maxima are kept as hypotheses. Coordinates are centred on
   the centroid of each minutiae set, so translation votes are not smeared by
   rotation errors.
2. For every hypothesis: transform A into B's frame and build a ONE-TO-ONE
   pairing (Hungarian assignment) restricted to pairs closer than `dist_tol`
   pixels, with direction difference below `angle_tol_deg` and compatible
   type. The transform is then refined by a least-squares rigid fit (Kabsch)
   on the paired minutiae and the pairing is recomputed. The hypothesis with
   most pairs wins.
3. Scores are normalised by the OVERLAP region only (minutiae of A that fall
   inside B's mask after alignment, and vice versa), so partially overlapping
   impressions are not penalised for minutiae that cannot be matched anyway.

Scores (both in MatchResult)
    score      = m^2 / (nA_overlap * nB_overlap)           in [0, 1]
                 (0 if an overlap contains fewer than `min_overlap_minutiae`)
    sig_score  = -log10( P[ >= m random matches ] )         in [0, 300]
                 binomial significance test with a random-match probability
                 estimated from the minutiae density in the overlap area.
                 Robust when the overlap is small; try both when computing EER.
m = number of paired minutiae.

Conventions (identical to minutiae.py)
--------------------------------------
* Input minutiae: (N, 5) float array [x, y, angle, type, quality] or a list
  of minutiae.Minutia objects. x = column, y = row (y grows downward); angle
  in radians in [0, 2*pi), counter-clockwise as seen on screen; type 0 =
  ending, 1 = bifurcation.
* Internally the matcher works in a y-up ("math") frame so that rotation
  matrices have their usual meaning; the reported `transform` is converted
  back to IMAGE pixel coordinates: [x', y', 1]^T = transform @ [x, y, 1]^T
  maps a minutia position of A into B's image.
* All tolerances are in pixels / degrees of the images being compared. The
  defaults assume 500 dpi images (ridge period ~9 px, FVC2002-like).

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import binom

TWO_PI = 2.0 * math.pi
_BIG = 1e6


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class MatcherConfig:
    # pairing tolerances
    dist_tol: float = 12.0              # px, max distance after alignment
    angle_tol_deg: float = 25.0         # max direction difference after alignment
    use_types: bool = True              # ending only pairs with ending, etc.
    # Hough voting
    rot_bin_deg: float = 10.0
    trans_bin: float = 8.0              # px
    max_rotation_deg: Optional[float] = None   # e.g. 60 to forbid large rotations
    n_hypotheses: int = 8               # strongest accumulator peaks to test
    refine_iterations: int = 2
    # input filtering / guards
    min_quality: float = 0.0
    min_minutiae: int = 3               # fewer than this in A or B -> score 0
    min_overlap_minutiae: int = 4       # smaller overlap -> `score` = 0
    overlap_step: int = 3               # mask sub-sampling for the overlap area
    max_accumulator_cells: int = 20_000_000


@dataclass
class Template:
    """Minutiae of one fingerprint (+ optional foreground mask)."""

    minutiae: np.ndarray                     # (N, 5): x, y, angle, type, quality
    mask: Optional[np.ndarray] = None        # bool (H, W), True = fingerprint

    def __len__(self) -> int:
        return int(self.minutiae.shape[0])


@dataclass
class MatchResult:
    score: float                             # m^2 / (nA_ov * nB_ov)
    sig_score: float                         # -log10 binomial tail probability
    n_matched: int
    n_a_overlap: int
    n_b_overlap: int
    overlap_area: float                      # px^2 (approximate)
    rotation: float                          # radians, CCW on screen, A -> B
    transform: np.ndarray                    # 3x3, image coordinates, A -> B
    pairs: List[Tuple[int, int]]             # (index in A, index in B)
    mean_pair_distance: float                # px, over the matched pairs
    n_hypotheses: int                        # alignments that were evaluated

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation)


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def as_array(minutiae) -> np.ndarray:
    """Convert a list of minutiae.Minutia (or an array) into an (N, 5) float64
    array [x, y, angle, type, quality]."""
    if isinstance(minutiae, Template):
        return minutiae.minutiae
    if isinstance(minutiae, np.ndarray):
        arr = np.asarray(minutiae, dtype=np.float64).reshape(-1, minutiae.shape[-1] if minutiae.ndim > 1 else 5)
        if arr.shape[1] == 4:                      # no quality column
            arr = np.hstack([arr, np.ones((arr.shape[0], 1))])
        if arr.shape[1] != 5:
            raise ValueError(f"minutiae array must have 5 columns, got {arr.shape[1]}")
        return arr
    rows = []
    for m in minutiae:
        t = getattr(m, "type", 0)
        t = 0 if (t == 0 or t == "ending") else 1
        rows.append((float(m.x), float(m.y), float(m.angle) % TWO_PI, t, float(getattr(m, "quality", 1.0))))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def make_template(minutiae, mask: Optional[np.ndarray] = None) -> Template:
    """Build a Template from minutiae.py output (list of Minutia or array)."""
    arr = as_array(minutiae).copy()
    if arr.size:
        arr[:, 2] = np.mod(arr[:, 2], TWO_PI)
    m = None if mask is None else np.asarray(mask).astype(bool)
    return Template(arr, m)


def _to_template(obj, mask=None) -> Template:
    if isinstance(obj, Template):
        return obj if mask is None else Template(obj.minutiae, np.asarray(mask).astype(bool))
    return make_template(obj, mask)


# ----------------------------------------------------------------------------
# Geometry helpers (math frame: x right, y UP)
# ----------------------------------------------------------------------------
def _wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % TWO_PI - math.pi


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class _Prep:
    pts: np.ndarray        # (N, 2) math-frame positions (x, -y)
    c: np.ndarray          # (N, 2) positions centred on the centroid
    centroid: np.ndarray   # (2,)
    ang: np.ndarray        # (N,)
    typ: np.ndarray        # (N,) int
    mask: Optional[np.ndarray]


def _prepare(t: Template, min_quality: float) -> _Prep:
    arr = t.minutiae
    if arr.size and min_quality > 0:
        arr = arr[arr[:, 4] >= min_quality]
    if arr.shape[0] == 0:
        z = np.zeros((0, 2))
        return _Prep(z, z, np.zeros(2), np.zeros(0), np.zeros(0, dtype=int), t.mask)
    pts = np.column_stack([arr[:, 0], -arr[:, 1]])
    centroid = pts.mean(axis=0)
    return _Prep(pts, pts - centroid, centroid, np.mod(arr[:, 2], TWO_PI),
                 arr[:, 3].astype(int), t.mask)


def _type_compat(ta: np.ndarray, tb: np.ndarray, use_types: bool) -> np.ndarray:
    if not use_types:
        return np.ones((ta.size, tb.size), dtype=bool)
    return ta[:, None] == tb[None, :]


# ----------------------------------------------------------------------------
# Step 1: Hough voting
# ----------------------------------------------------------------------------
def _hough_hypotheses(A: _Prep, B: _Prep, cfg: MatcherConfig) -> List[Tuple[float, np.ndarray, int]]:
    """Return up to cfg.n_hypotheses candidate transforms (dtheta, t, votes),
    with p_B_centred ~= R(dtheta) @ p_A_centred + t."""
    dth = (B.ang[None, :] - A.ang[:, None]) % TWO_PI
    compat = _type_compat(A.typ, B.typ, cfg.use_types)
    if cfg.max_rotation_deg is not None:
        compat &= np.abs(_wrap(dth)) <= math.radians(cfg.max_rotation_deg)
    ia, ib = np.nonzero(compat)
    if ia.size == 0:
        return []
    d = dth[ia, ib]
    c, s = np.cos(d), np.sin(d)
    ax, ay = A.c[ia, 0], A.c[ia, 1]
    tx = B.c[ib, 0] - (c * ax - s * ay)
    ty = B.c[ib, 1] - (s * ax + c * ay)

    nrot = max(int(round(360.0 / cfg.rot_bin_deg)), 4)
    rb = np.floor(d / (TWO_PI / nrot)).astype(int) % nrot
    xb = np.floor((tx - tx.min()) / cfg.trans_bin).astype(int)
    yb = np.floor((ty - ty.min()) / cfg.trans_bin).astype(int)
    nx, ny = int(xb.max()) + 1, int(yb.max()) + 1
    if nrot * nx * ny > cfg.max_accumulator_cells:
        raise MemoryError("Hough accumulator too large; increase trans_bin")

    acc = np.zeros((nrot, nx, ny), dtype=np.float32)
    np.add.at(acc, (rb, xb, yb), 1.0)
    modes = ("wrap", "constant", "constant")
    acc_s = ndimage.uniform_filter(acc, size=3, mode=modes) * 27.0
    peak_mask = (acc_s == ndimage.maximum_filter(acc_s, size=3, mode=modes)) & (acc_s > 0)
    coords = np.argwhere(peak_mask)
    if coords.shape[0] == 0:
        return []
    order = np.argsort(-acc_s[peak_mask])[: cfg.n_hypotheses]
    coords = coords[order]

    hyps = []
    for r, x, y in coords:
        dr = np.abs(rb - r)
        sel = (np.minimum(dr, nrot - dr) <= 1) & (np.abs(xb - x) <= 1) & (np.abs(yb - y) <= 1)
        if not sel.any():
            continue
        dth_est = math.atan2(np.sin(d[sel]).mean(), np.cos(d[sel]).mean()) % TWO_PI
        t = np.array([tx[sel].mean(), ty[sel].mean()])
        hyps.append((dth_est, t, int(sel.sum())))
    return hyps


# ----------------------------------------------------------------------------
# Step 2: pairing under a given transform + refinement
# ----------------------------------------------------------------------------
def _pair_under(A: _Prep, B: _Prep, dth: float, t: np.ndarray, cfg: MatcherConfig):
    """One-to-one pairing of A (transformed) with B. Returns (rows, cols, mean_cost)."""
    R = _rot(dth)
    pa = A.c @ R.T + t
    D = cdist(pa, B.c)
    dang = np.abs(_wrap(A.ang[:, None] + dth - B.ang[None, :]))
    atol = math.radians(cfg.angle_tol_deg)
    ok = (D <= cfg.dist_tol) & (dang <= atol) & _type_compat(A.typ, B.typ, cfg.use_types)
    if not ok.any():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), math.inf
    cost = np.where(ok, D / cfg.dist_tol + dang / atol, _BIG)
    rows, cols = linear_sum_assignment(cost)
    good = cost[rows, cols] < _BIG
    rows, cols = rows[good], cols[good]
    mean_cost = float(cost[rows, cols].mean()) if rows.size else math.inf
    return rows, cols, mean_cost


def _kabsch(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Least-squares rigid transform Y ~= R X + t. Returns (theta, t)."""
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    H = (X - mx).T @ (Y - my)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:              # guard against reflections
        Vt[-1] *= -1
        R = Vt.T @ U.T
    theta = math.atan2(R[1, 0], R[0, 0]) % TWO_PI
    return theta, my - R @ mx


def _best_pairing(A: _Prep, B: _Prep, dth: float, t: np.ndarray, cfg: MatcherConfig):
    rows, cols, mc = _pair_under(A, B, dth, t, cfg)
    best = (rows.size, -mc, dth, t, rows, cols)
    for _ in range(cfg.refine_iterations):
        if rows.size < 3:
            break
        th_new, t_new = _kabsch(A.c[rows], B.c[cols])
        if abs(_wrap(th_new - dth)) > math.radians(30.0):
            break                            # refinement drifted: keep previous
        dth, t = th_new, t_new
        rows, cols, mc = _pair_under(A, B, dth, t, cfg)
        cand = (rows.size, -mc, dth, t, rows, cols)
        if cand[:2] > best[:2]:
            best = cand
    return best


# ----------------------------------------------------------------------------
# Step 3: overlap and scores
# ----------------------------------------------------------------------------
def _inside_mask(mask: np.ndarray, x_img: np.ndarray, y_img: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xi = np.rint(x_img).astype(int)
    yi = np.rint(y_img).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(x_img.shape, dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def _inside_hull(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3 or query.shape[0] == 0:
        return np.ones(query.shape[0], dtype=bool)
    try:
        return Delaunay(points).find_simplex(query) >= 0
    except Exception:  # degenerate (collinear) point set
        return np.ones(query.shape[0], dtype=bool)


def _hull_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def _overlap(A: _Prep, B: _Prep, dth: float, t: np.ndarray, cfg: MatcherConfig):
    """(n_a_overlap, n_b_overlap, area_px2) for the alignment A -> B."""
    R = _rot(dth)
    a_in_b = A.c @ R.T + t            # A minutiae in B's centred frame
    b_in_a = (B.c - t) @ R            # B minutiae in A's centred frame (R^T v)

    if B.mask is not None:
        pb = a_in_b + B.centroid
        a_inside = _inside_mask(B.mask, pb[:, 0], -pb[:, 1])
    else:
        a_inside = _inside_hull(B.c, a_in_b)
    if A.mask is not None:
        pa = b_in_a + A.centroid
        b_inside = _inside_mask(A.mask, pa[:, 0], -pa[:, 1])
    else:
        b_inside = _inside_hull(A.c, b_in_a)

    # overlap area: pixels of B's mask whose pre-image lies inside A's mask
    area = 0.0
    if A.mask is not None and B.mask is not None:
        s = max(int(cfg.overlap_step), 1)
        ys, xs = np.nonzero(B.mask[::s, ::s])
        if ys.size:
            xb, yb = xs * s, ys * s
            pc = np.column_stack([xb - B.centroid[0], -yb - B.centroid[1]])
            pa = (pc - t) @ R + A.centroid
            inside = _inside_mask(A.mask, pa[:, 0], -pa[:, 1])
            area = float(inside.sum()) * s * s
    if area <= 0.0:
        area = min(_hull_area(A.c), _hull_area(B.c))
    return int(a_inside.sum()), int(b_inside.sum()), area


def _significance(m: int, n_trials: int, n_targets: int, area: float, cfg: MatcherConfig) -> float:
    if m <= 0 or n_trials <= 0 or n_targets <= 0 or area <= 0:
        return 0.0
    lam = n_targets * math.pi * cfg.dist_tol ** 2 / area
    p = 1.0 - math.exp(-lam * (math.radians(cfg.angle_tol_deg) / math.pi))
    if cfg.use_types:
        p *= 0.55                      # endings dominate: about half of the pairs are type-compatible
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    tail = float(binom.sf(m - 1, n_trials, p))
    return float(np.clip(-math.log10(max(tail, 1e-300)), 0.0, 300.0))


def _image_transform(A: _Prep, B: _Prep, dth: float, t: np.ndarray) -> np.ndarray:
    """3x3 matrix mapping A image coordinates to B image coordinates."""
    R = _rot(dth)
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = B.centroid + t - R @ A.centroid
    F = np.diag([1.0, -1.0, 1.0])
    return F @ M @ F


def transform_points(transform: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply MatchResult.transform to (N, 2) image coordinates."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    h = np.column_stack([xy, np.ones(xy.shape[0])])
    return (h @ transform.T)[:, :2]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def _empty_result(n_hyp: int = 0) -> MatchResult:
    return MatchResult(0.0, 0.0, 0, 0, 0, 0.0, 0.0, np.eye(3), [], 0.0, n_hyp)


class MinutiaeMatcher:
    """Matcher with a fixed configuration.

        matcher = MinutiaeMatcher(MatcherConfig())
        res = matcher.match(template_a, template_b)
        res.score, res.sig_score, res.n_matched, res.transform
    """

    def __init__(self, config: Optional[MatcherConfig] = None):
        self.config = config or MatcherConfig()

    def match(self, a, b, mask_a: Optional[np.ndarray] = None,
              mask_b: Optional[np.ndarray] = None) -> MatchResult:
        """Match fingerprint A against B. `a` / `b` may be Template objects,
        (N, 5) arrays or lists of minutiae.Minutia (masks are then given via
        mask_a / mask_b; without masks a convex-hull approximation of the
        overlap is used)."""
        cfg = self.config
        ta, tb = _to_template(a, mask_a), _to_template(b, mask_b)
        A, B = _prepare(ta, cfg.min_quality), _prepare(tb, cfg.min_quality)
        if A.pts.shape[0] < cfg.min_minutiae or B.pts.shape[0] < cfg.min_minutiae:
            return _empty_result()

        hyps = _hough_hypotheses(A, B, cfg)
        if not hyps:
            return _empty_result()

        best = None
        for dth, t, _votes in hyps:
            cand = _best_pairing(A, B, dth, t, cfg)
            if best is None or cand[:2] > best[:2]:
                best = cand
        m, neg_cost, dth, t, rows, cols = best
        if m == 0:
            return _empty_result(len(hyps))

        n_a, n_b, area = _overlap(A, B, dth, t, cfg)
        n_a, n_b = max(n_a, m), max(n_b, m)          # matched pairs are inside by definition
        if min(n_a, n_b) < cfg.min_overlap_minutiae:
            score = 0.0
        else:
            score = float(m * m) / float(n_a * n_b)
        sig = _significance(m, n_a, n_b, area, cfg)

        R = _rot(dth)
        pa = A.c[rows] @ R.T + t
        mean_dist = float(np.linalg.norm(pa - B.c[cols], axis=1).mean())
        return MatchResult(
            score=score, sig_score=sig, n_matched=int(m), n_a_overlap=int(n_a),
            n_b_overlap=int(n_b), overlap_area=float(area), rotation=float(dth),
            transform=_image_transform(A, B, dth, t),
            pairs=[(int(i), int(j)) for i, j in zip(rows, cols)],
            mean_pair_distance=mean_dist, n_hypotheses=len(hyps),
        )


def match_minutiae(a, b, mask_a: Optional[np.ndarray] = None, mask_b: Optional[np.ndarray] = None,
                   config: Optional[MatcherConfig] = None) -> MatchResult:
    """One-shot convenience wrapper around MinutiaeMatcher.match()."""
    return MinutiaeMatcher(config).match(a, b, mask_a, mask_b)


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def compute_far_frr(genuine: Sequence[float], impostor: Sequence[float],
                    n_thresholds: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FAR/FRR curves for a similarity score (higher = more similar).
    A pair is ACCEPTED when score >= threshold. Returns (thresholds, far, frr)."""
    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    if g.size == 0 or i.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    allv = np.concatenate([g, i])
    lo, hi = float(allv.min()), float(allv.max())
    if hi == lo:
        hi = lo + 1.0
    # thresholds include values just above the max so that FAR reaches 0
    th = np.unique(np.concatenate([np.linspace(lo, hi, n_thresholds), np.unique(allv), [hi + 1e-9]]))
    far = np.array([(i >= x).mean() for x in th])
    frr = np.array([(g < x).mean() for x in th])
    return th, far, frr


def compute_eer(genuine: Sequence[float], impostor: Sequence[float]) -> Tuple[float, float]:
    """Equal error rate (fraction in [0, 1]) and the threshold where FAR and
    FRR are closest. At that threshold EER is taken as (FAR + FRR) / 2."""
    th, far, frr = compute_far_frr(genuine, impostor)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(th[idx])