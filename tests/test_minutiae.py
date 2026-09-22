"""Tests for minutiae.py (pytest). Works whether the module lives in the
package (fingerprint_dataset.minutiae) or next to this file."""
import math

import numpy as np
import pytest

try:
    from fingerprint_dataset import minutiae as M
except ImportError:  # pragma: no cover
    import minutiae as M


def _dislocation_image(charge: int, invert: bool = False, size: int = 240, period: float = 9.0):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = size / 2
    phase = 2 * np.pi / period * xx + charge * np.arctan2(yy - c, xx - c)
    img = (127 + (-100 if invert else 100) * np.cos(phase)).astype(np.uint8)
    mask = (xx - c) ** 2 + (yy - c) ** 2 < (0.44 * size) ** 2
    return img, mask


def _skeleton_at(sk, m, dist, ang, r=1):
    x = m.x + dist * math.cos(ang)
    y = m.y - dist * math.sin(ang)
    xi, yi = int(round(x)), int(round(y))
    return bool(sk[max(yi - r, 0): yi + r + 1, max(xi - r, 0): xi + r + 1].any())


# --- helpers -----------------------------------------------------------------
def test_angle_helpers():
    assert M.angle_from_vector(1, 0) == pytest.approx(0.0)
    assert M.angle_from_vector(0, -1) == pytest.approx(math.pi / 2)   # up on screen
    assert M.angle_from_vector(0, 1) == pytest.approx(3 * math.pi / 2)
    assert M.angle_difference(0.1, 2 * math.pi - 0.1) == pytest.approx(0.2)
    assert abs(M.angle_difference(0.0, math.pi)) == pytest.approx(math.pi)


def test_array_roundtrip():
    ms = [M.Minutia(10, 20, 1.0, M.ENDING, 0.5), M.Minutia(30, 40, 4.0, M.BIFURCATION, 0.9)]
    back = M.minutiae_from_array(M.minutiae_to_array(ms))
    assert [m.type for m in back] == [M.ENDING, M.BIFURCATION]
    assert back[1].x == pytest.approx(30) and back[1].angle == pytest.approx(4.0, abs=1e-5)
    assert M.minutiae_to_array([]).shape == (0, 5)


def test_period_estimation_ignores_nan():
    pm = np.full((50, 50), 10.0)
    pm[:10] = np.nan
    pm[10:20] = -1
    assert M.estimate_ridge_period(pm, np.ones((50, 50), bool)) == pytest.approx(10.0)
    assert M.estimate_ridge_period(None, np.ones((5, 5), bool), default=9.0) == 9.0


# --- crossing number / pruning -------------------------------------------------
def test_crossing_number():
    s = np.zeros((20, 20), bool)
    s[10, 3:17] = True
    s[4:10, 10] = True
    cn = M.crossing_number(s)
    assert cn[10, 10] == 3
    assert cn[10, 3] == cn[10, 16] == cn[4, 10] == 1
    assert cn[10, 6] == 2


def test_prune_removes_short_spur_keeps_long_branch_and_line():
    sk = np.zeros((60, 80), bool)
    sk[30, 5:75] = True
    sk[24:30, 40] = True
    out = M.prune_skeleton(sk, 8)
    assert not out[24:30, 40].any() and out[30, 5:75].all()
    sk[10:30, 40] = True  # now a long branch
    assert M.prune_skeleton(sk, 8)[10:30, 40].all()


def test_prune_islands():
    sk = np.zeros((40, 40), bool)
    sk[10, 10:15] = True
    assert not M.prune_skeleton(sk, 8).any()
    assert M.prune_skeleton(sk, 8, remove_islands=False).any()


# --- synthetic fingerprints with a single known dislocation ---------------------
@pytest.mark.parametrize("method", ["skimage", "zhang_suen"])
@pytest.mark.parametrize("charge", [1, -1])
def test_single_dislocation_gives_one_ending_with_correct_direction(charge, method):
    if method == "skimage" and M._sk_skeletonize is None:
        pytest.skip("scikit-image not installed")
    img, mask = _dislocation_image(charge)
    res = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(polarity="bright"),
                                           skeleton_method=method)
    assert len(res.minutiae) == 1
    m = res.minutiae[0]
    assert m.type == M.ENDING
    assert math.hypot(m.x - 120, m.y - 120) < 8
    # ridge body lies ahead of the tip, nothing behind it
    assert _skeleton_at(res.skeleton_pruned, m, 6, m.angle)
    assert not _skeleton_at(res.skeleton_pruned, m, 6, m.angle + math.pi)


def test_inverted_polarity_gives_bifurcation_with_stem_behind():
    img, mask = _dislocation_image(1, invert=True)
    res = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(polarity="bright"))
    assert len(res.minutiae) == 1
    m = res.minutiae[0]
    assert m.type == M.BIFURCATION
    assert _skeleton_at(res.skeleton_pruned, m, 6, m.angle + math.pi)   # the stem


def test_auto_polarity_uses_reference():
    img, mask = _dislocation_image(1)
    raw_dark_ridges = 255 - img  # enhanced has bright ridges, raw has dark ridges
    _, pol = M.binarize_enhanced(img, mask, polarity="auto", reference=raw_dark_ridges)
    assert pol == "bright"
    _, pol = M.binarize_enhanced(img, mask, polarity="auto", reference=img)
    assert pol == "dark"


# --- filtering -----------------------------------------------------------------
E, B = M.ENDING, M.BIFURCATION
FULL = np.ones((200, 200), bool)


def test_filter_border_and_quality():
    assert M.filter_minutiae([M.Minutia(3, 100, 0, E, 1)], FULL, 9.0) == []
    cfg = M.MinutiaeConfig(min_quality=0.5)
    assert M.filter_minutiae([M.Minutia(100, 100, 0, E, 0.4)], FULL, 9.0, cfg) == []


def test_filter_duplicates_keep_best():
    out = M.filter_minutiae([M.Minutia(100, 100, 0, B, .5), M.Minutia(101, 100, 0, B, .9)], FULL, 9.0)
    assert len(out) == 1 and out[0].quality == pytest.approx(.9)


def test_filter_broken_ridge_only_when_facing():
    facing = [M.Minutia(80, 100, 0.0, E, .9), M.Minutia(73, 100, math.pi, E, .9)]
    assert M.filter_minutiae(facing, FULL, 9.0) == []
    same_dir = [M.Minutia(80, 100, 0.0, E, .9), M.Minutia(73, 100, 0.0, E, .9)]
    assert len(M.filter_minutiae(same_dir, FULL, 9.0)) == 2


def test_filter_bridge_spur_and_far_pairs():
    assert M.filter_minutiae([M.Minutia(100, 100, 0, B, .9), M.Minutia(105, 100, 1, B, .9)], FULL, 9.0) == []
    assert M.filter_minutiae([M.Minutia(100, 100, 0, B, .9), M.Minutia(104, 100, 1, E, .9)], FULL, 9.0) == []
    assert len(M.filter_minutiae([M.Minutia(60, 100, 0, E, 1), M.Minutia(140, 100, 0, B, 1)], FULL, 9.0)) == 2


def test_filter_cap():
    ms = [M.Minutia(30 + 20 * i, 100, 0, E, i / 10) for i in range(8)]
    out = M.filter_minutiae(ms, FULL, 9.0, M.MinutiaeConfig(max_minutiae=3))
    assert len(out) == 3 and min(m.quality for m in out) >= 0.5


# --- robustness ------------------------------------------------------------------
def test_degenerate_inputs():
    assert M.extract_minutiae_from_enhanced(np.zeros((50, 50), np.uint8), np.zeros((50, 50), bool)).minutiae == []
    assert M.extract_minutiae_from_enhanced(np.full((50, 50), 128, np.uint8), np.ones((50, 50), bool)).minutiae == []
    with pytest.raises(ValueError):
        M.extract_minutiae_from_enhanced(np.zeros((5, 5)), np.zeros((4, 4), bool))


def test_nan_period_and_float_image():
    img, mask = _dislocation_image(1)
    pm = np.full(img.shape, np.nan)
    res = M.extract_minutiae_from_enhanced(img.astype(np.float32) / 255, mask, period_map=pm,
                                           config=M.MinutiaeConfig(polarity="bright"))
    assert res.period == pytest.approx(9.0)
    assert len(res.minutiae) == 1


# --- additions: hole filling inside the mask, direct end/bifurcation angles, ranges ---
def test_small_holes_filled_only_inside_mask():
    enh = np.full((80, 80), 200, np.uint8)      # bright background = valley for polarity "dark"
    mask = np.zeros((80, 80), bool)
    mask[10:70, 10:70] = True
    enh[10:70, 10:70] = 50                      # one big dark (ridge) area inside the mask
    enh[38:41, 38:41] = 200                     # 3x3 bright hole inside it
    out, pol = M.binarize_enhanced(enh, mask, polarity="dark", sigma=25.0,
                                   smooth=False, min_object_area=1, min_hole_area=30)
    assert pol == "dark"
    assert out[39, 39]                          # the hole was filled
    assert not out[~mask].any()                 # nothing outside the mask


def test_extract_line_endings_have_opposite_body_directions():
    sk = np.zeros((60, 80), bool)
    sk[30, 10:51] = True
    mask = np.ones_like(sk)
    ms = M.extract_minutiae(sk, mask, track_length=10, border_margin=0.0)
    assert sorted(m.type for m in ms) == [M.ENDING, M.ENDING]
    left = min(ms, key=lambda m: m.x)
    right = max(ms, key=lambda m: m.x)
    assert left.angle == pytest.approx(0.0, abs=0.05) or left.angle == pytest.approx(2 * math.pi, abs=0.05)
    assert right.angle == pytest.approx(math.pi, abs=0.05)


def test_extract_y_junction_angle_points_away_from_stem():
    sk = np.zeros((80, 80), bool)
    for k in range(0, 25):
        sk[40 - k, 40 - k] = True     # NW branch (includes junction at k=0)
        sk[40 - k, 40 + k] = True     # NE branch
    sk[40:66, 40] = True              # stem going down
    mask = np.ones_like(sk)
    ms = [m for m in M.extract_minutiae(sk, mask, track_length=12, border_margin=0.0)
          if m.type == M.BIFURCATION]
    assert len(ms) == 1
    assert (ms[0].x, ms[0].y) == (40, 40)
    assert ms[0].angle == pytest.approx(math.pi / 2, abs=0.1)   # up on screen


def test_outputs_inside_image_and_ranges():
    yy, xx = np.mgrid[0:296, 0:400].astype(float)
    phase = 2 * np.pi / 9.0 * np.hypot(xx - 200, (yy - 150) * 0.8)
    for cx, cy, q in [(150, 110, 1), (260, 190, -1), (200, 230, 1)]:
        phase = phase + q * np.arctan2(yy - cy, xx - cx)
    img = (127 + 100 * np.cos(phase)).astype(np.uint8)
    mask = ((xx - 200) / 180) ** 2 + ((yy - 150) / 125) ** 2 < 1
    res = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(polarity="bright"))
    assert len(res.raw_minutiae) >= len(res.minutiae) >= 3
    for m in res.raw_minutiae + res.minutiae:
        assert 0 <= m.x < img.shape[1] and 0 <= m.y < img.shape[0]
        assert 0.0 <= m.angle < 2 * math.pi
        assert 0.0 <= m.quality <= 1.0
        assert mask[int(round(m.y)), int(round(m.x))]


def test_filter_stats_and_pipeline_result_stats():
    stats = {}
    ms = [M.Minutia(80, 100, 0.0, E, .9), M.Minutia(73, 100, math.pi, E, .9),      # broken ridge
          M.Minutia(150, 60, 0, B, .9), M.Minutia(155, 60, 1, B, .9),               # bridge
          M.Minutia(3, 100, 0, E, 1)]                                               # border
    out = M.filter_minutiae(ms, FULL, 9.0, stats=stats)
    assert out == []
    assert stats["broken_ridge"] == 2 and stats["bridge"] == 2 and stats["border"] == 1
    img, mask = _dislocation_image(1)
    res = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(polarity="bright"))
    assert set(res.filter_stats) >= {"border", "duplicate", "broken_ridge", "bridge", "spur", "cap"}


def test_broken_ridge_with_gap_of_1p5_periods_is_removed():
    gap = 1.5 * 9.0
    facing = [M.Minutia(100 + gap, 100, 0.0, E, .9), M.Minutia(100, 100, math.pi, E, .9)]
    assert M.filter_minutiae(facing, FULL, 9.0) == []


def test_remove_short_components():
    sk = np.zeros((60, 100), bool)
    sk[10, 5:20] = True          # 15 px fragment
    sk[30, 5:90] = True          # long ridge
    out, n = M.remove_short_components(sk, 27)
    assert n == 1 and not out[10].any() and out[30, 5:90].all()
    out, n = M.remove_short_components(sk, 0)
    assert n == 0 and out.sum() == sk.sum()


def test_pipeline_drops_isolated_fragment_endings():
    img = np.zeros((120, 260), np.uint8)
    img[40:44, 20:240] = 200      # long bright ridge
    img[80:84, 100:120] = 200     # isolated 20 px fragment
    mask = np.ones(img.shape, bool)
    mask[:8] = mask[-8:] = False
    mask[:, :8] = mask[:, -8:] = False
    on = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(
        polarity="bright", border_margin_factor=0.0))
    off = M.extract_minutiae_from_enhanced(img, mask, config=M.MinutiaeConfig(
        polarity="bright", border_margin_factor=0.0, min_component_period_factor=0.0))
    assert on.n_fragments_removed == 1 and off.n_fragments_removed == 0
    assert len(off.raw_minutiae) == len(on.raw_minutiae) + 2     # the fragment's two endings