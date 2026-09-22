# tests/test_frequency.py
import numpy as np
import pytest

from fingerprint_dataset.frequency import (
    preprocess_image,
    rotate_window,
    x_signature,
    peak_valley_distances,
    estimate_period_at_point,
    fill_missing,
    estimate_frequency_field,
)


def make_striped_image(shape=(120, 120), period=12, angle_degrees=0):
    """
    Build a synthetic image with parallel dark/light stripes (fake ridges)
    at a known period and angle, so the expected frequency is known
    ahead of time.
    """
    from scipy import ndimage

    height, width = shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    stripes = (np.sin(2 * np.pi * y / period) > 0).astype(np.float64) * 255

    if angle_degrees != 0:
        stripes = ndimage.rotate(stripes, angle_degrees, reshape=False, mode="reflect", order=1)

    return stripes.astype(np.uint8)


class TestPreprocessImage:
    def test_output_shape_matches_input(self):
        image = make_striped_image()
        result = preprocess_image(image)
        assert result.shape == image.shape

    def test_smooths_single_pixel_noise(self):
        image = make_striped_image()
        noisy = image.copy()
        noisy[60, 60] = 255 if image[60, 60] == 0 else 0  # flip one pixel

        result = preprocess_image(noisy)
        # after smoothing, the spike should be pulled toward its neighbors
        assert abs(int(result[60, 60]) - int(image[60, 60])) < abs(int(noisy[60, 60]) - int(image[60, 60]))


class TestRotateWindow:
    def test_output_has_requested_size(self):
        image = make_striped_image()
        window = rotate_window(image, center=(60, 60), angle=0.0, window_size=(23, 43))
        assert window.shape == (43, 23)  # (height, width)

    def test_zero_angle_keeps_horizontal_stripes_horizontal(self):
        image = make_striped_image(period=12, angle_degrees=0)
        window = rotate_window(image, center=(60, 60), angle=0.0, window_size=(23, 43))
        # rows should vary (stripes visible), columns within a row should be fairly uniform
        row_std = window.std(axis=1).mean()
        col_std = window.std(axis=0).mean()
        assert row_std < col_std


class TestXSignature:
    def test_output_length_matches_window_height(self):
        window = np.zeros((40, 20), dtype=np.uint8)
        signal = x_signature(window)
        assert signal.shape == (40,)

    def test_uniform_window_gives_constant_signal(self):
        window = np.full((10, 10), 100, dtype=np.uint8)
        signal = x_signature(window)
        assert np.allclose(signal, signal[0])

    def test_striped_window_gives_varying_signal(self):
        window = make_striped_image(shape=(40, 20), period=10)
        signal = x_signature(window)
        assert signal.std() > 0


class TestPeakValleyDistances:
    def test_recovers_known_period(self):
        # pure sine wave with period 12: peaks and valleys should be ~12 apart
        y = np.arange(200)
        signal = np.sin(2 * np.pi * y / 12)

        distances = peak_valley_distances(signal, period_min=5, period_max=20)

        assert len(distances) > 0
        assert abs(distances.mean() - 12) < 1.5

    def test_flat_signal_gives_no_distances(self):
        signal = np.full(100, 50.0)
        distances = peak_valley_distances(signal, period_min=5, period_max=20)
        assert len(distances) == 0

    def test_filters_out_of_range_distances(self):
        y = np.arange(200)
        signal = np.sin(2 * np.pi * y / 12)
        # a period of ~12 should be excluded by a narrow valid range
        distances = peak_valley_distances(signal, period_min=50, period_max=60)
        assert len(distances) == 0


class TestEstimatePeriodAtPoint:
    def test_recovers_known_period_on_synthetic_stripes(self):
        image = make_striped_image(shape=(150, 150), period=12, angle_degrees=0)
        period = estimate_period_at_point(
            image, center=(75, 75), angle=0.0,
            window_size=(23, 43), period_min=5, period_max=20,
        )
        assert period is not None
        assert abs(period - 12) < 2

    def test_returns_none_for_flat_region(self):
        image = np.full((150, 150), 128, dtype=np.uint8)
        period = estimate_period_at_point(
            image, center=(75, 75), angle=0.0,
            window_size=(23, 43), period_min=5, period_max=20,
        )
        assert period is None


class TestFillMissing:
    def test_fills_gap_between_valid_neighbors(self):
        field = np.zeros((10, 10))
        valid = np.zeros((10, 10), dtype=bool)

        field[3, :] = 5.0
        valid[3, :] = True
        field[7, :] = 5.0
        valid[7, :] = True

        result = fill_missing(field, valid, max_iterations=30)

        assert result[5, 5] > 0  # gap between the two valid rows got filled

    def test_already_fully_valid_field_is_unchanged(self):
        field = np.full((5, 5), 3.0)
        valid = np.ones((5, 5), dtype=bool)

        result = fill_missing(field, valid)

        assert np.allclose(result, field)


class TestEstimateFrequencyField:
    def test_output_shape_matches_input(self):
        image = make_striped_image(shape=(150, 150), period=12)
        mask = np.ones((150, 150), dtype=bool)
        orientations = np.zeros((150, 150))

        result = estimate_frequency_field(image, orientations, mask, step=10, border=20)

        assert result.shape == image.shape

    def test_frequency_is_inverse_of_known_period(self):
        image = make_striped_image(shape=(150, 150), period=12)
        mask = np.ones((150, 150), dtype=bool)
        orientations = np.zeros((150, 150))

        result = estimate_frequency_field(image, orientations, mask, step=10, border=20)

        center_region = result[60:90, 60:90]
        valid_values = center_region[center_region > 0]
        assert valid_values.size > 0
        assert abs(valid_values.mean() - 1 / 12) < 0.02

    def test_background_is_zeroed(self):
        image = make_striped_image(shape=(150, 150), period=12)
        mask = np.zeros((150, 150), dtype=bool)
        mask[40:110, 40:110] = True
        orientations = np.zeros((150, 150))

        result = estimate_frequency_field(image, orientations, mask, step=10, border=20)

        assert (result[~mask] == 0).all()
def test_nonzero_angle_correctly_aligns_diagonal_stripes():
    """
    Build stripes tilted at 30 degrees and check that rotate_window,
    given that same angle, produces horizontal stripes. If the angle
    convention is wrong, the output stripes will still be tilted
    (or tilted in the wrong direction).
    """
    true_angle_rad = np.radians(30)
    image = make_striped_image(shape=(150, 150), period=14, angle_degrees=30)

    window = rotate_window(image, center=(75, 75), angle=true_angle_rad, window_size=(23, 43))

    row_std = window.std(axis=1).mean()
    col_std = window.std(axis=0).mean()
    assert row_std < col_std, "stripes are not horizontal after rotation -- angle convention is likely wrong"