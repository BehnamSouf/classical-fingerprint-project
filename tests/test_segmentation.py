# tests/test_segmentation.py
import numpy as np
import pytest

from fingerprint_dataset.segmentation import (
    gradient_magnitude,
    create_mask,
    clean_mask,
    apply_mask,
    segment,
    _keep_largest_component,
)


@pytest.fixture
def uniform_image():
    """A flat, textureless image — should behave like pure background."""
    return np.full((64, 64), 128, dtype=np.uint8)


@pytest.fixture
def textured_square_image():
    """A noisy (high-variance) square in the middle of a flat background,
    simulating a fingerprint region surrounded by background."""
    image = np.full((64, 64), 128, dtype=np.uint8)
    rng = np.random.default_rng(seed=0)
    image[16:48, 16:48] = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    return image


class TestGradientMagnitude:
    def test_output_shape_matches_input(self, textured_square_image):
        result = gradient_magnitude(textured_square_image)
        assert result.shape == textured_square_image.shape

    def test_uniform_image_has_near_zero_magnitude(self, uniform_image):
        result = gradient_magnitude(uniform_image)
        assert np.allclose(result, 0, atol=1e-6)

    def test_textured_region_has_higher_magnitude_than_flat_background(self, textured_square_image):
        result = gradient_magnitude(textured_square_image)
        center = result[16:48, 16:48].mean()
        corner = result[:8, :8].mean()
        assert center > corner


class TestCreateMask:
    def test_uniform_image_yields_empty_mask(self, uniform_image):
        magnitude = gradient_magnitude(uniform_image)
        mask = create_mask(magnitude)
        assert not mask.any()

    def test_textured_region_is_flagged_foreground(self, textured_square_image):
        magnitude = gradient_magnitude(textured_square_image)
        mask = create_mask(magnitude)
        # at least the center of the textured square should be foreground
        assert mask[30:34, 30:34].all()

    def test_output_is_boolean(self, textured_square_image):
        magnitude = gradient_magnitude(textured_square_image)
        mask = create_mask(magnitude)
        assert mask.dtype == bool


class TestKeepLargestComponent:
    def test_keeps_only_largest_component(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:2, 0:2] = True      # small component, 4 px
        mask[10:16, 10:16] = True  # large component, 36 px

        result = _keep_largest_component(mask)

        assert not result[0:2, 0:2].any()
        assert result[10:16, 10:16].all()

    def test_single_component_is_unchanged(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        result = _keep_largest_component(mask)
        assert np.array_equal(result, mask)

    def test_empty_mask_is_unchanged(self):
        mask = np.zeros((10, 10), dtype=bool)
        result = _keep_largest_component(mask)
        assert not result.any()


class TestCleanMask:
    def test_fills_small_hole(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:16, 4:16] = True
        mask[9:11, 9:11] = False  # small hole inside the foreground

        result = clean_mask(mask, closing_iterations=2, opening_iterations=1)

        assert result[9:11, 9:11].all()

    def test_removes_small_isolated_speck(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True  # main region
        mask[0, 0] = True          # single-pixel noise speck

        result = clean_mask(mask, closing_iterations=2, opening_iterations=2)

        assert not result[0, 0]
        assert result[14, 14]  # center of main region survives


class TestApplyMask:
    def test_background_pixels_are_zeroed(self):
        image = np.full((10, 10), 200, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True

        result = apply_mask(image, mask)

        assert (result[~mask] == 0).all()
        assert (result[mask] == 200).all()

    def test_does_not_mutate_original_image(self):
        image = np.full((5, 5), 100, dtype=np.uint8)
        mask = np.zeros((5, 5), dtype=bool)

        apply_mask(image, mask)

        assert (image == 100).all()  # original untouched


class TestSegmentEndToEnd:
    def test_returns_boolean_mask_of_correct_shape(self, textured_square_image):
        mask = segment(textured_square_image)
        assert mask.shape == textured_square_image.shape
        assert mask.dtype == bool

    def test_uniform_image_yields_no_foreground(self, uniform_image):
        mask = segment(uniform_image)
        assert not mask.any()

    def test_textured_region_is_mostly_detected(self, textured_square_image):
        mask = segment(textured_square_image)
        detected_fraction = mask[16:48, 16:48].mean()
        assert detected_fraction > 0.5