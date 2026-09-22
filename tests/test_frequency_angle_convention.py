# tests/test_frequency_angle_convention.py

import numpy as np
import pytest

from fingerprint_dataset.frequency import (
    estimate_period_at_point,
    rotate_window,
)
from fingerprint_dataset.orientation import estimate_orientation


def make_striped_image(shape=(150, 150), period=14.0, angle_degrees=30.0):
    """angle_degrees is the counterclockwise ridge-tangent angle."""
    y, x = np.indices(shape, dtype=np.float64)
    y -= (shape[0] - 1) / 2.0
    x -= (shape[1] - 1) / 2.0

    angle = np.deg2rad(angle_degrees)
    normal_coordinate = x * np.sin(angle) + y * np.cos(angle)

    return 127.5 + 127.5 * np.cos(
        2.0 * np.pi * normal_coordinate / period
    )


def angular_error(actual, expected):
    """Return the absolute pi-periodic angular difference."""
    return abs((actual - expected + np.pi / 2.0) % np.pi - np.pi / 2.0)


@pytest.mark.parametrize("angle_degrees", [0, 30, 45, 60, 90, 120, 150])
def test_orientation_matches_stripe_convention(angle_degrees):
    image = make_striped_image(angle_degrees=angle_degrees)
    mask = np.ones(image.shape, dtype=bool)

    orientations, strengths = estimate_orientation(image, mask)
    center = (75, 75)

    expected = np.deg2rad(angle_degrees)
    actual = orientations[center]
    error = angular_error(actual, expected)

    assert error < np.deg2rad(2.0), (
        f"Expected {angle_degrees:.2f} degrees, "
        f"got {np.rad2deg(actual):.2f} degrees; "
        f"error={np.rad2deg(error):.2f} degrees"
    )
    assert strengths[center] > 0.9


def test_nonzero_angle_correctly_aligns_diagonal_stripes():
    image = make_striped_image(
        shape=(150, 150),
        period=14.0,
        angle_degrees=30.0,
    )

    window = rotate_window(
        image,
        center=(75, 75),
        angle=np.deg2rad(30.0),
        window_size=(23, 43),
    )

    assert window.shape == (43, 23)

    row_std = window.std(axis=1).mean()
    col_std = window.std(axis=0).mean()

    print(f"\nrow_std={row_std:.6f}, col_std={col_std:.6f}")

    assert row_std < col_std, (
        "Stripes are not horizontal after rotation: "
        f"row_std={row_std:.6f}, col_std={col_std:.6f}"
    )


@pytest.mark.parametrize("angle_degrees", [0, 30, 45, 60, 90, 120, 150])
def test_rotation_aligns_estimated_orientation(angle_degrees):
    image = make_striped_image(angle_degrees=angle_degrees)
    mask = np.ones(image.shape, dtype=bool)

    orientations, _ = estimate_orientation(image, mask)
    center = (75, 75)

    window = rotate_window(
        image,
        center=center,
        angle=orientations[center],
        window_size=(23, 43),
    )

    assert window.shape == (43, 23)

    row_std = window.std(axis=1).mean()
    col_std = window.std(axis=0).mean()

    assert col_std > 1.0, "The aligned window has insufficient contrast."
    assert row_std < 0.1 * col_std, (
        f"Input angle={angle_degrees:.2f} degrees; "
        f"estimated angle={np.rad2deg(orientations[center]):.2f} degrees; "
        f"row_std={row_std:.6f}, col_std={col_std:.6f}"
    )


@pytest.mark.parametrize("angle_degrees", [0, 30, 45, 60, 90, 120, 150])
def test_period_estimation_with_estimated_orientation(angle_degrees):
    expected_period = 14.0
    image = make_striped_image(
        period=expected_period,
        angle_degrees=angle_degrees,
    )
    mask = np.ones(image.shape, dtype=bool)

    orientations, _ = estimate_orientation(image, mask)
    center = (75, 75)

    actual_period = estimate_period_at_point(
        image,
        center=center,
        angle=orientations[center],
        window_size=(23, 43),
        period_min=5,
        period_max=20,
        min_valid_distances=4,
    )

    assert actual_period is not None, (
        f"No period estimated at angle={angle_degrees:.2f} degrees."
    )
    assert actual_period == pytest.approx(expected_period, abs=1.0), (
        f"Input angle={angle_degrees:.2f} degrees; "
        f"expected period={expected_period:.2f}; "
        f"actual period={actual_period:.2f}"
    )
