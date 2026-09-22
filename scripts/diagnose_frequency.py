from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fingerprint_dataset.dataset import FVCDataset
from fingerprint_dataset.orientation import estimate_orientation
from fingerprint_dataset.segmentation import segment
from fingerprint_dataset.frequency import rotate_window, x_signature


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASET_ROOT = Path("data/FVC2002_DB2_B")

# We deliberately do not hard-code an image filename.
# The first sample discovered by FVCDataset will be used.
WINDOW_SIZE = (23, 43)


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------

def main() -> None:
    dataset = FVCDataset(DATASET_ROOT)

    if not dataset.samples:
        raise RuntimeError(
            f"No fingerprint samples found in: {DATASET_ROOT}"
        )

    # Use the first image actually discovered by FVCDataset.
    sample = dataset.samples[0]

    print(f"Using image: {sample.path}")

    image = sample.load().astype(np.float64)

    # ---------------------------------------------------------------
    # 1. Segmentation
    # ---------------------------------------------------------------

    mask = segment(image)

    if not np.any(mask):
        raise RuntimeError(
            "Segmentation mask contains no foreground pixels."
        )

    # ---------------------------------------------------------------
    # 2. Pixel-level orientation field
    # ---------------------------------------------------------------

    orientations, strengths = estimate_orientation(
        image,
        mask,
    )

    # ---------------------------------------------------------------
    # 3. Select a valid point near the center of the fingerprint
    # ---------------------------------------------------------------

    ys, xs = np.where(mask)

    center_y = int(np.mean(ys))
    center_x = int(np.mean(xs))

    distances = (
        (ys - center_y) ** 2
        + (xs - center_x) ** 2
    )

    index = int(np.argmin(distances))

    y = int(ys[index])
    x = int(xs[index])

    angle = float(orientations[y, x])
    strength = float(strengths[y, x])

    # ---------------------------------------------------------------
    # Print diagnostic information
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("FREQUENCY DIAGNOSTIC")
    print("=" * 60)

    print(f"Image: {sample.path.name}")
    print(f"Selected point: (y={y}, x={x})")

    print()
    print("Orientation")
    print("-" * 60)
    print(f"Angle (radians): {angle:.6f}")
    print(f"Angle (degrees): {np.degrees(angle):.2f}")
    print(f"Strength:        {strength:.4f}")

    # ---------------------------------------------------------------
    # 4. Extract original patch
    # ---------------------------------------------------------------

    height, width = WINDOW_SIZE

    half_height = height // 2
    half_width = width // 2

    original_patch = image[
        y - half_height : y - half_height + height,
        x - half_width : x - half_width + width,
    ]

    if original_patch.shape != (height, width):
        raise RuntimeError(
            f"Original patch has wrong shape: "
            f"{original_patch.shape}, "
            f"expected {(height, width)}"
        )

    # ---------------------------------------------------------------
    # 5. Rotate patch using the exact function from frequency.py
    # ---------------------------------------------------------------

    rotated_patch = rotate_window(
        image,
        (y, x),
        angle,
        WINDOW_SIZE,
    )

    print()
    print("Patches")
    print("-" * 60)
    print(f"Original patch shape: {original_patch.shape}")
    print(f"Rotated patch shape:  {rotated_patch.shape}")

    # ---------------------------------------------------------------
    # 6. X-signature
    # ---------------------------------------------------------------

    signature = x_signature(rotated_patch)

    print()
    print("X-signature")
    print("-" * 60)
    print(f"Signature length: {len(signature)}")
    print(f"Minimum:          {signature.min():.2f}")
    print(f"Maximum:          {signature.max():.2f}")
    print(f"Mean:             {signature.mean():.2f}")
    print(f"Std:              {signature.std():.2f}")

    # ---------------------------------------------------------------
    # 7. Visualize everything
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5),
    )

    # Original patch
    axes[0].imshow(
        original_patch,
        cmap="gray",
    )
    axes[0].set_title("1. Original patch")
    axes[0].axis("off")

    # Rotated patch
    axes[1].imshow(
        rotated_patch,
        cmap="gray",
    )
    axes[1].set_title(
        f"2. Rotated patch\n"
        f"angle = {np.degrees(angle):.2f}°"
    )
    axes[1].axis("off")

    # X-signature
    axes[2].plot(signature)
    axes[2].set_title("3. X-signature")
    axes[2].set_xlabel("Row")
    axes[2].set_ylabel("Intensity sum")
    axes[2].grid(True)

    # Selected point
    axes[3].imshow(
        image,
        cmap="gray",
    )
    axes[3].scatter(
        [x],
        [y],
        s=50,
        marker="x",
    )
    axes[3].set_title(
        f"4. Selected point\n"
        f"({y}, {x})"
    )
    axes[3].axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # 8. Save diagnostic image
    # ---------------------------------------------------------------

    output_path = Path("out/frequency_diagnostic.png")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print()
    print(f"Diagnostic image saved to:")
    print(output_path)

    plt.show()


if __name__ == "__main__":
    main()