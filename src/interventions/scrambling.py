"""Patch scrambling for the spatial-layout intervention."""

import numpy as np


def patch_shuffle(
    img_np: np.ndarray,
    patch_size: int = 16,
    rng: np.random.Generator | None = None,
    remainder: str = "preserve",
) -> np.ndarray:
    """Randomly permute non-overlapping square patches across the image grid.

    Args:
        img_np: [H, W, 3] image array
        patch_size: Square patch width/height in pixels
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if remainder not in {"preserve", "zero", "crop"}:
        raise ValueError("remainder must be 'preserve', 'zero', or 'crop'")
    h, w, _ = img_np.shape
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    cropped = img_np[:new_h, :new_w, :].copy()

    patches = []
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patches.append(cropped[i : i + patch_size, j : j + patch_size, :])

    generator = rng if rng is not None else np.random.default_rng()
    order = generator.permutation(len(patches))

    shuffled_img = np.zeros_like(cropped)
    idx = 0
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            shuffled_img[i : i + patch_size, j : j + patch_size, :] = patches[
                order[idx]
            ]
            idx += 1

    if remainder == "crop":
        return shuffled_img
    final_img = img_np.copy() if remainder == "preserve" else np.zeros_like(img_np)
    final_img[:new_h, :new_w, :] = shuffled_img
    return final_img
