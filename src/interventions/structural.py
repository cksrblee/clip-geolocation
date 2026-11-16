"""Appearance-reduced edge and macro-blur transformations."""

import cv2
import numpy as np


def get_canny_edges(
    img_np: np.ndarray, low_thresh: int = 100, high_thresh: int = 200
) -> np.ndarray:
    """Return a three-channel Canny edge map."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def get_macro_blur(img_np: np.ndarray, sigma: int = 15) -> np.ndarray:
    """Apply strong Gaussian blur while retaining coarse color and layout."""
    return cv2.GaussianBlur(img_np, (0, 0), sigma)
