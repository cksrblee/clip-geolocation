"""OpenCLIP construction shared by every experiment script."""

from __future__ import annotations

import open_clip


def create_model_and_transforms(model_name: str, pretrained: str, device: str):
    """Load OpenCLIP model and preprocessing transforms."""
    return open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
