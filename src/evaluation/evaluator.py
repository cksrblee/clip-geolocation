"""Shared inference routines for training and intervention evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import top1_accuracy


HEAD_STRATEGIES = {"LP-T", "LP-C"}


def _get_logit_scale(model) -> torch.Tensor | float:
    if hasattr(model, "logit_scale") and model.logit_scale is not None:
        return model.logit_scale.exp()
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "logit_scale") and visual.logit_scale is not None:
        return visual.logit_scale.exp()
    return 100.0


def classification_logits(model, strategy: str, images, text_features):
    if strategy in HEAD_STRATEGIES:
        return model(images)
    if text_features is None:
        raise ValueError(f"Text features are required for {strategy}")
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    scale = _get_logit_scale(model)
    return scale * (image_features @ text_features.T)


def predict_dataset(
    model,
    dataset,
    strategy: str,
    text_features,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    predictions: list[int] = []
    targets: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = classification_logits(
                model, strategy, images.to(device, non_blocking=True), text_features
            )
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return np.asarray(predictions), np.asarray(targets)


def evaluate_dataset(*args, **kwargs) -> tuple[list[int], list[int], float]:
    predictions, targets = predict_dataset(*args, **kwargs)
    return predictions.tolist(), targets.tolist(), top1_accuracy(predictions, targets)
