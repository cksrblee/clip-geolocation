"""Trainable classification head used by encoder-adaptation strategies."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualClassifier(nn.Module):
    """Classify normalized CLIP image embeddings with a trainable linear head."""

    def __init__(self, clip_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.clip_model = clip_model
        feature_dim = getattr(clip_model.visual, "output_dim", None)
        if feature_dim is None:
            projection = getattr(clip_model.visual, "proj", None)
            if projection is None:
                raise ValueError("Cannot determine the CLIP visual embedding dimension")
            feature_dim = int(projection.shape[-1])
        self.classifier = nn.Linear(int(feature_dim), num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        return self.classifier(features)
