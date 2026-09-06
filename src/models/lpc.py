"""
Linear Probe Classifier (LP-C) Adaptation Strategy.
Applies a linear classification head on top of frozen CLIP visual features.
"""

import torch
import torch.nn as nn


class LPC_Model(nn.Module):
    """Linear Probe Classifier with frozen backbone and trained linear head."""

    def __init__(self, clip_model: nn.Module, num_classes: int = 8):
        super().__init__()
        self.clip_model = clip_model
        # Freeze visual and text backbones
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if hasattr(clip_model.visual, "output_dim"):
            feat_dim = clip_model.visual.output_dim
        elif hasattr(clip_model, "visual") and hasattr(clip_model.visual, "proj"):
            feat_dim = (
                clip_model.visual.proj.shape[1]
                if clip_model.visual.proj is not None
                else 768
            )
        else:
            feat_dim = 768

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.classifier(image_features)
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self
