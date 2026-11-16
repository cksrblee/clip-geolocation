"""
Linear Probe Text (LP-T) Adaptation Strategy.
Applies learnable affine transform (scale and bias) to image features while classifying against text features.
"""

import torch
import torch.nn as nn


class LPT_Model(nn.Module):
    """Linear Probe Text model that transforms image embeddings to match frozen text embeddings."""

    def __init__(self, clip_model: nn.Module, text_features: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        # Freeze underlying CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.register_buffer("text_features", text_features.detach().clone())
        feat_dim = text_features.shape[1]

        self.scale = nn.Parameter(torch.ones(feat_dim))
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.clip_model.eval()
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)

        image_features = image_features * self.scale + self.bias
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if hasattr(self.clip_model, "logit_scale"):
            logit_scale = self.clip_model.logit_scale.exp()
        elif hasattr(self.clip_model, "visual") and hasattr(
            self.clip_model.visual, "logit_scale"
        ):
            logit_scale = self.clip_model.visual.logit_scale.exp()
        else:
            logit_scale = 100.0

        logits = logit_scale * (image_features @ self.text_features.T)
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self
