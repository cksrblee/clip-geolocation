"""Configure a fresh OpenCLIP model for one paper adaptation strategy."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .adaptive import VisualClassifier
from .lora import setup_lora_vit
from .lpc import LPC_Model
from .lpt import LPT_Model


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def setup_model_strategy(
    clip_model: nn.Module,
    strategy: str,
    text_features: Optional[torch.Tensor] = None,
    model_name: str = "ViT-L-14",
    num_classes: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 16,
) -> nn.Module:
    _set_requires_grad(clip_model, False)

    if strategy == "Zero-shot":
        clip_model.eval()
        return clip_model
    if strategy == "LP-T":
        if text_features is None:
            raise ValueError("text_features are required for LP-T")
        return LPT_Model(clip_model, text_features)
    if strategy == "LP-C":
        return LPC_Model(clip_model, num_classes=num_classes)
    if strategy == "PU":
        if "ViT" not in model_name:
            raise ValueError("The paper's PU strategy is defined for a ViT backbone")
        try:
            final_block = clip_model.visual.transformer.resblocks[-1]
            final_norm = clip_model.visual.ln_post
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                "Visual encoder does not expose the expected ViT layers"
            ) from exc
        _set_requires_grad(final_block, True)
        _set_requires_grad(final_norm, True)
        return VisualClassifier(clip_model, num_classes)
    if strategy == "LoRA":
        if "ViT" not in model_name:
            raise ValueError("The paper's LoRA strategy is defined for a ViT backbone")
        clip_model = setup_lora_vit(clip_model, r=lora_r, alpha=lora_alpha)
        return VisualClassifier(clip_model, num_classes)
    if strategy == "Full-FT":
        # The reported experiment fine-tunes the full visual encoder, with text fixed.
        _set_requires_grad(clip_model.visual, True)
        return VisualClassifier(clip_model, num_classes)
    raise ValueError(f"Unknown adaptation strategy: {strategy}")


def trainable_parameters(model: nn.Module):
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Selected strategy has no trainable parameters")
    return parameters
