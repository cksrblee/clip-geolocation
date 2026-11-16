"""Configure an OpenCLIP model for one adaptation strategy."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .lora import setup_lora_vit
from .lpc import LPC_Model
from .lpt import LPT_Model


LOGGER = logging.getLogger(__name__)


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
    """Configure frozen/adapted parameters matching the 26-3-Summer baseline setup.

    Zero-shot, PU, LoRA, and Full-FT classify via dot product against frozen
    text_features; LP-T learns feature-wise scale/bias in text-score space;
    LP-C trains a linear classifier head on frozen image embeddings.
    """
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
        if "ViT" in model_name or hasattr(clip_model.visual, "transformer"):
            _set_requires_grad(clip_model.visual.transformer.resblocks[-1], True)
            _set_requires_grad(clip_model.visual.ln_post, True)
        elif hasattr(clip_model.visual, "layer4") and hasattr(
            clip_model.visual, "attnpool"
        ):
            _set_requires_grad(clip_model.visual.layer4, True)
            _set_requires_grad(clip_model.visual.attnpool, True)
        else:
            raise ValueError("Visual encoder does not expose expected PU layers")
        return clip_model
    if strategy == "LoRA":
        if "RN" in model_name or hasattr(clip_model.visual, "layer4"):
            LOGGER.info(
                "Skipping PEFT LoRA for ResNet backbone (%s); falling back to PU.",
                model_name,
            )
            return setup_model_strategy(
                clip_model, "PU", text_features=text_features, model_name=model_name
            )
        try:
            return setup_lora_vit(clip_model, r=lora_r, alpha=lora_alpha)
        except Exception as exc:
            LOGGER.warning(
                "PEFT LoRA setup failed (%s); falling back to PU.", exc
            )
            return setup_model_strategy(
                clip_model, "PU", text_features=text_features, model_name=model_name
            )
    if strategy == "Full-FT":
        _set_requires_grad(clip_model.visual, True)
        return clip_model
    raise ValueError(f"Unknown adaptation strategy: {strategy}")


def trainable_parameters(model: nn.Module):
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Selected strategy has no trainable parameters")
    return parameters
